#!/usr/bin/env python3
"""Download McAuley-Lab/Amazon-Reviews-2023 → data/raw/reviews.jsonl.

Why this dataset (not mteb/amazon_reviews_multi):
    The MTEB variant is a *sentiment-classification* benchmark — it strips
    product_id, product name and timestamp, leaving only {text, star-label}.
    That makes the per-product / per-period analysis the agent is built around
    (S2: plan→retrieve→cluster→summarize per product) impossible. The McAuley
    2023 dataset keeps `parent_asin` (product_id), `timestamp` and links to a
    metadata table with product titles — exactly what we need.

Sampling strategy (important):
    A category has 100k+ products. Sampling reviews uniformly would yield
    mostly one-review products → nothing to cluster. Instead we pick a set of
    *popular* products (by `rating_number` from the meta table) and cap reviews
    per product, producing dense per-product clusters suitable for the agent.

Domain (Sports & Outdoors gear):
    We pull the whole Sports_and_Outdoors category — a broad mix of athletic
    gear (footwear, apparel, accessories, fitness equipment, outdoor kit). No
    product-type filter is applied: keyword filtering on this category is
    unreliable (e.g. "shoe laces", "ice cleats", "boot socks" all read as
    footwear), so the agent's domain is framed as general sports & outdoors
    gear rather than footwear. `_select_products` keeps the first popular
    products by rating_number, whatever their type.

Usage:
    uv run python scripts/download_dataset.py
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import fsspec

DATASET_NAME = "McAuley-Lab/Amazon-Reviews-2023"
# Sports & Outdoors gear domain — broad athletic/outdoor category, no
# product-type filter (see module docstring). Swap freely (e.g.
# Clothing_Shoes_and_Jewelry, Amazon_Fashion).
CATEGORY = "Sports_and_Outdoors"

TARGET_TOTAL = 12_000  # ≥ 10k requirement (S1.T2 AC)
MIN_PRODUCT_RATINGS = 50  # only "popular" products → reviews actually cluster
MAX_PRODUCTS = 600  # cap selected products (keeps meta dict small)
MAX_REVIEWS_PER_PRODUCT = 60  # spread reviews across products, avoid 1-product domination

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUTPUT_FILE = DATA_DIR / "reviews.jsonl"

# `datasets` >= 4 dropped loading scripts, and this repo ships one, so we read
# the per-category JSONL files directly off the Hub filesystem instead. These
# `raw/` paths exist for every category (unlike the partial parquet exports).
_HF_PREFIX = f"hf://datasets/{DATASET_NAME}"
META_FILE = f"{_HF_PREFIX}/raw/meta_categories/meta_{CATEGORY}.jsonl"
REVIEW_FILE = f"{_HF_PREFIX}/raw/review_categories/{CATEGORY}.jsonl"


def _stream(data_file: str) -> Iterator[dict]:
    """Stream a JSONL file off the Hub line-by-line, one parsed record at a time.

    We deliberately bypass `datasets.load_dataset("json", ...)`: its Arrow JSON
    builder infers a column's type from the first rows and then fails to cast
    later rows when Amazon's metadata is heterogeneous/sparse — e.g. a field
    seen as `null` early on (inferred null-type) later holds a struct, raising
    `Couldn't cast struct<...> to null`. Plain `json.loads` per line has no such
    schema, handles any shape, and stays O(1) in memory.
    """
    with fsspec.open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _select_products() -> dict[str, dict]:
    """Pick popular products from the meta table → {parent_asin: meta-subset}."""
    print(
        f"[meta] selecting up to {MAX_PRODUCTS} popular products "
        f"with >={MIN_PRODUCT_RATINGS} ratings ...",
        flush=True,
    )
    selected: dict[str, dict] = {}
    for rec in _stream(META_FILE):
        asin = rec.get("parent_asin")
        title = (rec.get("title") or "").strip()
        ratings = rec.get("rating_number") or 0
        if not asin or not title or ratings < MIN_PRODUCT_RATINGS:
            continue
        selected[asin] = {
            "product_title": title,
            "main_category": rec.get("main_category"),
            "product_avg_rating": rec.get("average_rating"),
            "product_rating_count": ratings,
        }
        if len(selected) >= MAX_PRODUCTS:
            break
    print(f"[meta] {len(selected)} products selected", flush=True)
    return selected


def _to_iso(timestamp_ms: int | None) -> str | None:
    """Amazon timestamps are ms since epoch -> ISO-8601 UTC (pydantic-parseable)."""
    if not timestamp_ms:
        return None
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC).isoformat()


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    products = _select_products()
    if not products:
        raise SystemExit(f"No products with >={MIN_PRODUCT_RATINGS} ratings in {CATEGORY}")

    per_product: dict[str, int] = {}
    total = 0
    print(f"[reviews] streaming {CATEGORY} reviews -> {TARGET_TOTAL} docs ...", flush=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as out:
        for rec in _stream(REVIEW_FILE):
            asin = rec.get("parent_asin")
            meta = products.get(asin)
            if meta is None:
                continue
            if per_product.get(asin, 0) >= MAX_REVIEWS_PER_PRODUCT:
                continue
            text = (rec.get("text") or "").strip()
            if not text:
                continue

            idx = per_product.get(asin, 0)
            title = (rec.get("title") or "").strip()
            doc = {
                "id": f"{asin}__{idx}",
                "source": f"{DATASET_NAME}/{CATEGORY}",
                "lang": "en",
                "product_id": asin,
                "region": "US",
                "created_at": _to_iso(rec.get("timestamp")),
                "raw_text": f"{title}\n\n{text}" if title else text,
                "metadata": {
                    "rating": rec.get("rating"),
                    "helpful_vote": rec.get("helpful_vote"),
                    "verified_purchase": rec.get("verified_purchase"),
                    **meta,
                },
            }
            out.write(json.dumps(doc, ensure_ascii=False) + "\n")
            per_product[asin] = idx + 1
            total += 1
            if total % 2_000 == 0:
                print(f"[reviews] {total} written ...", flush=True)
            if total >= TARGET_TOTAL:
                break

    products_covered = len(per_product)
    print(f"\nDone -- {total} reviews across {products_covered} products -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
    sys.exit(0)
