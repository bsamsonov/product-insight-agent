"""Helper toolkit for building & validating golden-set eval files.

Why this exists: a golden set is only useful if its `expected_answer_substrings`
actually occur in the corpus (so they can occur in a grounded answer) and its
`expected_chunk_ids` actually exist as indexed chunks. Both facts are *mechanical*
— they can be checked against `data/raw/reviews.jsonl` rather than guessed. This
script turns that mechanical work into commands so an author (human or a smaller
model) never has to invent grounded values from memory.

Chunk-id scheme (see packages/ingestion/.../chunker.py): a chunk id is
``f"{review_id}__c{position}"`` and a review id is ``f"{product_id}__{n}"``.
Most reviews are short (< target tokens) → exactly one chunk per review
(``..._c0``). This script uses the *real* chunker, so the chunk ids it prints are
exactly the ids the indexer puts into Qdrant.

Commands
--------
    uv run python scripts/golden_set_tools.py products [--top N] [--min-reviews K]
        List candidate products (most reviews first) with category, review count
        and the most frequent content words — your menu for questions + substrings.

    uv run python scripts/golden_set_tools.py product <PRODUCT_ID>
        Everything you need to write cases anchored to one product: its title,
        review count, frequent content words (substring candidates) and the full
        list of its chunk ids (the gold `expected_chunk_ids`).

    uv run python scripts/golden_set_tools.py terms <PRODUCT_ID>...
        Frequent content words shared across the given products (for a
        cross-product / thematic question).

    uv run python scripts/golden_set_tools.py draft [--out PATH] [--top N] [--min-reviews K]
        Cheapest path: auto-generate a complete, grounded draft set with ZERO LLM
        cost — one templated question per product, substrings from its frequent
        words, real chunk ids. Output is immediately valid; an LLM only needs to
        naturalize awkward phrasings afterwards. Default out: golden_set_v2.jsonl.

    uv run python scripts/golden_set_tools.py validate <golden_set.jsonl>
        Gate before shipping a set. Checks every case: substrings occur in the
        corpus (with counts), chunk ids exist, ids are unique, schema is intact.
        Exits non-zero if anything fails — wire it into your loop.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_REVIEWS = _PROJECT_ROOT / "data" / "raw" / "reviews.jsonl"

# Common English function words — never useful as grounded substrings.
_STOP_WORDS = (
    "the a an and or but if then else for to of in on at by with from into over under "
    "this that these those is are was were be been being have has had do does did will would "
    "can could should may might must not no yes it its it's i you he she we they them his her "
    "your my our their what which who whom when where why how all any some more most very much "
    "just like so than too also only out up down off about again very really get got use used "
    "using one two product great good well time even still back even make made buy bought "
    "item items thing things little bit lot well"
)
_STOP = set(_STOP_WORDS.split())

_WORD = re.compile(r"[a-z][a-z'-]{2,}")


def _iter_reviews() -> list[dict]:
    if not _REVIEWS.exists():
        sys.exit(f"corpus not found: {_REVIEWS} — run scripts/download_dataset.py first")
    rows = []
    with _REVIEWS.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _freq_terms(texts: list[str], k: int = 25) -> list[tuple[str, int]]:
    """Most frequent content words across `texts`, counted once per text.

    Counting once per text (document frequency) rather than raw frequency keeps a
    single ranty review from dominating — a good substring is one that *many*
    reviewers use, so it reliably shows up in a summarized answer.
    """
    df: Counter[str] = Counter()
    for t in texts:
        seen = {w for w in _WORD.findall(t.lower()) if w not in _STOP}
        df.update(seen)
    return df.most_common(k)


def _real_chunk_ids(reviews: list[dict]) -> list[str]:
    """Exact chunk ids the indexer would produce for these reviews."""
    from poc.core.models import Document
    from poc.ingestion.chunker import RecursiveTokenChunker
    from poc.ingestion.normalizer import normalize

    chunker = RecursiveTokenChunker()
    ids: list[str] = []
    for r in reviews:
        doc = Document(
            id=r["id"],
            source=r.get("source", ""),
            lang=r.get("lang", "en"),
            raw_text=r["raw_text"],
            metadata=r.get("metadata", {}),
        )
        ids.extend(c.id for c in chunker.chunk(normalize(doc)))
    return ids


def cmd_products(top: int, min_reviews: int) -> None:
    rows = _iter_reviews()
    by_pid: dict[str, list[dict]] = {}
    for r in rows:
        by_pid.setdefault(r["product_id"], []).append(r)
    ranked = sorted(by_pid.items(), key=lambda kv: len(kv[1]), reverse=True)
    shown = 0
    for pid, revs in ranked:
        if len(revs) < min_reviews:
            continue
        m = revs[0]["metadata"]
        terms = ", ".join(w for w, _ in _freq_terms([r["raw_text"] for r in revs], 12))
        print(f"\n{pid}  [{len(revs)} reviews]  {m.get('main_category')}")
        print(f"  title: {m.get('product_title', '')[:90]}")
        print(f"  terms: {terms}")
        shown += 1
        if shown >= top:
            break


def cmd_product(pid: str) -> None:
    rows = [r for r in _iter_reviews() if r["product_id"] == pid]
    if not rows:
        sys.exit(f"no reviews for product_id={pid}")
    m = rows[0]["metadata"]
    print(f"product_id : {pid}")
    print(f"title      : {m.get('product_title', '')}")
    print(f"category   : {m.get('main_category')}")
    print(f"reviews    : {len(rows)}")
    print("\nfrequent content words (substring candidates, df = #reviews using it):")
    for w, c in _freq_terms([r["raw_text"] for r in rows], 25):
        print(f"  {c:3d}  {w}")
    ids = _real_chunk_ids(rows)
    print(f"\nexpected_chunk_ids ({len(ids)} chunks) — JSON array:")
    print(json.dumps(ids))


def cmd_terms(pids: list[str]) -> None:
    rows = _iter_reviews()
    wanted = set(pids)
    texts = [r["raw_text"] for r in rows if r["product_id"] in wanted]
    if not texts:
        sys.exit(f"no reviews for products={pids}")
    print(f"shared frequent words across {len(pids)} products ({len(texts)} reviews):")
    for w, c in _freq_terms(texts, 30):
        print(f"  {c:3d}  {w}")


# Sentiment/filler words: fine as substrings (they occur) but make awkward question
# subjects, so they are kept out of the question's aspect slot.
_FILLER = {
    "great",
    "good",
    "love",
    "nice",
    "perfect",
    "awesome",
    "happy",
    "really",
    "little",
    "pretty",
    "definitely",
    "overall",
    "well",
    "stuff",
    "far",
    "before",
    "because",
    "other",
    "seems",
    "expected",
    "i've",
}

# (category, question template, intent words to prefer as substrings when grounded).
# Templates only use {a} (top aspect term) and {title} (cleaned product title), so any
# product fits. Intent words are included as substrings only if they occur in *that*
# product's reviews, keeping every substring grounded.
_TEMPLATES = [
    ("product_feature", "What do reviewers say about the {a} of the {title}?", []),
    (
        "product_quality",
        "How do customers describe the quality and {a} of the {title}?",
        ["quality", "durable", "sturdy", "material"],
    ),
    (
        "product_issue",
        "Are there complaints or problems mentioned about the {title}?",
        ["cheap", "flimsy", "broke", "break", "small", "plastic", "disappointed", "return"],
    ),
    (
        "product_value",
        "Do reviewers feel the {title} is good value for the price?",
        ["price", "value", "worth", "money", "cheap"],
    ),
    (
        "product_use",
        "What do customers say about using the {title}?",
        ["easy", "works", "fit", "comfortable", "use"],
    ),
]


def _short_title(title: str) -> str:
    """Trim a marketing title to a short, natural noun phrase for a question."""
    for sep in [" - ", " | ", " \u2013 ", ","]:
        if sep in title:
            title = title.split(sep)[0]
    words = title.split()
    return " ".join(words[:6]).strip()


def _pick_subs(
    term_list: list[str], term_set: set[str], intent: list[str], n: int = 3
) -> list[str]:
    """Choose grounded substrings: intent words that occur, then top content terms."""
    chosen: list[str] = [w for w in intent if w in term_set]
    for w in term_list:
        if len(chosen) >= n:
            break
        if w not in chosen and w not in _FILLER:
            chosen.append(w)
    # Final fallback: allow filler if a product's vocabulary is tiny.
    for w in term_list:
        if len(chosen) >= 2:
            break
        if w not in chosen:
            chosen.append(w)
    return chosen[:n]


def cmd_draft(out: Path, top: int, min_reviews: int) -> None:
    """Emit a grounded draft golden set: one case per product, templates rotated.

    Zero LLM cost — questions are templated, substrings come from each product's
    frequent words, chunk ids come from the real chunker. The result is immediately
    valid (run `validate`); an LLM only needs to naturalize phrasing afterwards.
    """
    rows = _iter_reviews()
    by_pid: dict[str, list[dict]] = {}
    for r in rows:
        by_pid.setdefault(r["product_id"], []).append(r)
    ranked = sorted(by_pid.items(), key=lambda kv: len(kv[1]), reverse=True)

    lines: list[str] = []
    n = 0
    for pid, revs in ranked:
        if len(revs) < min_reviews:
            continue
        if n >= top:
            break
        term_pairs = _freq_terms([r["raw_text"] for r in revs], 25)
        term_list = [w for w, _ in term_pairs]
        term_set = set(term_list)
        aspect = next((w for w in term_list if w not in _FILLER), term_list[0] if term_list else "")
        if not aspect:
            continue
        category, template, intent = _TEMPLATES[n % len(_TEMPLATES)]
        title = _short_title(revs[0]["metadata"].get("product_title", "this product"))
        question = template.format(a=aspect, title=title)
        subs = _pick_subs(term_list, term_set, intent)
        case = {
            "id": f"q{n + 1:03d}",
            "question": question,
            "expected_answer_substrings": subs,
            "expected_chunk_ids": _real_chunk_ids(revs),
            "filters": {},
            "metadata": {"category": category, "product_id": pid},
        }
        lines.append(json.dumps(case, ensure_ascii=False))
        n += 1

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {n} draft cases → {out}")
    print(f"next: uv run python scripts/golden_set_tools.py validate {out}")


def cmd_validate(path: Path) -> None:
    if not path.exists() and not path.suffix:
        path = _PROJECT_ROOT / "packages" / "evals" / "data" / f"{path.name}.jsonl"
    if not path.exists():
        sys.exit(f"golden set not found: {path}")

    rows = _iter_reviews()
    corpus_text = "\n".join(r["raw_text"] for r in rows).lower()
    valid_chunk_ids = set(_real_chunk_ids(rows))

    cases = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    errors: list[str] = []
    warnings: list[str] = []
    seen_ids: set[str] = set()

    required = {"id", "question", "expected_answer_substrings", "expected_chunk_ids"}
    for i, c in enumerate(cases):
        tag = c.get("id", f"#{i}")
        missing = required - c.keys()
        if missing:
            errors.append(f"[{tag}] missing fields: {sorted(missing)}")
            continue
        if c["id"] in seen_ids:
            errors.append(f"[{tag}] duplicate id")
        seen_ids.add(c["id"])

        subs = c["expected_answer_substrings"]
        if not subs:
            warnings.append(f"[{tag}] no expected_answer_substrings")
        for s in subs:
            n = corpus_text.count(s.lower())
            if n == 0:
                errors.append(f"[{tag}] substring not in corpus: {s!r}")
            elif n < 3:
                warnings.append(f"[{tag}] substring rare ({n}x): {s!r}")

        chunks = c["expected_chunk_ids"]
        if not chunks:
            warnings.append(f"[{tag}] no expected_chunk_ids (citation_precision will be 0)")
        for cid in chunks:
            if cid not in valid_chunk_ids:
                errors.append(f"[{tag}] chunk id not in corpus: {cid!r}")

    print(f"validated {len(cases)} cases against {len(rows)} reviews")
    for w in warnings:
        print(f"  WARN  {w}")
    for e in errors:
        print(f"  FAIL  {e}")
    if errors:
        sys.exit(f"\n{len(errors)} error(s), {len(warnings)} warning(s) — NOT shippable")
    print(f"\nOK — 0 errors, {len(warnings)} warning(s)")


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)
    cmd, rest = args[0], args[1:]
    if cmd == "products":
        top = int(_opt(rest, "--top", "30"))
        min_reviews = int(_opt(rest, "--min-reviews", "20"))
        cmd_products(top, min_reviews)
    elif cmd == "product":
        cmd_product(rest[0])
    elif cmd == "terms":
        cmd_terms(rest)
    elif cmd == "draft":
        default_out = _PROJECT_ROOT / "packages" / "evals" / "data" / "golden_set_v2.jsonl"
        out = Path(_opt(rest, "--out", str(default_out)))
        top = int(_opt(rest, "--top", "35"))
        min_reviews = int(_opt(rest, "--min-reviews", "40"))
        cmd_draft(out, top, min_reviews)
    elif cmd == "validate":
        cmd_validate(Path(rest[0]))
    else:
        sys.exit(f"unknown command: {cmd}\n{__doc__}")


def _opt(rest: list[str], name: str, default: str) -> str:
    return rest[rest.index(name) + 1] if name in rest else default


if __name__ == "__main__":
    main()
