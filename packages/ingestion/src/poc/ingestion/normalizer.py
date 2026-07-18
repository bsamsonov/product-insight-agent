from __future__ import annotations

import unicodedata

from fast_langdetect import detect
from poc.core.models import Document


def normalize(doc: Document) -> Document:
    """
    Normalize the raw text of a document, stripping unnecessary whitespace
    and normalizing its Unicode format. Additionally, detect and update the
    language of the document, if not already specified.
    """
    raw_text = unicodedata.normalize("NFC", doc.raw_text.strip())
    lang = doc.lang.strip() if doc.lang else ""

    if not lang:
        result = detect(raw_text[:500], low_memory=True)
        lang = result.get("lang", "").lower()

    return doc.model_copy(update={"raw_text": raw_text, "lang": lang})
