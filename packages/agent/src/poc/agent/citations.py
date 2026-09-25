"""Parse inline chunk citations out of generated answers."""

from __future__ import annotations

import re

_BRACKET_RE = re.compile(r"\[([^\]]+)\]")
_SEPARATOR_RE = re.compile(r"[,;\s]+")


def cited_chunk_ids(text: str) -> list[str]:
    """Return cited ids in order of first appearance, without duplicates.

    Models cite one chunk per bracket (``[a]``) or several at once (``[a, b; c]``); both
    forms are split into individual ids. Callers still check each id against the
    retrieved chunks, so non-citation brackets are harmless.
    """
    seen: set[str] = set()
    ids: list[str] = []
    for group in _BRACKET_RE.findall(text):
        for token in _SEPARATOR_RE.split(group):
            if token and token not in seen:
                seen.add(token)
                ids.append(token)
    return ids
