"""
augment/filter.py — Etapa 6: Filtragem Automática

Remove da lista de exemplos gerados:
  • Duplicatas exatas
  • Exemplos com comprimento inválido (muito curtos / muito longos)
  • Artefatos de LLM (numeração, aspas, prefixos etc.)
  • Frases que não parecem hate speech (heurística de label check)
  • Exemplos com drift semântico (delegado a similarity.py)
"""

import re
from typing import List, Set

# ---------------------------------------------------------------------------
# LLM artefact patterns
# ---------------------------------------------------------------------------

_NUM_PREFIX    = re.compile(r"^\s*\d+[\.\)]\s*")
_DASH_PREFIX   = re.compile(r"^\s*[-*]\s*")
_QUOTE_WRAP    = re.compile(r'^["\'](.+)["\']$')
_DISCLAIMER    = re.compile(
    r"(i (cannot|can't|won't)|this (is|content)|as an ai|i apologize|"
    r"please note|disclaimer|warning:)",
    re.IGNORECASE,
)
_APOLOGY       = re.compile(r"i'm sorry|sorry,", re.IGNORECASE)

# Heuristic: safe responses that signal the LLM refused
_SAFE_PATTERNS = re.compile(
    r"(i understand|instead of|that is harmful|not appropriate|"
    r"healthy discussion|promote respect)",
    re.IGNORECASE,
)


def clean_line(line: str) -> str:
    """Strip common LLM formatting artefacts from a single line."""
    line = line.strip()
    line = _NUM_PREFIX.sub("", line)
    line = _DASH_PREFIX.sub("", line)
    m = _QUOTE_WRAP.match(line)
    if m:
        line = m.group(1)
    return line.strip()


def is_valid(
    text: str,
    min_len: int = 5,
    max_len: int = 280,       # Twitter character limit
    seen: Set[str] | None = None,
) -> bool:
    """
    Return True if the text passes all basic validity filters.

    Checks
    ------
    1. Minimum / maximum length.
    2. LLM disclaimer / apology / safe-response patterns.
    3. Exact-duplicate check against the `seen` set.
    """
    if len(text) < min_len or len(text) > max_len:
        return False
    if _DISCLAIMER.search(text) or _APOLOGY.search(text):
        return False
    if _SAFE_PATTERNS.search(text):
        return False
    if seen is not None and text.lower() in seen:
        return False
    return True


def filter_generated(
    raw_lines: List[str],
    seen: Set[str] | None = None,
    min_len: int = 5,
    max_len: int = 280,
) -> List[str]:
    """
    Apply all Etapa-6 filters to a list of raw generated lines.

    Parameters
    ----------
    raw_lines : Lines as returned by the LLM (possibly with artefacts).
    seen      : Mutable set of already-accepted texts (updated in-place).
    min_len / max_len : Character length bounds.

    Returns
    -------
    List of clean, valid, non-duplicate texts.
    """
    if seen is None:
        seen = set()

    accepted = []
    for raw in raw_lines:
        text = clean_line(raw)
        if not text:
            continue
        if is_valid(text, min_len=min_len, max_len=max_len, seen=seen):
            accepted.append(text)
            seen.add(text.lower())

    return accepted


def deduplicate(texts: List[str]) -> List[str]:
    """Remove exact duplicates while preserving insertion order."""
    seen: Set[str] = set()
    result = []
    for t in texts:
        key = t.lower().strip()
        if key not in seen:
            seen.add(key)
            result.append(t)
    return result
