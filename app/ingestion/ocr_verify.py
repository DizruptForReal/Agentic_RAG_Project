"""Deciding in code whether a value read from a scanned page can be trusted.

The rule is word-level: a value is VERIFIED when the OCR words that spell it are on the page and every one of them
was read with confidence >= the threshold. Words elsewhere on the page do not matter (a smudged signature does
not make a clearly printed "COMMERCIAL" uncertain), and nothing depends on the model quoting text perfectly:
the model's quote is only used, when it can be found, to pick the right occurrence of a value.

Two entry points:
* verify_value(value, pages, ...)       - a field the extractor read (receipt date, usage classification)
* check_claim(claim, marked_passage)     - a sub-question that cites a scanned passage: are the words the claim
                                            rests on legible, and is no value in the claim missing or uncertain?
"""
from __future__ import annotations

import re
from typing import Any

from app.agent.models import Verification
from app.ingestion.chunker import clean_text

_TOKEN = re.compile(r"[A-Za-z0-9]+")
_MARK = re.compile(r"\[\[\?([^\]]*)\]\]")
_PAGE_REF = re.compile(r"\(\s*page\s*\d+\s*\)", re.I)
STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "unit", "units", "says", "shows", "show", "states", "state",
    "declares", "declared", "lists", "listed", "page", "document", "receipt", "date", "value", "according", "per",
    "is", "are", "was", "were", "has", "have", "been", "its", "their", "there", "which", "under", "into", "onto",
}


def key(word: str) -> str:
    """Comparable form of one word: lower-case letters and digits only ('05-03-2025' -> '05032025')."""
    return "".join(_TOKEN.findall(clean_text(word).lower()))


def strip_markers(text: str) -> str:
    """Remove OCR uncertainty markers and page references the model may have copied into a quote."""
    return _PAGE_REF.sub(" ", _MARK.sub(lambda m: m.group(1), text or ""))


def _keyed(words: list[tuple[str, float | None]]) -> list[tuple[str, float | None]]:
    return [(k, c) for k, c in ((key(w), c) for w, c in words) if k]


def _find(seq: list[str], keyed: list[tuple[str, float | None]]) -> list[int]:
    n = len(seq)
    return [i for i in range(len(keyed) - n + 1) if [k for k, _ in keyed[i:i + n]] == seq] if n else []


def verify_value(value: str, pages: list[dict[str, Any]], threshold: float, quote: str | None = None,
                 page_hint: int | None = None) -> tuple[Verification, float | None, str, int | None]:
    """Returns (verification, lowest confidence of the value's words, note, page where it was found)."""
    vkeys = [k for k in (key(w) for w in strip_markers(value).split()) if k]
    if not vkeys:
        return Verification.UNVERIFIED, None, "empty value", None
    qkeys = [k for k in (key(w) for w in strip_markers(quote or "").split()) if k]
    ordered = sorted(pages, key=lambda p: p["page"] != page_hint)   # the model's page first, others after
    candidates: list[tuple[int, list[float], bool]] = []            # (page, confidences, inside the quote)
    for p in ordered:
        if p["method"] != "ocr" or not p.get("words"):
            pkeys = [key(w) for w in strip_markers(p["text"]).split()]
            if _find(vkeys, [(k, None) for k in pkeys if k]):
                return Verification.VERIFIED, None, "native text layer", p["page"]
            continue
        keyed = _keyed(p["words"])
        quote_spans = [(i, i + len(qkeys)) for i in _find(qkeys, keyed)] if len(qkeys) > len(vkeys) else []
        for i in _find(vkeys, keyed):
            confs = [c for _, c in keyed[i:i + len(vkeys)] if c is not None]
            inside = any(a <= i and i + len(vkeys) <= b for a, b in quote_spans)
            candidates.append((p["page"], confs, inside))
    if not candidates:
        return Verification.UNVERIFIED, None, "value not found on any page", None
    # Prefer the occurrence inside the model's quote; a very short value (e.g. "1") must be located that way.
    inside = [c for c in candidates if c[2]]
    if not inside and len("".join(vkeys)) < 3:
        return Verification.UNVERIFIED, None, "value too short to locate without its quote", None
    pool = inside or candidates
    clean = [c for c in pool if c[1] and min(c[1]) >= threshold]
    if clean:
        page, confs, _ = max(clean, key=lambda c: min(c[1]))
        return Verification.VERIFIED, min(confs), f"OCR confidence {min(confs):.0f}", page
    page, confs, _ = max(pool, key=lambda c: min(c[1]) if c[1] else -1)
    low = min(confs) if confs else None
    return (Verification.UNVERIFIED, low,
            f"OCR confidence {low:.0f} below {threshold:.0f}" if low is not None else "no OCR confidence", page)


def passage_words(marked_text: str) -> tuple[set[str], set[str]]:
    """(clearly read word keys, uncertain word keys) of a passage whose uncertain words are marked [[?w]]."""
    low = {key(m) for m in _MARK.findall(marked_text) if key(m)}
    rest = _MARK.sub(" ", marked_text)
    high = {key(w) for w in rest.split() if key(w)}
    return high, low


def _value_like(word: str) -> bool:
    """Words that carry a value in a claim: anything with a digit, or a code/label written in capitals."""
    w = clean_text(word).strip(".,;:()[]'\"")
    return any(ch.isdigit() for ch in w) or (len(w) >= 3 and w.isalpha() and w.isupper())


def check_claim(claim: str, marked_text: str) -> tuple[bool, str]:
    """Does a scanned passage legibly support this claim? True only if
      * every value-like word of the claim (dates, numbers, codes, CAPITALISED labels) is a clearly read word of
        the passage - a value that is absent (the model 'fixed' a digit) or uncertain fails;
      * no word of the claim is one of the passage's uncertain words;
      * the claim shares at least one meaningful word with the passage.
    Word-level: '2025' does not match inside 'R-2025-04417'."""
    words = [w for w in strip_markers(claim).split() if key(w)]
    if not words:
        return False, "no claim text to check"
    high, low = passage_words(marked_text)
    ck = {key(w) for w in words}
    if ck & low:
        return False, f"claim rests on uncertain OCR words: {', '.join(sorted(ck & low))}"
    values = [w for w in words if _value_like(w)]
    missing = [w for w in values if key(w) not in high]
    if missing:
        return False, f"claim values not clearly on the page: {', '.join(missing)}"
    meaningful = {key(w) for w in words if len(key(w)) >= 4 and key(w) not in STOPWORDS} | {key(w) for w in values}
    shared = meaningful & high
    if not shared:
        return False, "claim shares no clearly read word with the passage"
    return True, f"claim words read clearly: {', '.join(sorted(shared))}"
