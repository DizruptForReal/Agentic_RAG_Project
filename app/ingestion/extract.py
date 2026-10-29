"""Targeted field extraction from a document (the `extract_only` path: no embedding, no indexing).

The LLM proposes values plus the verbatim source text. Code then decides trust:
* source text not found on the stated page          -> UNVERIFIED (ungrounded extraction)
* page is OCR and any matching word is low-confidence -> UNVERIFIED, with the minimum confidence
* otherwise                                          -> VERIFIED
"""
from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

from app.agent.models import EvidenceItem, EvidenceKind, Verification
from app.ingestion.chunker import clean_text
from app.llm.base import LLMClient, call_structured


class ExtractedField(BaseModel):
    name: str
    found: bool
    value: str | None = None
    page: int | None = None
    source_text: str | None = None


class ExtractionResult(BaseModel):
    fields: list[ExtractedField]


EXTRACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "fields": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "found": {"type": "boolean"},
                    "value": {"type": ["string", "null"]},
                    "page": {"type": ["integer", "null"]},
                    "source_text": {"type": ["string", "null"],
                                    "description": "Exact text copied from the page that contains the value."},
                },
                "required": ["name", "found"],
            },
        }
    },
    "required": ["fields"],
}

EXTRACT_SYSTEM = """You extract specific fields from a document's pages.
Rules:
- Only report what is written on the pages. Never infer, complete or correct values.
- Words wrapped like [[?word]] were read by OCR with LOW confidence. Copy them as they appear in
  value and source_text; do not "fix" them.
- If a field is not present, or is only partially legible, set found=false or copy the partial value.
- value: exactly as written on the page (no reformatting). source_text: the short line containing it (one line
  is enough). Always give the page number.
"""

_TOKEN = re.compile(r"[A-Za-z0-9]+")


def _norm(s: str) -> str:
    return " ".join(clean_text(s).lower().split())


def _key(word: str) -> str:
    return "".join(_TOKEN.findall(clean_text(word).lower()))


def _aligned_confs(source_text: str, words: list[tuple[str, float | None]]) -> list[float]:
    """Confidences of the OCR words that form the source text (contiguous match)."""
    target = [k for k in (_key(w) for w in source_text.split()) if k]
    keyed = [(k, c) for k, c in ((_key(w), c) for w, c in words) if k]
    n = len(target)
    for i in range(len(keyed) - n + 1):
        if [k for k, _ in keyed[i:i + n]] == target:
            return [c for _, c in keyed[i:i + n] if c is not None]
    return []


def ground_field(field: ExtractedField, pages: list[dict[str, Any]], threshold: float) -> tuple[Verification, float | None, str]:
    """Return (verification, min_ocr_conf, note) for one extracted field, and set `field.page`.

    Trust comes from the OCR words, never from the model: the value must be spelled by words on a page of the
    document, each read with confidence >= threshold (see app/ingestion/ocr_verify.py). The model's quote and
    page number are only hints for finding the right occurrence; a quote with copied [[?..]] markers, a page
    reference or a missing page number no longer turns a clearly read value into UNVERIFIED."""
    from app.ingestion.ocr_verify import strip_markers, verify_value

    value = strip_markers(field.value or "").strip()
    if not value:
        return Verification.UNVERIFIED, None, "no value given"
    quote = strip_markers(field.source_text or "")
    if quote.strip() and _norm(quote) and any(_norm(quote) in _norm(strip_markers(p["text"])) for p in pages) and \
            not set(_TOKEN.findall(_norm(value))) <= set(_TOKEN.findall(_norm(quote))):
        return Verification.UNVERIFIED, None, "value is not part of its quoted source text"
    verification, conf, note, page = verify_value(value, pages, threshold, quote=quote, page_hint=field.page)
    if page is not None:
        field.page = page
    return verification, conf, note


class FieldExtractor:
    def __init__(self, llm: LLMClient, conf_threshold: float, model: str | None = None):
        self.llm, self.threshold, self.model = llm, conf_threshold, model

    def extract(self, doc: dict[str, Any], pages: list[dict[str, Any]], fields: list[str]
                ) -> tuple[list[EvidenceItem], list[str], list[str]]:
        """Returns (evidence, not_found_fields, errors)."""
        body = "\n\n".join(f"--- page {p['page']} ({p['method']}) ---\n{p['marked_text']}" for p in pages)
        user = f"Document: {doc['title'] or doc['filename']}\nFields to extract: {fields}\n\n{body}"
        res = call_structured(self.llm, purpose="extract_fields", system=EXTRACT_SYSTEM, user=user,
                              schema=EXTRACT_SCHEMA, model_cls=ExtractionResult, model=self.model)
        if res.value is None:
            return [], [], res.errors or ["extraction failed"]
        evidence, not_found = [], []
        for f in res.value.fields:
            if not f.found or not f.value:
                not_found.append(f.name)
                continue
            verification, conf, note = ground_field(f, pages, self.threshold)
            flag = "" if verification == Verification.VERIFIED else f" [UNVERIFIED: {note}]"
            evidence.append(EvidenceItem(
                kind=EvidenceKind.FIELD, doc_id=doc["doc_id"], doc_name=doc["title"] or doc["filename"],
                page=f.page, field_name=f.name, verification=verification, ocr_confidence=conf,
                text=f"{f.name} = {f.value!r} (source: {f.source_text!r}){flag}",
            ))
        missing = [n for n in fields if n not in {f.name for f in res.value.fields}]
        return evidence, not_found + missing, []
