"""Page-aware chunking with deterministic ids and OCR-confidence tracking.

Low-confidence OCR words are marked inline as ``[[?word]]`` so the LLM can see exactly which
tokens are uncertain. ``clean_text`` removes the markers for embedding and grounding checks.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

MARK_RE = re.compile(r"\[\[\?(.*?)\]\]")


def mark_word(word: str, conf: float | None, threshold: float) -> str:
    return f"[[?{word}]]" if conf is not None and conf < threshold else word


def clean_text(text: str) -> str:
    return MARK_RE.sub(r"\1", text)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    page: int
    idx: int
    text: str            # with [[?..]] markers
    min_conf: float | None
    verification: str    # VERIFIED | UNVERIFIED

    def to_dict(self) -> dict:
        return asdict(self)


def chunk_page(doc_id: str, page: int, words: list[tuple[str, float | None]], *, size: int, overlap: int,
               conf_threshold: float) -> list[Chunk]:
    if not words:
        return []
    step = max(1, size - overlap)
    chunks: list[Chunk] = []
    start, idx = 0, 0
    while start < len(words):
        window = words[start:start + size]
        confs = [c for _, c in window if c is not None]
        min_conf = min(confs) if confs else None
        unverified = min_conf is not None and min_conf < conf_threshold
        text = " ".join(mark_word(w, c, conf_threshold) for w, c in window)
        chunks.append(Chunk(
            chunk_id=f"{doc_id}:{page}:{idx}", doc_id=doc_id, page=page, idx=idx, text=text,
            min_conf=min_conf, verification="UNVERIFIED" if unverified else "VERIFIED",
        ))
        if start + size >= len(words):
            break
        start += step
        idx += 1
    return chunks


# ---------------------------------------------------------------------------------- section-aware
NUMBERED_HEADING_RE = re.compile(r"^\d{1,2}(?:\.\d+)?\.?\s+[A-Z][^.:;]{1,70}$")   # "7. Claims procedure"
SCHEDULE_HEADING_RE = re.compile(r"^schedule\s+[a-z]\b.{0,70}$", re.I)            # "Schedule B - ..."


def is_heading(line: str) -> bool:
    line = clean_text(line).strip()
    return bool(line) and len(line.split()) <= 10 and bool(
        NUMBERED_HEADING_RE.match(line) or SCHEDULE_HEADING_RE.match(line))


def _page_lines(page: dict) -> list[list[tuple[str, float | None]]]:
    """Re-attach per-word confidences to the page's line structure (words are stored in reading order)."""
    words = list(page["words"])
    out, k = [], 0
    for line in page["text"].split("\n"):
        toks = line.split()
        if not toks:
            continue
        seg = []
        for t in toks:
            conf = words[k][1] if k < len(words) else None
            seg.append((t, conf))
            k += 1
        out.append(seg)
    return out


def chunk_document(doc_id: str, pages: list[dict], *, size: int, overlap: int, conf_threshold: float,
                   title: str = "") -> list[dict]:
    """Split at section headings (carried across pages), then window long sections.
    Each chunk keeps its page and gets a `section` label used for retrieval context."""
    chunks: list[dict] = []
    heading = title or "Introduction"
    for page in pages:
        segments: list[tuple[str, list[tuple[str, float | None]]]] = []
        current: list[tuple[str, float | None]] = []
        for line in _page_lines(page):
            text = " ".join(w for w, _ in line)
            if is_heading(text):
                if current:
                    segments.append((heading, current))
                heading, current = clean_text(text).strip(), list(line)
            else:
                current.extend(line)
        if current:
            segments.append((heading, current))
        idx = 0
        for section, words in segments:
            for c in chunk_page(doc_id, page["page"], words, size=size, overlap=overlap,
                                conf_threshold=conf_threshold):
                d = c.to_dict()
                d["idx"], d["chunk_id"], d["section"] = idx, f"{doc_id}:{page['page']}:{idx}", section
                chunks.append(d)
                idx += 1
    return chunks


def context_text(chunk: dict) -> str:
    """Text used for embedding and reranking: document and section give the passage its context."""
    head = " - ".join(x for x in (chunk.get("doc_name"), chunk.get("section")) if x)
    return f"{head}\n{clean_text(chunk['text'])}" if head else clean_text(chunk["text"])
