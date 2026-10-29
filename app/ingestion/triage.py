"""Decide, per page, whether native text is usable or OCR is needed (avoid unnecessary OCR)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass
class PageTriage:
    page: int            # 1-based
    method: str          # "native" | "ocr"
    native_chars: int

    def to_dict(self) -> dict:
        return asdict(self)


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXT


def triage_file(path: Path, min_chars: int) -> list[PageTriage]:
    path = Path(path)
    if is_image(path):
        return [PageTriage(page=1, method="ocr", native_chars=0)]
    if path.suffix.lower() == ".txt":
        return [PageTriage(page=1, method="native", native_chars=len(path.read_text(encoding="utf-8")))]
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"unsupported file type: {path.suffix}")
    try:
        import pymupdf as fitz
    except ImportError:  # older PyMuPDF
        import fitz

    out = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            chars = len(page.get_text("text").strip())
            out.append(PageTriage(page=i, method="native" if chars >= min_chars else "ocr", native_chars=chars))
    return out
