"""OCR engine interface and the Tesseract implementation (per-word confidence)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from PIL import Image, ImageOps


class OCRError(Exception):
    """OCR engine failed (missing binary, crash, unreadable image)."""


@dataclass
class OcrWord:
    text: str
    conf: float  # 0..100


@dataclass
class OcrPage:
    lines: list[list[OcrWord]] = field(default_factory=list)

    @property
    def words(self) -> list[OcrWord]:
        return [w for line in self.lines for w in line]

    @property
    def text(self) -> str:
        return "\n".join(" ".join(w.text for w in line) for line in self.lines)

    @property
    def mean_conf(self) -> float | None:
        ws = self.words
        return sum(w.conf for w in ws) / len(ws) if ws else None


class OCREngine(Protocol):
    def ocr(self, image: Image.Image) -> OcrPage: ...


def preprocess(image: Image.Image) -> Image.Image:
    """Light, information-preserving clean-up. No hard binarisation: that would turn a smudge
    into a confident wrong character instead of a low-confidence one."""
    img = ImageOps.exif_transpose(image).convert("L")
    if img.width < 1000:  # tiny images OCR badly
        scale = 1000 / img.width
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    return ImageOps.autocontrast(img, cutoff=1)


class TesseractOCR:
    def __init__(self, tesseract_cmd: str | None = None, lang: str = "eng"):
        import pytesseract

        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self._pt = pytesseract
        self.lang = lang

    def ocr(self, image: Image.Image) -> OcrPage:
        try:
            data = self._pt.image_to_data(preprocess(image), lang=self.lang, output_type=self._pt.Output.DICT)
        except self._pt.TesseractNotFoundError as exc:
            raise OCRError("tesseract binary not found; install it or set TESSERACT_CMD") from exc
        except Exception as exc:  # noqa: BLE001 - surface any engine crash as OCRError
            raise OCRError(f"tesseract failed: {exc}") from exc
        lines: dict[tuple[int, int, int], list[OcrWord]] = {}
        for i, text in enumerate(data["text"]):
            conf = float(data["conf"][i])
            if not text.strip() or conf < 0:
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            lines.setdefault(key, []).append(OcrWord(text.strip(), conf))
        return OcrPage([lines[k] for k in sorted(lines)])
