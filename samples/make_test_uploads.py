"""Generate the upload files used by the behaviour test pack (docs/behaviour_test_pack.md).

    python -m samples.make_test_uploads            # writes samples/test_uploads/
    python -m samples.make_test_uploads --no-check # skip the Tesseract self-check

Each file exists to trigger one behaviour. The self-check OCRs the image files and asserts that the words
each test depends on really are high (or low) confidence, so a file cannot silently stop testing what it is
meant to test. Ground truth (see make_samples.py): the Acme X200 (AQ-X200-55871) is COMMERCIAL, so units
bought from 1 Jan 2025 get 12 months; labour only for the first 6 months. The failure date in the main
test question is 14 March 2026.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageFilter
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from samples.make_samples import BODY, H1, H2, OCR_THRESHOLD, _degrade, _ocr_words, _render_lines, _word_after

OUT = Path(__file__).parent / "test_uploads"


def _receipt_lines(date: str, receipt_no: str, serial: str = "AQ-X200-55871", item: str = "Aquaflow X200 Pump",
                   customer: str = "Acme Facilities Ltd", amount: str = "GBP 4,850.00") -> list[str]:
    return ["AQUAFLOW PUMPS LTD", "SALES RECEIPT", "", f"Receipt No: {receipt_no}", f"Date: {date}",
            f"Customer: {customer}", f"Item: {item}", f"Serial: {serial}", "Qty: 1", f"Amount: {amount}",
            "Paid: Bank transfer"]


def _image(lines: list[str], seed: int, size=(1000, 900), angle: float = 0.4) -> Image.Image:
    img, *_ = _render_lines(lines, size=size, font_size=34, top=60, left=90, gap=64)
    return _degrade(img, seed=seed, angle=angle)


def _pdf(path: Path, flow: list) -> None:
    SimpleDocTemplate(str(path), pagesize=A4, title=path.stem, author="Aquaflow (fictional)",
                      leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm).build(flow)


def _table(rows):
    t = Table(rows, colWidths=[(A4[0] - 4 * cm) / len(rows[0])] * len(rows[0]))
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, "#555555"), ("BACKGROUND", (0, 0), (-1, 0), "#dddddd")]))
    return t


# ------------------------------------------------------------------------------------------- files
def receipt_covered() -> None:
    """Clean receipt, bought 20-03-2025: 12-month commercial warranty runs to 20-03-2026 -> failure covered."""
    _image(_receipt_lines("20-03-2025", "R-2025-05120"), seed=21).save(OUT / "receipt_x200_clean_covered.png")


def receipt_blurred() -> None:
    """Whole receipt photographed out of focus: text is still there but no value can be trusted."""
    img = _image(_receipt_lines("20-03-2025", "R-2025-05120"), seed=22)
    small = img.resize((img.width // 2, img.height // 2))          # low-resolution phone photo ...
    img = small.resize(img.size).filter(ImageFilter.GaussianBlur(3.5))  # ... out of focus
    img.save(OUT / "receipt_x200_blurred.png")


def receipt_wrong_serial() -> None:
    """Legible receipt, but for a different X200 (serial AQ-X200-60112) bought by another customer."""
    lines = _receipt_lines("20-03-2025", "R-2025-05188", serial="AQ-X200-60112", customer="Bexley Dairies Ltd")
    _image(lines, seed=23).save(OUT / "receipt_x200_wrong_serial.png")


def warranty_card_x300() -> None:
    """X300 Extended Care registration card (residential): bought 02-09-2024, registered 15-09-2024."""
    lines = ["AQUAFLOW X300 EXTENDED CARE", "WARRANTY REGISTRATION CARD", "",
             "Card No: EC-300-77215", "Model: Aquaflow X300", "Serial: AQ-X300-10457",
             "Owner: R. Patel", "Use: Residential", "Purchase date: 02-09-2024",
             "Registered on: 15-09-2024", "Plan: Extended Care 36 months",
             "Annual service required"]
    _image(lines, seed=24, size=(1000, 960), angle=-0.3).save(OUT / "warranty_card_x300.png")


def installation_certificate() -> None:
    """Native-text PDF (no OCR needed): Acme's X200 installed by an Aquaflow Certified Installer."""
    flow = [Paragraph("Installation Certificate - Aquaflow X200", H1),
            Paragraph("Certificate No. IC-2025-0388. Issued by Northline Pump Services Ltd.", BODY), Spacer(1, 8),
            _table([["Item", "Detail"], ["Unit", "Aquaflow X200 Industrial Pump"], ["Serial", "AQ-X200-55871"],
                    ["Customer / site", "Acme Facilities Ltd, Unit 4 Harbour Park, plant room B"],
                    ["Installation date", "22 March 2025"], ["Installer", "D. Okafor"],
                    ["Aquaflow Certified Installer no.", "ACI-2291 (valid to 31 December 2026)"]]),
            Spacer(1, 10),
            Paragraph("Checks completed: level concrete base; isolation valves on inlet and outlet; non-return "
                      "valve on outlet; dry-run protection switch fitted and tested; motor protection circuit "
                      "breaker set to the nameplate current.", BODY)]
    _pdf(OUT / "installation_certificate_x200.pdf", flow)


def conflicting_purchase_records() -> None:
    """Native-text PDF whose two pages disagree on the purchase date of the same unit, with nothing saying
    which is authoritative. 02-02-2025 -> expired 02-02-2026 (not covered); 20-03-2025 -> covered."""
    flow = [Paragraph("Aquaflow Pumps Ltd - Sales Receipt", H1),
            _table([["Field", "Value"], ["Receipt No", "R-2025-05120"], ["Date of purchase", "20 March 2025"],
                    ["Customer", "Acme Facilities Ltd"], ["Serial", "AQ-X200-55871"], ["Amount", "GBP 4,850.00"]]),
            PageBreak(),
            Paragraph("Aquaflow Pumps Ltd - Customer Account Statement", H1),
            Paragraph("Account: Acme Facilities Ltd. Statement period: January - March 2025.", BODY), Spacer(1, 8),
            _table([["Line", "Detail"], ["Invoice", "INV-2025-0219"], ["Date of purchase", "2 February 2025"],
                    ["Item", "Aquaflow X200 Industrial Pump, serial AQ-X200-55871"],
                    ["Amount", "GBP 4,850.00"]])]
    _pdf(OUT / "purchase_records_conflicting.pdf", flow)


def _long_date(ddmmyyyy: str) -> str:
    """'15-01-2026' -> '15 January 2026'. Variable-width dates stop Tesseract from splitting the log into
    columns (aligned monospace columns are read column by column, separating each date from its task)."""
    from datetime import datetime

    return datetime.strptime(ddmmyyyy, "%d-%m-%Y").strftime("%d %B %Y").lstrip("0")


def maintenance_log_scan(pages: int = 6) -> None:
    """Multi-page SCANNED PDF (image only, every page needs OCR): slow enough to interrupt mid-processing,
    and answerable once indexed (bearings last regreased at 8,000 h on 14-01-2026)."""
    import io

    from reportlab.platypus import Image as RLImage

    from datetime import date, timedelta

    start, weeks = date(2025, 4, 3), 41                       # 03-04-2025 .. 15-01-2026
    rows = [("03-04-2025", "120", "Commissioning check", "D. Okafor")]
    prev = 120
    for w in range(1, weeks + 1):
        d = start + timedelta(weeks=w)
        hours = 120 + round(w * (8000 - 120) / weeks)
        if hours // 4000 > prev // 4000:
            task = "Regrease bearings"
        else:
            task = "Monthly vibration check" if w % 4 == 0 else "Weekly leak and noise check"
        rows.append((d.strftime("%d-%m-%Y"), str(hours), task, "S. Hale"))
        prev = hours
    per = -(-len(rows) // pages)
    flow = []
    for p in range(pages):
        chunk = rows[p * per:(p + 1) * per]
        lines = ["X200 MAINTENANCE LOG - AQ-X200-55871", f"Acme Facilities Ltd - sheet {p + 1} of {pages}", "",
                 "Entries: date, operating hours, task, engineer"] + [f"{_long_date(d)} - {h} h - {t} ({by})" for d, h, t, by in chunk]
        img = _degrade(_render_lines(lines, font_size=30, top=140, gap=62)[0], seed=40 + p, angle=0.3)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        flow += [RLImage(buf, width=A4[0] - 4 * cm, height=(A4[0] - 4 * cm) * img.height / img.width)]
        if p < pages - 1:
            flow.append(PageBreak())
    _pdf(OUT / "maintenance_log_x200_scanned.pdf", flow)


def corpus_receipts() -> None:
    """The two receipts from the main demo (samples/), copied here under descriptive names so every test-pack
    upload is in one folder. Same bytes as the originals, so the app treats them as the same documents."""
    src = Path(__file__).parent
    shutil.copyfile(src / "receipt_nonhinge.png", OUT / "receipt_x200_clean_expired.png")   # 05-03-2025: not covered
    shutil.copyfile(src / "receipt_hinge.png", OUT / "receipt_x200_smudged_digit.png")      # 1?-03-2025: uncertain


def duplicates_and_unsupported() -> None:
    # Same bytes, different name: dedupe is by content hash, not filename.
    shutil.copyfile(OUT / "receipt_x200_clean_covered.png", OUT / "receipt_x200_clean_covered_COPY.png")
    (OUT / "unsupported_notes.csv").write_text("serial,purchase_date\nAQ-X200-55871,20-03-2025\n", encoding="utf-8")


# ------------------------------------------------------------------------------------------- self-check
def check() -> None:
    problems = []

    def word(name, label):
        return _word_after(_ocr_words(OUT / name), label)

    clean = word("receipt_x200_clean_covered.png", "Date")
    print(f"clean receipt date     : {clean}")
    if not clean or clean[0] != "20-03-2025" or clean[1] < OCR_THRESHOLD:
        problems.append("clean receipt: date must OCR exactly with HIGH confidence")
    wrong = word("receipt_x200_wrong_serial.png", "Serial")
    print(f"wrong-serial receipt   : {wrong}")
    if not wrong or wrong[0] != "AQ-X200-60112" or wrong[1] < OCR_THRESHOLD:
        problems.append("wrong-serial receipt: serial must OCR exactly with HIGH confidence")
    card = _ocr_words(OUT / "warranty_card_x300.png")
    pdate = next((card[i + 1] for i, (t, _) in enumerate(card[:-1]) if t.lower() == "date:"
                  and card[i - 1][0].lower() == "purchase"), None)
    print(f"warranty card purchase : {pdate}")
    if not pdate or pdate[0] != "02-09-2024" or pdate[1] < OCR_THRESHOLD:
        problems.append("warranty card: purchase date must OCR exactly with HIGH confidence")
    blurred = _ocr_words(OUT / "receipt_x200_blurred.png")
    confs = [c for _, c in blurred]
    good = [t for t, c in blurred if c >= OCR_THRESHOLD]
    dates = [(t, c) for t, c in blurred if re.search(r"\d{2}-\d{2}-\d{4}", t)]
    print(f"blurred receipt        : {len(blurred)} words, {len(good)} >= {OCR_THRESHOLD:.0f}; date-like {dates}")
    if any(c >= OCR_THRESHOLD for _, c in dates) or len(good) > len(blurred) // 2:
        problems.append("blurred receipt: dates and most words must be LOW confidence (blur more)")
    if not confs:
        problems.append("blurred receipt: OCR found no words at all (blur less, so it is 'uncertain', not empty)")
    if problems:
        print("SELF-CHECK FAILED:\n  " + "\n  ".join(problems))
        sys.exit(1)
    print("self-check passed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-check", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    for build in (receipt_covered, receipt_blurred, receipt_wrong_serial, warranty_card_x300,
                  installation_certificate, conflicting_purchase_records, maintenance_log_scan,
                  corpus_receipts, duplicates_and_unsupported):
        build()
    print(f"test uploads written to {OUT}")
    if not args.no_check:
        check()


if __name__ == "__main__":
    main()
