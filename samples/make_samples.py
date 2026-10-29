"""Generate the demo corpus deterministically.

    python -m samples.make_samples            # build everything into samples/
    python -m samples.make_samples --no-check # skip the Tesseract self-check

All content is hand-written here so ground truth is fixed. Brands and companies are fictional.
After generating the receipts the script runs Tesseract and asserts that the smudged word really is
low-confidence (and the clean date really is high-confidence), so the demo cannot silently stop
demonstrating what it is meant to demonstrate.
"""
from __future__ import annotations

import argparse
import io
import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image as RLImage, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT = Path(__file__).parent
OCR_THRESHOLD = 80.0

STYLES = getSampleStyleSheet()
H1, H2, BODY = STYLES["Title"], STYLES["Heading2"], STYLES["BodyText"]

# ----------------------------------------------------------------------------- corpus text
# Ground-truth invariants the demo and tests rely on (do not break them when editing):
#  * Policy v1 section 2: 24 months; Amendment (eff. 2025-01-01) supersedes section 2: commercial = 12 months.
#  * Usage classification only appears on the SCANNED page 3 of the contract (Schedule B): COMMERCIAL.
#  * No document contains the purchase date of serial AQ-X200-55871 (only the receipt does).
#  * The X300 manual is a decoy (36-month Extended Care, X300 only). Nothing mentions an X500.

POLICY_V1 = [
    ("h1", "Aquaflow Pumps Ltd - Limited Warranty Policy"),
    ("p", "Document ID: AQF-WP-1.0. Version 1.0. Effective date: 1 January 2023. Issued by Aquaflow Pumps Ltd, "
          "Unit 9 Riverside Works, Leeds. This policy describes the limited warranty that Aquaflow provides for "
          "its X-series pumps. It should be read together with the service manual of the relevant model."),
    ("h2", "1. Scope and covered products"),
    ("p", "This policy applies to Aquaflow X-series pumps sold new by Aquaflow Pumps Ltd or its authorised "
          "dealers on or after the effective date: the X100 domestic booster pump, the X200 industrial pump "
          "and the X300 residential circulation pump, together with genuine Aquaflow control panels supplied "
          "with them."),
    ("p", "The following are not covered by this policy: accessories and spare parts made by third parties; "
          "units bought second-hand from a seller who is not an authorised dealer (unless the warranty was "
          "transferred under section 9); and units that have been refurbished or modified by anyone other "
          "than Aquaflow or an authorised service agent."),
    ("p", "Leased units. Units supplied by Aquaflow under an equipment lease are covered in the same way as "
          "purchased units; the warranty runs from the purchase date on the original sales receipt for the unit."),
    ("p", "Other models. Only the models named in this section are covered. A model that is not named here "
          "(for example a model launched after the effective date) is covered only if Aquaflow publishes an "
          "amendment that adds it to this policy."),
    ("h2", "2. Warranty period"),
    ("p", "Aquaflow warrants each unit against failure caused by defects in materials or workmanship "
          "for twenty-four (24) months from the date of purchase shown on the original sales receipt "
          "or invoice."),
    ("p", "The warranty therefore ends twenty-four months after the purchase date; for example, a unit bought "
          "on 10 May 2023 is covered until 10 May 2025. A claim can be made at any time during the warranty "
          "period, provided it is submitted within thirty (30) days of the failure (see section 7)."),
    ("h2", "3. What is covered"),
    ("p", "3.1 Covered components. During the warranty period the warranty covers failure caused by a defect "
          "in any of the following: the motor and motor windings; the pump housing and volute; the impeller; "
          "the shaft and bearings (bearings only for their first 8,000 operating hours); the integrated "
          "control panel and pressure sensor; and the factory-fitted wiring and terminal box. Mechanical seals "
          "are covered against manufacturing defects for the first six (6) months only; after that, seal "
          "replacement is normal wear."),
    ("p", "3.2 Remedies. For a valid claim Aquaflow will, at its discretion, repair the unit, replace it with "
          "a new or factory-reconditioned equivalent, or, if neither repair nor replacement is practicable, "
          "refund the purchase price shown on the receipt."),
    ("p", "3.3 Costs covered. The warranty covers genuine replacement parts, labour carried out by an "
          "Aquaflow-authorised service agent, return shipping within the UK mainland, and an on-site call-out "
          "for units weighing more than 25 kg (such as the X200)."),
    ("p", "3.4 Replaced parts. A part repaired or replaced under this warranty is covered for the remainder "
          "of the original warranty period or for ninety (90) days, whichever is longer."),
    ("break", None),
    ("h2", "4. Exclusions"),
    ("p", "The warranty does not cover normal wear of consumable parts as defined in the relevant "
          "service manual, damage caused by dry running, improper installation, use of unapproved "
          "fluids, or units whose serial number has been removed or altered."),
    ("p", "The warranty also does not cover: operation above the rated pressure or temperature stated in the "
          "service manual; damage from freezing, flooding, lightning or power surges; failure caused by a lack "
          "of the maintenance required by the service manual; cosmetic damage that does not affect operation; "
          "and consequential losses such as downtime, lost production or water damage to property."),
    ("p", "Approved fluids are clean water and water-glycol mixtures containing no more than 30% glycol. Any "
          "other fluid is an unapproved fluid unless Aquaflow has confirmed it in writing."),
    ("h2", "5. Conditions"),
    ("p", "5.1 Registration. Owners should register the unit online within thirty (30) days of purchase. An "
          "unregistered unit is still covered, but the owner must then prove the purchase date at claim time."),
    ("p", "5.2 Installation. X200 units, and any X-series unit used commercially, must be installed by an "
          "Aquaflow Certified Installer. X100 and residential X300 units may be installed by any qualified "
          "plumber following the installation section of the service manual."),
    ("p", "5.3 Maintenance. The owner must carry out the maintenance listed in the service manual. For the "
          "X200 a written maintenance log must be kept and made available on request."),
    ("h2", "6. Proof of purchase"),
    ("p", "Every claim must be accompanied by the original sales receipt showing the purchase date and "
          "the serial number of the unit. Claims without a legible purchase date cannot be assessed."),
    ("break", None),
    ("h2", "7. Claims procedure"),
    ("p", "Claims must be submitted within thirty (30) days of the failure. Aquaflow will repair or "
          "replace the failed unit at its discretion."),
    ("p", "To make a claim: (1) stop using the unit and isolate it from the power supply; (2) submit the claim "
          "through the Aquaflow service portal or by e-mail to the warranty team; (3) include the serial "
          "number, a copy of the sales receipt, a description of the failure, photographs, and, for the X200, "
          "the maintenance log; (4) keep the unit available for inspection."),
    ("p", "Aquaflow acknowledges claims within two (2) business days and completes its assessment within ten "
          "(10) business days of receiving all documents. Approved repairs are completed within fifteen (15) "
          "business days where parts are in stock."),
    ("h2", "8. Disputes"),
    ("p", "If a claim is rejected the owner may request a review within thirty (30) days of the decision. "
          "Where the parties still disagree, the unit is assessed by an independent engineer whose fee is paid "
          "by Aquaflow if the claim is then accepted."),
    ("h2", "9. Transfer of warranty"),
    ("p", "The warranty may be transferred once to a subsequent owner, for the remainder of the warranty "
          "period, provided the original sales receipt is handed over and Aquaflow is notified within thirty "
          "(30) days of the transfer."),
    ("h2", "10. Limitation of liability"),
    ("p", "Aquaflow's total liability under this warranty is limited to the purchase price of the unit. This "
          "warranty is in addition to, and does not affect, the owner's statutory rights."),
    ("h2", "11. Definitions"),
    ("p", "Date of purchase: the date printed on the original sales receipt or invoice. Failure: a loss of "
          "function caused by a defect in materials or workmanship. Normal wear: gradual degradation of "
          "consumable parts as defined in the service manual. Authorised service agent: a repairer approved "
          "by Aquaflow. Certified installer: an installer holding a current Aquaflow installation certificate."),
    ("h2", "12. Key time limits"),
    ("p", "The time limits in this policy - how long the warranty lasts, when to register, how long you have "
          "to claim and how long each stage takes - are summarised below."),
    ("table", [["What", "Time limit"], ["Warranty period", "24 months from the date of purchase"],
               ["Mechanical seals (defects)", "first 6 months only"], ["Bearings", "first 8,000 operating hours"],
               ["Product registration", "within 30 days of purchase"],
               ["Submitting a claim", "within 30 days of the failure"],
               ["Claim acknowledged", "within 2 business days"],
               ["Assessment completed", "within 10 business days of receiving all documents"],
               ["Approved repair completed", "within 15 business days (parts in stock)"],
               ["Replaced or repaired parts", "rest of the warranty or 90 days, whichever is longer"],
               ["Review of a rejected claim", "request within 30 days of the decision"],
               ["Transfer to a new owner", "notify Aquaflow within 30 days of the transfer"]]),
]

AMENDMENT = [
    ("h1", "Amendment No. 1 to the Aquaflow Limited Warranty Policy"),
    ("p", "Document ID: AQF-WP-1.0-A1. Effective date: 1 January 2025."),
    ("p", "This amendment supersedes Section 2 (Warranty period) of policy AQF-WP-1.0 for all units "
          "purchased on or after 1 January 2025. All other sections of AQF-WP-1.0 remain in force, except "
          "where this amendment states otherwise."),
    ("h2", "1. Commercial-use units"),
    ("p", "For units installed or used in commercial or industrial settings, the warranty period is "
          "reduced to twelve (12) months from the date of purchase."),
    ("p", "This replaces the 24-month warranty period of section 2 of AQF-WP-1.0 for commercial-use units "
          "purchased on or after 1 January 2025."),
    ("h2", "2. Residential-use units"),
    ("p", "For units used in private residential settings the warranty period remains twenty-four (24) "
          "months from the date of purchase."),
    ("h2", "3. Usage classification"),
    ("p", "Whether a unit is commercial or residential is determined by the customer contract or lease "
          "agreement under which the unit was supplied."),
    ("break", None),
    ("h2", "4. Labour costs for commercial-use units"),
    ("p", "For commercial-use units purchased on or after 1 January 2025, labour by an authorised service "
          "agent is covered for the first six (6) months from the date of purchase only. From month seven "
          "until the end of the warranty period, parts remain covered but labour is charged at Aquaflow's "
          "standard rates. This replaces the labour part of section 3.3 of AQF-WP-1.0 for these units."),
    ("h2", "5. Maintenance log for commercial-use units"),
    ("p", "For every commercial-use unit, claims must include the maintenance log required by the service "
          "manual. A claim without the log is rejected unless the failure is clearly unrelated to maintenance."),
    ("h2", "6. Sections not changed"),
    ("p", "Sections 1, 3 (except labour costs as described above), 4, 5, 6, 7, 8, 9, 10, 11 and 12 of "
          "AQF-WP-1.0 are unchanged. Units purchased before 1 January 2025 remain under the original section 2."),
    ("h2", "7. Summary of warranty periods after this amendment"),
    ("table", [["Unit and purchase date", "Warranty period", "Labour covered"],
               ["Any unit bought before 1 Jan 2025", "24 months", "whole period"],
               ["Residential unit bought from 1 Jan 2025", "24 months", "whole period"],
               ["Commercial unit bought from 1 Jan 2025", "12 months", "first 6 months"]]),
]

MANUAL_X200 = [
    ("h1", "Aquaflow X200 Industrial Pump - Service Manual (extract)"),
    ("p", "Document ID: AQF-SM-X200, revision C. This extract covers specifications, installation, approved "
          "fluids, maintenance, failure classification, error codes and warranty notes for the X200."),
    ("h2", "1. Specifications"),
    ("p", "The X200 industrial pump has a rated pressure of 10 bar, a maximum flow rate of "
          "42 m3/h, a 5.5 kW 400 V three-phase motor and a maximum fluid temperature of 80 C. It weighs 64 kg. "
          "The full specification is in the table below."),
    ("table", [["Parameter", "X200"], ["Rated pressure", "10 bar"], ["Motor", "5.5 kW, 400 V, 3-phase"],
               ["Max flow", "42 m3/h"], ["Max fluid temperature", "80 C"], ["Weight", "64 kg"],
               ["Seal service interval", "8,000 h"], ["Bearing service interval", "4,000 h (regrease)"]]),
    ("h2", "2. Installation requirements"),
    ("p", "The X200 must be installed by an Aquaflow Certified Installer on a level concrete base, with an "
          "isolation valve on each side, a non-return valve on the outlet and a dry-run protection switch. "
          "The motor must be protected by a motor protection circuit breaker set to the nameplate current."),
    ("h2", "3. Approved fluids"),
    ("p", "Clean water and water-glycol mixtures with at most 30% glycol. Seawater, oils, solvents and fluids "
          "containing abrasive solids are not approved and will damage the seals and impeller."),
    ("break", None),
    ("h2", "4. Maintenance schedule"),
    ("table", [["Interval", "Task"], ["Weekly", "Check for leaks and unusual noise"],
               ["Monthly", "Check vibration and motor temperature"], ["Every 4,000 h", "Regrease bearings"],
               ["Every 8,000 h", "Replace mechanical seal"], ["Annually", "Inspection by an authorised agent"]]),
    ("p", "Record every task in the maintenance log with the date, operating hours and the name of the person "
          "who carried it out. The log is required for warranty claims on the X200."),
    ("h2", "5. Failure versus normal wear"),
    ("p", "A failure is a loss of function caused by a defect. Motor seizure, motor winding failure and "
          "cracking of the pump housing at or below the rated pressure are classed as failures."),
    ("p", "Normal wear covers the gradual degradation of consumable parts: mechanical seals, impeller "
          "wear rings and bearings with more than 8,000 operating hours. Normal wear is not a failure."),
    ("break", None),
    ("h2", "6. Error codes"),
    ("table", [["Code", "Meaning", "Action"], ["E01", "Dry run detected", "Stop; check suction supply"],
               ["E02", "Overpressure", "Check outlet valves"], ["E03", "Motor over-temperature",
                                                               "Check ventilation and load"],
               ["E04", "Pressure sensor fault", "Call service agent"], ["E05", "Phase loss", "Check supply"]]),
    ("p", "Repeated E01 or E02 events are stored in the controller and are reviewed during warranty "
          "assessment, because damage from dry running or overpressure is excluded from the warranty."),
    ("h2", "7. Warranty"),
    ("p", "Warranty terms for the X200 are governed by the Aquaflow Limited Warranty Policy and any "
          "amendment in force at the date of purchase. This manual does not record purchase dates; the "
          "purchase date must be taken from the sales receipt."),
]

MANUAL_X300 = [
    ("h1", "Aquaflow X300 Residential Pump - Service Manual (extract)"),
    ("p", "Document ID: AQF-SM-X300, revision A. The X300 is a residential heating and hot-water "
          "circulation pump."),
    ("h2", "1. Specifications"),
    ("p", "The X300 residential pump has a rated pressure of 6 bar, a maximum flow rate of 3.5 m3/h, a 0.4 kW "
          "230 V single-phase motor and a maximum fluid temperature of 95 C. It weighs 7 kg."),
    ("h2", "7. Failure versus normal wear"),
    ("p", "Motor seizure and housing cracks are classed as failures. Seal wear is normal wear. Limescale "
          "build-up from untreated hard water is not a failure."),
    ("h2", "8. Maintenance"),
    ("p", "Bleed air from the pump at installation and after draining the system. Have the pump inspected "
          "once a year together with the boiler service."),
    ("h2", "9. Warranty"),
    ("p", "X300 units registered under the X300 Extended Care programme carry a thirty-six (36) month "
          "warranty from the date of purchase. Extended Care is available for the X300 model only."),
    ("p", "Extended Care requires online registration within thirty (30) days of purchase and an annual "
          "service by a qualified engineer. It does not apply to units used commercially."),
]

CONTRACT_P1 = [
    ("h1", "Equipment Lease Agreement"),
    ("p", "Agreement reference: ACME-LEASE-2025-014."),
    ("p", "Between Aquaflow Pumps Ltd (the Lessor) and Acme Facilities Ltd (the Lessee), company number "
          "08812234, registered office Unit 4 Harbour Park, Hull."),
    ("h2", "1. Equipment"),
    ("p", "One (1) Aquaflow X200 Industrial Pump, serial number AQ-X200-55871, together with its control "
          "panel and installation kit, as listed in Schedule A."),
    ("h2", "2. Term"),
    ("p", "The lease runs for thirty-six (36) months from delivery. Delivery and purchase date: as per "
          "the original sales receipt, which is not reproduced in this agreement."),
    ("h2", "3. Payments"),
    ("p", "The Lessee pays a monthly rental of GBP 310.00 by direct debit on the first business day of each "
          "month. Late payments carry interest at 4% above the Bank of England base rate."),
    ("break", None),
    ("h2", "4. Maintenance"),
    ("p", "The Lessee shall service the unit at the intervals stated in the service manual and keep a "
          "maintenance log available for inspection."),
    ("h2", "5. Insurance"),
    ("p", "The Lessee shall insure the equipment against loss and damage for its full replacement value and "
          "name the Lessor as loss payee."),
    ("h2", "6. Warranty"),
    ("p", "Warranty claims are handled under the Aquaflow Limited Warranty Policy. The usage "
          "classification of the unit is declared in Schedule B."),
    ("h2", "7. Termination"),
    ("p", "Either party may terminate on ninety (90) days' written notice after the first twelve months. On "
          "termination the Lessee returns the equipment in good working order, fair wear and tear excepted."),
    ("h2", "Schedule A - Equipment and site"),
    ("p", "Aquaflow X200 Industrial Pump, serial AQ-X200-55871, with control panel CP-200 and installation kit. "
          "Site: Acme Facilities Ltd, Unit 4 Harbour Park, plant room B."),
]

CONTRACT_SCANNED_P3 = [
    "SCHEDULE B - EQUIPMENT USAGE DECLARATION",
    "",
    "Agreement: ACME-LEASE-2025-014",
    "Unit: Aquaflow X200, serial AQ-X200-55871",
    "Site: Acme Facilities Ltd, Unit 4 Harbour Park",
    "Usage classification: COMMERCIAL",
    "Purpose: facility water circulation",
    "",
    "Signed for the Lessee: J. Morgan",
]

RECEIPT_LINES = [
    "AQUAFLOW PUMPS LTD",
    "SALES RECEIPT",
    "",
    "Receipt No: {receipt_no}",
    "Date: {date}",
    "Customer: Acme Facilities Ltd",
    "Item: Aquaflow X200 Pump",
    "Serial: AQ-X200-55871",
    "Qty: 1",
    "Amount: GBP 4,850.00",
    "Paid: Bank transfer",
]

MANIFEST = [
    {"file": "warranty_policy_v1.pdf", "title": "Aquaflow Limited Warranty Policy v1.0 (AQF-WP-1.0)",
     "doc_type": "policy", "product": "X-series", "effective_date": "2023-01-01", "supersedes": None},
    {"file": "warranty_amendment_2025.pdf", "title": "Warranty Policy Amendment No. 1 (AQF-WP-1.0-A1)",
     "doc_type": "amendment", "product": "X-series", "effective_date": "2025-01-01",
     "supersedes": "AQF-WP-1.0 section 2, for units purchased on or after 2025-01-01"},
    {"file": "service_manual_x200.pdf", "title": "Aquaflow X200 Service Manual (AQF-SM-X200)",
     "doc_type": "manual", "product": "X200", "effective_date": None, "supersedes": None},
    {"file": "customer_contract_acme.pdf", "title": "Acme Facilities equipment lease (ACME-LEASE-2025-014)",
     "doc_type": "contract", "product": "X200", "effective_date": "2025-03-01", "supersedes": None},
    {"file": "service_manual_x300.pdf", "title": "Aquaflow X300 Service Manual (AQF-SM-X300)",
     "doc_type": "manual", "product": "X300", "effective_date": None, "supersedes": None},
]


# ----------------------------------------------------------------------------- helpers
def _font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/Library/Fonts/Courier New.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/cour.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def _degrade(img: Image.Image, seed: int, angle: float) -> Image.Image:
    """Make a clean render look scanned: slight rotation, blur, noise, lowered contrast."""
    rnd = random.Random(seed)
    img = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=245)
    img = img.filter(ImageFilter.GaussianBlur(0.6))
    px = img.load()
    w, h = img.size
    for _ in range(w * h // 60):
        x, y = rnd.randrange(w), rnd.randrange(h)
        px[x, y] = max(0, min(255, px[x, y] + rnd.randint(-60, 60)))
    return img.point(lambda v: int(40 + v * 0.8))


def _render_lines(lines: list[str], size=(1240, 1754), font_size=34, top=140, left=110, gap=58):
    img = Image.new("L", size, 255)
    draw = ImageDraw.Draw(img)
    font = _font(font_size)
    positions = []
    y = top
    for line in lines:
        draw.text((left, y), line, font=font, fill=20)
        positions.append((left, y))
        y += gap
    return img, draw, font, positions


def _smudge(img: Image.Image, box: tuple[int, int, int, int], seed: int) -> Image.Image:
    """Grey blurred blob over one region (one character)."""
    rnd = random.Random(seed)
    layer = Image.new("L", img.size, 0)
    d = ImageDraw.Draw(layer)
    x0, y0, x1, y1 = box
    for _ in range(14):
        dx, dy = rnd.randint(-3, 3), rnd.randint(-3, 3)
        d.ellipse((x0 - 2 + dx, y0 + dy, x1 + 2 + dx, y1 + dy), fill=rnd.randint(200, 240))
    layer = layer.filter(ImageFilter.GaussianBlur(3))
    grey = Image.new("L", img.size, 120)
    return Image.composite(grey, img, layer)


# ----------------------------------------------------------------------------- PDFs
def _flow(blocks):
    flow = []
    for kind, content in blocks:
        if kind == "h1":
            flow.append(Paragraph(content, H1))
        elif kind == "h2":
            flow.append(Paragraph(content, H2))
        elif kind == "p":
            flow += [Paragraph(content, BODY), Spacer(1, 0.25 * cm)]
        elif kind == "table":
            width = (A4[0] - 4 * cm) / len(content[0])
            t = Table(content, colWidths=[width] * len(content[0]))
            t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, "#555555"),
                                   ("BACKGROUND", (0, 0), (-1, 0), "#dddddd")]))
            flow += [t, Spacer(1, 0.4 * cm)]
        elif kind == "break":
            flow.append(PageBreak())
    return flow


def _pdf(name: str, blocks, extra_flow=None) -> None:
    doc = SimpleDocTemplate(str(OUT / name), pagesize=A4, title=name, author="Aquaflow (fictional)",
                            leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm)
    doc.build(_flow(blocks) + (extra_flow or []))


def build_pdfs() -> None:
    _pdf("warranty_policy_v1.pdf", POLICY_V1)
    _pdf("warranty_amendment_2025.pdf", AMENDMENT)
    _pdf("service_manual_x200.pdf", MANUAL_X200)
    _pdf("service_manual_x300.pdf", MANUAL_X300)

    # Contract: pages 1-2 have a text layer, page 3 is an image-only "scan". The usage line must stay
    # legible; the signature line is smudged so the page also contains a genuinely uncertain word.
    img, _, font, pos = _render_lines(CONTRACT_SCANNED_P3, font_size=34, top=160, gap=66)
    sig = len(CONTRACT_SCANNED_P3) - 1
    x, y = pos[sig]
    line = CONTRACT_SCANNED_P3[sig]
    start = line.index("Morgan")
    bbox = font.getbbox(line)
    img = _smudge(img, (int(x + font.getlength(line[:start + 1])), y + bbox[1],
                        int(x + font.getlength(line[:start + 4])), y + bbox[3]), seed=5)
    img = _degrade(img, seed=3, angle=0.3)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    scan = RLImage(buf, width=A4[0] - 4 * cm, height=(A4[0] - 4 * cm) * img.height / img.width)
    _pdf("customer_contract_acme.pdf", CONTRACT_P1, [PageBreak(), scan])


# ----------------------------------------------------------------------------- receipts
def _receipt(date: str, receipt_no: str, smudge_line: str, smudge_char_index: int, seed: int) -> Image.Image:
    lines = [ln.format(date=date, receipt_no=receipt_no) for ln in RECEIPT_LINES]
    img, _, font, pos = _render_lines(lines, size=(1000, 900), font_size=34, top=60, left=90, gap=64)
    row = next(i for i, ln in enumerate(lines) if ln.startswith(smudge_line))
    text = lines[row]
    x, y = pos[row]
    x0 = x + font.getlength(text[:smudge_char_index])
    x1 = x + font.getlength(text[:smudge_char_index + 1])
    bbox = font.getbbox(text)
    img = _smudge(img, (int(x0), y + bbox[1], int(x1), y + bbox[3]), seed)
    return _degrade(img, seed=seed, angle=0.6)


def build_receipts() -> None:
    # Hinge: true date 17-03-2025, the "7" is smudged. 12 vs 17 decides coverage.
    hinge = _receipt("17-03-2025", "R-2025-04417", "Date:", len("Date: 1"), seed=11)
    hinge.save(OUT / "receipt_hinge.png")
    # Non-hinge: clean date, smudge on the receipt number (irrelevant to the verdict).
    nonhinge = _receipt("05-03-2025", "R-2025-03392", "Receipt No:", len("Receipt No: R-2025-03"), seed=12)
    nonhinge.save(OUT / "receipt_nonhinge.png")


# ----------------------------------------------------------------------------- self-check
def _ocr_words(path: Path) -> list[tuple[str, float]]:
    import pytesseract

    data = pytesseract.image_to_data(Image.open(path).convert("L"), output_type=pytesseract.Output.DICT)
    return [(t, float(c)) for t, c in zip(data["text"], data["conf"]) if t.strip() and float(c) >= 0]


def _word_after(words, label: str) -> tuple[str, float] | None:
    for i, (t, _) in enumerate(words):
        if t.rstrip(":").lower() == label.lower() and i + 1 < len(words):
            return words[i + 1]
    return None


def _ocr_contract_scan() -> list[tuple[str, float]]:
    import pymupdf
    import pytesseract

    with pymupdf.open(OUT / "customer_contract_acme.pdf") as doc:
        if doc[2].get_text("text").strip():
            raise SystemExit("contract page 3 must have no text layer")
        pix = doc[2].get_pixmap(dpi=300)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples).convert("L")
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    return [(t, float(c)) for t, c in zip(data["text"], data["conf"]) if t.strip() and float(c) >= 0]


EXPECTED_PAGES = {"warranty_policy_v1.pdf": 4, "warranty_amendment_2025.pdf": 2, "service_manual_x200.pdf": 3,
                  "customer_contract_acme.pdf": 3, "service_manual_x300.pdf": 1}


def check_layout() -> None:
    import pymupdf

    for name, n in EXPECTED_PAGES.items():
        with pymupdf.open(OUT / name) as doc:
            if len(doc) != n:
                raise SystemExit(f"{name}: expected {n} pages, got {len(doc)} - adjust the page breaks")


def check_receipts() -> None:
    contract = _ocr_contract_scan()
    usage = _word_after(contract, "classification")
    print(f"contract usage word  : {usage}")
    if not usage or usage[0] != "COMMERCIAL" or usage[1] < OCR_THRESHOLD + 5:
        print("SELF-CHECK FAILED: contract usage classification must OCR as COMMERCIAL with high confidence")
        sys.exit(1)
    hinge = _ocr_words(OUT / "receipt_hinge.png")
    non = _ocr_words(OUT / "receipt_nonhinge.png")
    h_date = _word_after(hinge, "Date")
    n_date = _word_after(non, "Date")
    n_no = _word_after(non, "No")
    print(f"hinge    date word   : {h_date}")
    print(f"nonhinge date word   : {n_date}")
    print(f"nonhinge receipt no. : {n_no}")
    problems = []
    if not h_date or h_date[1] >= OCR_THRESHOLD:
        problems.append("hinge receipt: date should be read with LOW confidence")
    if not n_date or n_date[1] < OCR_THRESHOLD or n_date[0] != "05-03-2025":
        problems.append("non-hinge receipt: date should be read exactly with HIGH confidence")
    if not n_no or n_no[1] >= OCR_THRESHOLD:
        problems.append("non-hinge receipt: receipt number should be LOW confidence")
    if problems:
        print("SELF-CHECK FAILED:\n  " + "\n  ".join(problems))
        sys.exit(1)
    print("self-check passed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-check", action="store_true", help="skip the Tesseract self-check")
    args = ap.parse_args()
    build_pdfs()
    build_receipts()
    (OUT / "manifest.json").write_text(json.dumps(MANIFEST, indent=2), encoding="utf-8")
    print(f"samples written to {OUT}")
    check_layout()
    if not args.no_check:
        check_receipts()


if __name__ == "__main__":
    main()
