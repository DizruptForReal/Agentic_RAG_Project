"""Uncertain OCR must never become a verified fact."""
from __future__ import annotations

import shutil

import pytest

from app.agent.models import Verification
from app.ingestion.extract import ExtractedField, FieldExtractor, ground_field
from tests.conftest import SAMPLES
from tests.fakes import ScriptedLLM, ocr_page

PAGE = {"page": 1, "method": "ocr", "text": "Date: 1-03-2025\nReceipt No: R-2025-03892",
        "words": [("Date:", 96.0), ("1-03-2025", 62.0), ("Receipt", 95.0), ("No:", 93.0), ("R-2025-03892", 91.0)]}


def f(value, source, page=1):
    return ExtractedField(name="x", found=True, value=value, page=page, source_text=source)


def test_low_confidence_word_is_unverified():
    v, conf, note = ground_field(f("1-03-2025", "Date: 1-03-2025"), [PAGE], 80)
    assert v == Verification.UNVERIFIED and conf == 62.0 and "below" in note


def test_high_confidence_word_is_verified():
    v, conf, _ = ground_field(f("R-2025-03892", "Receipt No: R-2025-03892"), [PAGE], 80)
    assert v == Verification.VERIFIED and conf == 91.0


def test_unrelated_low_confidence_word_does_not_taint_a_field():
    # "2025" also appears inside the low-confidence date; contiguous alignment keeps them apart
    v, _, _ = ground_field(f("R-2025-03892", "Receipt No: R-2025-03892"), [PAGE], 80)
    assert v == Verification.VERIFIED


def test_value_not_on_page_is_unverified():
    v, _, note = ground_field(f("17-03-2025", "Date: 17-03-2025"), [PAGE], 80)  # model "fixed" the digit
    assert v == Verification.UNVERIFIED and "not found" in note


def test_native_text_is_verified():
    native = {"page": 1, "method": "native", "text": "Effective date: 1 January 2025.", "words": []}
    v, conf, _ = ground_field(f("1 January 2025", "Effective date: 1 January 2025."), [native], 80)
    assert v == Verification.VERIFIED and conf is None


def test_extractor_reports_not_found_and_malformed_output():
    doc = {"doc_id": "d_1", "title": "receipt", "filename": "r.png"}
    pages = [{**PAGE, "marked_text": PAGE["text"]}]
    llm = ScriptedLLM({"extract_fields": [{"fields": [{"name": "serial", "found": False}]}]})
    ev, not_found, errors = FieldExtractor(llm, 80).extract(doc, pages, ["serial", "amount"])
    assert ev == [] and set(not_found) == {"serial", "amount"} and errors == []

    bad = ScriptedLLM({"extract_fields": ["nope", "still nope"]})
    ev, _, errors = FieldExtractor(bad, 80).extract(doc, pages, ["serial"])
    assert ev == [] and len(errors) == 2


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")
def test_real_tesseract_flags_the_smudged_date():
    from PIL import Image

    from app.ingestion.ocr import TesseractOCR

    words = TesseractOCR().ocr(Image.open(SAMPLES / "receipt_hinge.png")).words
    date = next(words[i + 1] for i, w in enumerate(words) if w.text == "Date:")
    assert date.conf < 80, date
    clean = TesseractOCR().ocr(Image.open(SAMPLES / "receipt_nonhinge.png")).words
    date = next(clean[i + 1] for i, w in enumerate(clean) if w.text == "Date:")
    assert date.text == "05-03-2025" and date.conf >= 80


def test_field_on_a_page_with_an_unrelated_smudge_is_verified_even_without_a_page_number():
    """Live test 15: 'Usage classification: COMMERCIAL' was read with high confidence on the scanned contract
    page, whose signature is smudged. The extractor gave no page number, so the value stayed UNVERIFIED and the
    answer was refused. The quote is now located on the document's pages itself."""
    native = {"page": 1, "method": "native", "text": "Equipment Lease Agreement ...", "words": []}
    scan = ocr_page("SCHEDULE B - EQUIPMENT USAGE DECLARATION\nUsage classification: COMMERCIAL\n"
                    "Signed for the Lessee: J. Morgan", low={"Morgan": 17.0})
    page3 = {"page": 3, "method": "ocr", "text": scan.text, "words": [(w.text, w.conf) for w in scan.words]}
    field = f("COMMERCIAL", "Usage classification: COMMERCIAL")
    field.page = None
    v, conf, note = ground_field(field, [native, page3], 80)
    assert v == Verification.VERIFIED and conf >= 80 and field.page == 3


def test_value_not_inside_its_quote_is_unverified():
    # RESIDENTIAL is on the page, but not where the model says the value comes from
    scan = ocr_page("Usage classification: COMMERCIAL\nResidential units: see section 2")
    page = {"page": 1, "method": "ocr", "text": scan.text, "words": [(w.text, w.conf) for w in scan.words]}
    v, _, note = ground_field(f("RESIDENTIAL", "Usage classification: COMMERCIAL"), [page], 80)
    assert v == Verification.UNVERIFIED and "not part of" in note
    v, _, note = ground_field(f("RESIDENTIAL", "Usage classification: RESIDENTIAL"), [PAGE], 80)
    assert v == Verification.UNVERIFIED                                # not on the page at all


def test_quote_with_copied_markers_and_page_reference_still_verifies():
    """Live inv_e8b2892f5f28: the quote included '[[?Maan]] (page 3)' and was rejected as 'not verbatim',
    although COMMERCIAL was read at 96. Trust now comes from the value's own OCR words."""
    scan = ocr_page("Usage classification: COMMERCIAL\nPurpose: facility water circulation\n"
                    "Signed for the Lessee: J. Maan", low={"Maan": 17.0}, default_conf=96.0)
    page3 = {"page": 3, "method": "ocr", "text": scan.text, "words": [(w.text, w.conf) for w in scan.words]}
    field = f("COMMERCIAL", "Usage classification: COMMERCIAL Purpose: facility water circulation Signed for the "
                            "Lessee: J. [[?Maan]] (page 3)", page=None)
    v, conf, _ = ground_field(field, [page3], 80)
    assert v == Verification.VERIFIED and conf == 96.0 and field.page == 3


def test_digit_inside_another_word_does_not_verify_a_value():
    v, _, _ = ground_field(f("2025", "Receipt No: R-2025-03892"), [PAGE], 80)
    assert v == Verification.UNVERIFIED                                # '2025' is not a word on that page


# ------------------------------------------------------------------ claim-level check of an uncertain passage
def _finalize_citing_scan(answer_text):
    from tests.fakes import parse_view

    def plan(user):
        v = parse_view(user)
        scan = [e["id"] for e in v["evidence_ledger"] if "Usage classification" in e["text"]][:1]
        assert scan, "the scanned Schedule B passage should be in the ledger"
        return {"sub_questions": [{"id": "q1", "text": "Is the unit classified as commercial use?",
                                   "status": "SUPPORTED", "evidence_ids": scan, "note": "Schedule B: COMMERCIAL"}],
                "action": {"type": "finalize", "args": {}, "reason": "done"},
                "answer": {"text": answer_text.format(scan[0]), "citations": scan}}
    return plan


def test_passage_flagged_by_an_unrelated_smudge_is_verified_at_field_level(container, llm, settings, seeded):
    """Live inv_1636492bcd2e: the planner cited only the scanned Schedule B passage, which is UNVERIFIED because
    the signature is smudged; the answer was refused. The system now re-reads the claim word by word."""
    settings.initial_search = container.investigator.s.initial_search = True
    llm.push("plan_step", _finalize_citing_scan("The unit is used commercially [{}]."))
    llm.push("extract_fields", {"fields": [{"name": "Is the unit classified as commercial use?", "found": True,
                                            "value": "COMMERCIAL", "page": None,
                                            "source_text": "Usage classification: COMMERCIAL"}]})
    state = container.investigator.run(container.investigator.create("How is the Acme X200 classified for use?").id)
    q1 = state.sub_questions[0]
    assert q1.status.value == "SUPPORTED" and "verified word by word" in q1.note
    assert llm.count("extract_fields") == 0          # checked in code from the OCR words, no model call needed
    assert state.final.outcome.value == "ANSWERED"
    assert not any("UNVERIFIED" in c for c in state.final.caveats)      # no contradicting page-wide caveat


def test_genuinely_uncertain_claims_stay_unverified():
    from app.ingestion.ocr_verify import check_claim

    hinge = "SALES RECEIPT Receipt No: R-2025-04417 Date: [[?1-03-2025]] Serial: AQ-X200-55871"
    assert not check_claim("Receipt shows 17-03-2025", hinge)[0]       # model "fixed" the smudged digit
    assert not check_claim("Purchase date 1-03-2025", hinge)[0]        # rests on the uncertain word
    assert not check_claim("bought in 2025", hinge)[0]                 # '2025' only inside R-2025-04417
    assert not check_claim("Receipt shows a purchase date", hinge)[0]  # no value, nothing legible to check
    assert check_claim("Serial AQ-X200-55871", hinge)[0]               # the serial itself is clearly read
    contract = "Usage classification: COMMERCIAL Signed for the Lessee: J. [[?Maan]]"
    assert check_claim("Schedule B says COMMERCIAL", contract)[0]
    assert not check_claim("Schedule B says RESIDENTIAL", contract)[0]
    assert not check_claim("signed by J. Maan", contract)[0]


def test_guard_keeps_an_uncertain_date_unverified_but_accepts_a_clear_one():
    from app.agent.guards import validate_subquestions
    from app.agent.models import EvidenceKind, InvestigationState, LedgerEntry, SubQuestion, SubQStatus

    def state(text):
        s = InvestigationState(id="inv_t", question="q")
        s.ledger = [LedgerEntry(id="e1", kind=EvidenceKind.CHUNK, doc_id="r", doc_name="receipt", page=1, text=text,
                                verification=Verification.UNVERIFIED, ocr_confidence=18.0, rerank_score=0.5,
                                retrieval_rank=1)]
        return s

    q = SubQuestion(id="q1", text="purchase date?", status=SubQStatus.SUPPORTED, evidence_ids=["e1"],
                    note="Receipt shows 05-03-2025")
    clean = state("Receipt No: [[?R-2025-03§92]] Date: 05-03-2025 Serial: AQ-X200-55871")
    fixed, _ = validate_subquestions([q], clean, 0.03)
    assert fixed[0].status == SubQStatus.SUPPORTED and clean.claim_verified == ["e1"]
    smudged = state("Receipt No: R-2025-04417 Date: [[?1-03-2025]]")
    fixed, _ = validate_subquestions([q.model_copy(update={"note": "Receipt shows 17-03-2025"})], smudged, 0.03)
    assert fixed[0].status == SubQStatus.UNVERIFIED
