"""Claude client contract (mocked SDK) and an ingestion+extraction run with the REAL Tesseract."""
from __future__ import annotations

import re
import shutil
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.container import assemble
from app.llm.base import LLMError
from tests.conftest import SAMPLES
from tests.fakes import HashEmbedder, InMemoryIndex, OverlapReranker, ScriptedLLM


class _FakeMessages:
    def __init__(self, content=None, exc=None):
        self.content, self.exc, self.kwargs = content, exc, None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.exc:
            raise self.exc
        return SimpleNamespace(content=self.content)


def _client(messages):
    pytest.importorskip("anthropic")
    from app.llm.claude import ClaudeClient

    c = ClaudeClient(api_key="sk-test", model="claude-sonnet-5-5")
    c.client = SimpleNamespace(messages=messages)
    return c


def test_claude_client_forces_tool_and_returns_input():
    msgs = _FakeMessages([SimpleNamespace(type="tool_use", input={"ok": 1})])
    out = _client(msgs).complete_json(purpose="plan_step", system="s", user="u", schema={"type": "object"})
    assert out == {"ok": 1}
    assert msgs.kwargs["tool_choice"] == {"type": "tool", "name": "plan_step"}
    assert msgs.kwargs["model"] == "claude-sonnet-5-5"


def test_claude_client_returns_text_when_no_tool_call():
    msgs = _FakeMessages([SimpleNamespace(type="text", text="sorry")])
    assert _client(msgs).complete_json(purpose="p", system="s", user="u", schema={}) == "sorry"


def test_claude_client_wraps_api_errors():
    anthropic = pytest.importorskip("anthropic")
    import httpx

    err = anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com"))
    with pytest.raises(LLMError):
        _client(_FakeMessages(exc=err)).complete_json(purpose="p", system="s", user="u", schema={})


def test_missing_api_key_is_explicit():
    pytest.importorskip("anthropic")
    from app.llm.claude import ClaudeClient

    with pytest.raises(LLMError):
        ClaudeClient(api_key="", model="m")


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")
def test_real_ocr_extraction_marks_smudged_date_unverified(tmp_path):
    from app.ingestion.ocr import TesseractOCR

    def copy_date_line(user: str):
        line = next(ln for ln in user.splitlines() if "Date:" in ln)
        m = re.search(r"Date:\s*(\S+)", line)
        raw = m.group(0)
        value = re.sub(r"\[\[\?|\]\]", "", m.group(1))
        return {"fields": [{"name": "purchase_date", "found": True, "value": value, "page": 1, "source_text": raw}]}

    llm = ScriptedLLM({"extract_fields": [copy_date_line, copy_date_line]})
    c = assemble(Settings(data_dir=tmp_path, _env_file=None), llm=llm, index=InMemoryIndex(),
                 embedder=HashEmbedder(), reranker=OverlapReranker(), ocr=TesseractOCR())
    results = {}
    for name in ("receipt_hinge.png", "receipt_nonhinge.png"):
        doc, _ = c.pipeline.register(name, (SAMPLES / name).read_bytes(), scope="session:t")
        c.pipeline.ensure_extracted(doc["doc_id"])
        ev, _, _ = c.investigator.registry.tools["process_document"].extractor.extract(
            doc, c.pipeline.page_views(doc["doc_id"]), ["purchase_date"])
        results[name] = ev[0].verification.value
    assert results == {"receipt_hinge.png": "UNVERIFIED", "receipt_nonhinge.png": "VERIFIED"}
