from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.container import Container, assemble
from app.main import create_app
from tests.fakes import CONTRACT_SCAN, FakeOCR, HashEmbedder, InMemoryIndex, OverlapReranker, ScriptedLLM

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


@pytest.fixture
def settings(tmp_path) -> Settings:
    # initial_search=False keeps the scripted scenarios deterministic; the fast path has its own tests.
    return Settings(data_dir=tmp_path, anthropic_api_key="test", tool_backoff_s=0.0, relevance_floor=0.2,
                    max_steps=12, initial_search=False, auto_read_attachments=False,
                    _env_file=None)


@pytest.fixture
def llm() -> ScriptedLLM:
    return ScriptedLLM()


@pytest.fixture
def ocr() -> FakeOCR:
    return FakeOCR(default=CONTRACT_SCAN)


@pytest.fixture
def index() -> InMemoryIndex:
    return InMemoryIndex()


@pytest.fixture
def container(settings, llm, ocr, index) -> Container:
    return assemble(settings, llm=llm, index=index, embedder=HashEmbedder(), reranker=OverlapReranker(),
                    ocr=ocr, sleep=lambda s: None)


def seed(container: Container) -> dict[str, str]:
    """Index the sample corpus globally. Returns {filename: doc_id}."""
    ids = {}
    for m in json.loads((SAMPLES / "manifest.json").read_text()):
        content = (SAMPLES / m["file"]).read_bytes()
        meta = {k: m[k] for k in ("title", "doc_type", "product", "effective_date", "supersedes")}
        doc, _ = container.pipeline.register(m["file"], content, scope="global", meta=meta)
        container.pipeline.ensure_indexed(doc["doc_id"])
        ids[m["file"]] = doc["doc_id"]
    return ids


@pytest.fixture
def seeded(container) -> dict[str, str]:
    return seed(container)


@pytest.fixture
def client(container):
    with TestClient(create_app(container)) as c:
        yield c
