"""The setup doctor (scripts/doctor.py): read-only checks with an exact fix for every problem."""
from __future__ import annotations

import httpx

from scripts import doctor


def _fake_get(routes):
    def get(url, timeout=None):
        for suffix, resp in routes.items():
            if url.endswith(suffix):
                if isinstance(resp, Exception):
                    raise resp
                return httpx.Response(200, json=resp) if not isinstance(resp, httpx.Response) else resp
        raise httpx.ConnectError("refused")
    return get


def test_elasticsearch_running_and_corpus_seeded(monkeypatch):
    monkeypatch.setattr(httpx, "get", _fake_get({":9200": {"version": {"number": "8.15.3"}},
                                                 "/chunks/_count": {"count": 40}}))
    es, corpus = doctor.check_elasticsearch("http://localhost:9200", "chunks")
    assert es.status == "OK" and "8.15.3" in es.detail
    assert corpus.status == "OK" and "40 chunks" in corpus.detail


def test_elasticsearch_down_gives_the_compose_command(monkeypatch):
    monkeypatch.setattr(httpx, "get", _fake_get({}))
    es, corpus = doctor.check_elasticsearch("http://localhost:9200", "chunks")
    assert es.status == "FAIL" and "docker compose up -d elasticsearch" in es.fix
    assert corpus.status == "WARN"


def test_unseeded_corpus_points_to_the_seed_script(monkeypatch):
    monkeypatch.setattr(httpx, "get", _fake_get({":9200": {"version": {"number": "8.15.3"}},
                                                 "/chunks/_count": httpx.Response(404, json={})}))
    _, corpus = doctor.check_elasticsearch("http://localhost:9200", "chunks")
    assert corpus.status == "WARN" and corpus.fix == ["python -m scripts.seed_corpus"]


def test_ollama_with_model_downloaded(monkeypatch):
    monkeypatch.setattr(httpx, "get", _fake_get({"/api/version": {"version": "0.12.6"},
                                                 "/api/tags": {"models": [{"name": "gpt-oss:20b"}]}}))
    olla, model = doctor.check_ollama("http://localhost:11434/v1", "gpt-oss:20b")
    assert olla.status == "OK" and model.status == "OK"


def test_missing_model_and_old_ollama_are_failures_with_fixes(monkeypatch):
    monkeypatch.setattr(httpx, "get", _fake_get({"/api/version": {"version": "0.5.1"},
                                                 "/api/tags": {"models": [{"name": "llama3:latest"}]}}))
    olla, model = doctor.check_ollama("http://localhost:11434/v1", "gpt-oss:20b")
    assert olla.status == "FAIL" and "too old" in olla.detail
    assert model.status == "FAIL" and model.fix == ["ollama pull gpt-oss:20b"]


def test_ollama_not_running(monkeypatch):
    monkeypatch.setattr(httpx, "get", _fake_get({}))
    olla, model = doctor.check_ollama("http://localhost:11434/v1", "gpt-oss:20b")
    assert olla.status == "FAIL" and "ollama.com/download" in olla.fix[0]


def test_tesseract_not_found_explains_the_fix(monkeypatch):
    import pytesseract

    def missing():
        raise pytesseract.TesseractNotFoundError()

    monkeypatch.setattr(pytesseract, "get_tesseract_version", missing)
    r = doctor.check_tesseract(None)
    assert r.status == "FAIL" and r.fix


def test_docker_missing(monkeypatch):
    monkeypatch.setattr(doctor.shutil, "which", lambda name: None)
    r = doctor.check_docker()
    assert r.status == "FAIL" and "docker-desktop" in r.fix[0]


def test_version_parsing():
    assert doctor._version_tuple("0.12.6") >= doctor.OLLAMA_MIN
    assert doctor._version_tuple("0.11.10-rc1") >= doctor.OLLAMA_MIN
    assert doctor._version_tuple("0.10.1") < doctor.OLLAMA_MIN
