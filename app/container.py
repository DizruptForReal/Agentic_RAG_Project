"""Wiring. `build_container` creates real components; tests build the same Container with fakes."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

from app.agent.loop import Investigator
from app.config import Settings
from app.db.repo import Repo
from app.ingestion.extract import FieldExtractor
from app.ingestion.ocr import OCREngine
from app.ingestion.pipeline import IngestionPipeline
from app.llm.base import LLMClient
from app.retrieval.base import Embedder, Reranker, SearchIndex
from app.retrieval.retriever import HybridRetriever
from app.tools.ask_user import AskUserTool
from app.tools.base import ToolRegistry
from app.tools.get_metadata import GetMetadataTool
from app.tools.process_document import ProcessDocumentTool
from app.tools.search_kb import SearchTool

log = logging.getLogger(__name__)


@dataclass
class Container:
    settings: Settings
    repo: Repo
    llm: LLMClient
    index: SearchIndex
    embedder: Embedder
    reranker: Reranker
    ocr: OCREngine
    pipeline: IngestionPipeline
    retriever: HybridRetriever
    registry: ToolRegistry
    investigator: Investigator

    def startup(self) -> dict:
        """Create tables, clear stale locks, mark interrupted investigations, resume documents."""
        self.repo.init()
        stale = self.repo.reset_stale_locks()
        interrupted = self.investigator.mark_interrupted()
        resumed = self.pipeline.resume_pending()
        self._warn_stale_settings()
        if self.settings.warm_up_models:
            threading.Thread(target=self._warm_up, name="model-warm-up", daemon=True).start()
        if stale or interrupted or resumed:
            log.info("recovery: stale locks=%s interrupted=%s resumed=%s", stale, interrupted, resumed)
        return {"stale_locks": stale, "interrupted": interrupted, "resumed_documents": resumed}

    def _warn_stale_settings(self) -> None:
        import os
        from pathlib import Path

        env = Path(".env").read_text(encoding="utf-8", errors="ignore") if Path(".env").exists() else ""
        if "RERANK_FLOOR" in os.environ or any(l.strip().startswith("RERANK_FLOOR") for l in env.splitlines()):
            log.warning("RERANK_FLOOR is no longer used (it rejected correct passages); RELEVANCE_FLOOR=%s applies. "
                        "Remove the RERANK_FLOOR line from .env.", self.settings.relevance_floor)

    def _warm_up(self) -> None:
        """Load the embedding and reranker models (and a local Ollama model) in the background so the first
        question is not slow."""
        try:
            self.embedder.embed_query("warm up")
            self.reranker.score("warm up", ["warm up"])
            log.info("retrieval models loaded")
        except Exception as exc:  # noqa: BLE001
            log.warning("model warm-up failed (models will load on first use): %s", exc)
        self._warm_up_ollama()

    def _warm_up_ollama(self) -> None:
        from app.llm.openai_compat import is_ollama, ollama_root

        s = self.settings
        if s.llm_provider.lower() not in ("ollama", "openai_compat") or not is_ollama(s.llm_base_url):
            return
        try:
            import httpx

            # An empty generate request loads the model into GPU memory and keeps it there.
            body = {"model": s.llm_model, "keep_alive": s.ollama_keep_alive}
            if s.llm_provider.lower() == "ollama":
                body["options"] = {"num_ctx": s.ollama_num_ctx}   # load with the context requests will use
            httpx.post(f"{ollama_root(s.llm_base_url)}/api/generate", timeout=300, json=body).raise_for_status()
            log.info("Ollama model %s loaded (kept for %s)", s.llm_model, s.ollama_keep_alive)
        except Exception as exc:  # noqa: BLE001
            log.warning("could not preload the Ollama model %s (is Ollama running? `ollama pull %s`): %s",
                        s.llm_model, s.llm_model, exc)


def assemble(settings: Settings, *, llm: LLMClient, index: SearchIndex, embedder: Embedder,
             reranker: Reranker, ocr: OCREngine, sleep=None) -> Container:
    repo = Repo(settings.db_path)
    repo.init()
    pipeline = IngestionPipeline(repo, ocr, index, embedder, settings)
    retriever = HybridRetriever(index, embedder, reranker, candidates=settings.retrieve_candidates,
                                top_k=settings.top_k, rrf_k=settings.rrf_k)
    extractor = FieldExtractor(llm, settings.ocr_conf_threshold, model=settings.llm_model_extract)
    tools = [SearchTool(retriever, settings.relevance_floor), GetMetadataTool(repo),
             ProcessDocumentTool(repo, pipeline, extractor), AskUserTool()]
    kwargs = {"sleep": sleep} if sleep else {}
    registry = ToolRegistry(tools, retries=settings.tool_retries, backoff_s=settings.tool_backoff_s, **kwargs)
    investigator = Investigator(repo=repo, llm=llm, registry=registry, pipeline=pipeline, settings=settings)
    return Container(settings, repo, llm, index, embedder, reranker, ocr, pipeline, retriever, registry, investigator)


def fallback_models(s: Settings) -> list[str]:
    raw = s.llm_fallback_models.strip()
    if raw.lower() == "none":
        return []
    if raw:
        return [m.strip() for m in raw.split(",") if m.strip()]
    if "api.groq.com" in s.llm_base_url and s.llm_model == "openai/gpt-oss-120b":
        return ["openai/gpt-oss-20b"]       # separate per-model quota on Groq
    return []


OLLAMA_DEFAULT_URL = "http://localhost:11434/v1"
OLLAMA_DEFAULT_MODEL = "gpt-oss:20b"


def apply_provider_defaults(s: Settings) -> None:
    """LLM_PROVIDER=ollama needs no other LLM settings: local URL, gpt-oss:20b, no key, longer timeout."""
    from app.llm.openai_compat import is_ollama

    if s.llm_provider.lower() == "ollama":
        s.llm_base_url = s.llm_base_url or OLLAMA_DEFAULT_URL
        s.llm_model = s.llm_model or OLLAMA_DEFAULT_MODEL
    if is_ollama(s.llm_base_url):
        s.llm_timeout_s = max(s.llm_timeout_s, 300.0)


def build_llm(s: Settings) -> LLMClient:
    apply_provider_defaults(s)
    provider = s.llm_provider.lower()
    if provider == "anthropic":
        from app.llm.claude import ClaudeClient

        return ClaudeClient(s.anthropic_api_key, s.llm_model, s.llm_max_tokens, s.llm_timeout_s)
    if provider == "ollama":
        # Native Ollama API: schema-constrained output, reasoning effort, context and keep-alive per request.
        from app.llm.ollama import OllamaClient

        # num_predict covers hidden reasoning as well as the reply, so allow more than for hosted models
        return OllamaClient(s.llm_base_url, s.llm_model, max_tokens=max(s.llm_max_tokens, 8192),
                            timeout_s=s.llm_timeout_s,
                            num_ctx=s.ollama_num_ctx, keep_alive=s.ollama_keep_alive,
                            reasoning_effort=s.llm_reasoning_effort)
    if provider in ("openai_compat", "openai", "groq", "gemini", "openrouter", "cerebras"):
        from app.llm.openai_compat import OpenAICompatClient

        return OpenAICompatClient(s.llm_base_url, s.llm_api_key, s.llm_model, max_tokens=s.llm_max_tokens,
                                  timeout_s=s.llm_timeout_s, mode=s.llm_json_mode,
                                  reasoning_effort=s.llm_reasoning_effort, fallback_models=fallback_models(s),
                                  max_wait_s=s.llm_max_wait_s)
    raise ValueError(f"unknown LLM_PROVIDER '{s.llm_provider}'")


def build_container(settings: Settings | None = None) -> Container:
    from app.ingestion.ocr import TesseractOCR
    from app.retrieval.embedder import BGEEmbedder
    from app.retrieval.es_client import ElasticIndex
    from app.retrieval.reranker import BGEReranker

    s = settings or Settings()
    return assemble(
        s,
        llm=build_llm(s),
        index=ElasticIndex(s.es_url, s.es_index, s.embed_dims, s.es_timeout_s),
        embedder=BGEEmbedder(s.embed_model),
        reranker=BGEReranker(s.rerank_model),
        ocr=TesseractOCR(s.tesseract_cmd),
    )
