"""Application settings. Every value can be overridden in `.env` or the environment."""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- storage -----------------------------------------------------------
    data_dir: Path = Path("data")
    db_path: Path | None = None          # default: <data_dir>/investigator.db
    upload_dir: Path | None = None       # default: <data_dir>/uploads

    # --- elasticsearch -----------------------------------------------------
    es_url: str = "http://localhost:9200"
    es_index: str = "chunks"
    es_timeout_s: float = 10.0

    # --- llm ---------------------------------------------------------------
    llm_provider: str = "ollama"          # ollama (default, local) | openai_compat (any OpenAI-compatible API) | anthropic
    llm_base_url: str = ""                # empty = http://localhost:11434/v1 for ollama
    llm_api_key: str = ""                 # openai_compat only (leave empty for Ollama)
    llm_json_mode: str = "auto"           # openai_compat only: auto | tools | json
    llm_fallback_models: str = ""         # comma list used when the primary model is rate-limited;
                                          # empty = automatic for some hosted providers, "none" = off
    llm_reasoning_effort: str | None = None  # openai_compat: low|medium|high; auto "low" for gpt-oss models
    anthropic_api_key: str = ""
    llm_model: str = "gpt-oss:20b"
    llm_model_extract: str | None = None  # optional separate model for field extraction
    llm_max_tokens: int = 4096           # reasoning models spend part of this on hidden reasoning
    llm_timeout_s: float = 60.0           # raised to at least 300 s for a local Ollama model (first load, long prompts)
    ollama_keep_alive: str = "30m"        # Ollama: keep the model loaded between questions
    ollama_num_ctx: int = 16384           # Ollama: context window, sent with every request
    llm_max_wait_s: float = 65.0          # longest rate-limit wait worth sitting through (per-minute limits
                                          # recover within ~60 s; daily quotas take minutes -> fail fast)

    # --- embeddings / reranker ----------------------------------------------
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dims: int = 384
    rerank_model: str = "BAAI/bge-reranker-base"

    # --- ocr ---------------------------------------------------------------
    tesseract_cmd: str | None = None      # e.g. C:\Program Files\Tesseract-OCR\tesseract.exe
    ocr_dpi: int = 300
    ocr_conf_threshold: float = 80.0      # words below this are UNVERIFIED

    # --- ingestion ---------------------------------------------------------
    text_layer_min_chars: int = 40        # fewer native chars on a page -> OCR that page
    chunk_words: int = 220
    chunk_overlap: int = 30

    # --- retrieval ---------------------------------------------------------
    retrieve_candidates: int = 20
    warm_up_models: bool = True           # load BGE models at startup instead of on the first question
    top_k: int = 6
    rrf_k: int = 60
    # Chunks scoring below this (bge-reranker-base, 0..1) cannot SUPPORT a sub-question. Calibrated on real
    # scores: correct passages for casually phrased questions scored 0.035-0.15, unrelated ones < 0.03.
    # Re-check with: python -m scripts.eval_retrieval. (Renamed from RERANK_FLOOR, whose old 0.2 default
    # rejected correct passages; an old RERANK_FLOOR line in .env is now ignored.)
    relevance_floor: float = 0.03

    # --- agent -------------------------------------------------------------
    max_steps: int = 12
    initial_search: bool = True           # search the question before the first LLM call (fast path)
    initial_top_k: int = 10              # wider first look: paraphrased questions often rank the right section 4-10
    auto_read_attachments: bool = True    # index + search an attached document before the first planner call
    min_actions_before_ask: int = 3       # investigation actions required before ask_user is allowed
    max_user_questions: int = 1           # per investigation; beyond this the agent finalizes and explains
    max_blocked_repeats: int = 3
    max_consecutive_failures: int = 3
    tool_retries: int = 2
    tool_backoff_s: float = 0.5

    def model_post_init(self, __context) -> None:  # noqa: D401
        if self.db_path is None:
            self.db_path = self.data_dir / "investigator.db"
        if self.upload_dir is None:
            self.upload_dir = self.data_dir / "uploads"
