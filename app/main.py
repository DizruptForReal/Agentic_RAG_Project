"""FastAPI application factory.

    uvicorn app.main:app --reload
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse

from app.api import documents, investigations
from app.container import Container, build_container

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def create_app(container: Container | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        c = app.state.container = container or build_container()
        app.state.recovery = c.startup()
        yield

    app = FastAPI(title="Evidence Investigator", version="1.0.0", lifespan=lifespan,
                  description="Agentic investigation over Elasticsearch, SQLite metadata and on-demand OCR.")
    app.include_router(investigations.router)
    app.include_router(documents.router)

    ui_file = Path(__file__).parent / "static" / "index.html"

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/ui")

    @app.get("/ui", include_in_schema=False)
    def ui():
        return FileResponse(ui_file, media_type="text/html")

    @app.get("/health", tags=["meta"])
    def health():
        c: Container = app.state.container
        try:
            es = c.index.count()
            es_ok = True
        except Exception as exc:  # noqa: BLE001
            es, es_ok = str(exc), False
        return {"status": "ok" if es_ok else "degraded", "elasticsearch": {"ok": es_ok, "chunks": es},
                "llm_model": c.settings.llm_model, "recovery": app.state.recovery}

    return app


app = create_app()
