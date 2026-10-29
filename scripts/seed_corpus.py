"""Index the sample corpus into Elasticsearch + SQLite (global scope).

    python -m scripts.seed_corpus

Idempotent: re-running skips documents that are already indexed (same bytes = same doc_id).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from app.config import Settings
from app.container import build_container

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


def wait_for_es(container, seconds: int = 90) -> None:
    deadline = time.time() + seconds
    while True:
        try:
            container.index.ensure_index()
            return
        except Exception as exc:  # noqa: BLE001
            if time.time() > deadline:
                sys.exit(f"Elasticsearch not reachable at {container.settings.es_url}: {exc}")
            print("waiting for Elasticsearch ...")
            time.sleep(3)


def main() -> None:
    c = build_container(Settings())
    c.startup()
    wait_for_es(c)
    manifest = json.loads((SAMPLES / "manifest.json").read_text(encoding="utf-8"))
    print(f"{'file':32} {'doc_id':20} {'status':9} pages(ocr)")
    for m in manifest:
        meta = {k: m[k] for k in ("title", "doc_type", "product", "effective_date", "supersedes")}
        doc, dup = c.pipeline.register(m["file"], (SAMPLES / m["file"]).read_bytes(), scope="global", meta=meta)
        doc = c.pipeline.ensure_indexed(doc["doc_id"], scope="global")
        ocr = [t["page"] for t in json.loads(doc["triage_json"]) if t["method"] == "ocr"]
        print(f"{m['file']:32} {doc['doc_id']:20} {doc['status']:9} {doc['page_count']}({ocr or '-'})"
              + ("  [already present]" if dup else ""))
    print(f"chunks in index: {c.index.count()}")


if __name__ == "__main__":
    main()
