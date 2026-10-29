"""get_metadata: document records from SQLite (type, product, effective date, supersedes, state)."""
from __future__ import annotations

import json
from typing import ClassVar

from pydantic import BaseModel

from app.agent.models import EvidenceItem, EvidenceKind, Observation, ObsStatus
from app.db.repo import Repo
from app.tools.base import ToolContext


class MetadataArgs(BaseModel):
    doc_ids: list[str] | None = None
    doc_type: str | None = None
    product: str | None = None


class GetMetadataTool:
    name: ClassVar[str] = "get_metadata"
    description: ClassVar[str] = (
        "Look up document metadata (title, doc_type, product, effective_date, supersedes, processing "
        "status, OCR need) by doc_ids or filters. Use it to resolve conflicts between versions.")
    Args: ClassVar[type[BaseModel]] = MetadataArgs

    def __init__(self, repo: Repo):
        self.repo = repo

    def run(self, args: MetadataArgs, ctx: ToolContext) -> Observation:
        scopes = ["global", f"session:{ctx.investigation_id}"]
        docs = self.repo.list_documents(doc_ids=args.doc_ids, doc_type=args.doc_type, product=args.product)
        docs = [d for d in docs if d["scope"] in scopes or d["doc_id"] in ctx.attachments]
        if not docs:
            return Observation(tool=self.name, status=ObsStatus.EMPTY, summary="no matching documents")
        evidence = []
        for d in docs:
            triage = json.loads(d["triage_json"] or "[]")
            ocr_pages = [t["page"] for t in triage if t["method"] == "ocr"]
            text = (f"title={d['title']!r}; doc_type={d['doc_type']}; product={d['product']}; "
                    f"document_effective_date={d['effective_date']} (when this DOCUMENT took effect - not a "
                    f"purchase, delivery or installation date of any unit); supersedes={d['supersedes']}; "
                    f"pages={d['page_count']}; ocr_pages={ocr_pages}; status={d['status']}")
            evidence.append(EvidenceItem(kind=EvidenceKind.METADATA, doc_id=d["doc_id"],
                                         doc_name=d["title"] or d["filename"], text=text))
        return Observation(tool=self.name, status=ObsStatus.OK, summary=f"{len(docs)} document records",
                           evidence=evidence)
