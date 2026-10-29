"""SQLite repository for documents, pages, chunks and investigations.

One short-lived connection per operation (safe across FastAPI worker threads), WAL mode and a busy timeout.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from app.agent.models import InvestigationState, InvStatus, utcnow

SCHEMA = (Path(__file__).parent / "schema.sql").read_text(encoding="utf-8")

DOC_FIELDS = {
    "sha256", "filename", "mime", "path", "size_bytes", "title", "doc_type", "product", "effective_date",
    "supersedes", "scope", "status", "target_stage", "processing", "error", "error_stage", "page_count",
    "triage_json",
}


class Repo:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 30000")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)
            cols = {r[1] for r in c.execute("PRAGMA table_info(chunks)")}
            if "section" not in cols:  # migrate databases created before section-aware chunking
                c.execute("ALTER TABLE chunks ADD COLUMN section TEXT")

    # ------------------------------------------------------------ documents
    def insert_document(self, doc: dict[str, Any]) -> None:
        now = utcnow()
        row = {**doc, "created_at": now, "updated_at": now}
        cols = ", ".join(row)
        marks = ", ".join(f":{k}" for k in row)
        with self._conn() as c:
            c.execute(f"INSERT INTO documents ({cols}) VALUES ({marks})", row)

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        with self._conn() as c:
            r = c.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        return dict(r) if r else None

    def get_document_by_hash(self, sha256: str) -> dict[str, Any] | None:
        with self._conn() as c:
            r = c.execute("SELECT * FROM documents WHERE sha256 = ?", (sha256,)).fetchone()
        return dict(r) if r else None

    def list_documents(self, *, doc_ids: list[str] | None = None, doc_type: str | None = None,
                       product: str | None = None, visible_scopes: list[str] | None = None) -> list[dict]:
        sql, args = "SELECT * FROM documents WHERE 1=1", []
        if doc_ids:
            sql += f" AND doc_id IN ({','.join('?' * len(doc_ids))})"
            args += doc_ids
        if doc_type:
            sql += " AND lower(doc_type) = lower(?)"
            args.append(doc_type)
        if product:
            sql += " AND lower(product) = lower(?)"
            args.append(product)
        if visible_scopes:
            sql += f" AND scope IN ({','.join('?' * len(visible_scopes))})"
            args += visible_scopes
        with self._conn() as c:
            return [dict(r) for r in c.execute(sql + " ORDER BY created_at", args).fetchall()]

    def update_document(self, doc_id: str, **fields: Any) -> None:
        bad = set(fields) - DOC_FIELDS
        if bad:
            raise ValueError(f"unknown document fields: {bad}")
        fields["updated_at"] = utcnow()
        sets = ", ".join(f"{k} = :{k}" for k in fields)
        with self._conn() as c:
            c.execute(f"UPDATE documents SET {sets} WHERE doc_id = :doc_id", {**fields, "doc_id": doc_id})

    def try_lock(self, doc_id: str) -> bool:
        """Atomically claim a document for processing. False if someone else holds it."""
        with self._conn() as c:
            cur = c.execute(
                "UPDATE documents SET processing = 1, updated_at = ? WHERE doc_id = ? AND processing = 0",
                (utcnow(), doc_id),
            )
            return cur.rowcount == 1

    def unlock(self, doc_id: str) -> None:
        with self._conn() as c:
            c.execute("UPDATE documents SET processing = 0 WHERE doc_id = ?", (doc_id,))

    def reset_stale_locks(self) -> list[str]:
        """Called at startup: any lock left behind belongs to a process that died."""
        with self._conn() as c:
            ids = [r[0] for r in c.execute("SELECT doc_id FROM documents WHERE processing = 1")]
            c.execute("UPDATE documents SET processing = 0 WHERE processing = 1")
        return ids

    def documents_needing_resume(self) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM documents WHERE target_stage IS NOT NULL AND status != target_stage"
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------ pages
    def upsert_page(self, doc_id: str, page: int, method: str, text: str, mean_conf: float | None,
                    words: list[tuple[str, float | None]]) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO pages (doc_id, page, method, text, mean_conf, words_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (doc_id, page, method, text, mean_conf, json.dumps(words)),
            )

    def get_pages(self, doc_id: str) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM pages WHERE doc_id = ? ORDER BY page", (doc_id,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["words"] = [tuple(w) for w in json.loads(d.pop("words_json"))]
            out.append(d)
        return out

    def text_mentions(self, needle: str, scopes: list[str], doc_ids: list[str]) -> bool:
        """Does any extracted page of a visible document contain `needle` (case-insensitive)?"""
        sql = ("SELECT 1 FROM pages p JOIN documents d ON d.doc_id = p.doc_id WHERE instr(upper(p.text), ?) > 0 "
               f"AND (d.scope IN ({','.join('?' * len(scopes))})")
        args: list[Any] = [needle.upper(), *scopes]
        if doc_ids:
            sql += f" OR d.doc_id IN ({','.join('?' * len(doc_ids))})"
            args += doc_ids
        with self._conn() as c:
            return c.execute(sql + ") LIMIT 1", args).fetchone() is not None

    # ------------------------------------------------------------ chunks
    def replace_chunks(self, doc_id: str, chunks: list[dict[str, Any]]) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            c.executemany(
                "INSERT INTO chunks (chunk_id, doc_id, page, idx, text, section, min_conf, verification) "
                "VALUES (:chunk_id, :doc_id, :page, :idx, :text, :section, :min_conf, :verification)",
                [{"section": None, **c} for c in chunks],
            )

    def get_chunks(self, doc_id: str) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM chunks WHERE doc_id = ? ORDER BY page, idx", (doc_id,)).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------ investigations
    def save_investigation(self, state: InvestigationState) -> None:
        state.updated_at = utcnow()
        outcome = state.final.outcome.value if state.final else None
        with self._conn() as c:
            c.execute(
                "INSERT INTO investigations (id, question, status, outcome, state_json, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET status = excluded.status, "
                "outcome = excluded.outcome, state_json = excluded.state_json, updated_at = excluded.updated_at",
                (state.id, state.question, state.status.value, outcome, state.model_dump_json(),
                 state.created_at, state.updated_at),
            )

    def get_investigation(self, inv_id: str) -> InvestigationState | None:
        with self._conn() as c:
            r = c.execute("SELECT status, state_json FROM investigations WHERE id = ?", (inv_id,)).fetchone()
        if not r:
            return None
        state = InvestigationState.model_validate_json(r[1])
        state.status = InvStatus(r[0])  # the column is authoritative (claim_investigation updates it)
        return state

    def claim_investigation(self, inv_id: str) -> bool:
        """Atomically move an investigation to RUNNING. False if it is already running elsewhere."""
        with self._conn() as c:
            cur = c.execute(
                "UPDATE investigations SET status = 'RUNNING', updated_at = ? WHERE id = ? AND status != 'RUNNING'",
                (utcnow(), inv_id),
            )
            return cur.rowcount == 1

    def list_investigations(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("SELECT id, question, status, outcome, created_at, updated_at FROM investigations "
                             "ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def investigation_ids_with_status(self, status: str) -> list[str]:
        with self._conn() as c:
            return [r[0] for r in c.execute("SELECT id FROM investigations WHERE status = ?", (status,))]
