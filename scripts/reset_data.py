"""Reset local data.

    python -m scripts.reset_data --history --yes   # clear past investigations and session uploads, KEEP the corpus
    python -m scripts.reset_data --yes             # wipe everything (index + database + files), then:
    python -m scripts.seed_corpus

--history: deletes every investigation and every document that is not part of the global corpus (uploads made
during investigations), with their pages, chunks, index entries and files. The seeded corpus stays indexed, so
no re-seeding is needed.

Needed after the sample documents change: old document versions would otherwise stay indexed next to the
new ones. Stop the API first (on Windows an open database file cannot be deleted).
"""
from __future__ import annotations

import argparse
import shutil
import sys

from app.config import Settings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="confirm deletion")
    ap.add_argument("--history", action="store_true", help="only clear investigations and session uploads")
    args = ap.parse_args()
    s = Settings()
    if args.history:
        return reset_history(s, args.yes)
    targets = [s.db_path, s.db_path.with_name(s.db_path.name + "-wal"), s.db_path.with_name(s.db_path.name + "-shm")]
    print(f"This deletes the Elasticsearch index '{s.es_index}' at {s.es_url},")
    print(f"the SQLite database {s.db_path} and uploaded files in {s.upload_dir}.")
    if not args.yes:
        sys.exit("Nothing deleted. Re-run with --yes to confirm.")
    try:
        from elasticsearch import Elasticsearch

        es = Elasticsearch(s.es_url, request_timeout=10)
        es.indices.delete(index=s.es_index, ignore_unavailable=True)
        print(f"deleted index {s.es_index}")
    except Exception as exc:  # noqa: BLE001
        print(f"could not delete the index (is Elasticsearch running?): {exc}")
    for path in targets:
        try:
            if path.exists():
                path.unlink()
                print(f"deleted {path}")
        except PermissionError:
            sys.exit(f"{path} is in use - stop the API (Ctrl+C in the uvicorn terminal) and try again")
    if s.upload_dir.exists():
        shutil.rmtree(s.upload_dir)
        print(f"deleted {s.upload_dir}")
    print("done - now run: python -m scripts.seed_corpus")


def reset_history(s: Settings, yes: bool) -> None:
    import sqlite3
    from pathlib import Path

    if not s.db_path.exists():
        sys.exit(f"no database at {s.db_path} - nothing to clear")
    con = sqlite3.connect(s.db_path, timeout=5)
    docs = con.execute("SELECT doc_id, filename, path FROM documents WHERE scope != 'global'").fetchall()
    n_inv = con.execute("SELECT COUNT(*) FROM investigations").fetchone()[0]
    n_corpus = con.execute("SELECT COUNT(*) FROM documents WHERE scope = 'global'").fetchone()[0]
    print(f"This deletes {n_inv} investigation(s) and {len(docs)} session upload(s): "
          f"{', '.join(f for _, f, _ in docs) or 'none'}.")
    print(f"The {n_corpus} corpus document(s) stay indexed.")
    if not yes:
        sys.exit("Nothing deleted. Re-run with --yes to confirm.")
    ids = [d for d, _, _ in docs]
    if ids:
        try:
            from elasticsearch import Elasticsearch

            es = Elasticsearch(s.es_url, request_timeout=10)
            r = es.delete_by_query(index=s.es_index, query={"terms": {"doc_id": ids}}, refresh=True,
                                   ignore_unavailable=True)
            print(f"removed {r.get('deleted', 0)} indexed chunk(s) of session uploads")
        except Exception as exc:  # noqa: BLE001
            sys.exit(f"could not reach Elasticsearch to remove the uploads' chunks ({exc}); start it and retry")
    try:
        with con:
            marks = ",".join("?" * len(ids))
            if ids:
                for table in ("chunks", "pages", "documents"):
                    con.execute(f"DELETE FROM {table} WHERE doc_id IN ({marks})", ids)
            con.execute("DELETE FROM investigations")
    except sqlite3.OperationalError as exc:
        sys.exit(f"database is busy ({exc}) - stop the API (Ctrl+C in the uvicorn terminal) and try again")
    con.close()
    for _, _, path in docs:
        try:
            Path(path).unlink(missing_ok=True)
        except OSError as exc:
            print(f"could not delete {path}: {exc}")
    print("done - start the API again: uvicorn app.main:app --port 8000")


if __name__ == "__main__":
    main()
