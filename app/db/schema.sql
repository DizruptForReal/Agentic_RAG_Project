PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS documents (
    doc_id         TEXT PRIMARY KEY,           -- d_<first 16 hex of sha256>
    sha256         TEXT NOT NULL UNIQUE,       -- duplicate detection
    filename       TEXT NOT NULL,
    mime           TEXT NOT NULL,
    path           TEXT NOT NULL,
    size_bytes     INTEGER NOT NULL,
    title          TEXT,
    doc_type       TEXT,                       -- policy | amendment | manual | contract | receipt | other
    product        TEXT,
    effective_date TEXT,                       -- ISO date
    supersedes     TEXT,                       -- free text reference, e.g. "AQF-WP-1.0 section 2"
    scope          TEXT NOT NULL DEFAULT 'global',  -- global | session:<investigation_id>
    status         TEXT NOT NULL,              -- last COMPLETED stage
    target_stage   TEXT,                       -- stage the pipeline is trying to reach
    processing     INTEGER NOT NULL DEFAULT 0, -- 1 while a worker holds the document
    error          TEXT,
    error_stage    TEXT,
    page_count     INTEGER,
    triage_json    TEXT,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pages (
    doc_id     TEXT NOT NULL,
    page       INTEGER NOT NULL,
    method     TEXT NOT NULL,                  -- native | ocr
    text       TEXT NOT NULL,
    mean_conf  REAL,
    words_json TEXT NOT NULL,                  -- [[word, conf|null], ...]
    PRIMARY KEY (doc_id, page)
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     TEXT PRIMARY KEY,             -- <doc_id>:<page>:<idx>
    doc_id       TEXT NOT NULL,
    page         INTEGER NOT NULL,
    idx          INTEGER NOT NULL,
    text         TEXT NOT NULL,
    section      TEXT,
    min_conf     REAL,
    verification TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS investigations (
    id         TEXT PRIMARY KEY,
    question   TEXT NOT NULL,
    status     TEXT NOT NULL,
    outcome    TEXT,
    state_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_inv_status ON investigations(status);
CREATE INDEX IF NOT EXISTS idx_docs_status ON documents(status);
