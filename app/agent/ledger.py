"""Evidence ledger: the only things an answer is allowed to cite."""
from __future__ import annotations

from app.agent.models import EvidenceItem, InvestigationState, LedgerEntry


def add_evidence(state: InvestigationState, items: list[EvidenceItem]) -> list[str]:
    """Add items (deduplicated) and return their ledger ids in order."""
    by_key = {e.dedupe_key(): e for e in state.ledger}
    ids = []
    for item in items:
        key = item.dedupe_key()
        existing = by_key.get(key)
        if existing:
            # keep the best relevance score seen for a re-retrieved chunk
            if item.rerank_score is not None and (existing.rerank_score or -1) < item.rerank_score:
                existing.rerank_score = item.rerank_score
            if item.retrieval_rank is not None and (existing.retrieval_rank or 10**6) > item.retrieval_rank:
                existing.retrieval_rank = item.retrieval_rank
            ids.append(existing.id)
            continue
        entry = LedgerEntry(id=f"e{len(state.ledger) + 1}", **item.model_dump())
        state.ledger.append(entry)
        by_key[key] = entry
        ids.append(entry.id)
    return ids


def get_entries(state: InvestigationState, ids: list[str]) -> tuple[list[LedgerEntry], list[str]]:
    """Return (found entries, unknown ids)."""
    index = {e.id: e for e in state.ledger}
    return [index[i] for i in ids if i in index], [i for i in ids if i not in index]
