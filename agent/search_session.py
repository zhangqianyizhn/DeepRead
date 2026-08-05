from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class SeenChunk:
    first_seen_round: int
    times_seen: int = 1


def normalize_requested_top_k(value: Any, default: int, maximum: int) -> int:
    """Normalize an agent-provided top_k without allowing unbounded context growth."""
    try:
        requested = int(value) if value is not None else int(default)
    except (TypeError, ValueError):
        requested = int(default)
    return max(1, min(requested, max(1, int(maximum))))


def chunk_id_for_result(result: Dict[str, Any]) -> Optional[str]:
    """Return a stable ID for the search hit anchor, excluding expanded neighbors."""
    ref = result.get("ref") or {}
    doc_id = ref.get("doc_id")
    node_id = ref.get("node_id")
    paragraph_index = ref.get("hit_paragraph_index")
    if paragraph_index is None:
        paragraph_indexes = ref.get("paragraph_indexes") or []
        paragraph_index = paragraph_indexes[0] if paragraph_indexes else None
    if doc_id is None or node_id is None or paragraph_index is None:
        return None
    return f"{doc_id}/{node_id}/{int(paragraph_index)}"


@dataclass
class SearchSessionState:
    """Per-agent-session retrieval history used for visible result pagination."""

    seen_chunks: Dict[str, SeenChunk] = field(default_factory=dict)

    def candidate_top_k(self, requested_top_k: int, candidate_limit: int) -> int:
        # In the worst case every previously seen chunk is ranked before the next
        # unseen hit, so retrieve requested + seen candidates in one backend call.
        return max(
            requested_top_k,
            min(int(candidate_limit), requested_top_k + len(self.seen_chunks)),
        )

    def paginate(
        self,
        search_result: Dict[str, Any],
        *,
        requested_top_k: int,
        round_id: int,
        candidate_top_k: int,
    ) -> Dict[str, Any]:
        """Return compact markers for repeats and full content for unseen hits."""
        if not isinstance(search_result, dict) or not search_result.get("ok", False):
            return search_result

        ranked_results = []
        new_result_count = 0
        seen_result_count = 0
        scanned_count = 0

        for rank, result in enumerate(search_result.get("results") or [], start=1):
            if new_result_count >= requested_top_k:
                break
            scanned_count += 1
            chunk_id = chunk_id_for_result(result)
            if chunk_id is None:
                # Preserve malformed/legacy hits rather than silently hiding them.
                ranked_results.append(result)
                new_result_count += 1
                continue

            prior = self.seen_chunks.get(chunk_id)
            if prior is not None:
                prior.times_seen += 1
                ranked_results.append(
                    {
                        "rank": rank,
                        "chunk_id": chunk_id,
                        "ref": result.get("ref"),
                        "score": result.get("score"),
                        "status": "ALREADY_SEEN_FULL_TEXT_AVAILABLE_IN_HISTORY",
                        "first_seen_round": prior.first_seen_round,
                        "times_seen": prior.times_seen,
                    }
                )
                seen_result_count += 1
                continue

            tagged = dict(result)
            tagged["chunk_id"] = chunk_id
            tagged["rank"] = rank
            tagged["status"] = "NEW_RESULT"
            ranked_results.append(tagged)
            new_result_count += 1
            self.seen_chunks[chunk_id] = SeenChunk(first_seen_round=round_id)

        out = dict(search_result)
        out["results"] = ranked_results
        out["pagination"] = {
            "requested_new_results": requested_top_k,
            "returned_new_results": new_result_count,
            "returned_seen_markers": seen_result_count,
            "scanned_candidates": scanned_count,
            "candidate_top_k": candidate_top_k,
            "total_unique_chunks_seen_in_session": len(self.seen_chunks),
            "hint": (
                "results preserves backend rank: NEW_RESULT entries contain full text; "
                "ALREADY_SEEN entries are compact references whose full text is already "
                "present in this conversation."
            ),
        }
        return out


@dataclass
class RetrievalStrategyState:
    """Detect repeated use of one retrieval channel without blocking the agent."""

    stagnation_threshold: int = 3
    recent_attempts: List[Tuple[str, str, Optional[str]]] = field(default_factory=list)
    seen_doc_ids: Set[str] = field(default_factory=set)

    def annotate(
        self,
        search_result: Dict[str, Any],
        *,
        tool_name: str,
        scope: str,
        doc_id: Optional[str],
    ) -> Dict[str, Any]:
        """Attach a visible warning after repeated use of the same search strategy."""
        if not isinstance(search_result, dict) or not search_result.get("ok", False):
            return search_result

        normalized_scope = str(scope or "full")
        normalized_doc_id = str(doc_id) if doc_id is not None else None
        signature = (str(tool_name), normalized_scope, normalized_doc_id)
        self.recent_attempts.append(signature)

        threshold = max(2, int(self.stagnation_threshold))
        if len(self.recent_attempts) > threshold:
            self.recent_attempts = self.recent_attempts[-threshold:]

        result_doc_ids = {
            str((item.get("ref") or {}).get("doc_id"))
            for item in (search_result.get("results") or [])
            if (item.get("ref") or {}).get("doc_id") is not None
        }
        new_doc_ids = sorted(result_doc_ids - self.seen_doc_ids)
        self.seen_doc_ids.update(result_doc_ids)

        same_channel_streak = 0
        for previous in reversed(self.recent_attempts):
            if previous == signature:
                same_channel_streak += 1
            else:
                break

        strategy = {
            "status": "CONTINUE",
            "same_channel_streak": same_channel_streak,
            "new_document_ids": new_doc_ids,
        }
        if same_channel_streak >= threshold:
            strategy.update(
                {
                    "status": "STRATEGY_STAGNATION",
                    "reason": (
                        f"{tool_name} has been used {same_channel_streak} consecutive "
                        f"times with scope={normalized_scope!r} and doc_id="
                        f"{normalized_doc_id!r}. New chunks do not necessarily mean "
                        "that the retrieval strategy is making semantic progress."
                    ),
                    "recommended_actions": [
                        "Switch retrieval method (for example BM25 to vector or regex).",
                        "Inspect the structure of a promising document before continuing.",
                        "Change scope or document instead of paging the same ranking further.",
                        "Answer now if the evidence already supports the requested conclusion.",
                    ],
                }
            )

        out = dict(search_result)
        out["retrieval_strategy"] = strategy
        return out
