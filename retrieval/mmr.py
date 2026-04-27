from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class MMRResult:
    doc_id: str
    relevance_score: float
    novelty_score: float
    mmr_score: float
    rank: int
    metadata: dict = field(default_factory=dict)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _cosine_matrix(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(candidates, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    normed = candidates / norms
    q_norm = query / (np.linalg.norm(query) + 1e-10)
    return normed @ q_norm


def run_mmr(
    query_embedding: np.ndarray,
    candidate_ids: list[str],
    candidate_embeddings: np.ndarray,
    candidate_scores: list[float],
    top_k: int = 10,
    lambda_param: float = 0.7,
    metadata: Optional[list[dict]] = None,
) -> list[MMRResult]:
    if not candidate_ids:
        return []

    if metadata is None:
        metadata = [{} for _ in candidate_ids]

    n = len(candidate_ids)
    top_k = min(top_k, n)

    relevance = _cosine_matrix(query_embedding, candidate_embeddings)

    fused_relevance = 0.6 * relevance + 0.4 * np.array(
        [s / (max(candidate_scores) + 1e-10) for s in candidate_scores]
    )

    selected_indices: list[int] = []
    remaining = list(range(n))
    results: list[MMRResult] = []

    for rank in range(top_k):
        if not remaining:
            break

        best_idx = None
        best_mmr = -float("inf")
        best_novelty = 0.0

        for i in remaining:
            rel = float(fused_relevance[i])

            if not selected_indices:
                novelty = 1.0
            else:
                sel_embs = candidate_embeddings[selected_indices]
                sims_to_selected = [
                    _cosine_sim(candidate_embeddings[i], sel_embs[j])
                    for j in range(len(sel_embs))
                ]
                novelty = 1.0 - max(sims_to_selected)

            mmr = lambda_param * rel + (1.0 - lambda_param) * novelty

            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = i
                best_novelty = novelty

        selected_indices.append(best_idx)
        remaining.remove(best_idx)

        results.append(
            MMRResult(
                doc_id=candidate_ids[best_idx],
                relevance_score=round(float(fused_relevance[best_idx]), 4),
                novelty_score=round(best_novelty, 4),
                mmr_score=round(best_mmr, 4),
                rank=rank + 1,
                metadata=metadata[best_idx],
            )
        )

    return results