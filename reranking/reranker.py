from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import yaml
from sentence_transformers import CrossEncoder


@dataclass
class RerankResult:
    doc_id: str
    doc_text: str
    reranker_score: float
    rank_before: int
    rank_after: int
    category: Optional[str]
    difficulty: Optional[str]
    metadata: dict


def _load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _extract_doc_text(doc: dict) -> str:
    parts = []
    if doc.get("question"):
        parts.append(doc["question"].strip())
    if doc.get("answer"):
        parts.append(doc["answer"].strip())
    return " | ".join(parts) if parts else doc.get("doc_text", "")


class CrossEncoderReranker:
    def __init__(self, config_path: str = "retrieval/configs/retrieval_config.yaml"):
        cfg = _load_config(config_path)
        rerank_cfg = cfg.get("reranker", {})

        model_name: str = rerank_cfg.get("model_id", "BAAI/bge-reranker-base")
        max_length: int = rerank_cfg.get("max_length", 512)
        device: Optional[str] = rerank_cfg.get("device", None)
        batch_size: int = rerank_cfg.get("batch_size", 16)

        self.model = CrossEncoder(
            model_name,
            max_length=max_length,
            device=device,
        )
        self.model_name = model_name
        self.batch_size = batch_size
        self._last_reranker_ms: float = 0.0

    def rerank(
        self,
        query: str,
        docs: list[dict],
        top_k: int = 3,
    ) -> tuple[list[RerankResult], float]:
        if not docs:
            self._last_reranker_ms = 0.0
            return [], 0.0

        t0 = time.perf_counter()

        pairs = [(query, _extract_doc_text(doc)) for doc in docs]
        raw_scores: list[float] = self.model.predict(
            pairs,
            show_progress_bar=False,
            batch_size=self.batch_size,
        ).tolist()

        t1 = time.perf_counter()
        self._last_reranker_ms = round((t1 - t0) * 1000, 2)

        indexed = list(zip(raw_scores, range(len(docs)), docs))
        indexed.sort(key=lambda x: x[0], reverse=True)

        results: list[RerankResult] = []
        for rank_after, (score, rank_before, doc) in enumerate(indexed[:top_k], 1):
            results.append(
                RerankResult(
                    doc_id=doc.get("doc_id", str(rank_before)),
                    doc_text=_extract_doc_text(doc),
                    reranker_score=round(float(score), 6),
                    rank_before=rank_before + 1,
                    rank_after=rank_after,
                    category=doc.get("category"),
                    difficulty=doc.get("difficulty"),
                    metadata={
                        k: doc[k]
                        for k in doc
                        if k not in ("question", "answer", "doc_id", "category", "difficulty")
                    },
                )
            )

        top_score = results[0].reranker_score if results else 0.0
        return results, top_score

    @property
    def reranker_ms(self) -> float:
        return self._last_reranker_ms