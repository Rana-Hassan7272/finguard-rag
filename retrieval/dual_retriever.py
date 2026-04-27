"""
Dual-source retrieval over QA + PDF corpora.
"""

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from retrieval.bm25_retriever import BM25Retriever
from retrieval.fusion import fuse
from retrieval.normalization import tokenize
from retrieval.vector_retriever import VectorRetriever

CONFIG_PATH = Path("retrieval/configs/retrieval_config.yaml")
PDF_FAISS_INDEX_PATH = Path("retrieval/artifacts/pdf_faiss.index")
PDF_FAISS_META_PATH = Path("retrieval/artifacts/pdf_faiss_meta.pkl")
PDF_BM25_PATH = Path("retrieval/artifacts/pdf_bm25.pkl")


@dataclass
class DualRetrievalResult:
    docs: list[dict]
    diagnostics: dict


def _load_cfg() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _intent_source_weights(intent: str) -> tuple[float, float]:
    if intent == "legal":
        return (0.35, 0.65)  # qa, pdf
    if intent == "practical":
        return (0.70, 0.30)
    return (0.50, 0.50)


def _language_retriever_weights(cfg: dict, language: str) -> tuple[float, float]:
    lw = cfg["fusion"]["language_weights"].get(language) or cfg["fusion"]["language_weights"]["english"]
    return float(lw["bm25_weight"]), float(lw["vector_weight"])


class DualRetriever:
    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = cfg or _load_cfg()
        self.qa_vector = VectorRetriever(self.cfg)
        self.qa_bm25 = BM25Retriever(self.cfg)

        self.embedder = None
        self.pdf_faiss = None
        self.pdf_docs = []
        self.pdf_bm25_payload = None

    def _ensure_pdf_loaded(self) -> None:
        if self.embedder is None:
            from sentence_transformers import SentenceTransformer

            model_name = self.cfg["embedding_model"].get("local_path") or self.cfg["embedding_model"]["hub_id"]
            self.embedder = SentenceTransformer(model_name)

        if self.pdf_faiss is None:
            import faiss

            self.pdf_faiss = faiss.read_index(str(PDF_FAISS_INDEX_PATH))
            with open(PDF_FAISS_META_PATH, "rb") as f:
                meta = pickle.load(f)
            self.pdf_docs = meta.get("docs", [])

        if self.pdf_bm25_payload is None:
            with open(PDF_BM25_PATH, "rb") as f:
                self.pdf_bm25_payload = pickle.load(f)

    def _retrieve_pdf(self, query: str, k: int, bm25_w: float, vec_w: float, category_filter: Optional[str]) -> list[dict]:
        self._ensure_pdf_loaded()

        q_vec = self.embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        vec_scores, vec_idx = self.pdf_faiss.search(q_vec, k)
        vec_ranked = [(int(i), float(s)) for i, s in zip(vec_idx[0], vec_scores[0]) if i != -1]

        bm25_scores = self.pdf_bm25_payload["bm25"].get_scores(tokenize(query))
        bm25_idx = np.argsort(bm25_scores)[::-1][:k]
        bm25_ranked = [(int(i), float(bm25_scores[i])) for i in bm25_idx if bm25_scores[i] > 0]

        fused: dict[int, float] = {}
        for rank, (row, _) in enumerate(bm25_ranked, start=1):
            fused[row] = fused.get(row, 0.0) + bm25_w / (60 + rank)
        for rank, (row, _) in enumerate(vec_ranked, start=1):
            fused[row] = fused.get(row, 0.0) + vec_w / (60 + rank)

        rows = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]
        out = []
        for row, score in rows:
            if 0 <= row < len(self.pdf_docs):
                d = dict(self.pdf_docs[row])
                if category_filter and d.get("category") != category_filter:
                    continue
                d["_rrf_score"] = float(score)
                d["_source"] = "pdf"
                out.append(d)
        return out

    def retrieve(
        self,
        query: str,
        language: str = "english",
        query_intent: str = "practical",
        category_filter: Optional[str] = None,
        top_k: int = 10,
    ) -> DualRetrievalResult:
        bm25_w, vec_w = _language_retriever_weights(self.cfg, language)
        qa_w, pdf_w = _intent_source_weights(query_intent)

        qa_vector = self.qa_vector.retrieve(query=query, k=top_k, category_filter=category_filter)
        qa_bm25 = self.qa_bm25.retrieve(query=query, k=top_k, category_filter=category_filter)
        qa_fused = fuse(qa_vector, qa_bm25, bm25_weight=bm25_w, vector_weight=vec_w, rrf_k=60, top_k=top_k)

        qa_docs = []
        for r in qa_fused:
            doc = dict(r.doc)
            doc["_rrf_score"] = float(r.rrf_score) * qa_w
            doc["_source"] = "qa"
            qa_docs.append(doc)

        pdf_docs = self._retrieve_pdf(query, top_k, bm25_w, vec_w, category_filter)
        for d in pdf_docs:
            d["_rrf_score"] = float(d["_rrf_score"]) * pdf_w

        merged = qa_docs + pdf_docs
        merged.sort(key=lambda d: d.get("_rrf_score", 0.0), reverse=True)
        merged = merged[:top_k]

        return DualRetrievalResult(
            docs=merged,
            diagnostics={
                "language": language,
                "query_intent": query_intent,
                "qa_weight": qa_w,
                "pdf_weight": pdf_w,
                "qa_docs": len(qa_docs),
                "pdf_docs": len(pdf_docs),
                "pdf_vector_mode": "faiss",
            },
        )


if __name__ == "__main__":
    dr = DualRetriever()
    out = dr.retrieve("zakat calculation in pakistan", language="english", query_intent="legal", top_k=10)
    print(json.dumps(out.diagnostics, indent=2))
    print(f"Retrieved docs: {len(out.docs)}")
