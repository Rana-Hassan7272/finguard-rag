from __future__ import annotations

import csv
import json
import time
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

try:
    from retrieval.normalization import normalize_query
    from retrieval.language import detect_language, get_fusion_weights
    from retrieval.vector_retriever import VectorRetriever
    from retrieval.bm25_retriever import BM25Retriever
    from retrieval.fusion import fuse
    from retrieval.mmr import run_mmr
except ModuleNotFoundError:
    # Allow direct script execution: python retrieval/evaluate_retrieval.py
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from retrieval.normalization import normalize_query
    from retrieval.language import detect_language, get_fusion_weights
    from retrieval.vector_retriever import VectorRetriever
    from retrieval.bm25_retriever import BM25Retriever
    from retrieval.fusion import fuse
    from retrieval.mmr import run_mmr

EVAL_DIR = Path("retrieval/eval")

LANGUAGES = ("urdu", "roman_urdu", "english")
SYSTEMS = ("vector_only", "hybrid", "hybrid_mmr")
PHASE4_SYSTEMS = ("hybrid_mmr_rerank", "hybrid_mmr_rerank_gate")


def _load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _compute_metrics(ranked_ids: list[str], relevant_id: str) -> dict:
    hit1 = int(len(ranked_ids) >= 1 and ranked_ids[0] == relevant_id)
    hit3 = int(relevant_id in ranked_ids[:3])
    hit10 = int(relevant_id in ranked_ids[:10])
    mrr = 0.0
    for rank, doc_id in enumerate(ranked_ids[:10], 1):
        if doc_id == relevant_id:
            mrr = 1.0 / rank
            break
    return {"hit1": hit1, "hit3": hit3, "hit10": hit10, "mrr": mrr}


def _aggregate(metric_lists: list[dict]) -> dict:
    if not metric_lists:
        return {"acc@1": 0.0, "acc@3": 0.0, "acc@10": 0.0, "mrr": 0.0, "n": 0}
    n = len(metric_lists)
    return {
        "acc@1": round(sum(m["hit1"] for m in metric_lists) / n, 4),
        "acc@3": round(sum(m["hit3"] for m in metric_lists) / n, 4),
        "acc@10": round(sum(m["hit10"] for m in metric_lists) / n, 4),
        "mrr": round(sum(m["mrr"] for m in metric_lists) / n, 4),
        "n": n,
    }


class RetrievalEvaluator:
    def __init__(self, config_path: str = "retrieval/configs/retrieval_config.yaml"):
        self.cfg = _load_config(config_path)
        self.vector_k: int = self.cfg["retrieval"]["vector_k"]
        self.hybrid_k: int = self.cfg["retrieval"]["hybrid_candidate_k"]
        self.mmr_k: int = self.cfg["retrieval"]["mmr_output_k"]
        self.mmr_lambda: float = self.cfg["mmr"]["lambda"]
        self.rrf_k: int = self.cfg["fusion"]["rrf_k"]
        self.reranker_k: int = self.cfg["reranker"]["output_k"]
        self.reranker_threshold: float = self.cfg["reranker"]["confidence_threshold"]

        self._vector = VectorRetriever(self.cfg)
        self._bm25 = BM25Retriever(self.cfg)
        self._reranker = None

    def _load_reranker(self):
        if self._reranker is not None:
            return
        from sentence_transformers import CrossEncoder

        self._reranker = CrossEncoder(self.cfg["reranker"]["model_id"])

    def _resolve_relevant_doc_id(self, rec: dict) -> str:
        """
        Test split records often carry source `id` (int), while retrievers return `doc_id`.
        Resolve source id -> doc_id via docstore.
        """
        if not self._vector.is_loaded:
            self._vector.load()

        docstore = self._vector.docstore
        if "doc_id" in rec and rec["doc_id"] in docstore:
            return rec["doc_id"]

        src = str(rec.get("id", ""))
        if not src:
            return ""
        for doc_id, doc in docstore.items():
            if str(doc.get("source_id", "")) == src:
                return doc_id
        return ""

    def _vector_only(self, query_norm: str) -> list[str]:
        results = self._vector.retrieve(query_norm, k=10)
        return [r.doc_id for r in results]

    def _hybrid(self, query_norm: str, lang: str) -> list[str]:
        bm25_w, vector_w = get_fusion_weights(lang, self.cfg)
        vec = self._vector.retrieve(query_norm, k=self.vector_k)
        bm25 = self._bm25.retrieve(query_norm, k=self.hybrid_k)
        fused = fuse(
            vector_results=vec, bm25_results=bm25,
            vector_weight=vector_w,
            bm25_weight=bm25_w,
            rrf_k=self.rrf_k,
            top_k=self.hybrid_k,
        )
        return [r.doc_id for r in fused]

    def _hybrid_mmr(self, query_norm: str, lang: str) -> list[str]:
        bm25_w, vector_w = get_fusion_weights(lang, self.cfg)
        vec = self._vector.retrieve(query_norm, k=self.vector_k)
        bm25 = self._bm25.retrieve(query_norm, k=self.hybrid_k)
        fused = fuse(
            vector_results=vec, bm25_results=bm25,
            vector_weight=vector_w,
            bm25_weight=bm25_w,
            rrf_k=self.rrf_k,
            top_k=self.hybrid_k,
        )
        candidate_ids = [r.doc_id for r in fused]
        candidate_scores = [r.rrf_score for r in fused]
        candidate_embeddings = self._vector.encode_docs(candidate_ids)
        query_embedding = self._vector.encode_query(query_norm)
        mmr_results = run_mmr(
            query_embedding=query_embedding,
            candidate_ids=candidate_ids,
            candidate_embeddings=candidate_embeddings,
            candidate_scores=candidate_scores,
            top_k=self.mmr_k,
            lambda_param=self.mmr_lambda,
        )
        return [r.doc_id for r in mmr_results]

    def _hybrid_mmr_rerank(self, query_norm: str, lang: str) -> tuple[list[str], float]:
        bm25_w, vector_w = get_fusion_weights(lang, self.cfg)
        vec = self._vector.retrieve(query_norm, k=self.vector_k)
        bm25 = self._bm25.retrieve(query_norm, k=self.hybrid_k)
        fused = fuse(
            vector_results=vec, bm25_results=bm25,
            vector_weight=vector_w,
            bm25_weight=bm25_w,
            rrf_k=self.rrf_k,
            top_k=self.hybrid_k,
        )
        candidate_ids = [r.doc_id for r in fused]
        candidate_scores = [r.rrf_score for r in fused]
        candidate_embeddings = self._vector.encode_docs(candidate_ids)
        query_embedding = self._vector.encode_query(query_norm)
        mmr_results = run_mmr(
            query_embedding=query_embedding,
            candidate_ids=candidate_ids,
            candidate_embeddings=candidate_embeddings,
            candidate_scores=candidate_scores,
            top_k=self.mmr_k,
            lambda_param=self.mmr_lambda,
        )
        rerank_doc_ids = [r.doc_id for r in mmr_results[: self.cfg["reranker"]["input_k"]]]
        if not rerank_doc_ids:
            return [], 0.0
        self._load_reranker()
        docstore = self._vector.docstore
        pairs = [(query_norm, docstore.get(did, {}).get("question", "")) for did in rerank_doc_ids]
        scores = self._reranker.predict(pairs, batch_size=self.cfg["reranker"].get("batch_size", 16))
        ranked = sorted(zip(rerank_doc_ids, scores), key=lambda x: float(x[1]), reverse=True)
        top_score = float(ranked[0][1]) if ranked else 0.0
        top_ids = [doc_id for doc_id, _ in ranked[: self.reranker_k]]
        return top_ids, top_score

    def evaluate(
        self,
        test_path: str,
        results_json_path: Path = EVAL_DIR / "results.json",
        results_csv_path: Path = EVAL_DIR / "results.csv",
    ) -> dict:
        records = _load_jsonl(test_path)

        all_systems = SYSTEMS + PHASE4_SYSTEMS
        all_metrics: dict[str, list[dict]] = {s: [] for s in all_systems}
        lang_metrics: dict[str, dict[str, list[dict]]] = {
            s: defaultdict(list) for s in all_systems
        }
        gate_rejections = 0
        per_query_rows: list[dict] = []

        for rec in records:
            query_raw = rec.get("query", "")
            relevant_id = self._resolve_relevant_doc_id(rec)
            if not relevant_id:
                continue

            query_norm = normalize_query(query_raw)
            lang = detect_language(query_norm).label

            t0 = time.perf_counter()
            vec_ids = self._vector_only(query_norm)
            t_vec = time.perf_counter()

            hyb_ids = self._hybrid(query_norm, lang)
            t_hyb = time.perf_counter()

            hyb_mmr_ids = self._hybrid_mmr(query_norm, lang)
            t_mmr = time.perf_counter()
            rerank_ids, top_reranker_score = self._hybrid_mmr_rerank(query_norm, lang)
            t_rerank = time.perf_counter()
            gate_pass = top_reranker_score >= self.reranker_threshold if rerank_ids else False
            rerank_gate_ids = rerank_ids if gate_pass else []
            if not gate_pass:
                gate_rejections += 1

            row = {
                "query_id": relevant_id,
                "query": query_raw[:80],
                "language": lang,
                "latency_vector_ms": round((t_vec - t0) * 1000, 1),
                "latency_hybrid_ms": round((t_hyb - t_vec) * 1000, 1),
                "latency_mmr_ms": round((t_mmr - t_hyb) * 1000, 1),
                "latency_rerank_ms": round((t_rerank - t_mmr) * 1000, 1),
                "top_reranker_score": round(top_reranker_score, 4),
                "gate_pass": int(gate_pass),
            }

            for system, ranked_ids in [
                ("vector_only", vec_ids),
                ("hybrid", hyb_ids),
                ("hybrid_mmr", hyb_mmr_ids),
                ("hybrid_mmr_rerank", rerank_ids),
                ("hybrid_mmr_rerank_gate", rerank_gate_ids),
            ]:
                m = _compute_metrics(ranked_ids, relevant_id)
                all_metrics[system].append(m)
                lang_metrics[system][lang].append(m)
                row[f"{system}_hit@1"] = m["hit1"]
                row[f"{system}_hit@3"] = m["hit3"]
                row[f"{system}_hit@10"] = m["hit10"]
                row[f"{system}_mrr"] = round(m["mrr"], 4)

            per_query_rows.append(row)

        overall: dict[str, dict] = {}
        by_language: dict[str, dict[str, dict]] = {s: {} for s in all_systems}

        for system in all_systems:
            overall[system] = _aggregate(all_metrics[system])
            for lang in LANGUAGES:
                by_language[system][lang] = _aggregate(lang_metrics[system].get(lang, []))

        results = {
            "overall": overall,
            "by_language": by_language,
            "n_queries": len(records),
            "test_file": test_path,
            "gate": {
                "confidence_threshold": self.reranker_threshold,
                "rejections": gate_rejections,
                "rejection_rate": round(gate_rejections / len(records), 4) if records else 0.0,
            },
        }

        EVAL_DIR.mkdir(parents=True, exist_ok=True)

        with open(results_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        if per_query_rows:
            fieldnames = list(per_query_rows[0].keys())
            with open(results_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(per_query_rows)

        _print_results(results, results_json_path, results_csv_path)
        return results


def _print_results(results: dict, results_json_path: Path, results_csv_path: Path) -> None:
    print("\n" + "=" * 65)
    print("RETRIEVAL EVALUATION RESULTS")
    print("=" * 65)
    print(f"{'System':<20} {'Acc@1':>7} {'Acc@3':>7} {'Acc@10':>8} {'MRR':>7} {'N':>5}")
    print("-" * 65)
    for system, m in results["overall"].items():
        print(
            f"{system:<20} {m['acc@1']:>7.4f} {m['acc@3']:>7.4f} "
            f"{m['acc@10']:>8.4f} {m['mrr']:>7.4f} {m['n']:>5}"
        )

    print("\n── Per-language breakdown ──")
    for lang in LANGUAGES:
        print(f"\n  [{lang}]")
        print(f"  {'System':<20} {'Acc@1':>7} {'Acc@3':>7} {'MRR':>7} {'N':>5}")
        print(f"  {'-'*50}")
        for system in SYSTEMS + PHASE4_SYSTEMS:
            m = results["by_language"][system].get(lang, {})
            if m.get("n", 0) == 0:
                continue
            print(
                f"  {system:<20} {m['acc@1']:>7.4f} {m['acc@3']:>7.4f} "
                f"{m['mrr']:>7.4f} {m['n']:>5}"
            )
    if "gate" in results:
        g = results["gate"]
        print(
            f"\nGate: threshold={g['confidence_threshold']}, "
            f"rejections={g['rejections']}, rejection_rate={g['rejection_rate']}"
        )
    print("=" * 65)
    print(f"Results saved → {results_json_path}")
    print(f"Per-query CSV  → {results_csv_path}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test",
        default="data/processed/splits/test.jsonl",
        help="Path to test JSONL file",
    )
    parser.add_argument(
        "--config",
        default="retrieval/configs/retrieval_config.yaml",
        help="Path to retrieval config",
    )
    parser.add_argument(
        "--out_prefix",
        default="results",
        help="Output prefix under retrieval/eval (e.g., 'results_in_corpus')",
    )
    args = parser.parse_args()

    evaluator = RetrievalEvaluator(config_path=args.config)
    out_json = EVAL_DIR / f"{args.out_prefix}.json"
    out_csv = EVAL_DIR / f"{args.out_prefix}.csv"
    evaluator.evaluate(test_path=args.test, results_json_path=out_json, results_csv_path=out_csv)