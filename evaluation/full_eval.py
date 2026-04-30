from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

try:
    from retrieval.normalization import normalize_query
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from retrieval.normalization import normalize_query

EVAL_DIR = Path("evaluation/results")
OUT_JSON = EVAL_DIR / "full_eval_results.json"
OUT_CSV  = EVAL_DIR / "full_eval_results.csv"


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


def _mrr(ranked_ids: list[str], relevant_id: str, k: int = 10) -> float:
    for rank, doc_id in enumerate(ranked_ids[:k], 1):
        if doc_id == relevant_id:
            return 1.0 / rank
    return 0.0


def _hits(ranked_ids: list[str], relevant_id: str) -> dict:
    return {
        "hit1":  int(len(ranked_ids) >= 1 and ranked_ids[0] == relevant_id),
        "hit3":  int(relevant_id in ranked_ids[:3]),
        "hit10": int(relevant_id in ranked_ids[:10]),
        "mrr":   _mrr(ranked_ids, relevant_id),
    }


def _agg(metrics: list[dict]) -> dict:
    if not metrics:
        return {"acc@1": 0.0, "acc@3": 0.0, "acc@10": 0.0, "mrr": 0.0, "n": 0}
    n = len(metrics)
    return {
        "acc@1":  round(sum(m["hit1"]  for m in metrics) / n, 4),
        "acc@3":  round(sum(m["hit3"]  for m in metrics) / n, 4),
        "acc@10": round(sum(m["hit10"] for m in metrics) / n, 4),
        "mrr":    round(sum(m["mrr"]   for m in metrics) / n, 4),
        "n": n,
    }


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(float(np.percentile(values, 95)), 1)


def _p50(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(float(np.percentile(values, 50)), 1)


class AblationEvaluator:
    def __init__(self, config_path: str = "retrieval/configs/retrieval_config.yaml"):
        self.cfg = _load_config(config_path)
        self.config_path = config_path
        self._systems: dict[str, callable] = {}
        self._build_systems()

    def _build_systems(self) -> None:
        from retrieval.bm25_retriever import BM25Retriever
        from retrieval.dual_retriever import DualRetriever
        from retrieval.fusion import fuse
        from retrieval.language import detect_and_get_weights
        from retrieval.mmr import run_mmr
        from retrieval.vector_retriever import VectorRetriever
        from reranking.reranker import CrossEncoderReranker
        from metadata.category_detector import CategoryDetector
        from metadata.filter_policy import FilterPolicy

        cfg = self.cfg
        vector_k = cfg["retrieval"]["vector_k"]
        hybrid_k = cfg["retrieval"]["hybrid_candidate_k"]
        mmr_k = cfg["mmr"]["output_k"]
        mmr_lam   = cfg["mmr"]["lambda"]
        rrf_k     = cfg["fusion"]["rrf_k"]

        bm25_r = BM25Retriever(cfg)
        vector_r = VectorRetriever(cfg)
        vector_r.load()
        bm25_r.load()
        reranker  = CrossEncoderReranker(self.config_path)
        detector  = CategoryDetector(cfg)
        policy    = FilterPolicy(cfg, known_categories=vector_r.available_categories())
        dual_r    = DualRetriever(cfg)

        def bm25_only(query_norm: str, _lang: str) -> tuple[list[str], float]:
            out = bm25_r.retrieve(query_norm, k=10)
            return [r.doc_id for r in out], 0.0

        def embedding_only(query_norm: str, _lang: str) -> tuple[list[str], float]:
            out = vector_r.retrieve(query_norm, k=10)
            return [r.doc_id for r in out], 0.0

        def hybrid(query_norm: str, lang: str) -> tuple[list[str], float]:
            _, bm25_w, vec_w = detect_and_get_weights(query_norm, cfg)
            vec = vector_r.retrieve(query_norm, k=vector_k)
            bm25 = bm25_r.retrieve(query_norm, k=hybrid_k)
            fused = fuse(
                vector_results=vec,
                bm25_results=bm25,
                vector_weight=vec_w,
                bm25_weight=bm25_w,
                rrf_k=rrf_k,
                top_k=hybrid_k,
            )
            return [r.doc_id for r in fused], 0.0

        def hybrid_mmr(query_norm: str, lang: str) -> tuple[list[str], float]:
            _, bm25_w, vec_w = detect_and_get_weights(query_norm, cfg)
            vec = vector_r.retrieve(query_norm, k=vector_k)
            bm25 = bm25_r.retrieve(query_norm, k=hybrid_k)
            fused = fuse(
                vector_results=vec,
                bm25_results=bm25,
                vector_weight=vec_w,
                bm25_weight=bm25_w,
                rrf_k=rrf_k,
                top_k=hybrid_k,
            )
            cids = [r.doc_id for r in fused]
            cscores = [r.rrf_score for r in fused]
            cembs = vector_r.encode_docs(cids)
            qemb = vector_r.encode_query(query_norm)
            mmr_res = run_mmr(
                query_embedding=qemb,
                candidate_ids=cids,
                candidate_embeddings=cembs,
                candidate_scores=cscores,
                top_k=mmr_k,
                lambda_param=mmr_lam,
            )
            return [r.doc_id for r in mmr_res], 0.0

        def hybrid_mmr_reranker(query_norm: str, lang: str) -> tuple[list[str], float]:
            ids, _ = hybrid_mmr(query_norm, lang)
            docstore = vector_r.docstore
            docs = [dict(docstore.get(i, {"doc_id": i})) for i in ids]
            reranked, _ = reranker.rerank(query_norm, docs, top_k=3)
            return [r.doc_id for r in reranked], 0.0

        def hybrid_mmr_reranker_filter(query_norm: str, lang: str) -> tuple[list[str], float]:
            det = detector.detect(query_norm)
            pol = policy.decide(det)
            cat = pol.category_filter if pol.apply_filter else None
            _, bm25_w, vec_w = detect_and_get_weights(query_norm, cfg)
            vec = vector_r.retrieve(query_norm, k=vector_k, category_filter=cat)
            bm25 = bm25_r.retrieve(query_norm, k=hybrid_k, category_filter=cat)
            fused = fuse(
                vector_results=vec,
                bm25_results=bm25,
                vector_weight=vec_w,
                bm25_weight=bm25_w,
                rrf_k=rrf_k,
                top_k=hybrid_k,
            )
            cids = [r.doc_id for r in fused]
            cscores = [r.rrf_score for r in fused]
            cembs = vector_r.encode_docs(cids)
            qemb = vector_r.encode_query(query_norm)
            mmr_res = run_mmr(
                query_embedding=qemb,
                candidate_ids=cids,
                candidate_embeddings=cembs,
                candidate_scores=cscores,
                top_k=mmr_k,
                lambda_param=mmr_lam,
            )
            ids = [r.doc_id for r in mmr_res]
            docs = [dict(vector_r.docstore.get(i, {"doc_id": i})) for i in ids]
            reranked, _ = reranker.rerank(query_norm, docs, top_k=3)
            return [r.doc_id for r in reranked], 0.0

        def full_dual(query_norm: str, lang: str) -> tuple[list[str], float]:
            result = dual_r.retrieve(query=query_norm, language=lang, top_k=10)
            ids = [r.get("doc_id") for r in result.docs[:3] if r.get("doc_id")]
            return ids, 0.0

        self._systems = {
            "01_bm25_only":                    bm25_only,
            "02_embedding_only":               embedding_only,
            "03_hybrid":                       hybrid,
            "04_hybrid_mmr":                   hybrid_mmr,
            "05_hybrid_mmr_reranker":          hybrid_mmr_reranker,
            "06_hybrid_mmr_reranker_filter":   hybrid_mmr_reranker_filter,
            "07_full_dual_index":              full_dual,
        }

    def evaluate(
        self,
        test_path: str,
        pdf_test_path: Optional[str] = None,
    ) -> dict:
        from retrieval.language import detect_language

        records = _load_jsonl(test_path)
        pdf_records = _load_jsonl(pdf_test_path) if pdf_test_path and Path(pdf_test_path).exists() else []
        all_records = records + pdf_records

        from retrieval.vector_retriever import VectorRetriever
        vector = VectorRetriever(self.cfg)
        vector.load()
        source_to_doc = {}
        for doc_id, doc in vector.docstore.items():
            src = str(doc.get("source_id", ""))
            if src:
                source_to_doc[src] = doc_id

        system_metrics:   dict[str, list[dict]]   = defaultdict(list)
        system_latencies: dict[str, list[float]]  = defaultdict(list)
        per_query_rows:   list[dict]               = []

        for rec in all_records:
            query_raw    = rec.get("query", "")
            relevant_id  = source_to_doc.get(str(rec.get("id", "")), rec.get("doc_id", ""))
            if not relevant_id:
                continue
            query_norm   = normalize_query(query_raw)
            lang = detect_language(query_norm).label

            row = {
                "query_id":   relevant_id,
                "query":      query_raw[:80],
                "language":   lang,
                "source":     rec.get("source_type", "qa"),
            }

            for sys_name, sys_fn in self._systems.items():
                try:
                    ranked_ids, latency_ms = sys_fn(query_norm, lang)
                except Exception as e:
                    ranked_ids, latency_ms = [], 0.0
                    print(f"[WARN] {sys_name} failed on '{query_raw[:40]}': {e}")

                m = _hits(ranked_ids, relevant_id)
                system_metrics[sys_name].append(m)
                system_latencies[sys_name].append(latency_ms)
                row[f"{sys_name}_hit1"]     = m["hit1"]
                row[f"{sys_name}_hit3"]     = m["hit3"]
                row[f"{sys_name}_mrr"]      = round(m["mrr"], 4)
                row[f"{sys_name}_latency"]  = round(latency_ms, 1)

            per_query_rows.append(row)

        system_labels = {
            "01_bm25_only":                  "Baseline BM25 only",
            "02_embedding_only":             "+ Fine-tuned embeddings",
            "03_hybrid":                     "+ Hybrid search",
            "04_hybrid_mmr":                 "+ MMR",
            "05_hybrid_mmr_reranker":        "+ Reranker",
            "06_hybrid_mmr_reranker_filter": "+ Metadata filter",
            "07_full_dual_index":            "+ PDF corpus dual-index (Full system)",
        }

        ablation: list[dict] = []
        for sys_name, label in system_labels.items():
            agg  = _agg(system_metrics[sys_name])
            lats = system_latencies[sys_name]
            ablation.append({
                "system":    sys_name,
                "label":     label,
                "acc@1":     agg["acc@1"],
                "acc@3":     agg["acc@3"],
                "acc@10":    agg["acc@10"],
                "mrr":       agg["mrr"],
                "p50_ms":    _p50(lats),
                "p95_ms":    _p95(lats),
                "n":         agg["n"],
            })

        results = {
            "n_qa_queries":  len(records),
            "n_pdf_queries": len(pdf_records),
            "ablation":      ablation,
        }

        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        if per_query_rows:
            with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(per_query_rows[0].keys()))
                writer.writeheader()
                writer.writerows(per_query_rows)

        _print_ablation(ablation)
        return results


def _print_ablation(ablation: list[dict]) -> None:
    print("\n" + "=" * 80)
    print("ABLATION TABLE")
    print("=" * 80)
    print(f"{'System':<45} {'Acc@1':>6} {'Acc@3':>6} {'MRR':>6} {'P50ms':>7} {'P95ms':>7}")
    print("-" * 80)
    for row in ablation:
        print(
            f"{row['label']:<45} "
            f"{row['acc@1']:>6.4f} "
            f"{row['acc@3']:>6.4f} "
            f"{row['mrr']:>6.4f} "
            f"{row['p50_ms']:>7.1f} "
            f"{row['p95_ms']:>7.1f}"
        )
    print("=" * 80)
    print(f"  Results → {OUT_JSON}")
    print(f"  CSV     → {OUT_CSV}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",     default="data/processed/splits/test.jsonl")
    parser.add_argument("--pdf_test", default="evaluation/pdf_test_queries.jsonl")
    parser.add_argument("--config",   default="retrieval/configs/retrieval_config.yaml")
    args = parser.parse_args()
    ev = AblationEvaluator(config_path=args.config)
    ev.evaluate(test_path=args.test, pdf_test_path=args.pdf_test)