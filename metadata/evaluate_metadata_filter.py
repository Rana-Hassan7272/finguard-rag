from __future__ import annotations

import csv
import json
import time
import sys
from collections import defaultdict
from pathlib import Path

import yaml

try:
    from retrieval.normalization import normalize_query
    from retrieval.language import detect_language, get_fusion_weights
    from retrieval.vector_retriever import VectorRetriever
    from retrieval.bm25_retriever import BM25Retriever
    from retrieval.fusion import fuse
    from retrieval.mmr import run_mmr
    from metadata.category_detector import CategoryDetector
    from metadata.filter_policy import FilterPolicy
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from retrieval.normalization import normalize_query
    from retrieval.language import detect_language, get_fusion_weights
    from retrieval.vector_retriever import VectorRetriever
    from retrieval.bm25_retriever import BM25Retriever
    from retrieval.fusion import fuse
    from retrieval.mmr import run_mmr
    from metadata.category_detector import CategoryDetector
    from metadata.filter_policy import FilterPolicy

EVAL_DIR = Path("retrieval/eval")
OUT_JSON = EVAL_DIR / "metadata_filter_results.json"
OUT_CSV = EVAL_DIR / "metadata_filter_results.csv"


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


def _aggregate(metrics: list[dict]) -> dict:
    if not metrics:
        return {"acc@1": 0.0, "acc@3": 0.0, "acc@10": 0.0, "mrr": 0.0, "n": 0}
    n = len(metrics)
    return {
        "acc@1": round(sum(m["hit1"] for m in metrics) / n, 4),
        "acc@3": round(sum(m["hit3"] for m in metrics) / n, 4),
        "acc@10": round(sum(m["hit10"] for m in metrics) / n, 4),
        "mrr": round(sum(m["mrr"] for m in metrics) / n, 4),
        "n": n,
    }


class MetadataFilterEvaluator:
    def __init__(self, config_path: str = "retrieval/configs/retrieval_config.yaml"):
        self.cfg = _load_config(config_path)
        self.vector_k: int = self.cfg["retrieval"]["vector_k"]
        self.hybrid_k: int = self.cfg["retrieval"]["hybrid_candidate_k"]
        self.mmr_k: int = self.cfg["retrieval"]["mmr_output_k"]
        self.mmr_lambda: float = self.cfg["mmr"]["lambda"]
        self.rrf_k: int = self.cfg["fusion"]["rrf_k"]

        self._vector = VectorRetriever(self.cfg)
        self._bm25 = BM25Retriever(self.cfg)
        self._detector = CategoryDetector(self.cfg)
        self._policy = FilterPolicy(self.cfg)

        # Build source_id -> doc_id for fair relevance matching.
        self._vector.load()
        self._source_to_doc = {
            str(doc.get("source_id")): doc_id
            for doc_id, doc in self._vector.docstore.items()
            if doc.get("source_id") is not None
        }
        self._policy.register_known_categories(self._vector.available_categories())

    def _retrieve(self, query_norm: str, lang: str, category_filter: str | None) -> list[str]:
        bm25_w, vector_w = get_fusion_weights(lang, self.cfg)
        vec = self._vector.retrieve(query_norm, k=self.vector_k, category_filter=category_filter)
        bm25 = self._bm25.retrieve(query_norm, k=self.hybrid_k, category_filter=category_filter)
        fused = fuse(
            vector_results=vec,
            bm25_results=bm25,
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

    def evaluate(self, test_path: str) -> dict:
        records = _load_jsonl(test_path)

        no_filter_metrics: list[dict] = []
        with_filter_metrics: list[dict] = []
        per_query_rows: list[dict] = []

        wrong_filter = 0
        filter_applied_count = 0
        lang_no: dict[str, list[dict]] = defaultdict(list)
        lang_with: dict[str, list[dict]] = defaultdict(list)

        no_filter_latencies: list[float] = []
        with_filter_latencies: list[float] = []

        for rec in records:
            query_raw = rec.get("query", "")
            relevant_id = self._source_to_doc.get(str(rec.get("id", "")), "")
            if not relevant_id:
                continue
            true_category = rec.get("category", "")
            query_norm = normalize_query(query_raw)
            lang = detect_language(query_norm).label

            detection = self._detector.detect(query_norm)
            policy = self._policy.decide(detection)

            t0 = time.perf_counter()
            no_filter_ids = self._retrieve(query_norm, lang, category_filter=None)
            t1 = time.perf_counter()
            no_filter_latencies.append((t1 - t0) * 1000)

            t2 = time.perf_counter()
            with_filter_ids = self._retrieve(
                query_norm, lang,
                category_filter=policy.category_filter if policy.apply_filter else None
            )
            t3 = time.perf_counter()
            with_filter_latencies.append((t3 - t2) * 1000)

            m_no = _compute_metrics(no_filter_ids, relevant_id)
            m_with = _compute_metrics(with_filter_ids, relevant_id)

            no_filter_metrics.append(m_no)
            with_filter_metrics.append(m_with)
            lang_no[lang].append(m_no)
            lang_with[lang].append(m_with)

            if policy.apply_filter:
                filter_applied_count += 1
                if policy.category_filter and true_category and policy.category_filter != true_category:
                    wrong_filter += 1

            per_query_rows.append({
                "query_id": relevant_id,
                "query": query_raw[:80],
                "language": lang,
                "true_category": true_category,
                "detected_category": detection.predicted_category or "",
                "detection_confidence": round(detection.confidence, 3),
                "filter_applied": policy.apply_filter,
                "filter_category": policy.category_filter or "",
                "wrong_filter": policy.apply_filter and bool(true_category) and policy.category_filter != true_category,
                "no_filter_hit1": m_no["hit1"],
                "no_filter_hit3": m_no["hit3"],
                "no_filter_mrr": round(m_no["mrr"], 4),
                "with_filter_hit1": m_with["hit1"],
                "with_filter_hit3": m_with["hit3"],
                "with_filter_mrr": round(m_with["mrr"], 4),
                "latency_no_filter_ms": round(no_filter_latencies[-1], 1),
                "latency_with_filter_ms": round(with_filter_latencies[-1], 1),
            })

        agg_no = _aggregate(no_filter_metrics)
        agg_with = _aggregate(with_filter_metrics)

        def delta(a: float, b: float) -> float:
            return round(b - a, 4)

        results = {
            "n_queries": len(records),
            "filter_applied_count": filter_applied_count,
            "filter_applied_rate": round(filter_applied_count / len(records), 4) if records else 0,
            "wrong_filter_count": wrong_filter,
            "wrong_filter_rate": round(wrong_filter / filter_applied_count, 4) if filter_applied_count else 0,
            "overall": {
                "no_filter": agg_no,
                "with_filter": agg_with,
                "delta": {
                    "acc@1": delta(agg_no["acc@1"], agg_with["acc@1"]),
                    "acc@3": delta(agg_no["acc@3"], agg_with["acc@3"]),
                    "acc@10": delta(agg_no["acc@10"], agg_with["acc@10"]),
                    "mrr": delta(agg_no["mrr"], agg_with["mrr"]),
                    "latency_ms": round(
                        sum(with_filter_latencies) / len(with_filter_latencies) -
                        sum(no_filter_latencies) / len(no_filter_latencies), 2
                    ) if no_filter_latencies else 0,
                },
            },
            "by_language": {},
        }

        for lang in ("urdu", "roman_urdu", "english"):
            a = _aggregate(lang_no.get(lang, []))
            b = _aggregate(lang_with.get(lang, []))
            results["by_language"][lang] = {
                "no_filter": a,
                "with_filter": b,
                "delta": {
                    "acc@1": delta(a["acc@1"], b["acc@1"]),
                    "mrr": delta(a["mrr"], b["mrr"]),
                },
            }

        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        if per_query_rows:
            fieldnames = list(per_query_rows[0].keys())
            with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(per_query_rows)

        _print_results(results)
        return results


def _print_results(r: dict) -> None:
    print("\n" + "=" * 65)
    print("METADATA FILTER EVALUATION")
    print("=" * 65)
    o = r["overall"]
    print(f"\n  Filter applied: {r['filter_applied_count']}/{r['n_queries']} "
          f"({r['filter_applied_rate']:.1%})")
    print(f"  Wrong filter  : {r['wrong_filter_count']} ({r['wrong_filter_rate']:.1%} of filtered)\n")
    print(f"  {'Metric':<12} {'No Filter':>10} {'With Filter':>12} {'Delta':>8}")
    print(f"  {'-'*48}")
    for k in ("acc@1", "acc@3", "acc@10", "mrr"):
        nf = o["no_filter"][k]
        wf = o["with_filter"][k]
        d = o["delta"][k]
        sign = "+" if d >= 0 else ""
        print(f"  {k:<12} {nf:>10.4f} {wf:>12.4f} {sign}{d:>7.4f}")
    lat = o["delta"]["latency_ms"]
    sign = "+" if lat >= 0 else ""
    print(f"  {'latency_ms':<12} {'—':>10} {'—':>12} {sign}{lat:>7.1f}")
    print(f"\n  Results → {OUT_JSON}")
    print(f"  CSV     → {OUT_CSV}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default="data/processed/splits/test.jsonl")
    parser.add_argument("--config", default="retrieval/configs/retrieval_config.yaml")
    args = parser.parse_args()
    evaluator = MetadataFilterEvaluator(config_path=args.config)
    evaluator.evaluate(test_path=args.test)