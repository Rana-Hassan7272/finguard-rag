from __future__ import annotations

import csv
import json
import sys
import time
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
OUT_JSON = EVAL_DIR / "latency_profile.json"
OUT_CSV  = EVAL_DIR / "latency_profile.csv"


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


def _percentiles(values: list[float]) -> dict:
    if not values:
        return {"p50": 0.0, "p75": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0}
    arr = np.array(values)
    return {
        "p50":  round(float(np.percentile(arr, 50)), 2),
        "p75":  round(float(np.percentile(arr, 75)), 2),
        "p90":  round(float(np.percentile(arr, 90)), 2),
        "p95":  round(float(np.percentile(arr, 95)), 2),
        "p99":  round(float(np.percentile(arr, 99)), 2),
        "mean": round(float(arr.mean()), 2),
        "min":  round(float(arr.min()), 2),
        "max":  round(float(arr.max()), 2),
    }


STAGES = [
    "normalize_ms",
    "lang_detect_ms",
    "cache_lookup_ms",
    "category_detect_ms",
    "qa_vector_ms",
    "qa_bm25_ms",
    "pdf_vector_ms",
    "pdf_bm25_ms",
    "rrf_fusion_ms",
    "mmr_ms",
    "reranker_ms",
    "gate_ms",
    "llm_ms",
    "cache_store_ms",
    "total_ms",
]


class LatencyProfiler:
    def __init__(self, config_path: str = "retrieval/configs/retrieval_config.yaml"):
        self.cfg = _load_config(config_path)
        self.config_path = config_path

    def _time_stage(self, fn, *args, **kwargs) -> tuple[any, float]:
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        ms = round((time.perf_counter() - t0) * 1000, 2)
        return result, ms

    def profile_query(self, query_raw: str) -> dict:
        timings: dict[str, float] = {s: 0.0 for s in STAGES}
        out = self._pipeline.answer(query_raw)
        rdiag = out.retrieval.diagnostics
        gdiag = out.diagnostics.get("generation", {})

        stage = rdiag.get("stage_latency_ms", {})
        timings["normalize_ms"] = float(stage.get("normalize_route_expand_ms", 0.0))
        timings["lang_detect_ms"] = float(stage.get("language_ms", 0.0))
        timings["category_detect_ms"] = float(stage.get("category_detect_ms", 0.0))
        timings["rrf_fusion_ms"] = float(stage.get("fusion_ms", 0.0))
        timings["mmr_ms"] = float(stage.get("mmr_ms", 0.0))
        # End-to-end total must include retrieval + generation.
        # Previous code only used retrieval total, which understated latency.
        timings["total_ms"] = float(out.retrieval.total_ms) + float(out.generation.total_ms)

        # Dual mode currently exposes aggregate retrieval time and source mix.
        dual_ms = float(stage.get("dual_retrieve_ms", 0.0))
        source_mix = rdiag.get("source_mix", {})
        qa_docs = source_mix.get("qa_docs", 0) or source_mix.get("qa", 0)
        pdf_docs = source_mix.get("pdf_docs", 0) or source_mix.get("pdf", 0)
        total_docs = max(qa_docs + pdf_docs, 1)
        timings["qa_vector_ms"] = round(dual_ms * (qa_docs / total_docs), 2)
        timings["pdf_vector_ms"] = round(dual_ms * (pdf_docs / total_docs), 2)

        gstage = gdiag.get("stage_ms", {}) if isinstance(gdiag, dict) else {}
        timings["cache_lookup_ms"] = float(gstage.get("cache_lookup_ms", 0.0))
        timings["reranker_ms"] = float(gstage.get("reranker_ms", 0.0))
        timings["gate_ms"] = float(gstage.get("gate_ms", 0.0))
        llm_meta = gdiag.get("llm", {}) if isinstance(gdiag, dict) else {}
        timings["llm_ms"] = float(gstage.get("llm_ms", llm_meta.get("latency_ms", 0.0)))
        timings["cache_store_ms"] = float(gstage.get("cache_store_ms", 0.0))
        timings["cache_hit"] = bool(gdiag.get("cache_hit", False))
        return timings

    def _init_components(self) -> None:
        from retrieval.pipeline import RetrievalPipeline

        self._pipeline = RetrievalPipeline(self.cfg)
        self._pipeline.load()

    def profile(
        self,
        test_path: str,
        n_queries: Optional[int] = None,
        warmup: int = 3,
    ) -> dict:
        records = _load_jsonl(test_path)
        if n_queries:
            records = records[:n_queries]

        print(f"Initializing pipeline components...")
        self._init_components()

        print(f"Running {warmup} warmup queries...")
        for rec in records[:warmup]:
            try:
                self.profile_query(rec.get("query", ""))
            except Exception:
                pass

        print(f"Profiling {len(records)} queries...")

        stage_timings: dict[str, list[float]] = defaultdict(list)
        per_query_rows: list[dict] = []
        cache_hits = 0
        lang_timings: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

        from retrieval.language import detect_language
        for rec in records:
            query_raw = rec.get("query", "")
            query_norm = normalize_query(query_raw)
            lang = detect_language(query_norm).label

            try:
                timings = self.profile_query(query_raw)
            except Exception as e:
                print(f"[WARN] profiling failed on '{query_raw[:40]}': {e}")
                continue

            if timings.get("cache_hit"):
                cache_hits += 1

            row = {"query": query_raw[:60], "language": lang}
            for stage in STAGES:
                v = timings.get(stage, 0.0)
                stage_timings[stage].append(v)
                lang_timings[lang][stage].append(v)
                row[stage] = v
            row["cache_hit"] = int(timings.get("cache_hit", False))
            per_query_rows.append(row)

        overall_perc: dict[str, dict] = {}
        for stage in STAGES:
            overall_perc[stage] = _percentiles(stage_timings[stage])

        by_language: dict[str, dict] = {}
        for lang, lt in lang_timings.items():
            by_language[lang] = {
                stage: _percentiles(lt[stage]) for stage in STAGES
            }

        n = len(per_query_rows)
        results = {
            "n_queries":      n,
            "cache_hits":     cache_hits,
            "cache_hit_rate": round(cache_hits / n, 4) if n else 0,
            "warmup_queries": warmup,
            "overall":        overall_perc,
            "by_language":    by_language,
        }

        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        if per_query_rows:
            with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(per_query_rows[0].keys()))
                writer.writeheader()
                writer.writerows(per_query_rows)

        _print_results(results)
        return results


def _print_results(r: dict) -> None:
    print("\n" + "=" * 70)
    print("LATENCY PROFILE")
    print("=" * 70)
    print(f"  Queries     : {r['n_queries']}")
    print(f"  Cache hits  : {r['cache_hits']} ({r['cache_hit_rate']:.1%})\n")

    print(f"  {'Stage':<22} {'P50ms':>7} {'P75ms':>7} {'P90ms':>7} {'P95ms':>7} {'P99ms':>7} {'Mean':>7}")
    print(f"  {'-'*64}")

    priority = [
        "normalize_ms", "lang_detect_ms", "cache_lookup_ms",
        "qa_vector_ms", "pdf_vector_ms", "qa_bm25_ms",
        "rrf_fusion_ms", "mmr_ms", "reranker_ms",
        "gate_ms", "llm_ms", "total_ms",
    ]
    for stage in priority:
        m = r["overall"].get(stage, {})
        if not m or m.get("mean", 0) == 0:
            continue
        label = stage.replace("_ms", "")
        print(
            f"  {label:<22} "
            f"{m['p50']:>7.1f} {m['p75']:>7.1f} {m['p90']:>7.1f} "
            f"{m['p95']:>7.1f} {m['p99']:>7.1f} {m['mean']:>7.1f}"
        )

    print(f"\n  Results → {OUT_JSON}")
    print(f"  CSV     → {OUT_CSV}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",    default="data/processed/splits/test.jsonl")
    parser.add_argument("--n",       type=int, default=None, help="Limit number of queries")
    parser.add_argument("--warmup",  type=int, default=3,    help="Warmup queries before timing")
    parser.add_argument("--config",  default="retrieval/configs/retrieval_config.yaml")
    args = parser.parse_args()
    profiler = LatencyProfiler(config_path=args.config)
    profiler.profile(test_path=args.test, n_queries=args.n, warmup=args.warmup)