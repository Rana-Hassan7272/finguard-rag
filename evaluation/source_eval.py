from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import yaml

try:
    from retrieval.normalization import normalize_query
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from retrieval.normalization import normalize_query

EVAL_DIR = Path("evaluation/results")
OUT_JSON = EVAL_DIR / "source_eval_results.json"
OUT_CSV  = EVAL_DIR / "source_eval_results.csv"

VALID_CATEGORIES = [
    "personal_finance", "islamic_finance", "financial_education", "banking",
    "investment", "loans_credit", "digital_finance", "bills_payments",
]

HIGH_CONFIDENCE_THRESHOLD = 0.70


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


def _safe_div(a: float, b: float) -> float:
    return round(a / b, 4) if b > 0 else 0.0


class SourceEvaluator:
    def __init__(self, config_path: str = "retrieval/configs/retrieval_config.yaml"):
        self.cfg = _load_config(config_path)
        self.config_path = config_path

    def evaluate(
        self,
        test_path: str,
        pdf_test_path: Optional[str] = None,
    ) -> dict:
        from retrieval.language import detect_language
        from retrieval.dual_retriever import DualRetriever
        from reranking.reranker import CrossEncoderReranker
        from reranking.confidence_gate import ConfidenceGate
        from metadata.category_detector import CategoryDetector

        dual_r   = DualRetriever(self.cfg)
        reranker = CrossEncoderReranker(self.config_path)
        gate     = ConfidenceGate(self.config_path)
        detector = CategoryDetector(self.cfg)

        qa_records  = _load_jsonl(test_path)
        pdf_records = _load_jsonl(pdf_test_path) if pdf_test_path and Path(pdf_test_path).exists() else []

        all_records = [
            {**r, "_source_type": "qa"}  for r in qa_records
        ] + [
            {**r, "_source_type": "pdf"} for r in pdf_records
        ]

        per_query_rows: list[dict] = []

        cat_stats:  dict[str, dict] = defaultdict(lambda: {
            "n": 0, "gate_passed": 0, "high_conf": 0,
            "qa_grounded": 0, "pdf_grounded": 0, "mixed": 0,
        })
        lang_stats: dict[str, dict] = defaultdict(lambda: {
            "n": 0, "gate_passed": 0, "high_conf": 0,
            "qa_grounded": 0, "pdf_grounded": 0,
        })
        overall = {
            "n": 0, "gate_passed": 0, "gate_failed": 0,
            "high_conf": 0, "qa_grounded": 0, "pdf_grounded": 0, "mixed": 0,
        }

        for rec in all_records:
            query_raw   = rec.get("query", "")
            true_cat    = rec.get("category", "unknown")
            source_type = rec.get("_source_type", "qa")

            query_norm = normalize_query(query_raw)
            lang = detect_language(query_norm).label

            try:
                result   = dual_r.retrieve(query=query_norm, language=lang, top_k=10)
                docs     = result.docs
                reranked, top_score = reranker.rerank(query_norm, docs, top_k=3)
                decision = gate.check(reranked, language=lang)
            except Exception as e:
                print(f"[WARN] pipeline failed on '{query_raw[:40]}': {e}")
                continue

            gate_passed  = decision.passed
            high_conf    = gate_passed and top_score >= HIGH_CONFIDENCE_THRESHOLD

            reranked_docs_by_id = {d.get("doc_id"): d for d in docs}
            qa_count = 0
            pdf_count = 0
            for r in reranked:
                d = reranked_docs_by_id.get(r.doc_id, {})
                if d.get("doc_type") == "pdf_chunk":
                    pdf_count += 1
                else:
                    qa_count += 1

            if pdf_count == 0:
                grounding = "qa"
            elif qa_count == 0:
                grounding = "pdf"
            else:
                grounding = "mixed"

            det       = detector.detect(query_norm)
            det_cat   = det.predicted_category or "unknown"
            det_conf  = round(det.confidence, 3)

            row = {
                "query":             query_raw[:80],
                "language":          lang,
                "true_category":     true_cat,
                "detected_category": det_cat,
                "det_confidence":    det_conf,
                "source_type":       source_type,
                "gate_passed":       int(gate_passed),
                "top_reranker_score":round(top_score, 4),
                "high_confidence":   int(high_conf),
                "grounding":         grounding,
                "qa_docs_in_top3":   qa_count,
                "pdf_docs_in_top3":  pdf_count,
            }
            per_query_rows.append(row)

            overall["n"]           += 1
            overall["gate_passed"] += int(gate_passed)
            overall["gate_failed"] += int(not gate_passed)
            overall["high_conf"]   += int(high_conf)
            overall[f"{grounding}_grounded" if grounding != "mixed" else "mixed"] += 1

            cs = cat_stats[true_cat]
            cs["n"]           += 1
            cs["gate_passed"] += int(gate_passed)
            cs["high_conf"]   += int(high_conf)
            if grounding == "qa":    cs["qa_grounded"]  += 1
            elif grounding == "pdf": cs["pdf_grounded"] += 1
            else:                    cs["mixed"]        += 1

            ls = lang_stats[lang]
            ls["n"]           += 1
            ls["gate_passed"] += int(gate_passed)
            ls["high_conf"]   += int(high_conf)
            if grounding == "qa":    ls["qa_grounded"]  += 1
            elif grounding == "pdf": ls["pdf_grounded"] += 1

        n = overall["n"] or 1

        overall_rates = {
            "n":                  overall["n"],
            "gate_pass_rate":     _safe_div(overall["gate_passed"], overall["n"]),
            "gate_fail_rate":     _safe_div(overall["gate_failed"], overall["n"]),
            "high_conf_rate":     _safe_div(overall["high_conf"],   overall["n"]),
            "qa_grounded_rate":   _safe_div(overall["qa_grounded"],  overall["n"]),
            "pdf_grounded_rate":  _safe_div(overall["pdf_grounded"], overall["n"]),
            "mixed_rate":         _safe_div(overall["mixed"],        overall["n"]),
        }

        by_category: dict[str, dict] = {}
        for cat, cs in cat_stats.items():
            cn = cs["n"] or 1
            by_category[cat] = {
                "n":                 cs["n"],
                "gate_pass_rate":    _safe_div(cs["gate_passed"], cs["n"]),
                "high_conf_rate":    _safe_div(cs["high_conf"],   cs["n"]),
                "qa_grounded_rate":  _safe_div(cs["qa_grounded"],  cs["n"]),
                "pdf_grounded_rate": _safe_div(cs["pdf_grounded"], cs["n"]),
                "mixed_rate":        _safe_div(cs["mixed"],        cs["n"]),
            }

        by_language: dict[str, dict] = {}
        for lang, ls in lang_stats.items():
            by_language[lang] = {
                "n":                 ls["n"],
                "gate_pass_rate":    _safe_div(ls["gate_passed"], ls["n"]),
                "high_conf_rate":    _safe_div(ls["high_conf"],   ls["n"]),
                "qa_grounded_rate":  _safe_div(ls["qa_grounded"],  ls["n"]),
                "pdf_grounded_rate": _safe_div(ls["pdf_grounded"], ls["n"]),
            }

        results = {
            "overall":        overall_rates,
            "by_category":    by_category,
            "by_language":    by_language,
            "high_conf_threshold_used": HIGH_CONFIDENCE_THRESHOLD,
        }

        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        if per_query_rows:
            with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(per_query_rows[0].keys()))
                writer.writeheader()
                writer.writerows(per_query_rows)

        _print_results(results)
        return results


def _print_results(r: dict) -> None:
    o = r["overall"]
    print("\n" + "=" * 65)
    print("SOURCE GROUNDING EVALUATION")
    print("=" * 65)
    print(f"\n  Total queries      : {o['n']}")
    print(f"  Gate pass rate     : {o['gate_pass_rate']:.1%}")
    print(f"  High confidence    : {o['high_conf_rate']:.1%}  (score ≥ {r['high_conf_threshold_used']})")
    print(f"  QA-grounded        : {o['qa_grounded_rate']:.1%}")
    print(f"  PDF-grounded       : {o['pdf_grounded_rate']:.1%}")
    print(f"  Mixed              : {o['mixed_rate']:.1%}")

    print("\n  ── By Category ──")
    print(f"  {'Category':<25} {'N':>4} {'Gate%':>6} {'HiConf%':>8} {'QA%':>6} {'PDF%':>6}")
    print(f"  {'-'*60}")
    for cat, m in r["by_category"].items():
        print(
            f"  {cat:<25} {m['n']:>4} "
            f"{m['gate_pass_rate']:>6.1%} "
            f"{m['high_conf_rate']:>8.1%} "
            f"{m['qa_grounded_rate']:>6.1%} "
            f"{m['pdf_grounded_rate']:>6.1%}"
        )

    print("\n  ── By Language ──")
    print(f"  {'Language':<15} {'N':>4} {'Gate%':>6} {'HiConf%':>8} {'QA%':>6} {'PDF%':>6}")
    print(f"  {'-'*50}")
    for lang, m in r["by_language"].items():
        print(
            f"  {lang:<15} {m['n']:>4} "
            f"{m['gate_pass_rate']:>6.1%} "
            f"{m['high_conf_rate']:>8.1%} "
            f"{m['qa_grounded_rate']:>6.1%} "
            f"{m['pdf_grounded_rate']:>6.1%}"
        )

    print(f"\n  Results → {OUT_JSON}")
    print(f"  CSV     → {OUT_CSV}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",     default="data/processed/splits/test.jsonl")
    parser.add_argument("--pdf_test", default="evaluation/pdf_test_queries.jsonl")
    parser.add_argument("--config",   default="retrieval/configs/retrieval_config.yaml")
    args = parser.parse_args()
    ev = SourceEvaluator(config_path=args.config)
    ev.evaluate(test_path=args.test, pdf_test_path=args.pdf_test)