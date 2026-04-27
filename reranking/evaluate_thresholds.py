from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

try:
    from retrieval.normalization import normalize_query
    from retrieval.language import detect_language
    from retrieval.pipeline import RetrievalPipeline
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from retrieval.normalization import normalize_query
    from retrieval.language import detect_language
    from retrieval.pipeline import RetrievalPipeline

SWEEP_OUTPUT = Path("retrieval/eval/reranker_threshold_sweep.csv")
SWEEP_THRESHOLDS = [round(t, 2) for t in np.arange(0.10, 0.85, 0.05)]


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


def _score_all_queries(records: list[dict], pipeline: RetrievalPipeline) -> list[dict]:
    # Build source_id -> doc_id mapping once.
    if not pipeline._vector.is_loaded:
        pipeline._vector.load()
    source_to_doc = {}
    for doc_id, doc in pipeline._vector.docstore.items():
        src = str(doc.get("source_id", ""))
        if src:
            source_to_doc[src] = doc_id

    scored = []
    for rec in records:
        query_raw = rec.get("query", "")
        relevant_id = source_to_doc.get(str(rec.get("id", "")), "")
        if not relevant_id:
            continue
        query_norm = normalize_query(query_raw)
        lang = detect_language(query_norm).label

        try:
            result = pipeline.run(query_norm)
            top_score = result.diagnostics.top_reranker_score
            reranked_ids = [r.get("doc_id") for r in result.diagnostics.reranker_results]
        except Exception:
            scored.append({
                "query_id": relevant_id,
                "language": lang,
                "top_reranker_score": 0.0,
                "relevant_in_reranked": False,
                "reranked_rank": None,
            })
            continue

        relevant_in_reranked = relevant_id in reranked_ids
        reranked_rank = (reranked_ids.index(relevant_id) + 1) if relevant_in_reranked else None

        scored.append({
            "query_id": relevant_id,
            "language": lang,
            "top_reranker_score": top_score,
            "relevant_in_reranked": relevant_in_reranked,
            "reranked_rank": reranked_rank,
        })

    return scored


def _sweep_thresholds(scored: list[dict], thresholds: list[float]) -> list[dict]:
    rows = []
    total = len(scored)

    for threshold in thresholds:
        accepted = [s for s in scored if s["top_reranker_score"] >= threshold]
        rejected = [s for s in scored if s["top_reranker_score"] < threshold]

        n_accepted = len(accepted)
        n_rejected = len(rejected)
        rejection_rate = round(n_rejected / total, 4) if total else 0.0

        true_positives = sum(1 for s in accepted if s["relevant_in_reranked"])
        false_positives = n_accepted - true_positives
        precision = round(true_positives / n_accepted, 4) if n_accepted else 0.0

        false_rejects = sum(1 for s in rejected if s["relevant_in_reranked"])
        false_reject_rate = round(false_rejects / total, 4) if total else 0.0

        by_lang: dict[str, dict] = {}
        for lang in ("urdu", "roman_urdu", "english"):
            lang_scored = [s for s in scored if s["language"] == lang]
            lang_accepted = [s for s in lang_scored if s["top_reranker_score"] >= threshold]
            lang_rejected = [s for s in lang_scored if s["top_reranker_score"] < threshold]
            lang_tp = sum(1 for s in lang_accepted if s["relevant_in_reranked"])
            by_lang[lang] = {
                "n": len(lang_scored),
                "accepted": len(lang_accepted),
                "rejected": len(lang_rejected),
                "precision": round(lang_tp / len(lang_accepted), 4) if lang_accepted else 0.0,
                "rejection_rate": round(len(lang_rejected) / len(lang_scored), 4) if lang_scored else 0.0,
                "false_reject_rate": round(
                    sum(1 for s in lang_rejected if s["relevant_in_reranked"]) / len(lang_scored), 4
                ) if lang_scored else 0.0,
            }

        row = {
            "threshold": threshold,
            "n_total": total,
            "n_accepted": n_accepted,
            "n_rejected": n_rejected,
            "rejection_rate": rejection_rate,
            "precision_on_accepted": precision,
            "false_reject_rate": false_reject_rate,
            "f1_proxy": round(
                2 * precision * (1 - false_reject_rate) / (precision + (1 - false_reject_rate) + 1e-10), 4
            ),
        }
        for lang, lm in by_lang.items():
            row[f"{lang}_precision"] = lm["precision"]
            row[f"{lang}_rejection_rate"] = lm["rejection_rate"]
            row[f"{lang}_false_reject_rate"] = lm["false_reject_rate"]

        rows.append(row)

    return rows


def _print_sweep(rows: list[dict]) -> None:
    print("\n" + "=" * 75)
    print("RERANKER THRESHOLD SWEEP")
    print(f"{'Threshold':>10} {'Accepted':>9} {'Rejected':>9} {'Precision':>10} {'FalseRej':>9} {'F1proxy':>8}")
    print("-" * 75)
    for r in rows:
        print(
            f"{r['threshold']:>10.2f} {r['n_accepted']:>9} {r['n_rejected']:>9} "
            f"{r['precision_on_accepted']:>10.4f} {r['false_reject_rate']:>9.4f} {r['f1_proxy']:>8.4f}"
        )

    best_f1 = max(rows, key=lambda x: x["f1_proxy"])
    best_prec = max(rows, key=lambda x: x["precision_on_accepted"])
    print("=" * 75)
    print(f"\n  Best F1-proxy  threshold: {best_f1['threshold']:.2f}  (F1={best_f1['f1_proxy']:.4f})")
    print(f"  Best precision threshold: {best_prec['threshold']:.2f}  (prec={best_prec['precision_on_accepted']:.4f})")
    print(f"\n  Recommendation:")
    print(f"    If you want max accuracy  → use {best_f1['threshold']:.2f}")
    print(f"    If you want high trust    → use {best_prec['threshold']:.2f}")
    print(f"\n  Output → {SWEEP_OUTPUT}\n")


def run_sweep(
    test_path: str = "data/processed/splits/test.jsonl",
    config_path: str = "retrieval/configs/retrieval_config.yaml",
    thresholds: Optional[list[float]] = None,
) -> list[dict]:
    if thresholds is None:
        thresholds = SWEEP_THRESHOLDS

    records = _load_jsonl(test_path)
    pipeline = RetrievalPipeline(config_path=config_path)

    print(f"Scoring {len(records)} queries through retrieval + reranker...")
    scored = _score_all_queries(records, pipeline)

    print(f"Sweeping {len(thresholds)} thresholds...")
    sweep_rows = _sweep_thresholds(scored, thresholds)

    SWEEP_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(sweep_rows[0].keys())
    with open(SWEEP_OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sweep_rows)

    _print_sweep(sweep_rows)
    return sweep_rows


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default="data/processed/splits/test.jsonl")
    parser.add_argument("--config", default="retrieval/configs/retrieval_config.yaml")
    args = parser.parse_args()
    run_sweep(test_path=args.test, config_path=args.config)