from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.normalization import normalize_query
from retrieval.language import detect_language
from retrieval.pipeline import RetrievalPipeline

SEP = "─" * 65


def run_demo(
    query: str,
    config_path: str = "retrieval/configs/retrieval_config.yaml",
    rerank_top_k: int = 3,
) -> None:
    print(f"\n{'═' * 65}")
    print("  FINGUARD RERANKING DEMO")
    print(f"{'═' * 65}")
    print(f"  Query : {query}")

    query_norm = normalize_query(query)
    lang_result = detect_language(query_norm)
    print(f"  Lang  : {lang_result.label} (conf={lang_result.confidence:.2f})")
    print(f"  Norm  : {query_norm}\n")

    print("Loading pipeline...")
    pipeline = RetrievalPipeline(config_path=config_path)

    print("\n" + SEP)
    print("  STEP 1 — RETRIEVAL + RERANK + GATE")
    print(SEP)
    result = pipeline.run(query_norm)
    docs = result.docs
    diag = result.diagnostics

    print(f"  Language weights → BM25: {diag.bm25_weight}, Vector: {diag.vector_weight}")
    print(f"  Timings: vector={diag.timings_ms.get('vector_retrieve')}ms | "
          f"bm25={diag.timings_ms.get('bm25_retrieve')}ms | "
          f"fusion={diag.timings_ms.get('rrf_fusion')}ms | "
          f"mmr={diag.timings_ms.get('mmr')}ms\n")

    print(f"  {'Rank':<5} {'Doc ID':<20} {'MMR Score':<12} {'Category':<22} {'Novelty'}")
    print(f"  {'-'*72}")
    for doc in docs:
        print(
            f"  {doc.get('rank', '?'):<5} "
            f"{doc.get('doc_id', '?'):<20} "
            f"{doc.get('mmr_score', 0):<12.4f} "
            f"{doc.get('category', '?'):<22} "
            f"{doc.get('novelty_score', 0):.4f}"
        )

    print("\n" + SEP)
    print(f"  STEP 2 — RERANKER OUTPUT (top-{rerank_top_k})")
    print(SEP)
    print(f"  {'New Rank':<9} {'Old Rank':<9} {'Doc ID':<20} {'Score':<10}")
    print(f"  {'-'*60}")
    for r in diag.reranker_results[:rerank_top_k]:
        old_rank = r.get("rank_before_rerank")
        new_rank = r.get("rank_after_rerank")
        move = (old_rank - new_rank) if isinstance(old_rank, int) and isinstance(new_rank, int) else 0
        indicator = f"(↑{move})" if move > 0 else f"(↓{abs(move)})" if move < 0 else "(=)"
        print(
            f"  {new_rank:<9} {old_rank:<9} {r.get('doc_id','?'):<20} "
            f"{r.get('reranker_score',0):<10.6f} {indicator}"
        )

    print("\n" + SEP)
    print("  STEP 3 — CONFIDENCE GATE")
    print(SEP)
    status = "✅ PASSED" if diag.gate_passed else "❌ FAILED"
    llm_status = "YES — proceed to LLM" if result.should_call_llm else "NO  — return fallback message"

    print(f"  Status         : {status}")
    print(f"  Top score      : {diag.top_reranker_score:.6f}")
    print(f"  Threshold      : {diag.gate_threshold}")
    print(f"  Reason         : {diag.gate_reason}")
    print(f"  LLM allowed?   : {llm_status}")

    if not diag.gate_passed:
        print(f"\n  Fallback msg   : {result.fallback_answer}")

    print(f"\n{'═' * 65}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finguard Reranking Demo")
    parser.add_argument("--query", type=str, required=True, help="Query to test")
    parser.add_argument("--config", type=str, default="retrieval/configs/retrieval_config.yaml")
    parser.add_argument("--top_k", type=int, default=3, help="Reranker top-k (default 3)")
    args = parser.parse_args()

    run_demo(query=args.query, config_path=args.config, rerank_top_k=args.top_k)