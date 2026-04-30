"""
observability/analyzer.py

Pandas-based analysis of the JSONL request log.

Computes:
  - Cache hit rates (L1, L2, miss)
  - P50/P95/P99 latency per stage and total
  - Gate pass/fail rates
  - Source mix (QA vs PDF contribution)
  - Category distribution
  - Failure query patterns (gate blocked, LLM error)
  - Per-language breakdown

Run:
  python observability/analyzer.py
  python observability/analyzer.py --log_file observability/logs/requests_2025-01-15.jsonl
  python observability/analyzer.py --tail 500  (last N requests only)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

DEFAULT_LOG_DIR = Path("observability/logs")


def load_log(log_path: Optional[Path] = None, tail: Optional[int] = None) -> "pd.DataFrame":
    try:
        import pandas as pd
    except ImportError:
        print("pip install pandas")
        sys.exit(1)

    if log_path is None:
        log_files = sorted(DEFAULT_LOG_DIR.glob("requests*.jsonl"))
        if not log_files:
            print(f"No log files found in {DEFAULT_LOG_DIR}")
            sys.exit(1)
        log_path = log_files[-1]

    print(f"Loading: {log_path}")
    records = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if not records:
        print("No records found in log file.")
        sys.exit(0)

    df = pd.DataFrame(records)

    if tail:
        df = df.tail(tail).reset_index(drop=True)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    for col in ["retrieval_ms", "reranker_ms", "llm_ms", "total_ms", "top_reranker_score",
                "category_confidence", "cache_similarity", "language_confidence"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["gate_passed", "filter_applied", "answer_generated"]:
        if col in df.columns:
            df[col] = df[col].astype(bool, errors="ignore")

    print(f"Loaded {len(df)} records\n")
    return df


def _pct(n: int, total: int) -> str:
    return f"{n} ({100*n/max(total,1):.1f}%)"


def _percentile_row(series: "pd.Series", label: str) -> str:
    p50 = series.quantile(0.50)
    p95 = series.quantile(0.95)
    p99 = series.quantile(0.99)
    mn = series.mean()
    return f"  {label:<20} mean={mn:>7.1f}ms  P50={p50:>7.1f}ms  P95={p95:>7.1f}ms  P99={p99:>7.1f}ms"


def report_overview(df: "pd.DataFrame") -> None:
    total = len(df)
    print("=" * 65)
    print("OVERVIEW")
    print("=" * 65)
    print(f"  Total requests    : {total}")
    if "timestamp" in df.columns and df["timestamp"].notna().any():
        t_min = df["timestamp"].min()
        t_max = df["timestamp"].max()
        span = (t_max - t_min).total_seconds()
        qps = total / max(span, 1)
        print(f"  Time range        : {t_min} → {t_max}")
        print(f"  Duration          : {span/3600:.2f}h")
        print(f"  Avg QPS           : {qps:.2f}")
    if "answer_generated" in df.columns:
        answered = df["answer_generated"].sum()
        print(f"  Answers generated : {_pct(answered, total)}")
    print()


def report_cache(df: "pd.DataFrame") -> None:
    print("=" * 65)
    print("CACHE HIT RATES")
    print("=" * 65)
    if "cache_level" not in df.columns:
        print("  No cache_level column found.\n")
        return

    total = len(df)
    l1 = (df["cache_level"] == "l1_hit").sum()
    l2 = (df["cache_level"] == "l2_hit").sum()
    miss = (df["cache_level"] == "miss").sum()

    print(f"  L1 hits (answer)  : {_pct(l1, total)}")
    print(f"  L2 hits (retriev) : {_pct(l2, total)}")
    print(f"  Misses            : {_pct(miss, total)}")
    print(f"  Combined hit rate : {100*(l1+l2)/max(total,1):.1f}%")

    if "cache_similarity" in df.columns:
        hits = df[df["cache_level"] != "miss"]["cache_similarity"].dropna()
        misses = df[df["cache_level"] == "miss"]["cache_similarity"].dropna()
        if not hits.empty:
            print(f"  Avg sim on hits   : {hits.mean():.4f}")
        if not misses.empty:
            print(f"  Avg sim on misses : {misses.mean():.4f}")
    print()


def report_latency(df: "pd.DataFrame") -> None:
    print("=" * 65)
    print("LATENCY (ms)")
    print("=" * 65)

    stage_cols = {
        "retrieval_ms": "Retrieval",
        "reranker_ms": "Reranker",
        "llm_ms": "LLM",
        "total_ms": "Total",
    }
    for col, label in stage_cols.items():
        if col in df.columns:
            s = df[col].dropna()
            if not s.empty:
                print(_percentile_row(s, label))

    if "total_ms" in df.columns and "cache_level" in df.columns:
        print()
        for level, label in [("l1_hit", "L1 cache hit"), ("l2_hit", "L2 cache hit"), ("miss", "Cache miss")]:
            subset = df[df["cache_level"] == level]["total_ms"].dropna()
            if not subset.empty:
                print(f"  {label:<20} mean={subset.mean():>7.1f}ms  P95={subset.quantile(0.95):>7.1f}ms  n={len(subset)}")
    print()


def report_gate(df: "pd.DataFrame") -> None:
    print("=" * 65)
    print("CONFIDENCE GATE")
    print("=" * 65)
    if "gate_passed" not in df.columns:
        print("  No gate_passed column.\n")
        return

    total = len(df)
    passed = df["gate_passed"].sum()
    blocked = total - passed
    print(f"  Passed  : {_pct(passed, total)}")
    print(f"  Blocked : {_pct(blocked, total)}")

    if "gate_reason" in df.columns:
        reasons = df[df["gate_passed"] == False]["gate_reason"].value_counts()
        if not reasons.empty:
            print("  Block reasons:")
            for reason, cnt in reasons.items():
                print(f"    {reason}: {cnt}")

    if "top_reranker_score" in df.columns:
        passed_scores = df[df["gate_passed"] == True]["top_reranker_score"].dropna()
        blocked_scores = df[df["gate_passed"] == False]["top_reranker_score"].dropna()
        if not passed_scores.empty:
            print(f"  Avg score (passed)  : {passed_scores.mean():.4f}")
        if not blocked_scores.empty:
            print(f"  Avg score (blocked) : {blocked_scores.mean():.4f}")
        high_conf = (df["top_reranker_score"] > 0.70).sum()
        print(f"  Score > 0.70        : {_pct(high_conf, total)}")
    print()


def report_source_mix(df: "pd.DataFrame") -> None:
    print("=" * 65)
    print("SOURCE MIX (QA vs PDF)")
    print("=" * 65)
    if "source_mix" not in df.columns:
        print("  No source_mix column.\n")
        return

    import pandas as pd

    mixes = df["source_mix"].dropna()
    if mixes.empty:
        print("  No source_mix data.\n")
        return

    def parse_mix(v):
        if isinstance(v, dict):
            return v
        if isinstance(v, str):
            try:
                return json.loads(v)
            except Exception:
                return {}
        return {}

    parsed = mixes.apply(parse_mix)
    # Support both schemas:
    # - {"qa": n, "pdf": n}
    # - {"qa_docs": n, "pdf_docs": n}
    qa_counts = parsed.apply(lambda m: int(m.get("qa", m.get("qa_docs", 0))))
    pdf_counts = parsed.apply(lambda m: int(m.get("pdf", m.get("pdf_docs", 0))))

    total_docs = qa_counts.sum() + pdf_counts.sum()
    print(f"  QA docs served    : {qa_counts.sum()} ({100*qa_counts.sum()/max(total_docs,1):.1f}%)")
    print(f"  PDF docs served   : {pdf_counts.sum()} ({100*pdf_counts.sum()/max(total_docs,1):.1f}%)")

    qa_only = ((qa_counts > 0) & (pdf_counts == 0)).sum()
    pdf_only = ((pdf_counts > 0) & (qa_counts == 0)).sum()
    mixed = ((qa_counts > 0) & (pdf_counts > 0)).sum()
    n = len(parsed)
    print(f"  QA-only answers   : {_pct(qa_only, n)}")
    print(f"  PDF-only answers  : {_pct(pdf_only, n)}")
    print(f"  Mixed answers     : {_pct(mixed, n)}")

    if pdf_counts.sum() == 0:
        print("  ⚠ PDF index contributing 0 docs — check dual_retriever config")
    print()


def report_language(df: "pd.DataFrame") -> None:
    print("=" * 65)
    print("PER-LANGUAGE BREAKDOWN")
    print("=" * 65)
    if "query_language" not in df.columns:
        print("  No query_language column.\n")
        return

    total = len(df)
    lang_counts = df["query_language"].value_counts()
    print("  Distribution:")
    for lang, cnt in lang_counts.items():
        print(f"    {lang:<15} {_pct(cnt, total)}")

    if "gate_passed" in df.columns and "total_ms" in df.columns:
        print()
        print("  Gate pass rate per language:")
        for lang in lang_counts.index:
            subset = df[df["query_language"] == lang]
            gate_rate = subset["gate_passed"].mean()
            avg_ms = subset["total_ms"].mean()
            print(f"    {lang:<15} gate_pass={gate_rate:.1%}  avg_ms={avg_ms:.0f}")
    print()


def report_category(df: "pd.DataFrame") -> None:
    print("=" * 65)
    print("CATEGORY DISTRIBUTION")
    print("=" * 65)
    if "category_detected" not in df.columns:
        print("  No category_detected column.\n")
        return

    total = len(df)
    cat_counts = df["category_detected"].fillna("none").value_counts()
    for cat, cnt in cat_counts.items():
        print(f"  {cat:<25} {_pct(cnt, total)}")

    if "filter_applied" in df.columns:
        filtered = df["filter_applied"].sum()
        print(f"\n  Filter applied    : {_pct(filtered, total)}")
    print()


def report_failures(df: "pd.DataFrame", top_n: int = 20) -> None:
    print("=" * 65)
    print(f"FAILURE QUERY PATTERNS (top {top_n})")
    print("=" * 65)
    if "gate_passed" not in df.columns:
        print("  No gate data.\n")
        return

    blocked = df[df["gate_passed"] == False]
    if blocked.empty:
        print("  No blocked queries — gate never triggered.\n")
        return

    print(f"  Total blocked: {len(blocked)}")
    print()

    if "query" in blocked.columns:
        print("  Sample blocked queries:")
        for q in blocked["query"].head(top_n):
            print(f"    • {q[:90]}")

    if "error" in df.columns:
        errors = df[df["error"].notna()]
        if not errors.empty:
            print(f"\n  Requests with errors: {len(errors)}")
            for e in errors["error"].value_counts().head(5).items():
                print(f"    {e[0]}: {e[1]}")
    print()


def full_report(df: "pd.DataFrame") -> None:
    report_overview(df)
    report_cache(df)
    report_latency(df)
    report_gate(df)
    report_source_mix(df)
    report_language(df)
    report_category(df)
    report_failures(df)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze FinGuard request logs")
    p.add_argument("--log_file", default=None, help="Path to specific JSONL log file")
    p.add_argument("--log_dir", default=str(DEFAULT_LOG_DIR), help="Log directory")
    p.add_argument("--tail", type=int, default=None, help="Analyze only last N records")
    p.add_argument("--section", default="all",
                   choices=["all", "overview", "cache", "latency", "gate",
                            "source", "language", "category", "failures"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    log_path = Path(args.log_file) if args.log_file else None
    if log_path is None:
        log_dir = Path(args.log_dir)
        files = sorted(log_dir.glob("requests*.jsonl"))
        if files:
            log_path = files[-1]

    df = load_log(log_path, tail=args.tail)

    dispatch = {
        "overview": report_overview,
        "cache": report_cache,
        "latency": report_latency,
        "gate": report_gate,
        "source": report_source_mix,
        "language": report_language,
        "category": report_category,
        "failures": report_failures,
    }

    if args.section == "all":
        full_report(df)
    else:
        dispatch[args.section](df)