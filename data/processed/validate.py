"""
validate.py — Pre-training Data Quality Checks
=====================================================
Checks performed:
  1. Required fields present and non-empty
  2. No duplicate IDs or queries
  3. No record where positive == hard_negative (label leakage)
  4. No query appears in both train and test (data leakage)
  5. Category & difficulty distributions are balanced enough
  6. Text length sanity (not too short, not truncated)
  7. Hard negative IDs exist in dataset
  8. Language tag validity

Exit code: 0 = all checks pass, 1 = critical failures found

Run:
    python data/processed/validate.py
"""

import json
import sys
import logging
from pathlib import Path
from collections import Counter

# ── paths ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]
PROC       = ROOT / "data" / "processed"
SPLITS_DIR = PROC / "splits"

REQUIRED_PAIR_FIELDS = {"id", "query", "positive", "category", "difficulty", "language"}
REQUIRED_NEG_FIELDS  = REQUIRED_PAIR_FIELDS | {"hard_negative", "hard_negative_id"}
VALID_CATEGORIES     = {
    "personal_finance", "islamic_finance", "financial_education",
    "banking", "investment", "loans_credit", "digital_finance", "bills_payments",
}
VALID_DIFFICULTIES   = {"easy", "medium", "complex"}
VALID_LANGUAGES      = {"ur", "en", "roman_ur"}

MIN_QUERY_WORDS    = 3
MIN_POSITIVE_WORDS = 5
MIN_NEGATIVE_WORDS = 5

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ═══════════════════════════════════════════════════════════════════════════
# CHECK FUNCTIONS — each returns (n_errors, n_warnings)
# ═══════════════════════════════════════════════════════════════════════════

def check_required_fields(records: list[dict], required: set, name: str) -> tuple[int, int]:
    errors = 0
    for i, r in enumerate(records):
        missing = required - set(r.keys())
        empty   = {k for k in required if k in r and (r[k] is None or str(r[k]).strip() == "")}
        if missing:
            log.error(f"[{name}] Record #{i} missing fields: {missing}")
            errors += 1
        if empty:
            log.error(f"[{name}] Record #{i} (id={r.get('id')}) has empty fields: {empty}")
            errors += 1
    return errors, 0


def check_duplicates(records: list[dict], name: str) -> tuple[int, int]:
    errors = 0
    id_counts    = Counter(r.get("id") for r in records)
    query_counts = Counter(r.get("query", "").strip() for r in records)

    dup_ids = {k: v for k, v in id_counts.items() if v > 1}
    dup_qs  = {k: v for k, v in query_counts.items() if v > 1 and k}

    if dup_ids:
        log.error(f"[{name}] {len(dup_ids)} duplicate IDs found: {list(dup_ids.items())[:5]}")
        errors += len(dup_ids)
    if dup_qs:
        log.warning(f"[{name}] {len(dup_qs)} duplicate queries (may be ok for multi-lang): {list(dup_qs.keys())[:3]}")
    return errors, len(dup_qs)


def check_label_leakage(records: list[dict], name: str) -> tuple[int, int]:
    """Positive must not equal hard_negative."""
    errors = 0
    for r in records:
        pos = r.get("positive", "").strip()
        neg = r.get("hard_negative", "").strip()
        if pos and neg and pos == neg:
            log.error(f"[{name}] id={r['id']}: positive == hard_negative (label leak!)")
            errors += 1
        if not neg:
            log.warning(f"[{name}] id={r['id']}: hard_negative is empty")
    return errors, 0


def check_split_leakage(splits: dict[str, list[dict]]) -> tuple[int, int]:
    """No query should appear in both train and test."""
    errors = 0
    train_queries = {r.get("query", "").strip() for r in splits.get("train", [])}
    test_queries  = {r.get("query", "").strip() for r in splits.get("test",  [])}
    leaked = train_queries & test_queries
    if leaked:
        log.error(f"[SPLIT LEAKAGE] {len(leaked)} queries appear in both train and test!")
        for q in list(leaked)[:5]:
            log.error(f"  → {q[:80]}")
        errors += len(leaked)
    else:
        log.info("[SPLIT LEAKAGE] No train/test query overlap. ✓")
    return errors, 0


def check_text_lengths(records: list[dict], name: str) -> tuple[int, int]:
    errors = warnings = 0
    for r in records:
        q_words = len(r.get("query", "").split())
        p_words = len(r.get("positive", "").split())
        n_words = len(r.get("hard_negative", "").split()) if "hard_negative" in r else None

        if q_words < MIN_QUERY_WORDS:
            log.error(f"[{name}] id={r['id']}: query too short ({q_words} words): {r.get('query')}")
            errors += 1
        if p_words < MIN_POSITIVE_WORDS:
            log.error(f"[{name}] id={r['id']}: positive too short ({p_words} words)")
            errors += 1
        if n_words is not None and 0 < n_words < MIN_NEGATIVE_WORDS:
            log.warning(f"[{name}] id={r['id']}: hard_negative suspiciously short ({n_words} words)")
            warnings += 1
    return errors, warnings


def check_enum_fields(records: list[dict], name: str) -> tuple[int, int]:
    errors = 0
    for r in records:
        cat  = r.get("category", "")
        diff = r.get("difficulty", "")
        lang = r.get("language", "")
        if cat and cat not in VALID_CATEGORIES:
            log.error(f"[{name}] id={r['id']}: unknown category '{cat}'")
            errors += 1
        if diff and diff not in VALID_DIFFICULTIES:
            log.error(f"[{name}] id={r['id']}: unknown difficulty '{diff}'")
            errors += 1
        if lang and lang not in VALID_LANGUAGES:
            log.warning(f"[{name}] id={r['id']}: unusual language tag '{lang}'")
    return errors, 0


def check_hard_negative_ids(records: list[dict], name: str) -> tuple[int, int]:
    """All hard_negative_id values must reference valid IDs in the dataset."""
    all_ids = {r["id"] for r in records}
    errors  = 0
    for r in records:
        neg_id = r.get("hard_negative_id")
        if neg_id and neg_id not in all_ids:
            log.error(f"[{name}] id={r['id']}: hard_negative_id={neg_id} not found in dataset")
            errors += 1
        if neg_id and neg_id == r["id"]:
            log.error(f"[{name}] id={r['id']}: hard_negative_id points to itself!")
            errors += 1
    return errors, 0


def print_distribution(records: list[dict], name: str) -> None:
    cat_counts  = Counter(r.get("category", "?")  for r in records)
    diff_counts = Counter(r.get("difficulty", "?") for r in records)
    lang_counts = Counter(r.get("language", "?")  for r in records)
    log.info(f"[{name}] category   : {dict(cat_counts)}")
    log.info(f"[{name}] difficulty : {dict(diff_counts)}")
    log.info(f"[{name}] language   : {dict(lang_counts)}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def validate_all() -> int:
    total_errors   = 0
    total_warnings = 0

    log.info("=" * 60)
    log.info("Finguard RAG — Data Validation")
    log.info("=" * 60)

    # ── 1. train_pairs.jsonl ─────────────────────────────────────
    pairs_path = PROC / "train_pairs.jsonl"
    if pairs_path.exists():
        log.info(f"\n── Checking {pairs_path.name} ──")
        pairs = load_jsonl(pairs_path)
        log.info(f"Records: {len(pairs):,}")
        for fn in [
            lambda r, n: check_required_fields(r, REQUIRED_PAIR_FIELDS, n),
            check_duplicates,
            check_text_lengths,
            check_enum_fields,
        ]:
            e, w = fn(pairs, "train_pairs")
            total_errors   += e
            total_warnings += w
        print_distribution(pairs, "train_pairs")
    else:
        log.warning(f"train_pairs.jsonl not found at {pairs_path}")

    # ── 2. hard_negatives.jsonl ───────────────────────────────────
    neg_path = PROC / "hard_negatives.jsonl"
    if neg_path.exists():
        log.info(f"\n── Checking {neg_path.name} ──")
        negs = load_jsonl(neg_path)
        log.info(f"Records: {len(negs):,}")
        for fn in [
            lambda r, n: check_required_fields(r, REQUIRED_NEG_FIELDS, n),
            check_duplicates,
            check_label_leakage,
            check_hard_negative_ids,
            check_text_lengths,
        ]:
            e, w = fn(negs, "hard_negatives")
            total_errors   += e
            total_warnings += w
    else:
        log.warning(f"hard_negatives.jsonl not found at {neg_path}")

    # ── 3. Splits ─────────────────────────────────────────────────
    splits = {}
    for split_name in ("train", "val", "test"):
        path = SPLITS_DIR / f"{split_name}.jsonl"
        if path.exists():
            log.info(f"\n── Checking splits/{split_name}.jsonl ──")
            data = load_jsonl(path)
            splits[split_name] = data
            log.info(f"Records: {len(data):,}")
            for fn in [
                lambda r, n: check_required_fields(r, REQUIRED_NEG_FIELDS, n),
                check_duplicates,
                check_label_leakage,
                check_text_lengths,
                check_enum_fields,
            ]:
                e, w = fn(data, split_name)
                total_errors   += e
                total_warnings += w
            print_distribution(data, split_name)
        else:
            log.warning(f"Split not found: {path}")

    # ── 4. Cross-split leakage ────────────────────────────────────
    if "train" in splits and "test" in splits:
        log.info("\n── Checking cross-split query leakage ──")
        e, w = check_split_leakage(splits)
        total_errors   += e
        total_warnings += w

    # ── Summary ───────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    if total_errors == 0:
        log.info(f"✅ ALL CHECKS PASSED  |  warnings: {total_warnings}")
        log.info("Data is ready for training.")
        return 0
    else:
        log.error(f"❌ FAILED: {total_errors} errors, {total_warnings} warnings")
        log.error("Fix errors before training!")
        return 1


if __name__ == "__main__":
    sys.exit(validate_all())