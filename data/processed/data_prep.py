"""
data_prep.py — Phase 1 Data Pipeline for Finguard RAG
======================================================
Steps:
  1. Load raw JSONL dataset
  2. Normalize text (Urdu / Roman Urdu / English)
  3. Build query/positive pairs (bilingual strategy)
  4. Mine hard negatives (category-aware, intent-diverse)
  5. Stratified train/val/test split (80/10/10)
  6. Save all processed files

Run:
    python src/data_prep.py
"""

import json
import re
import random
import logging
from pathlib import Path
from collections import defaultdict
from typing import Optional

import numpy as np

# ── reproducibility ─────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── paths ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[2]
RAW_FILE    = ROOT / "data" / "raw" / "urdu_finance_qa.jsonl"
TRAIN_PAIRS = ROOT / "data" / "processed" / "train_pairs.jsonl"
HARD_NEG    = ROOT / "data" / "processed" / "hard_negatives.jsonl"
SPLITS_DIR  = ROOT / "data" / "processed" / "splits"

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# 1. TEXT NORMALISATION
# ═══════════════════════════════════════════════════════════════════════════

# Urdu-specific character normalisations
URDU_REPLACEMENTS = [
    ("\u0624", "\u0648"),   # waw with hamza → waw
    ("\u0625", "\u0627"),   # alef with hamza below → alef
    ("\u0623", "\u0627"),   # alef with hamza above → alef
    ("\ufb50", "\u0627"),   # alef wasla
    ("\u00ab", '"'),        # guillemets
    ("\u00bb", '"'),
]

# Roman Urdu: collapse repeated chars common in informal writing
ROMAN_COLLAPSE = re.compile(r'(.)\1{2,}')

def normalize_text(text: str, lang_hint: str = "auto") -> str:
    """
    Light normalization that preserves meaning.
    lang_hint: 'ur', 'en', 'roman_ur', 'auto'
    """
    if not text or not isinstance(text, str):
        return ""

    # 1. Unicode normalization
    text = text.strip()

    # 2. Urdu-specific replacements
    for src, tgt in URDU_REPLACEMENTS:
        text = text.replace(src, tgt)

    # 3. Fix whitespace around punctuation
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s([،,؟?!۔.])', r'\1', text)

    # 4. Roman Urdu: collapse excessive repeated letters (kyyyya → kya)
    if lang_hint in ("roman_ur", "auto"):
        text = ROMAN_COLLAPSE.sub(r'\1\1', text)

    # 5. Remove HTML artifacts if any
    text = re.sub(r'<[^>]+>', '', text)

    # 6. Strip leading/trailing whitespace again
    return text.strip()


def detect_lang(text: str) -> str:
    """Detect if text is Urdu script, Roman Urdu, or English."""
    urdu_chars = sum(1 for c in text if '\u0600' <= c <= '\u06ff')
    if urdu_chars / max(len(text), 1) > 0.3:
        return "ur"
    # Simple heuristic: if common Roman Urdu words present
    roman_ur_markers = {"ka", "ki", "ke", "hai", "hain", "kya", "aur", "se", "ko", "mein"}
    words = set(text.lower().split())
    if len(words & roman_ur_markers) >= 2:
        return "roman_ur"
    return "en"


# ═══════════════════════════════════════════════════════════════════════════
# 2. BUILD QUERY / POSITIVE PAIRS
# ═══════════════════════════════════════════════════════════════════════════

def build_pair(record: dict) -> dict:
    """
    Bilingual query strategy:
      - If question_ur contains Urdu script → use question_ur as query
      - If question_ur is Roman Urdu only  → use question_ur as query (native form)
      - Always use answer_ur as positive (richer, native)
      - Store English counterparts for cross-lingual retrieval
    """
    q_ur  = normalize_text(record["question_ur"], lang_hint="auto")
    q_en  = normalize_text(record["question_en"], lang_hint="en")
    a_ur  = normalize_text(record["answer_ur"],   lang_hint="auto")
    a_en  = normalize_text(record["answer_en"],   lang_hint="en")

    lang = detect_lang(q_ur)

    return {
        "id":         record["id"],
        "query":      q_ur,          # primary query (Urdu / Roman Urdu)
        "query_en":   q_en,          # English query (for cross-lingual eval)
        "positive":   a_ur,          # primary positive (Urdu answer)
        "positive_en": a_en,         # English positive (secondary)
        "category":   record["category"],
        "difficulty": record["difficulty"],
        "language":   lang,
        "keywords":   record.get("keywords", []),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. HARD NEGATIVE MINING
# ═══════════════════════════════════════════════════════════════════════════

# Category groups: categories that are semantically closest to each other
# Hard negatives should come from these "confusable" pairs
CONFUSABLE_CATS = {
    "islamic_finance":    ["personal_finance", "loans_credit", "investment"],
    "personal_finance":   ["islamic_finance", "financial_education", "banking"],
    "loans_credit":       ["islamic_finance", "banking", "personal_finance"],
    "investment":         ["personal_finance", "financial_education", "banking"],
    "banking":            ["personal_finance", "loans_credit", "digital_finance"],
    "digital_finance":    ["banking", "bills_payments", "personal_finance"],
    "bills_payments":     ["digital_finance", "banking", "personal_finance"],
    "financial_education":["personal_finance", "investment", "banking"],
}


TOKEN_RE = re.compile(r"[\u0600-\u06FFA-Za-z0-9]+")


def tokenize_light(text: str) -> set[str]:
    return {t.lower() for t in TOKEN_RE.findall(text or "")}


def keyword_overlap(kw1: list, kw2: list) -> int:
    """Count shared keywords (lower-cased)."""
    s1 = {k.lower() for k in kw1}
    s2 = {k.lower() for k in kw2}
    return len(s1 & s2)


def jaccard_similarity(tokens_a: set[str], tokens_b: set[str]) -> float:
    if not tokens_a or not tokens_b:
        return 0.0
    inter = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return inter / max(union, 1)


def mine_hard_negatives(
    pairs: list[dict],
    n_candidates: int = 20,
    min_keyword_overlap: int = 0,
    max_keyword_overlap: int = 3,
) -> list[dict]:
    """
    Hard negative mining strategy:
      1. First look in confusable categories (same domain, different intent)
      2. Keep overlap in a medium band (confusable but not near-duplicate)
      3. Ensure negative answer ≠ positive answer (never leak ground truth)

    Each record gets exactly ONE hard negative (answer from another QA pair).
    """
    log.info("Mining hard negatives …")

    # Index by category for fast lookup
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        by_cat[p["category"]].append(p)

    result = []
    failed = 0
    overlap_hist = defaultdict(int)
    jaccard_scores = []

    for pair in pairs:
        own_id       = pair["id"]
        own_cat      = pair["category"]
        own_positive = pair["positive"]
        own_kws      = pair.get("keywords", [])
        own_query_tokens = tokenize_light(pair.get("query", ""))

        # 1. Build candidate pool: confusable categories first, then same category
        confusable = CONFUSABLE_CATS.get(own_cat, [])
        candidate_pool: list[dict] = []
        for cat in confusable:
            candidate_pool.extend(by_cat[cat])
        # Shuffle confusable pool
        random.shuffle(candidate_pool)

        # 2. Filter candidates
        valid = [
            c for c in candidate_pool
            if c["id"] != own_id                               # not self
            and c["positive"] != own_positive                  # not same answer
            and min_keyword_overlap <= keyword_overlap(own_kws, c.get("keywords", [])) <= max_keyword_overlap
        ]

        # 3. Fallback: same category, different record
        if len(valid) < 3:
            same_cat = [
                c for c in by_cat[own_cat]
                if c["id"] != own_id and c["positive"] != own_positive
            ]
            random.shuffle(same_cat)
            valid.extend(same_cat)

        if not valid:
            # Last resort: any other record
            valid = [c for c in pairs if c["id"] != own_id]
            random.shuffle(valid)

        # 4. Pick best candidate from top-N by lexical confusability.
        # We want semantically close but incorrect negatives.
        top_n = valid[:n_candidates]
        scored = []
        for c in top_n:
            overlap = keyword_overlap(own_kws, c.get("keywords", []))
            query_jaccard = jaccard_similarity(own_query_tokens, tokenize_light(c.get("query", "")))
            # Prefer confusable categories, then higher textual proximity.
            category_bonus = 0.1 if c.get("category") in confusable else 0.0
            score = query_jaccard + (0.05 * overlap) + category_bonus
            scored.append((score, overlap, query_jaccard, c))

        scored.sort(key=lambda x: x[0], reverse=True)

        if not scored:
            failed += 1
            result.append({**pair, "hard_negative": "", "hard_negative_id": None})
            continue

        _, chosen_overlap, chosen_jaccard, chosen = scored[0]
        overlap_hist[chosen_overlap] += 1
        jaccard_scores.append(chosen_jaccard)
        result.append({
            **pair,
            "hard_negative":    chosen["positive"],   # wrong answer for this query
            "hard_negative_id": chosen["id"],         # for traceability
        })

    log.info(f"Hard negative mining done. Failed: {failed}/{len(pairs)}")
    if jaccard_scores:
        log.info(
            "Hard negative quality → avg_query_jaccard=%.3f | overlap_hist=%s",
            float(np.mean(jaccard_scores)),
            dict(sorted(overlap_hist.items())),
        )
    return result


# ═══════════════════════════════════════════════════════════════════════════
# 4. STRATIFIED SPLIT
# ═══════════════════════════════════════════════════════════════════════════

def stratified_split(
    records: list[dict],
    train_ratio: float = 0.80,
    val_ratio:   float = 0.10,
    # test = remainder
) -> tuple[list, list, list]:
    """
    Stratified split by `category` to preserve class distribution.
    Returns (train, val, test).
    """
    by_cat: dict[str, list] = defaultdict(list)
    for r in records:
        by_cat[r["category"]].append(r)

    train, val, test = [], [], []

    for cat, items in by_cat.items():
        random.shuffle(items)
        n = len(items)
        if n <= 2:
            # Tiny buckets: keep data train-heavy and avoid empty test leakage checks.
            n_train = max(1, n - 1)
            n_val = n - n_train
        else:
            n_train = max(1, int(n * train_ratio))
            n_val   = max(1, int(n * val_ratio))
            if n_train + n_val >= n:
                n_val = max(1, n - n_train - 1)

        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

    # Shuffle splits so categories aren't contiguous
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    log.info(f"Split sizes → train: {len(train)}, val: {len(val)}, test: {len(test)}")
    return train, val, test


# ═══════════════════════════════════════════════════════════════════════════
# 5. I/O HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                log.warning(f"Skipping line {line_no}: {e}")
    return records


def load_json_or_jsonl(path: Path) -> list[dict]:
    """Auto-detect JSON array vs JSONL."""
    with open(path, encoding="utf-8") as f:
        first_char = f.read(1)
    if first_char == "[":
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return load_jsonl(path)


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info(f"Saved {len(records):,} records → {path}")


# ═══════════════════════════════════════════════════════════════════════════
# 6. MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline(raw_path: Path = RAW_FILE) -> None:
    log.info("=" * 60)
    log.info("Finguard RAG — Phase 1 Data Pipeline")
    log.info("=" * 60)

    # ── Step 1: Load ─────────────────────────────────────────────
    log.info(f"Loading dataset from: {raw_path}")
    raw = load_json_or_jsonl(raw_path)
    log.info(f"Loaded {len(raw):,} records")

    # ── Step 2: Deduplicate (keep first occurrence) ───────────────
    seen_ids  = set()
    seen_qs   = set()
    clean     = []
    for r in raw:
        rid = r.get("id")
        q   = r.get("question_ur", "").strip()
        if rid in seen_ids:
            log.warning(f"Duplicate ID skipped: {rid}")
            continue
        if q in seen_qs:
            log.warning(f"Duplicate question skipped: {q[:60]}")
            continue
        seen_ids.add(rid)
        seen_qs.add(q)
        clean.append(r)
    log.info(f"After dedup: {len(clean):,} records")

    # ── Step 3: Build query/positive pairs ────────────────────────
    log.info("Building query/positive pairs …")
    pairs = [build_pair(r) for r in clean]

    # Remove pairs with empty query or positive
    pairs = [p for p in pairs if p["query"] and p["positive"]]
    log.info(f"Valid pairs: {len(pairs):,}")

    # Save train_pairs.jsonl (before adding hard negatives)
    save_jsonl(pairs, TRAIN_PAIRS)

    # ── Step 4: Hard negative mining ─────────────────────────────
    pairs_with_neg = mine_hard_negatives(pairs)
    save_jsonl(pairs_with_neg, HARD_NEG)

    # ── Step 5: Splits ────────────────────────────────────────────
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train, val, test = stratified_split(pairs_with_neg)
    save_jsonl(train, SPLITS_DIR / "train.jsonl")
    save_jsonl(val,   SPLITS_DIR / "val.jsonl")
    save_jsonl(test,  SPLITS_DIR / "test.jsonl")

    # ── Summary ───────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Pipeline complete.")
    log.info(f"  train_pairs.jsonl   : {len(pairs):,}")
    log.info(f"  hard_negatives.jsonl: {len(pairs_with_neg):,}")
    log.info(f"  train.jsonl         : {len(train):,}")
    log.info(f"  val.jsonl           : {len(val):,}")
    log.info(f"  test.jsonl          : {len(test):,}")
    log.info("=" * 60)


if __name__ == "__main__":
    run_pipeline()