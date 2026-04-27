from __future__ import annotations

import json
import sys
import logging
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

REQUIRED_FIELDS = {"doc_id", "question", "answer", "category", "difficulty", "source_id"}
VALID_CATEGORIES = {
    "personal_finance", "islamic_finance", "financial_education", "banking",
    "investment", "loans_credit", "digital_finance", "bills_payments",
}
VALID_DIFFICULTIES = {"easy", "medium", "complex"}

MIN_QUESTION_WORDS = 3
MAX_QUESTION_WORDS = 60
MIN_ANSWER_WORDS = 5
MAX_ANSWER_WORDS = 300


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    log.error(f"Line {i}: JSON parse error — {e}")
    return records


def validate_corpus(corpus_path: str = "retrieval/artifacts/corpus.jsonl") -> int:
    path = Path(corpus_path)
    if not path.exists():
        log.error(f"Corpus file not found: {path}")
        return 1

    log.info(f"Validating corpus: {path}")
    docs = _load_jsonl(path)
    log.info(f"Loaded {len(docs)} documents")

    errors = 0
    warnings = 0

    seen_doc_ids: set[str] = set()
    seen_source_ids: set[str] = set()

    q_length_violations: list[str] = []
    a_length_violations: list[str] = []
    cat_counter: Counter = Counter()
    diff_counter: Counter = Counter()

    for i, doc in enumerate(docs):
        doc_id = doc.get("doc_id", f"<missing_doc_id_#{i}>")

        missing = REQUIRED_FIELDS - set(doc.keys())
        if missing:
            log.error(f"{doc_id}: missing required fields: {missing}")
            errors += 1

        empty = {f for f in REQUIRED_FIELDS if f in doc and not str(doc[f]).strip()}
        if empty:
            log.error(f"{doc_id}: empty required fields: {empty}")
            errors += 1

        if doc_id in seen_doc_ids:
            log.error(f"Duplicate doc_id: {doc_id}")
            errors += 1
        seen_doc_ids.add(doc_id)

        source_id = doc.get("source_id", "")
        if source_id in seen_source_ids:
            log.error(f"{doc_id}: duplicate source_id '{source_id}' — chunking policy violated (one QA = one doc)")
            errors += 1
        if source_id:
            seen_source_ids.add(source_id)

        if not doc_id.startswith("doc_"):
            log.warning(f"{doc_id}: doc_id does not follow 'doc_{{source_id}}' format")
            warnings += 1

        cat = doc.get("category", "")
        if cat and cat not in VALID_CATEGORIES:
            log.error(f"{doc_id}: invalid category '{cat}'")
            errors += 1
        if cat:
            cat_counter[cat] += 1

        diff = doc.get("difficulty", "")
        if diff and diff not in VALID_DIFFICULTIES:
            log.error(f"{doc_id}: invalid difficulty '{diff}'")
            errors += 1
        if diff:
            diff_counter[diff] += 1

        question = doc.get("question", "")
        q_words = len(question.split())
        if q_words < MIN_QUESTION_WORDS:
            q_length_violations.append(f"{doc_id}: question too short ({q_words} words)")
            errors += 1
        elif q_words > MAX_QUESTION_WORDS:
            log.warning(f"{doc_id}: question unusually long ({q_words} words)")
            warnings += 1

        answer = doc.get("answer", "")
        a_words = len(answer.split())
        if a_words < MIN_ANSWER_WORDS:
            a_length_violations.append(f"{doc_id}: answer too short ({a_words} words)")
            errors += 1
        elif a_words > MAX_ANSWER_WORDS:
            log.warning(f"{doc_id}: answer unusually long ({a_words} words)")
            warnings += 1

    for v in q_length_violations[:10]:
        log.error(v)
    if len(q_length_violations) > 10:
        log.error(f"... and {len(q_length_violations) - 10} more question length violations")

    for v in a_length_violations[:10]:
        log.error(v)
    if len(a_length_violations) > 10:
        log.error(f"... and {len(a_length_violations) - 10} more answer length violations")

    log.info(f"Category distribution: {dict(cat_counter)}")
    log.info(f"Difficulty distribution: {dict(diff_counter)}")

    missing_cats = VALID_CATEGORIES - set(cat_counter.keys())
    if missing_cats:
        log.warning(f"Missing categories in corpus (all 8 expected): {missing_cats}")
        warnings += 1

    log.info("=" * 60)
    if errors == 0:
        log.info(f"✅ Corpus valid — {len(docs)} documents, {warnings} warnings")
        log.info("Safe to proceed with build_vector_index and build_bm25_index.")
        return 0
    else:
        log.error(f"❌ Corpus invalid — {errors} errors, {warnings} warnings")
        log.error("Fix errors before indexing.")
        return 1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="retrieval/artifacts/corpus.jsonl")
    args = parser.parse_args()
    sys.exit(validate_corpus(corpus_path=args.corpus))