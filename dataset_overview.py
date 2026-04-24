import json
import re
from collections import Counter
from pathlib import Path
from statistics import median


def has_urdu(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text or ""))


def has_latin(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", text or ""))


def safe_words_count(text: str) -> int:
    return len((text or "").strip().split())


def load_records(path: Path):
    raw = path.read_text(encoding="utf-8").strip()

    if not raw:
        raise ValueError("Dataset file is empty.")

    # Support both JSON array and JSONL formats.
    if raw.startswith("["):
        records = json.loads(raw)
        if not isinstance(records, list):
            raise ValueError("Expected a JSON array.")
        return records

    records = []
    for idx, line in enumerate(raw.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at line {idx}: {exc}") from exc
    return records


def describe_lengths(values):
    if not values:
        return {"min": 0, "max": 0, "avg": 0.0, "median": 0}
    return {
        "min": min(values),
        "max": max(values),
        "avg": round(sum(values) / len(values), 2),
        "median": median(values),
    }


def main():
    dataset_path = Path("data/raw/urdu_finance_qa.jsonl")
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"File not found: {dataset_path}. Update path in script if needed."
        )

    records = load_records(dataset_path)
    total = len(records)

    required_fields = [
        "id",
        "question_ur",
        "question_en",
        "answer_ur",
        "answer_en",
        "category",
        "difficulty",
        "keywords",
        "source",
    ]

    missing_field_counts = Counter()
    category_counts = Counter()
    difficulty_counts = Counter()
    source_counts = Counter()
    keyword_sizes = []

    q_ur_word_lengths = []
    q_en_word_lengths = []
    a_ur_word_lengths = []
    a_en_word_lengths = []

    urdu_in_question_ur = 0
    latin_in_question_ur = 0
    latin_in_question_en = 0
    urdu_in_question_en = 0

    duplicate_ids = 0
    seen_ids = set()
    duplicate_question_pairs = 0
    seen_q_pairs = set()

    for rec in records:
        for field in required_fields:
            if field not in rec or rec[field] in ("", None, []):
                missing_field_counts[field] += 1

        rec_id = rec.get("id")
        if rec_id in seen_ids:
            duplicate_ids += 1
        else:
            seen_ids.add(rec_id)

        q_ur = str(rec.get("question_ur", "") or "")
        q_en = str(rec.get("question_en", "") or "")
        a_ur = str(rec.get("answer_ur", "") or "")
        a_en = str(rec.get("answer_en", "") or "")

        q_pair_key = (q_ur.strip().lower(), q_en.strip().lower())
        if q_pair_key in seen_q_pairs:
            duplicate_question_pairs += 1
        else:
            seen_q_pairs.add(q_pair_key)

        category_counts[str(rec.get("category", "unknown"))] += 1
        difficulty_counts[str(rec.get("difficulty", "unknown"))] += 1
        source_counts[str(rec.get("source", "unknown"))] += 1

        kws = rec.get("keywords", [])
        if isinstance(kws, list):
            keyword_sizes.append(len(kws))
        else:
            keyword_sizes.append(0)

        q_ur_word_lengths.append(safe_words_count(q_ur))
        q_en_word_lengths.append(safe_words_count(q_en))
        a_ur_word_lengths.append(safe_words_count(a_ur))
        a_en_word_lengths.append(safe_words_count(a_en))

        if has_urdu(q_ur):
            urdu_in_question_ur += 1
        if has_latin(q_ur):
            latin_in_question_ur += 1
        if has_latin(q_en):
            latin_in_question_en += 1
        if has_urdu(q_en):
            urdu_in_question_en += 1

    print("\n=== DATASET OVERVIEW ===")
    print(f"Path: {dataset_path}")
    print(f"Total records: {total}")
    print(f"Unique IDs: {len(seen_ids)}")
    print(f"Duplicate IDs: {duplicate_ids}")
    print(f"Duplicate question pairs: {duplicate_question_pairs}")

    print("\n=== FORMAT CHECK ===")
    print("Expected fields:", ", ".join(required_fields))
    if missing_field_counts:
        print("Missing/empty field counts:")
        for field, count in missing_field_counts.items():
            print(f"  - {field}: {count}")
    else:
        print("No missing/empty required fields found.")

    print("\n=== DISTRIBUTION ===")
    print("Categories:")
    for k, v in category_counts.most_common():
        print(f"  - {k}: {v}")
    print("Difficulty:")
    for k, v in difficulty_counts.most_common():
        print(f"  - {k}: {v}")
    print("Source:")
    for k, v in source_counts.most_common():
        print(f"  - {k}: {v}")

    print("\n=== LANGUAGE SIGNALS ===")
    print(f"question_ur contains Urdu script: {urdu_in_question_ur}/{total}")
    print(f"question_ur contains Latin letters: {latin_in_question_ur}/{total}")
    print(f"question_en contains Latin letters: {latin_in_question_en}/{total}")
    print(f"question_en contains Urdu script: {urdu_in_question_en}/{total}")

    print("\n=== TEXT LENGTHS (WORDS) ===")
    print(f"question_ur: {describe_lengths(q_ur_word_lengths)}")
    print(f"question_en: {describe_lengths(q_en_word_lengths)}")
    print(f"answer_ur:   {describe_lengths(a_ur_word_lengths)}")
    print(f"answer_en:   {describe_lengths(a_en_word_lengths)}")
    print(f"keywords per record: {describe_lengths(keyword_sizes)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
