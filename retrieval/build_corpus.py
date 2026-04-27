from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

REQUIRED_FIELDS = {"doc_id", "question", "answer", "category", "difficulty", "source_id"}
VALID_CATEGORIES = {
    "personal_finance", "islamic_finance", "financial_education", "banking",
    "investment", "loans_credit", "digital_finance", "bills_payments",
}
VALID_DIFFICULTIES = {"easy", "medium", "complex"}
DEDUP_THRESHOLD = 0.97

ARTIFACTS = Path("retrieval/artifacts")


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


def _load_json_or_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        first = f.read(1)
    if first == "[":
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return _load_jsonl(str(path))


def _make_doc_id(source_id: str | int) -> str:
    sid = str(source_id)
    if sid.isdigit():
        return f"doc_{int(sid):04d}"
    return f"doc_{sid}"


def _validate_schema(doc: dict, idx: int) -> list[str]:
    errors = []
    for field in REQUIRED_FIELDS:
        if field not in doc or not str(doc[field]).strip():
            errors.append(f"doc #{idx}: missing/empty required field '{field}'")
    if doc.get("category") and doc["category"] not in VALID_CATEGORIES:
        errors.append(f"doc #{idx} ({doc.get('doc_id')}): invalid category '{doc['category']}'")
    if doc.get("difficulty") and doc["difficulty"] not in VALID_DIFFICULTIES:
        errors.append(f"doc #{idx} ({doc.get('doc_id')}): invalid difficulty '{doc['difficulty']}'")
    return errors


def _cosine_sim_matrix(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    normed = embeddings / norms
    return normed @ normed.T


def _dedup_by_embedding(
    docs: list[dict],
    embedder,
    threshold: float = DEDUP_THRESHOLD,
) -> tuple[list[dict], list[str]]:
    log.info("Computing embeddings for deduplication sweep ...")
    questions = [d["question"] for d in docs]
    embeddings = embedder.encode(questions, batch_size=64, show_progress_bar=False)
    embeddings = embeddings.astype(np.float32)

    sim_matrix = _cosine_sim_matrix(embeddings)
    removed_ids: list[str] = []
    keep = [True] * len(docs)

    for i in range(len(docs)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(docs)):
            if keep[j] and sim_matrix[i, j] > threshold:
                log.warning(
                    f"Near-duplicate (sim={sim_matrix[i,j]:.4f}): "
                    f"{docs[i]['doc_id']} ≈ {docs[j]['doc_id']} — removing {docs[j]['doc_id']}"
                )
                removed_ids.append(docs[j]["doc_id"])
                keep[j] = False

    clean = [d for d, k in zip(docs, keep) if k]
    log.info(f"Dedup complete: {len(docs)} → {len(clean)} docs ({len(removed_ids)} removed)")
    return clean, removed_ids


def build_corpus(
    source_path: str,
    config_path: str = "retrieval/configs/retrieval_config.yaml",
    run_dedup: bool = True,
) -> dict:
    cfg = _load_config(config_path)
    model_name: str = cfg.get("embedding_model", {}).get(
        "local_path"
    ) or cfg.get("embedding_model", {}).get("hub_id", "hassan7272/urdu-finance-embeddings")

    log.info(f"Loading source data from {source_path}")
    raw = _load_json_or_jsonl(Path(source_path))
    log.info(f"Loaded {len(raw)} records")

    docs: list[dict] = []
    schema_errors: list[str] = []
    seen_source_ids: set[str] = set()
    seen_doc_ids: set[str] = set()

    for i, rec in enumerate(raw):
        source_id = str(rec.get("id", i))
        doc_id = _make_doc_id(source_id)

        if source_id in seen_source_ids:
            schema_errors.append(f"Duplicate source_id '{source_id}' at record #{i} — skipped")
            continue
        seen_source_ids.add(source_id)

        if doc_id in seen_doc_ids:
            schema_errors.append(f"doc_id collision '{doc_id}' at record #{i} — skipped")
            continue
        seen_doc_ids.add(doc_id)

        doc: dict = {
            "doc_id": doc_id,
            "question": rec.get("question_ur", rec.get("question", "")).strip(),
            "answer": rec.get("answer_ur", rec.get("answer", "")).strip(),
            "category": rec.get("category", ""),
            "difficulty": rec.get("difficulty", ""),
            "source_id": source_id,
        }
        for opt in ("question_en", "answer_en", "keywords"):
            if rec.get(opt):
                doc[opt] = rec[opt]

        errors = _validate_schema(doc, i)
        schema_errors.extend(errors)
        if not errors:
            docs.append(doc)

    if schema_errors:
        for e in schema_errors[:20]:
            log.warning(e)
        if len(schema_errors) > 20:
            log.warning(f"... and {len(schema_errors) - 20} more schema errors")

    log.info(f"Valid docs after schema validation: {len(docs)}")

    removed_ids: list[str] = []
    if run_dedup:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(model_name)
        docs, removed_ids = _dedup_by_embedding(docs, embedder)

    source_id_set = {d["source_id"] for d in docs}
    collisions = len(raw) - len(source_id_set) - len(removed_ids) - len(schema_errors)
    assert len({d["doc_id"] for d in docs}) == len(docs), "doc_id uniqueness violated after dedup"

    from collections import Counter
    cat_counts = dict(Counter(d["category"] for d in docs))
    diff_counts = dict(Counter(d["difficulty"] for d in docs))
    q_lengths = [len(d["question"].split()) for d in docs]
    a_lengths = [len(d["answer"].split()) for d in docs]

    stats = {
        "total_source_records": len(raw),
        "after_schema_validation": len(docs) + len(removed_ids),
        "after_dedup": len(docs),
        "removed_near_duplicates": len(removed_ids),
        "removed_doc_ids": removed_ids,
        "schema_errors": len(schema_errors),
        "category_counts": cat_counts,
        "difficulty_counts": diff_counts,
        "avg_question_length_words": round(float(np.mean(q_lengths)), 2) if q_lengths else 0,
        "avg_answer_length_words": round(float(np.mean(a_lengths)), 2) if a_lengths else 0,
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
        "source_path": source_path,
        "embedder_model": model_name,
    }

    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    corpus_path = ARTIFACTS / "corpus.jsonl"
    with open(corpus_path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    log.info(f"Saved corpus → {corpus_path}")

    docstore = {d["doc_id"]: d for d in docs}
    docstore_path = ARTIFACTS / "docstore.json"
    with open(docstore_path, "w", encoding="utf-8") as f:
        json.dump(docstore, f, ensure_ascii=False, indent=2)
    log.info(f"Saved docstore → {docstore_path}")

    stats_path = ARTIFACTS / "corpus_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    log.info(f"Saved corpus stats → {stats_path}")

    log.info(f"Build complete: {len(docs)} documents in corpus")
    return stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/raw/urdu_finance_qa.jsonl")
    parser.add_argument("--config", default="retrieval/configs/retrieval_config.yaml")
    parser.add_argument("--no-dedup", action="store_true")
    args = parser.parse_args()
    build_corpus(source_path=args.source, config_path=args.config, run_dedup=not args.no_dedup)