"""
train_embedding_kaggle.py — Phase 1 Training Entry Point
=========================================================
Designed for Kaggle GPU environment (P100 / T4 / 2x T4).

What this does:
  1. Loads config from embedding/embedding_train.yaml
  2. Loads train.jsonl + val.jsonl
  3. Builds SentenceTransformer training examples
  4. Trains with MultipleNegativesRankingLoss + hard negatives
  5. Evaluates on val split (cosine accuracy + IR metrics)
  6. Saves best model checkpoint

Run on Kaggle:
    !python training/train_embedding_kaggle.py
    # or with custom config:
    !python training/train_embedding_kaggle.py --config embedding/embedding_train.yaml

Expected outputs:
    models/finguard-embedding-v1/         ← final best model
    models/finguard-embedding-v1-final/   ← sentence-transformer format
    training_log.json                     ← evaluation history
"""

import os
import sys
import json
import math
import logging
import argparse
import time
from pathlib import Path
from typing import Optional

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    util,
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator

# ── paths (works on both Kaggle and local) ────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training.log"),
    ],
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def log_section(title: str) -> None:
    log.info("─" * 60)
    log.info(f"  {title}")
    log.info("─" * 60)


def pick_runtime_device() -> str:
    """
    Pick a safe runtime device for this environment.
    Kaggle sometimes provides P100 (sm_60), while torch wheels may only support sm_70+.
    """
    if not torch.cuda.is_available():
        log.info("CUDA not available. Using CPU.")
        return "cpu"

    major, minor = torch.cuda.get_device_capability(0)
    gpu_name = torch.cuda.get_device_name(0)
    log.info(f"Detected GPU: {gpu_name} (sm_{major}{minor})")

    # Current Kaggle torch wheels in some images require >= sm_70.
    if major < 7:
        log.warning(
            "GPU compute capability sm_%s%s is not supported by current torch build. Falling back to CPU.",
            major,
            minor,
        )
        return "cpu"

    return "cuda"


def normalize_scheduler_name(name: str) -> str:
    """
    Map config-friendly scheduler aliases to sentence-transformers names.
    """
    if not name:
        return "WarmupCosine"

    raw = str(name).strip()
    key = raw.lower().replace("-", "").replace("_", "")
    alias_map = {
        "cosine": "WarmupCosine",
        "warmupcosine": "WarmupCosine",
        "linear": "WarmupLinear",
        "warmuplinear": "WarmupLinear",
        "constant": "WarmupConstant",
        "warmupconstant": "WarmupConstant",
        "cosinewithhardrestarts": "WarmupCosineWithHardRestarts",
        "warmupcosinewithhardrestarts": "WarmupCosineWithHardRestarts",
    }
    return alias_map.get(key, raw)


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING EXAMPLE BUILDERS
# ═══════════════════════════════════════════════════════════════════════════

def build_mnr_examples(
    records: list[dict],
    query_col: str,
    positive_col: str,
    hard_neg_col: Optional[str],
    include_english: bool = True,
    english_weight: float = 0.3,
) -> list[InputExample]:
    """
    Build InputExample list for MultipleNegativesRankingLoss.

    MNR Loss expects: InputExample(texts=[anchor, positive])
    When hard negatives are present, we ALSO add:
             InputExample(texts=[anchor, hard_negative])
    with a label of 0 — handled via TripletLoss mode.

    Strategy:
      Primary  : (query_ur, answer_ur)       — native Urdu / Roman Urdu
      Secondary: (query_en, answer_en)        — English (if include_english=True)
      Hard neg : (query_ur, hard_negative)    — confusable wrong answer
    """
    examples = []

    for r in records:
        query    = r.get(query_col, "").strip()
        positive = r.get(positive_col, "").strip()
        hard_neg = r.get(hard_neg_col, "").strip() if hard_neg_col else ""

        if not query or not positive:
            continue

        # Primary Urdu pair
        examples.append(InputExample(texts=[query, positive]))

        # Hard negative as explicit negative candidate for MNR.
        # Order for 3 texts must be [anchor, positive, hard_negative].
        if hard_neg and hard_neg != positive:
            examples.append(InputExample(texts=[query, positive, hard_neg]))

        # English pair (cross-lingual boost)
        if include_english:
            q_en = r.get("query_en", "").strip()
            p_en = r.get("positive_en", "").strip()
            if q_en and p_en and np.random.random() < english_weight:
                examples.append(InputExample(texts=[q_en, p_en]))

    log.info(f"Built {len(examples):,} training examples")
    return examples


# ═══════════════════════════════════════════════════════════════════════════
# EVALUATORS
# ═══════════════════════════════════════════════════════════════════════════

def build_ir_evaluator(
    records: list[dict],
    query_col: str,
    positive_col: str,
    name: str = "val",
) -> InformationRetrievalEvaluator:
    """
    InformationRetrievalEvaluator:
      - Encodes all answers as corpus
      - For each query, retrieves top-K
      - Reports Accuracy@1, Accuracy@3, MRR@10
    """
    queries   = {}   # qid → query text
    corpus    = {}   # did → answer text
    relevant  = {}   # qid → set of relevant did

    for i, r in enumerate(records):
        qid = str(r.get("id", i))
        did = f"doc_{qid}"
        q   = r.get(query_col, "").strip()
        pos = r.get(positive_col, "").strip()
        if q and pos:
            queries[qid]  = q
            corpus[did]   = pos
            relevant[qid] = {did}

    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant,
        name=name,
        show_progress_bar=False,
        precision_recall_at_k=[1, 3, 10],
        mrr_at_k=[10],
        ndcg_at_k=[10],
        accuracy_at_k=[1, 3, 10],
    )


def compute_val_retrieval_metrics(
    model: SentenceTransformer,
    records: list[dict],
    query_col: str,
    positive_col: str,
) -> dict:
    """Compute retrieval metrics on val set with in-memory corpus."""
    pairs = []
    for i, r in enumerate(records):
        q = r.get(query_col, "").strip()
        p = r.get(positive_col, "").strip()
        if q and p:
            pairs.append((str(r.get("id", i)), q, p))

    if not pairs:
        return {"count": 0, "acc@1": 0.0, "acc@3": 0.0, "acc@10": 0.0, "mrr@10": 0.0}

    qids = [p[0] for p in pairs]
    queries = [p[1] for p in pairs]
    corpus = [p[2] for p in pairs]
    gold_index = {qid: idx for idx, qid in enumerate(qids)}

    q_emb = model.encode(queries, convert_to_tensor=True, show_progress_bar=False)
    c_emb = model.encode(corpus, convert_to_tensor=True, show_progress_bar=False)
    scores = util.cos_sim(q_emb, c_emb).cpu().numpy()

    hit1 = hit3 = hit10 = 0
    rr_sum = 0.0
    for i, qid in enumerate(qids):
        target = gold_index[qid]
        ranking = np.argsort(-scores[i])  # descending
        top10 = ranking[:10]

        if target in ranking[:1]:
            hit1 += 1
        if target in ranking[:3]:
            hit3 += 1
        if target in top10:
            hit10 += 1
            rank_pos = int(np.where(top10 == target)[0][0]) + 1
            rr_sum += 1.0 / rank_pos

    n = len(qids)
    metrics = {
        "count": n,
        "acc@1": round(hit1 / n, 4),
        "acc@3": round(hit3 / n, 4),
        "acc@10": round(hit10 / n, 4),
        "mrr@10": round(rr_sum / n, 4),
    }
    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# MAIN TRAINING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def train(config_path: str = "embedding/embedding_train.yaml") -> None:
    start_time = time.time()

    # ── Load config ──────────────────────────────────────────────
    log_section("Loading Config")
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = ROOT / config_file
    cfg = load_config(config_file)
    model_cfg    = cfg["model"]
    train_cfg    = cfg["training"]
    data_cfg     = cfg["data"]
    eval_cfg     = cfg["evaluation"]
    output_cfg   = cfg["output"]

    log.info(f"Base model   : {model_cfg['base_model']}")
    log.info(f"Max seq len  : {model_cfg['max_seq_length']}")
    log.info(f"Epochs       : {train_cfg['epochs']}")
    log.info(f"Batch size   : {train_cfg['batch_size']}")
    log.info(f"Grad accum   : {train_cfg['gradient_accumulation_steps']}")
    log.info(f"LR           : {train_cfg['learning_rate']}")

    # ── Load data ────────────────────────────────────────────────
    log_section("Loading Data")
    train_records = load_jsonl(ROOT / data_cfg["train_file"])
    val_records   = load_jsonl(ROOT / data_cfg["val_file"])
    log.info(f"Train records: {len(train_records):,}")
    log.info(f"Val records  : {len(val_records):,}")

    # ── Build model ──────────────────────────────────────────────
    log_section("Initializing Model")
    runtime_device = pick_runtime_device()
    model = SentenceTransformer(model_cfg["base_model"], device=runtime_device)
    model.max_seq_length = model_cfg["max_seq_length"]
    log.info(f"Model loaded: {model_cfg['base_model']} on {runtime_device}")

    # ── Build examples ───────────────────────────────────────────
    log_section("Building Training Examples")
    use_hard_neg = cfg["loss"].get("use_hard_negatives", True)
    train_examples = build_mnr_examples(
        records         = train_records,
        query_col       = data_cfg["query_col"],
        positive_col    = data_cfg["positive_col"],
        hard_neg_col    = data_cfg["hard_negative_col"] if use_hard_neg else None,
        include_english = data_cfg.get("include_english_pairs", True),
        english_weight  = data_cfg.get("english_pair_weight", 0.3),
    )

    # DataLoader — shuffle for MNR (in-batch diversity matters)
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg.get("dataloader_num_workers", 2),
        drop_last=True,   # MNR needs uniform batch sizes
    )

    # ── Loss function ────────────────────────────────────────────
    log_section("Configuring Loss")
    loss_fn = losses.MultipleNegativesRankingLoss(
        model=model,
        scale=cfg["loss"].get("scale", 20.0),
        similarity_fct=util.cos_sim,
    )
    log.info("Loss: MultipleNegativesRankingLoss (cosine similarity, scale=20)")

    # ── Evaluator ────────────────────────────────────────────────
    log_section("Building Evaluator")
    evaluator = build_ir_evaluator(
        records      = val_records,
        query_col    = data_cfg["query_col"],
        positive_col = data_cfg["positive_col"],
        name         = "val_ir",
    )
    log.info("Running baseline val retrieval before training …")
    baseline_metrics = compute_val_retrieval_metrics(
        model=model,
        records=val_records,
        query_col=data_cfg["query_col"],
        positive_col=data_cfg["positive_col"],
    )
    log.info(f"Baseline val metrics: {baseline_metrics}")

    # ── Warmup + scheduler ───────────────────────────────────────
    total_steps  = math.ceil(len(train_dataloader) / train_cfg["gradient_accumulation_steps"]) * train_cfg["epochs"]
    warmup_steps = int(total_steps * train_cfg["warmup_ratio"])
    log.info(f"Total steps  : {total_steps:,}")
    log.info(f"Warmup steps : {warmup_steps:,}")
    log.info(f"Eval every   : {eval_cfg['eval_steps']} steps")

    # ── Output path ──────────────────────────────────────────────
    save_path = ROOT / output_cfg["save_path"]
    save_path.mkdir(parents=True, exist_ok=True)

    # ── TRAIN ────────────────────────────────────────────────────
    log_section("Training Started")
    scheduler_name = normalize_scheduler_name(train_cfg.get("lr_scheduler", "cosine"))
    log.info(f"Scheduler    : {scheduler_name}")
    use_amp = train_cfg.get("fp16", True) and runtime_device == "cuda"
    if train_cfg.get("fp16", True) and runtime_device != "cuda":
        log.warning("fp16 requested but runtime device is CPU. Disabling AMP for this run.")

    model.fit(
        train_objectives     = [(train_dataloader, loss_fn)],
        evaluator            = evaluator,
        epochs               = train_cfg["epochs"],
        warmup_steps         = warmup_steps,
        evaluation_steps     = eval_cfg["eval_steps"],
        output_path          = str(save_path),
        save_best_model      = True,
        show_progress_bar    = True,
        optimizer_params     = {"lr": train_cfg["learning_rate"]},
        weight_decay         = train_cfg.get("weight_decay", 0.01),
        scheduler            = scheduler_name,
        use_amp              = use_amp,  # mixed precision only on supported CUDA
        checkpoint_path      = str(save_path / "checkpoints"),
        checkpoint_save_steps= output_cfg.get("checkpoint_steps", 200),
        checkpoint_save_total_limit = output_cfg.get("save_total_limit", 2),
    )

    # ── Final evaluation ─────────────────────────────────────────
    log_section("Final Evaluation (Best Checkpoint)")
    best_model = SentenceTransformer(str(save_path))
    final_score = evaluator(best_model)
    log.info(f"Final IR eval score: {final_score:.4f}")
    final_metrics = compute_val_retrieval_metrics(
        model=best_model,
        records=val_records,
        query_col=data_cfg["query_col"],
        positive_col=data_cfg["positive_col"],
    )
    log.info(f"Final val metrics: {final_metrics}")

    # ── Save in sentence-transformers format ──────────────────────
    final_path = ROOT / output_cfg.get("final_model_name", "finguard-embedding-final")
    best_model.save(str(final_path))
    log.info(f"Final model saved → {final_path}")

    # ── Save training log ─────────────────────────────────────────
    elapsed = time.time() - start_time
    training_log = {
        "base_model":    model_cfg["base_model"],
        "train_size":    len(train_records),
        "val_size":      len(val_records),
        "examples":      len(train_examples),
        "epochs":        train_cfg["epochs"],
        "batch_size":    train_cfg["batch_size"],
        "final_score":   round(float(final_score), 4),
        "val_metrics_before": baseline_metrics,
        "val_metrics_after": final_metrics,
        "elapsed_sec":   round(elapsed, 1),
        "save_path":     str(save_path),
    }
    log_path = ROOT / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)
    log.info(f"Training log → {log_path}")

    log.info("=" * 60)
    log.info(f"✅ Training complete in {elapsed/60:.1f} min")
    log.info(f"   Best model : {save_path}")
    log.info(f"   Final model: {final_path}")
    log.info("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finguard Embedding Training")
    parser.add_argument(
        "--config",
        type=str,
        default="embedding/embedding_train.yaml",
        help="Path to training config YAML",
    )
    args = parser.parse_args()

    # Kaggle environment detection
    if Path("/kaggle/input").exists():
        log.info("Running in Kaggle environment")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    train(config_path=args.config)