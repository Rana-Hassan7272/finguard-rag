# Phase 1 — Embedding Fine-Tuning (Finguard RAG)

## What This Phase Does

Fine-tunes a multilingual sentence-transformer on the **Urdu Finance QA** dataset
(1 510 records, 8 categories) using **Multiple Negatives Ranking Loss** + **hard negative mining**.

---

## Project Structure

```
finguard-rag/
├── data/
│   ├── raw/
│   │   └── urdu_finance_qa.jsonl          ← original dataset (DO NOT EDIT)
│   └── processed/
│       ├── train_pairs.jsonl              ← cleaned query/positive pairs
│       ├── hard_negatives.jsonl           ← pairs + hard_negative field
│       └── splits/
│           ├── train.jsonl                ← 80% (stratified by category)
│           ├── val.jsonl                  ← 10%
│           └── test.jsonl                 ← 10% (held-out, DO NOT TOUCH)
├── embedding/
│   └── embedding_train.yaml              ← all training hyperparameters
├── data/
│   └── processed/
│       ├── data_prep.py                  ← full preprocessing pipeline
│       └── validate.py                   ← data quality checks
├── training/
│   └── train_embedding_kaggle.py         ← single training entry point
├── requirements.txt
└── README_phase1.md                       ← this file
```

---

## Run Order

### Step 1 — Install Dependencies

```bash
pip install -r requirements.txt
```

On Kaggle, most packages are pre-installed. Only run if something is missing.

---

### Step 2 — Run Data Pipeline

```bash
python data/processed/data_prep.py
```

**What it does:**
1. Loads `data/raw/urdu_finance_qa.jsonl` (auto-detects JSON array vs JSONL)
2. Deduplicates records (removes 1 known duplicate question)
3. Normalizes Urdu / Roman Urdu / English text (spacing, punctuation, char variants)
4. Detects language per record (`ur` / `roman_ur` / `en`)
5. Builds `(query, positive)` pairs — Urdu query + Urdu answer as primary
6. Mines hard negatives: picks semantically confusable answers from adjacent categories
7. Stratified 80/10/10 split by `category`
8. Saves all output files

**Expected output:**
```
data/processed/train_pairs.jsonl         ~1508 records
data/processed/hard_negatives.jsonl      ~1508 records
data/processed/splits/train.jsonl        ~1207 records
data/processed/splits/val.jsonl          ~151  records
data/processed/splits/test.jsonl         ~150  records
```

---

### Step 3 — Validate Data Quality

```bash
python data/processed/validate.py
```

**Checks performed:**
- All required fields present and non-empty
- No duplicate IDs or queries within any split
- No label leakage (positive ≠ hard_negative)
- No train/test query overlap (data leakage check)
- Text lengths above minimum thresholds
- Category/difficulty/language values are valid
- All `hard_negative_id` values reference existing records

**Exit codes:**
- `0` = all checks pass → ready to train
- `1` = critical errors found → fix before training

---

### Step 4 — Train Embedding Model

```bash
# Local
python training/train_embedding_kaggle.py

# Custom config
python training/train_embedding_kaggle.py --config embedding/embedding_train.yaml

# Kaggle (add to notebook cell)
!python training/train_embedding_kaggle.py
```

**What it does:**
1. Loads `embedding/embedding_train.yaml`
2. Loads `splits/train.jsonl` + `splits/val.jsonl`
3. Builds `InputExample` list with:
   - Primary: `(query_ur, answer_ur)` — native Urdu/Roman Urdu
   - Hard negative: `(query_ur, hard_negative)` — explicit in-batch negative
   - English (30% chance): `(query_en, answer_en)` — cross-lingual boost
4. Trains with **MultipleNegativesRankingLoss** (scale=20, cosine similarity)
5. Evaluates every 100 steps using `InformationRetrievalEvaluator`
6. Saves best checkpoint automatically

**Expected outputs:**
```
models/finguard-embedding-v1/            ← best checkpoint (auto-saved)
models/finguard-embedding-v1-final/      ← sentence-transformers format
training.log                             ← full training log
training_log.json                        ← summary metrics
```

---

## Key Design Decisions

### Model Choice
`paraphrase-multilingual-mpnet-base-v2` — handles Urdu, Roman Urdu, and English
natively without any language-specific tokenization tricks.

### Loss Function
**Multiple Negatives Ranking Loss** treats every other `(query, positive)` pair
in the batch as a negative. Adding explicit hard negatives from confusable
categories makes the model discriminate between e.g. *zakat calculation* and
*loan interest calculation* — which a random negative would not force.

### Hard Negative Strategy
Negatives are drawn from **adjacent categories** (e.g., `islamic_finance` →
`loans_credit`, `personal_finance`) with low keyword overlap. This forces the
model to learn semantic intent, not just topic matching.

### Bilingual Training
Roman Urdu + script Urdu + English are all present in the dataset. Training
on all three forms (with English at 30% sampling rate) produces an embedding
space that handles real-world mixed-language user queries.

---

## Evaluation Metrics (reported on val split)

| Metric          | Meaning                                      |
|-----------------|----------------------------------------------|
| Accuracy@1      | Is the correct answer ranked #1?             |
| Accuracy@3      | Is the correct answer in top 3?              |
| Accuracy@10     | Is the correct answer in top 10?             |
| MRR@10          | Mean Reciprocal Rank (quality of ranking)    |

---

## Kaggle Tips

- Enable **GPU** (P100 or T4) — training takes ~15–25 min with GPU, ~3 hrs on CPU
- Enable **Internet** if loading the base model for the first time
- After first run, the model is cached; disable internet for subsequent runs
- Use `fp16: true` in config (already set) — cuts memory usage and speeds up ~2x

---

## What Comes Next (Phase 2)

With the fine-tuned model saved at `models/finguard-embedding-v1-final/`:

1. Build FAISS index over all 1510 answers
2. Implement Top-10 retrieval → MMR → Reranker → Top-3
3. Add BM25 hybrid search for keyword-sensitive Roman Urdu queries
4. Filter by `category` metadata before retrieval# finguard-rag
