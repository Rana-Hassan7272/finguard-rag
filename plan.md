# FinGuard RAG — Production Architecture Plan v3
> Phases 1–5 locked and complete. Phase 6 onward reflects updated stack.

---

## PHASE 1 — Embedding Model ✅ DONE

- Fine-tuned `paraphrase-multilingual-mpnet-base-v2` on 1,510 Urdu/Roman Urdu/English finance QA pairs
- Multiple Negatives Ranking Loss
- Domain-aware hard negative mining (confusable category map)
- Bilingual training pairs (Urdu primary + English 30%)
- Results: Accuracy@1 = 93.9%, MRR@10 = 0.966, trained in 7 min on Kaggle GPU
- Model published: `hassan7272/urdu-finance-embeddings`

---

## PHASE 2 — Vector Retrieval ✅ DONE

- FAISS IndexFlatIP with L2-normalized embeddings
- Top-10 retrieval
- Index versioning with corpus manifest
- Startup safety check: embedder version mismatch → refuse search

---

## PHASE 3 — Hybrid Search ✅ DONE

- BM25 (`rank_bm25`) + FAISS vector search
- Reciprocal Rank Fusion (RRF)
- Language-aware weights at runtime:
  - Roman Urdu: BM25=0.3, Vector=0.7
  - Urdu script: BM25=0.4, Vector=0.6
  - English: BM25=0.6, Vector=0.4
- MMR (Max Marginal Relevance) for diversity, lambda=0.7

---

## PHASE 4 — Reranking ✅ DONE

- Cross-encoder reranker: `BAAI/bge-reranker-base`
- Top-10 → reranker → Top-3
- Confidence gate: hard threshold 0.40
  - Below threshold → language-aware fallback message, LLM skipped
  - Gate reasons: `no_docs`, `score_below_threshold`, `empty_doc_text`
- Threshold sweep tool: precision vs false-reject tradeoff curve

---

## PHASE 5 — Metadata Filtering ✅ DONE

- Runtime category detection from query (keyword mapping, rule-based)
- Filter policy: apply pre-filter only if confidence ≥ 0.65
- Pre-filter reduces FAISS search space before retrieval
- Evaluation: with/without filter delta on acc@1, acc@3, MRR, latency
- Diagnostics: category detected, confidence, filter applied, reason
 CHUNKING

### ✅ Best Approach (Unchanged)
- Each QA pair = one document
- Store as: `{"question": ..., "answer": ..., "category": ..., "difficulty": ...}`
- Embed the **question** field for retrieval, return full QA pair as context

### ❌ Avoid
- Splitting answers randomly
- Token-based chunking
- Embedding question+answer together (dilutes signal)

---

---

# FinGuard RAG — Architecture Review & Revised Plan (Phase 6 Onward)
> Phases 1–5 locked. This document covers Phase 6+ only.
---

## PHASE 6 — PDF Corpus Integration

### Stack: LlamaIndex (ingestion only) + raw Python (indexing)

LlamaIndex handles PDF parsing, layout extraction, chunking, and metadata.
Its output feeds your existing FAISS + BM25 pipeline unchanged.
Nothing downstream of ingestion touches LlamaIndex.

### Why LlamaIndex here

- 40 PDFs × 60 pages average = ~2,400 pages → 6,000–9,000 chunks
- Pakistani financial PDFs have tables, mixed Urdu/English headers, footnotes, multi-column layouts
- `pymupdf` raw extraction requires 300–400 lines of brittle parser code you maintain per layout
- LlamaIndex `SimpleDirectoryReader` + `SentenceSplitter` handles all edge cases in ~20 lines

### Chunking Strategy

```
chunk_size    : 512 tokens
chunk_overlap : 100 tokens
section title : prepended to every chunk from that section
```

Section title prepending is the most important technique:
```
chunk_text = f"{section_title}\n\n{raw_chunk_text}"
```
This ensures the embedding captures topic context, not just a mid-paragraph fragment.

### PDF Document Schema

```json
{
  "doc_id":         "pdf_<8char_hash>",
  "doc_type":       "pdf_chunk",
  "retrieval_text": "Section Title\n\nchunk body text...",
  "chunk_text":     "chunk body text...",
  "source_file":    "meezan_bank_guide.pdf",
  "page_no":        42,
  "chunk_id":       4,
  "parent_section": "Zakat Deduction Rules",
  "category":       "islamic_finance",
  "difficulty":     null
}
```

### Dual-Index Architecture

QA and PDF corpora are indexed separately. Query runs against both in parallel.
Results merged via RRF before MMR. This is critical — unified indexing breaks BM25 scoring
(PDF chunks have more tokens → artificially higher TF scores over QA pairs).

```
Query
  ├── QA FAISS + BM25   → top-10 QA results
  └── PDF FAISS + BM25  → top-10 PDF chunk results
          ↓
    RRF merge with source routing weights
          ↓
    MMR over 20 candidates → top-10
          ↓
    Cross-encoder reranker → top-3
```

### Source Routing Weights (in `retrieval_config.yaml`)

```yaml
source_routing:
  practical_query:  { qa: 0.7, pdf: 0.3 }
  legal_query:      { qa: 0.3, pdf: 0.7 }
  default:          { qa: 0.5, pdf: 0.5 }
```

Legal query detection: keywords like `law`, `act`, `regulation`, `section`,
`SBP`, `SECP`, `ordinance`, `clause` → legal routing. Otherwise practical.

### Files to Build

```
ingestion/
  pdf_chunker.py           — LlamaIndex load + chunk + section title prepend
  pdf_corpus_builder.py    — convert LlamaIndex nodes to corpus schema, save JSONL
  infer_category.py        — keyword mapping to assign category to PDF chunks

retrieval/
  build_pdf_vector_index.py   — FAISS index for PDF chunks (separate from QA index)
  build_pdf_bm25_index.py     — BM25 index for PDF chunks
  dual_retriever.py           — parallel QA + PDF retrieval, RRF merge, source routing
```

### Skip

Hierarchical retrieval — adds two-stage complexity for no gain at this corpus size.
MMR already handles diversity. Revisit if corpus exceeds 50k chunks.

---

## PHASE 7 — Query Processing

### Query Normalization

- Lowercase Latin text
- Strip extra whitespace and punctuation noise
- Roman Urdu: collapse repeated chars (`kyaaaa` → `kyaa`)
- Urdu: fix Unicode character variants

### Query Routing (new — lives here, not in Phase 6)

Before retrieval, classify query intent:

```
normalize_query()
  → detect_language()
  → detect_category()        (Phase 5, already built)
  → detect_query_intent()    (practical vs legal)
  → source_routing_weights() (returns qa_weight, pdf_weight)
  → dual retriever
```

### Query Expansion (optional, add last)

Expand short single-term queries that underperform:

```
"zakat"   → "zakat calculation Pakistan nisab 2.5%"
"riba"    → "riba interest Islamic finance prohibition"
```

Use a keyword dictionary, not LLM — deterministic, zero latency, auditable.
Only add this after seeing systematic failures on short queries in your eval logs.

### Files to Build

```
query/
  router.py     — intent detection + source routing weights
  expander.py   — keyword expansion dictionary (add after eval, not before)
```

---

## PHASE 8 — Semantic Query Cache

### Why This Matters

60–70% of real-world finance queries are near-duplicates.
"zakat ka hisab", "zakat calculation", "how to calculate zakat" all need the same answer.
Running the full pipeline (embed → BM25 → FAISS → reranker → LLM) costs 200–800ms and LLM API fees each time.

### Architecture: Two Cache Levels

```
Incoming query
      ↓
Embed query (always — fast, ~5ms)
      ↓
Level 1: Answer cache
  cosine_sim(query_embed, cached_query_embeds) > 0.95?
      ↓ HIT → return cached final answer ⚡ (~5ms total)
      ↓ MISS
Level 2: Retrieval cache
  cosine_sim > 0.90?
      ↓ HIT → return cached doc_ids, skip to reranker ⚡ (~130ms total)
      ↓ MISS
Full pipeline → store in both caches
```

### Cache Invalidation (critical — plan v2 missed this)

Every cache entry stores `corpus_version` — a hash of the corpus manifest.
When corpus is rebuilt (new PDFs added), compute new hash → all old entries invalid automatically.
No manual cache clearing needed.

```python
entry = {
    "query_embedding": [...],
    "answer": "...",
    "doc_ids": [...],
    "corpus_version": "sha256_first8",
    "timestamp": "ISO-8601",
    "ttl_hours": 48,
}
```

### Implementation

- In-memory Python dict + numpy cosine sim — handles hundreds of QPS, no infra needed
- Move to Redis only at 1,000+ QPS
- TTL: 48 hours (corpus is mostly static)

### Files to Build

```
cache/
  semantic_cache.py   — two-level cache, cosine sim lookup, TTL, corpus version check
  cache_stats.py      — hit rate, miss rate, eviction count (feeds observability)
```

---

## PHASE 9 — LLM Generation

### Two Prompt Templates (not one)

Template selection is based on `doc_type` of top-1 retrieved document.

**QA-sourced answer:**
```
You are a helpful financial assistant for Pakistani users.
Answer ONLY using the provided context.
If context is insufficient, say: "I don't have enough information to answer this."
Keep answer to 2–3 sentences maximum.
Respond in the same language as the user's question.

CONTEXT:
{retrieved_docs}

QUESTION: {user_query}

ANSWER:
```

**PDF/policy-sourced answer:**
```
You are a helpful financial assistant for Pakistani users.
Answer ONLY using the provided context, which is drawn from official documents.
Cite the source document name briefly (e.g. "According to the Banking Act...").
If context is insufficient, say: "I don't have enough information to answer this."
Keep answer to 2–3 sentences maximum.
Respond in the same language as the user's question.

CONTEXT:
{retrieved_docs}

QUESTION: {user_query}

ANSWER:
```

### LLM Client

- Primary: Groq (Llama-3 70B or Mixtral) — fast inference, free tier, good for demo
- Fallback: OpenAI GPT-4o-mini — higher quality, costs money
- Config-driven model selection
- Retry with exponential backoff on timeout
- Hard timeout: 10 seconds — if exceeded, return confidence gate fallback message

### Files to Build

```
generation/
  prompt_builder.py   — selects template based on doc_type, formats context
  llm_client.py       — API wrapper, retry logic, timeout, model selection
  generator.py        — orchestrates gate check → prompt → LLM → response
```

---

## PHASE 10 — Observability & Logging

### Structured JSONL Log Per Request

```json
{
  "timestamp": "2025-01-15T10:23:45Z",
  "query": "zakat ka nisab kya hai",
  "query_language": "roman_urdu",
  "query_intent": "practical",
  "category_detected": "islamic_finance",
  "category_confidence": 0.87,
  "filter_applied": true,
  "cache_level": "miss",
  "corpus_version": "abc12345",
  "retrieval_ms": 45,
  "reranker_ms": 120,
  "llm_ms": 800,
  "total_ms": 965,
  "top_reranker_score": 0.87,
  "gate_passed": true,
  "gate_reason": "passed",
  "source_mix": { "qa": 2, "pdf": 1 },
  "retrieved_doc_ids": ["doc_042", "pdf_a3f9c1", "doc_107"],
  "answer_generated": true
}
```

`source_mix` is the most important new field — tells you if dual-index routing is working.
If it's always `{"qa": 3, "pdf": 0}`, your PDF index isn't contributing.

### Files to Build

```
observability/
  logger.py     — writes JSONL log entry per request, handles I/O errors gracefully
  analyzer.py   — pandas analysis: hit rates, P50/P95 latency, failure query patterns
```

---

## PHASE 11 — Evaluation

### Ablation Table

| System Configuration | Acc@1 | Acc@3 | MRR | P95 Latency |
|---|---|---|---|---|
| Baseline BM25 only | — | — | — | — |
| + Fine-tuned embeddings | — | — | — | — |
| + Hybrid search | — | — | — | — |
| + Reranker | — | — | — | — |
| + Metadata filter | — | — | — | — |
| + PDF corpus dual-index | — | — | — | — |
| + Semantic cache | — | — | — | — |
| **Full system** | 🔥 | 🔥 | 🔥 | 🔥 |

### Evaluation Splits

- QA-only queries (test set from Phase 1)
- PDF-grounded policy queries (manually curated 30 queries from PDF content)
- Mixed queries requiring both QA + PDF context

### Source Grounding Metrics

- % answers QA-grounded vs PDF-grounded
- % answers where confidence gate passed
- % answers where reranker score > 0.70 (high confidence)

### Files to Build

```
evaluation/
  full_eval.py           — complete ablation table across all system configs
  source_eval.py         — QA vs PDF grounding rates, per-category breakdown
  latency_profiler.py    — P50/P95 per stage across N queries
```

---

## PHASE 12 — Gradio UI (HuggingFace Space)

### UI Components

- Query input text box
- Language badge: `Roman Urdu` / `Urdu` / `English`
- Intent badge: `Practical` / `Legal/Policy`
- Cache indicator: `⚡ Cached` / `🔍 Retrieved`
- Category detected chip
- Retrieved docs panel — top-3 with:
  - Source badge: `[QA]` or `[PDF — filename.pdf p.42]`
  - Reranker score
  - Snippet of doc text
- Gate status: passed score or rejection reason + score
- Final answer
- Total latency in ms

### Files to Build

```
app.py    — single Gradio file, deployed to HuggingFace Spaces
```

---

## Final Architecture Flow

```
User Query
    ↓
Query Normalization
    ↓
Language Detection (ur / roman_ur / en)
    ↓
Semantic Cache L1 ────── HIT ──→ Return answer ⚡ 5ms
    ↓ MISS
Semantic Cache L2 ────── HIT ──→ Skip to reranker ⚡ 130ms
    ↓ MISS
Category Detection → Metadata Pre-filter
    ↓
Query Intent Detection → Source Routing Weights
    ↓
Parallel Retrieval:
  ├── QA FAISS + BM25 (language-weighted)
  └── PDF FAISS + BM25 (language-weighted)
    ↓
RRF Fusion (source-routing-weighted)
    ↓
Top-20 candidates → MMR → Top-10
    ↓
Cross-Encoder Reranker (BAAI/bge-reranker-base) → Top-3
    ↓
Confidence Gate (threshold 0.40)
  └── FAIL → language-aware fallback, skip LLM
    ↓ PASS
Prompt Builder (QA template vs PDF/policy template)
    ↓
LLM Generation (Groq primary / OpenAI fallback)
    ↓
Observability Logger (JSONL)
    ↓
Cache Store (L1 + L2)
    ↓
Return Answer + Diagnostics
```

---

## Tech Stack Summary

| Layer | Technology | Role |
|---|---|---|
| Embedding model | `sentence-transformers` (custom fine-tuned) | Encode queries + docs |
| PDF ingestion | `LlamaIndex` (SimpleDirectoryReader + SentenceSplitter) | Parse + chunk PDFs |
| Vector search | `FAISS` (IndexFlatIP) | Dense retrieval |
| Keyword search | `rank_bm25` | Sparse retrieval |
| Fusion | Custom RRF | Merge ranked lists |
| Diversity | Custom MMR | Remove duplicate results |
| Reranker | `sentence-transformers` CrossEncoder (`BAAI/bge-reranker-base`) | Score query-doc pairs |
| Cache | Custom in-memory (numpy cosine sim) | Two-level semantic cache |
| LLM | Groq (Llama-3 70B) / OpenAI fallback | Answer synthesis |
| UI | Gradio | HuggingFace Space demo |
| Logging | Custom JSONL | Observability |
| Config | YAML | All thresholds, weights, paths |
| Evaluation | Custom Python + pandas | Ablation table, latency profiling |

---

## What to Skip (Final List)

| Feature | Reason |
|---|---|
| GraphRAG | Your PDFs are procedural, not entity-graph-heavy. Add in v2 for legal multi-hop only. |
| LangChain | No benefit over raw Python for your custom components. Adds dependency weight. |
| LlamaIndex retrieval/query engine | Only use LlamaIndex for PDF ingestion. Your retrieval is better. |
| Self-RAG / Corrective RAG | Confidence gate does the same job. 3x latency cost not worth it. |
| Redis cache | In-memory handles your load. Redis only at 1,000+ QPS. |
| Multi-hop retrieval | Your queries are single-turn. Not needed. |
| Distillation | 93.9% Acc@1 — no performance problem to solve. |
| Sentence-level PDF chunking | Too granular, destroys financial explanation coherence. |
| Unified QA+PDF index | Breaks BM25 scoring. Dual index is mandatory. |

---

