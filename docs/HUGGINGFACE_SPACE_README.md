---
title: FinGuard RAG
emoji: 🏦
colorFrom: blue
colorTo: slate
sdk: gradio
app_file: app.py
pinned: false
license: apache-2.0
---

# FinGuard — Pakistani Finance Assistant

Ask questions about **Islamic finance**, **banking**, **digital payments**, **tax**, **investments**, and related topics — in **Urdu**, **Roman Urdu**, or **English**.

This Space runs a full **retrieval-augmented generation (RAG)** stack: hybrid search (dense + BM25), cross-encoder reranking, confidence gating, semantic caching, and an LLM for grounded answers when the model responds successfully. If the LLM call fails, the app can fall back to the **top retrieved QA answer** so you still see useful context.

---

## What you can try

- Zakat / Islamic banking basics  
- Easypaisa, JazzCash, bank accounts  
- Freelancing and tax (FBR / NTN — informational only)  
- Investments and savings concepts  

---

## Important disclaimers

- **Not legal, tax, or investment advice.** Outputs are for general education only. Verify anything critical with qualified professionals and official sources (e.g. FBR, SBP, SECP).  
- **Accuracy:** Responses depend on retrieved snippets and the LLM; errors or outdated details are possible.  
- **Language:** Mixed Urdu/Roman Urdu detection may occasionally mis-label script; answers still aim to match your question language.

---

## Tech stack (high level)

| Layer | Choice |
|--------|--------|
| Embeddings | [`hassan7272/urdu-finance-embeddings`](https://huggingface.co/hassan7272/urdu-finance-embeddings) |
| Vector search | FAISS (inner product / cosine on normalized vectors) |
| Keyword search | BM25 |
| Fusion | Weighted reciprocal rank fusion (language-aware weights) |
| Reranker | [`BAAI/bge-reranker-base`](https://huggingface.co/BAAI/bge-reranker-base) |
| LLM | Groq — Llama 3.3 70B Versatile (primary); optional OpenAI fallback if configured |
| UI | Gradio |

Artifacts (indexes, corpus snapshots) are loaded from a private Hugging Face **Dataset** at startup when they are not bundled in the repo.

---

## For Space maintainers (secrets)

Visitors do **not** supply API keys. Configure these under **Space → Settings → Secrets**:

| Secret | Purpose |
|--------|---------|
| `GROQ_API_KEY` | Primary LLM inference |
| `HF_TOKEN` | Download private artifact dataset + Hub models |
| `HF_ARTIFACTS_DATASET` | Dataset id holding `retrieval/artifacts/…` files (e.g. `your-org/finguard-artifacts`) |
| `OPENAI_API_KEY` | Optional fallback LLM |

Optional: `HF_ARTIFACTS_REVISION` to pin a dataset commit.

---

## Full source code

Development layout, evaluation scripts, and training notes live on GitHub — **replace this line with your repo URL**, e.g. `https://github.com/<you>/finguard-rag`.

---

## Citation / attribution

If you use this demo in research or blogs, cite the embedding model and reranker URLs above and mention FinGuard RAG. Model licenses apply to third-party checkpoints.
