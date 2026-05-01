"""
app.py
FinGuard RAG — Gradio UI for HuggingFace Spaces

Single-file deployment. All pipeline components are loaded once at startup
and reused across requests. Handles the full flow:
  normalize → cache → retrieve → rerank → gate → generate → log → display
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import yaml


def _repo_root() -> Path:
    """
    Directory that contains retrieval/, generation/, etc.

    - HF Spaces: app.py lives at repo root next to retrieval/ → use parent of __file__.
    - Local dev: app/app/app.py → repo root is one level up from app/.
    """
    here = Path(__file__).resolve().parent
    cfg = Path("retrieval") / "configs" / "retrieval_config.yaml"
    if (here / cfg).exists():
        return here
    parent = here.parent
    if (parent / cfg).exists():
        return parent
    # Spaces: never use parents[1] blindly — it can be filesystem "/" if app.py is at /app/app.py.
    return here


_ROOT = _repo_root()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("app")

CONFIG_PATH = _ROOT / "retrieval/configs/retrieval_config.yaml"

# ---------------------------------------------------------------------------
# Global pipeline components (loaded once at startup)
# ---------------------------------------------------------------------------

_cfg: Optional[dict] = None
_pipeline = None
_reranker = None
_generator = None
_obs_logger = None
_ready = False
_startup_error: Optional[str] = None


def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def startup() -> None:
    global _cfg, _pipeline, _reranker, _generator, _obs_logger, _ready, _startup_error
    try:
        # Relative paths like Path("retrieval/...") expect cwd == repo root on Spaces.
        os.chdir(_ROOT)

        # Local .env (Space injects secrets as env vars — no file needed there).
        try:
            from dotenv import load_dotenv

            load_dotenv(_ROOT / ".env")
        except Exception:
            pass

        # HF Spaces inject secrets as env vars — verify names match Space Settings (case-sensitive).
        log.info(
            "Env: GROQ_API_KEY=%s | HF_TOKEN=%s | HF_ARTIFACTS_DATASET=%s",
            "set" if os.environ.get("GROQ_API_KEY") else "MISSING",
            "set" if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") else "MISSING",
            os.environ.get("HF_ARTIFACTS_DATASET") or "(not set)",
        )

        log.info("FinGuard startup: loading configuration...")
        _cfg = _load_config()

        from deployment.ensure_hf_artifacts import ensure_hf_artifacts

        ensure_hf_artifacts(_ROOT, _cfg, log)

        log.info("Loading retrieval pipeline...")
        from retrieval.pipeline import RetrievalPipeline
        _pipeline = RetrievalPipeline(_cfg)
        _pipeline.load()

        log.info("Loading cross-encoder reranker (sentence-transformers, matches eval path)...")
        from sentence_transformers import CrossEncoder
        reranker_model = _cfg.get("reranker", {}).get("model_id", "BAAI/bge-reranker-base")
        _reranker = CrossEncoder(reranker_model)

        log.info("Building semantic cache and generator...")
        from cache.semantic_cache import build_cache_from_config
        from generation.generator import Generator
        from generation.llm_client import LLMClient
        _cache = build_cache_from_config(_cfg)
        _llm = LLMClient(_cfg)
        _generator = Generator(cfg=_cfg, cache=_cache, llm_client=_llm)

        log.info("Initializing observability logger...")
        from observability.logger import RequestLogger
        obs_cfg = _cfg.get("observability", {})
        _obs_logger = RequestLogger(
            log_dir=obs_cfg.get("log_dir", "observability/logs"),
            rotate_daily=obs_cfg.get("rotate_daily", True),
            enabled=obs_cfg.get("enabled", True),
        )

        _ready = True
        log.info("FinGuard startup complete ✓")

    except Exception as exc:
        _startup_error = str(exc)
        log.error(f"Startup failed: {exc}", exc_info=True)


# ---------------------------------------------------------------------------
# Query processing
# ---------------------------------------------------------------------------

def _source_badge(doc: dict) -> str:
    if doc.get("doc_type") == "pdf_chunk":
        fname = doc.get("source_file", "document.pdf")
        fname = fname.replace(".pdf", "")
        page = doc.get("page_no")
        page_str = f" p.{page}" if page else ""
        return f"PDF — {fname}{page_str}"
    return "QA"


def _doc_snippet(doc: dict, max_chars: int = 220) -> str:
    text = (
        doc.get("answer")
        or doc.get("retrieval_text")
        or doc.get("chunk_text")
        or doc.get("question")
        or ""
    ).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "…"
    return text


def _format_lang_label(lang: str) -> str:
    return {"urdu": "🇵🇰 Urdu", "roman_urdu": "🔤 Roman Urdu", "english": "🇬🇧 English"}.get(
        lang, f"🌐 {lang.title()}"
    )


def _format_intent_label(intent: str) -> str:
    return {"legal": "⚖️ Legal / Policy", "practical": "💡 Practical"}.get(
        intent, "💡 Practical"
    )


def process_query(query: str):
    """
    Main Gradio handler. Runs the full pipeline and returns UI component values.
    Yields intermediate status so the user sees progress during loading.
    """
    if not query or not query.strip():
        yield (
            "", "", "", "", "", "",
            _empty_docs_html(), "", "Please enter a question.", ""
        )
        return

    if not _ready:
        err = _startup_error or "System is still loading. Please wait a moment and try again."
        yield ("", "", "", "", "", "", _empty_docs_html(), "", f"⚠️ {err}", "")
        return

    t_total = time.time()

    # Yield a loading state immediately
    yield (
        "⏳ Detecting language...", "", "", "", "", "",
        _empty_docs_html(), "", "Processing…", ""
    )

    try:
        # ── Step 1: Retrieval pipeline ──────────────────────────────────
        from retrieval.normalization import normalize_query
        from retrieval.language import detect_language

        query_normalized = normalize_query(query)
        lang_result = detect_language(query_normalized)

        yield (
            _format_lang_label(lang_result.label), "", "", "", "", "",
            _empty_docs_html(), "", "Retrieving documents…", ""
        )

        # Pass raw query so route_query + expand_query match production pipeline.
        retrieval_output = _pipeline.run(query)
        diag = retrieval_output.diagnostics

        stage = diag.get("stage_latency_ms") or {}
        if isinstance(stage, dict) and stage.get("dual_retrieve_ms") is not None:
            retrieval_ms = float(stage.get("dual_retrieve_ms", 0) or 0)
        elif isinstance(stage, dict):
            retrieval_ms = float(stage.get("vector_ms", 0) or 0) + float(
                stage.get("bm25_ms", 0) or 0
            ) + float(stage.get("mmr_ms", 0) or 0)
        else:
            retrieval_ms = float(retrieval_output.total_ms or 0)

        yield (
            _format_lang_label(lang_result.label), "", "", "", "", "",
            _empty_docs_html(), "", "Reranking…", ""
        )

        # ── Step 2: Reranker ────────────────────────────────────────────
        # MMRResult stores the document dict under metadata["doc"] (dual + QA-only).
        candidate_docs = []
        for r in retrieval_output.docs:
            d = dict(r.metadata.get("doc", {}))
            if not d.get("doc_text"):
                d["doc_text"] = (
                    d.get("retrieval_text")
                    or d.get("chunk_text")
                    or d.get("answer")
                    or d.get("question")
                    or ""
                )
            candidate_docs.append(d)
        t_rerank = time.time()
        reranker_scores: list[float] = []
        reranked_docs: list[dict] = []

        if candidate_docs:
            rerank_q = diag.get("query_expanded") or query_normalized
            pairs = [(rerank_q, _doc_snippet(doc, max_chars=512)) for doc in candidate_docs]
            scores = _reranker.predict(pairs).tolist()
            ranked = sorted(zip(scores, candidate_docs), key=lambda x: x[0], reverse=True)
            reranked_docs = [doc for _, doc in ranked[:3]]
            reranker_scores = [score for score, _ in ranked[:3]]

        reranker_ms = (time.time() - t_rerank) * 1000

        yield (
            _format_lang_label(lang_result.label), "", "", "", "", "",
            _empty_docs_html(), "", "Generating answer…", ""
        )

        # ── Step 3: Query intent (from diagnostics or router) ───────────
        query_intent = diag.get("query_intent", "practical")

        # ── Step 4: Generator (gate + cache + LLM) ──────────────────────
        query_expanded = diag.get("query_expanded") or query_normalized
        query_embedding = _pipeline._vector_retriever.encode_query(query_expanded)

        t_gen = time.time()
        gen_output = _generator.generate(
            query=query_expanded,
            query_embedding=query_embedding,
            reranked_docs=reranked_docs,
            reranker_scores=reranker_scores,
            language=lang_result.label,
        )
        llm_ms = (time.time() - t_gen) * 1000

        total_ms = (time.time() - t_total) * 1000

        # ── Step 5: Build UI values ─────────────────────────────────────
        lang_html = _lang_badge_html(lang_result.label, lang_result.confidence)
        intent_html = _intent_badge_html(query_intent)
        cache_html = _cache_badge_html(gen_output.cache_level, gen_output.cache_hit)
        category_html = _category_chip_html(
            diag.get("category_detected"), diag.get("category_confidence", 0.0)
        )
        gate_html = _gate_html(gen_output.gate)
        docs_html = _docs_panel_html(reranked_docs, reranker_scores)
        answer_md = gen_output.answer or "⚠️ No answer generated."
        latency_html = _latency_html(retrieval_ms, reranker_ms, llm_ms, total_ms)

        # ── Step 6: Observability log ───────────────────────────────────
        if _obs_logger:
            from observability.logger import build_log_entry
            entry = build_log_entry(
                query=query,
                total_ms=total_ms,
                query_language=diag.get("language_detected", lang_result.label),
                language_confidence=diag.get(
                    "language_confidence", lang_result.confidence
                ),
                query_intent=query_intent,
                query_normalized=diag.get("query_normalized", query_normalized),
                category_detected=diag.get("category_detected"),
                category_confidence=diag.get("category_confidence", 0.0),
                filter_applied=diag.get("filter_applied", False),
                filter_reason=diag.get("filter_reason"),
                cache_level=gen_output.cache_level,
                cache_similarity=None,
                corpus_version=_cfg.get("embedding_model", {}).get("version_tag", "unknown"),
                retrieval_ms=retrieval_ms,
                reranker_ms=reranker_ms,
                llm_ms=llm_ms,
                top_reranker_score=reranker_scores[0] if reranker_scores else 0.0,
                gate_passed=gen_output.gate.passed,
                gate_reason=gen_output.gate.reason,
                retrieved_doc_ids=[d.get("doc_id", "") for d in reranked_docs],
                answer_generated=gen_output.answer_generated if hasattr(gen_output, "answer_generated") else bool(gen_output.answer),
                llm_provider=gen_output.llm_response.provider if gen_output.llm_response else None,
                llm_model=gen_output.llm_response.model if gen_output.llm_response else None,
                llm_tokens=(
                    (gen_output.llm_response.prompt_tokens or 0)
                    + (gen_output.llm_response.completion_tokens or 0)
                )
                if gen_output.llm_response
                else None,
                source_mix=diag.get("source_mix"),
            )
            _obs_logger.write(entry)

        yield (
            lang_html, intent_html, cache_html, category_html,
            gate_html, latency_html, docs_html, answer_md, "", ""
        )

    except Exception as exc:
        log.error(f"Query processing error: {exc}", exc_info=True)
        yield (
            "", "", "", "", "", "", _empty_docs_html(),
            "", f"⚠️ An error occurred: {exc}", ""
        )


# ---------------------------------------------------------------------------
# HTML builders
# ---------------------------------------------------------------------------

def _lang_badge_html(lang: str, confidence: float) -> str:
    colors = {
        "urdu": "#7C3AED",
        "roman_urdu": "#2563EB",
        "english": "#059669",
    }
    color = colors.get(lang, "#6B7280")
    label = _format_lang_label(lang)
    return (
        f'<span style="background:{color};color:#fff;padding:4px 12px;'
        f'border-radius:12px;font-size:13px;font-weight:600;">'
        f'{label}</span>'
        f'<span style="color:#9CA3AF;font-size:12px;margin-left:8px;">conf {confidence:.2f}</span>'
    )


def _intent_badge_html(intent: str) -> str:
    color = "#DC2626" if intent == "legal" else "#D97706"
    label = _format_intent_label(intent)
    return (
        f'<span style="background:{color};color:#fff;padding:4px 12px;'
        f'border-radius:12px;font-size:13px;font-weight:600;">{label}</span>'
    )


def _cache_badge_html(cache_level: Optional[int], cache_hit: bool) -> str:
    if cache_hit and cache_level == 1:
        return (
            '<span style="background:#065F46;color:#D1FAE5;padding:4px 12px;'
            'border-radius:12px;font-size:13px;font-weight:600;">⚡ L1 Cached (answer)</span>'
        )
    if cache_hit and cache_level == 2:
        return (
            '<span style="background:#1E40AF;color:#DBEAFE;padding:4px 12px;'
            'border-radius:12px;font-size:13px;font-weight:600;">⚡ L2 Cached (retrieval)</span>'
        )
    return (
        '<span style="background:#374151;color:#F9FAFB;padding:4px 12px;'
        'border-radius:12px;font-size:13px;font-weight:600;">🔍 Retrieved</span>'
    )


def _category_chip_html(category: Optional[str], confidence: float) -> str:
    if not category:
        return '<span style="color:#9CA3AF;font-size:13px;">No category detected</span>'
    icons = {
        "islamic_finance": "🕌",
        "banking_basics": "🏦",
        "digital_payments": "📱",
        "loans": "💳",
        "investment": "📈",
        "insurance": "🛡️",
        "tax": "🧾",
        "remittance": "💸",
    }
    icon = icons.get(category, "📂")
    label = category.replace("_", " ").title()
    conf_pct = int(confidence * 100)
    return (
        f'<span style="background:#F3F4F6;color:#111827;padding:4px 12px;'
        f'border-radius:20px;font-size:13px;border:1px solid #E5E7EB;">'
        f'{icon} {label}</span>'
        f'<span style="color:#9CA3AF;font-size:12px;margin-left:8px;">{conf_pct}% conf</span>'
    )


def _gate_html(gate) -> str:
    if gate.passed:
        score_pct = int(gate.top_score * 100)
        color = "#065F46"
        bg = "#ECFDF5"
        icon = "✅"
        label = f"Gate passed — score {score_pct}%"
    else:
        score_pct = int(gate.top_score * 100)
        color = "#991B1B"
        bg = "#FEF2F2"
        icon = "🚫"
        label = f"Gate blocked — {gate.reason} (score {score_pct}%)"
    return (
        f'<div style="background:{bg};color:{color};padding:8px 14px;'
        f'border-radius:8px;font-size:13px;border:1px solid {color}33;">'
        f'{icon} {label}</div>'
    )


def _docs_panel_html(docs: list[dict], scores: list[float]) -> str:
    if not docs:
        return _empty_docs_html()

    parts = []
    for i, (doc, score) in enumerate(zip(docs, scores), start=1):
        is_pdf = doc.get("doc_type") == "pdf_chunk"
        badge_color = "#1E40AF" if is_pdf else "#065F46"
        badge_bg = "#DBEAFE" if is_pdf else "#D1FAE5"
        source_label = _source_badge(doc)
        snippet = _doc_snippet(doc)
        score_pct = int(score * 100)
        score_color = "#065F46" if score >= 0.7 else "#D97706" if score >= 0.4 else "#991B1B"

        question_line = ""
        if not is_pdf and doc.get("question"):
            q = doc["question"][:120]
            question_line = (
                f'<div style="font-size:12px;color:#6B7280;margin-bottom:4px;">'
                f'Q: {q}</div>'
            )

        parts.append(
            f'<div style="border:1px solid #E5E7EB;border-radius:10px;'
            f'padding:12px 16px;margin-bottom:10px;background:#FAFAFA;">'
            f'  <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">'
            f'    <span style="font-weight:700;color:#374151;font-size:13px;">#{i}</span>'
            f'    <span style="background:{badge_bg};color:{badge_color};padding:2px 10px;'
            f'    border-radius:10px;font-size:12px;font-weight:600;">{source_label}</span>'
            f'    <span style="margin-left:auto;font-size:12px;font-weight:700;color:{score_color};">'
            f'    Score {score_pct}%</span>'
            f'  </div>'
            f'  {question_line}'
            f'  <div style="font-size:13px;color:#374151;line-height:1.5;">{snippet}</div>'
            f'</div>'
        )

    return "".join(parts)


def _empty_docs_html() -> str:
    return (
        '<div style="color:#9CA3AF;font-size:13px;padding:20px;text-align:center;">'
        'No documents retrieved yet.'
        '</div>'
    )


def _latency_html(retrieval_ms: float, reranker_ms: float, llm_ms: float, total_ms: float) -> str:
    def _bar(ms: float, max_ms: float = 1000) -> str:
        pct = min(100, int(ms / max_ms * 100))
        return (
            f'<div style="background:#E5E7EB;border-radius:4px;height:6px;width:100%;margin-top:2px;">'
            f'<div style="background:#2563EB;width:{pct}%;height:6px;border-radius:4px;"></div>'
            f'</div>'
        )

    return (
        f'<div style="font-size:12px;color:#374151;">'
        f'  <div style="display:flex;justify-content:space-between;margin-bottom:6px;">'
        f'    <span>Retrieval</span><span style="font-weight:600;">{retrieval_ms:.0f}ms</span>'
        f'  </div>{_bar(retrieval_ms)}'
        f'  <div style="display:flex;justify-content:space-between;margin:8px 0 2px;">'
        f'    <span>Reranker</span><span style="font-weight:600;">{reranker_ms:.0f}ms</span>'
        f'  </div>{_bar(reranker_ms)}'
        f'  <div style="display:flex;justify-content:space-between;margin:8px 0 2px;">'
        f'    <span>LLM</span><span style="font-weight:600;">{llm_ms:.0f}ms</span>'
        f'  </div>{_bar(llm_ms, max_ms=2000)}'
        f'  <div style="border-top:1px solid #E5E7EB;margin-top:10px;padding-top:8px;'
        f'  display:flex;justify-content:space-between;">'
        f'    <span style="font-weight:700;">Total</span>'
        f'    <span style="font-weight:700;color:#2563EB;">{total_ms:.0f}ms</span>'
        f'  </div>'
        f'</div>'
    )


def _startup_error_ui():
    import gradio as gr
    with gr.Blocks(title="FinGuard RAG — Error") as demo:
        gr.Markdown(f"## ⚠️ Startup Failed\n\n```\n{_startup_error}\n```\n\nCheck logs for details.")
    return demo


# ---------------------------------------------------------------------------
# Example queries
# ---------------------------------------------------------------------------

EXAMPLES = [
    ["زکوٰۃ کا نصاب کتنا ہے؟"],
    ["zakat ka hisab kaise karein"],
    ["meezan bank account kholne ke liye kya documents chahiye"],
    ["easypaisa se paise kaise bhejein"],
    ["riba aur interest mein kya farq hai"],
    ["how to file income tax return on FBR IRIS"],
    ["home loan ki qist kitni hogi 50 lakh pe"],
    ["pakistan stock exchange mein invest kaise karein"],
    ["roshan digital account kya hai overseas pakistanis ke liye"],
    ["what is the difference between murabaha and ijarah"],
]


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui():
    import gradio as gr

    css = """
    .gr-button-primary { background: #2563EB !important; border-color: #2563EB !important; }
    .answer-box textarea { font-size: 15px !important; line-height: 1.7 !important; }
    footer { display: none !important; }
    #title-block { text-align: center; padding-bottom: 8px; }
    """

    with gr.Blocks(
        title="FinGuard — Pakistani Finance Assistant",
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css=css,
    ) as demo:

        # ── Header ──────────────────────────────────────────────────────
        with gr.Column(elem_id="title-block"):
            gr.Markdown(
                """
                # 🏦 FinGuard — Pakistani Finance Assistant
                **Ask anything about Islamic finance, banking, investments, tax, digital payments, and more.**
                Supports Roman Urdu 🔤 · Urdu 🇵🇰 · English 🇬🇧
                """,
                elem_id="header",
            )

        # ── Input row ───────────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=5):
                query_input = gr.Textbox(
                    label="Your Question",
                    placeholder="e.g. zakat ka hisab kaise karein / What is nisab?",
                    lines=2,
                    max_lines=4,
                )
            with gr.Column(scale=1, min_width=120):
                submit_btn = gr.Button("Ask →", variant="primary", size="lg")
                clear_btn = gr.Button("Clear", size="sm")

        # ── Metadata badges row ─────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=1):
                lang_badge = gr.HTML(label="Language")
            with gr.Column(scale=1):
                intent_badge = gr.HTML(label="Intent")
            with gr.Column(scale=1):
                cache_badge = gr.HTML(label="Cache")
            with gr.Column(scale=1):
                category_chip = gr.HTML(label="Category")

        # ── Gate status ─────────────────────────────────────────────────
        gate_status = gr.HTML(label="Confidence Gate")

        # ── Main content: answer + retrieved docs + latency ─────────────
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("### 💬 Answer")
                answer_output = gr.Markdown(
                    value="*Answer will appear here...*",
                    elem_classes=["answer-box"],
                )

            with gr.Column(scale=2):
                gr.Markdown("### 📄 Retrieved Documents")
                docs_panel = gr.HTML(value=_empty_docs_html())

                gr.Markdown("### ⏱️ Latency")
                latency_display = gr.HTML()

        # ── Status / error line ─────────────────────────────────────────
        status_line = gr.Markdown(visible=False)
        error_line = gr.Markdown(visible=False)

        # ── Examples ────────────────────────────────────────────────────
        gr.Markdown("---\n### 📝 Example Questions")
        gr.Examples(
            examples=EXAMPLES,
            inputs=query_input,
            label="Click any example to load it",
        )

        # ── Footer ──────────────────────────────────────────────────────
        gr.Markdown(
            """
            ---
            <div style="text-align:center;color:#9CA3AF;font-size:12px;">
            FinGuard RAG · Embedding: <code>hassan7272/urdu-finance-embeddings</code> ·
            Reranker: <code>BAAI/bge-reranker-base</code> ·
            Built with FAISS · BM25 · RRF · MMR · Groq LLM
            </div>
            """,
        )

        # ── Event wiring ─────────────────────────────────────────────────
        outputs = [
            lang_badge, intent_badge, cache_badge, category_chip,
            gate_status, latency_display, docs_panel, answer_output,
            status_line, error_line,
        ]

        submit_btn.click(
            fn=process_query,
            inputs=query_input,
            outputs=outputs,
        )

        query_input.submit(
            fn=process_query,
            inputs=query_input,
            outputs=outputs,
        )

        def clear_all():
            return (
                "", "", "", "", "", "",
                _empty_docs_html(), "*Answer will appear here...*", "", ""
            )

        clear_btn.click(fn=clear_all, outputs=outputs)

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    startup()

    if not _ready:
        log.error(f"Startup failed: {_startup_error}")
        demo = _startup_error_ui()
    else:
        demo = build_ui()

    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
        show_error=True,
    )
else:
    startup()
    if _ready:
        demo = build_ui()
    else:
        import gradio as gr
        demo = _startup_error_ui()