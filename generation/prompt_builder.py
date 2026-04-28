"""
generation/prompt_builder.py

Builds the LLM prompt for the final answer generation step.

Two templates:
  - QA template    : for answers grounded in QA-pair documents
  - PDF template   : for answers grounded in official PDF policy documents

Template is selected based on the doc_type of the top-1 reranked document.
Context is formatted differently per source type so the LLM knows what
kind of material it is reading.
"""

import logging
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)

QA_TEMPLATE = """\
You are a helpful financial assistant for Pakistani users.
Answer ONLY using the provided context below.
If the context does not contain enough information to answer, say exactly: "I don't have enough information to answer this."
Keep your answer to 2-3 sentences maximum.
Respond in the same language as the user's question (Roman Urdu, Urdu, or English).
Do not add any information that is not present in the context.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

PDF_TEMPLATE = """\
You are a helpful financial assistant for Pakistani users.
Answer ONLY using the provided context below, which is drawn from official financial documents and regulations.
Where relevant, briefly cite the source document name (for example: "According to the FBR guide..." or "As per the Banking Policy...").
If the context does not contain enough information to answer, say exactly: "I don't have enough information to answer this."
Keep your answer to 2-3 sentences maximum.
Respond in the same language as the user's question (Roman Urdu, Urdu, or English).
Do not add any information that is not present in the context.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


@dataclass
class BuiltPrompt:
    prompt: str
    template_used: str
    doc_types_in_context: list[str]
    num_docs: int
    context_chars: int
    top_doc_source: Optional[str]


def _format_qa_doc(doc: dict, rank: int) -> str:
    question = doc.get("question", "").strip()
    answer = doc.get("answer", "").strip()
    if question and answer:
        return f"[Source {rank}]\nQ: {question}\nA: {answer}"
    return f"[Source {rank}]\n{answer or question}"


def _format_pdf_doc(doc: dict, rank: int) -> str:
    source_file = doc.get("source_file", "")
    if source_file:
        source_file = source_file.replace(".pdf", "").replace("_", " ").replace("-", " ").title()
    parent_section = doc.get("parent_section", "")
    text = doc.get("retrieval_text") or doc.get("chunk_text") or doc.get("answer", "")
    text = text.strip()

    header_parts = []
    if source_file:
        header_parts.append(source_file)
    if parent_section:
        header_parts.append(parent_section)
    header = " — ".join(header_parts) if header_parts else "Official Document"

    page = doc.get("page_no")
    page_str = f" (p.{page})" if page else ""

    return f"[Source {rank} — {header}{page_str}]\n{text}"


def _select_template(docs: list[dict]) -> str:
    if not docs:
        return "qa"
    top_doc = docs[0]
    doc_type = top_doc.get("doc_type", "")
    if doc_type == "pdf_chunk":
        return "pdf"
    return "qa"


def build_prompt(
    question: str,
    docs: list[dict],
    max_context_chars: int = 3000,
) -> BuiltPrompt:
    """
    Build the final LLM prompt from a question and ranked context documents.

    Parameters
    ----------
    question          : the user's original (normalized) query
    docs              : list of document dicts from the reranker, ranked best-first
                        each doc must have doc_type: "qa_pair" | "pdf_chunk"
    max_context_chars : hard cap on total context length sent to LLM
                        prevents token limit overflows on long PDF chunks

    Returns
    -------
    BuiltPrompt with the final prompt string and metadata about what was built
    """
    if not docs:
        log.warning("build_prompt called with empty docs list")
        context = "No relevant context available."
        template_name = "qa"
        formatted_parts = []
        doc_types = []
        top_source = None
    else:
        template_name = _select_template(docs)
        doc_types = [d.get("doc_type", "qa_pair") for d in docs]
        top_source = docs[0].get("source_file") or docs[0].get("doc_id")

        formatted_parts = []
        running_chars = 0

        for rank, doc in enumerate(docs, start=1):
            doc_type = doc.get("doc_type", "qa_pair")

            if doc_type == "pdf_chunk":
                formatted = _format_pdf_doc(doc, rank)
            else:
                formatted = _format_qa_doc(doc, rank)

            if running_chars + len(formatted) > max_context_chars:
                remaining = max_context_chars - running_chars
                if remaining > 200:
                    formatted = formatted[:remaining] + "..."
                    formatted_parts.append(formatted)
                break

            formatted_parts.append(formatted)
            running_chars += len(formatted)

        context = "\n\n".join(formatted_parts) if formatted_parts else "No relevant context available."

    template = PDF_TEMPLATE if template_name == "pdf" else QA_TEMPLATE
    prompt = template.format(context=context, question=question)

    return BuiltPrompt(
        prompt=prompt,
        template_used=template_name,
        doc_types_in_context=doc_types,
        num_docs=len(formatted_parts) if docs else 0,
        context_chars=len(context),
        top_doc_source=top_source,
    )


def build_fallback_message(language: str = "roman_urdu") -> str:
    """
    Return the confidence-gate fallback message in the appropriate language.
    Called when reranker score is below threshold — LLM is never invoked.
    """
    messages = {
        "urdu": "معذرت، مجھے اس سوال کا جواب دینے کے لیے کافی معلومات نہیں ملیں۔ براہ کرم سوال کو دوبارہ واضح طریقے سے پوچھیں۔",
        "roman_urdu": "Mujhe is sawal ka jawab dene ke liye kafi information nahi mili. Meherbani kar ke apna sawal dobara poochein.",
        "english": "I don't have enough information to answer this question. Please try rephrasing your query.",
    }
    return messages.get(language, messages["roman_urdu"])