"""
ingestion/pdf_chunker.py

Loads PDFs from a directory using LlamaIndex SimpleDirectoryReader,
splits into overlapping chunks via SentenceSplitter, prepends the
inferred section title to every chunk so the embedding captures topic
context rather than a bare mid-paragraph fragment.

Returns a list of raw LlamaIndex Node objects for pdf_corpus_builder.py
to convert into the canonical corpus schema.
"""

import hashlib
import logging
import re
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

CHUNK_SIZE = 512
CHUNK_OVERLAP = 100


def _infer_section_title(text: str) -> Optional[str]:
    lines = text.strip().splitlines()
    for line in lines[:6]:
        line = line.strip()
        if not line:
            continue
        if 10 <= len(line) <= 120:
            upper_ratio = sum(1 for c in line if c.isupper()) / max(len(line), 1)
            digit_ratio = sum(1 for c in line if c.isdigit()) / max(len(line), 1)
            if upper_ratio > 0.4 or re.match(r"^(\d+[\.\-]?\s+|[A-Z][A-Z\s]+$)", line):
                return line
            if digit_ratio < 0.5 and line.endswith((":", "—", "-")) is False:
                if len(line.split()) <= 12:
                    return line
    return None


def _prepend_title(title: Optional[str], body: str) -> str:
    if title:
        return f"{title}\n\n{body}"
    return body


def _file_category_hint(filename: str) -> Optional[str]:
    name = filename.lower()
    rules = [
        (["islamic", "jurisprudence", "sharia", "zakat", "halal"], "islamic_finance"),
        (["allied", "askari", "nbp", "banking", "consumer", "bank"], "banking"),
        (["easypaisa", "jazz", "digital payment", "digital_payment"], "digital_finance"),
        (["fbr", "income tax", "tax return", "filing"], "tax"),
        (["investment", "stock", "psx", "secp", "growth"], "investment"),
        (["saving", "money management", "finance guide", "pakistan invest"], "personal_finance"),
        (["virtual asset", "crypto"], "digital_finance"),
        (["asaan karobar", "ordinance", "act", "policy"], "financial_education"),
    ]
    for keywords, cat in rules:
        if any(k in name for k in keywords):
            return cat
    return None


def load_and_chunk(
    pdf_dir: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    file_filter: Optional[list[str]] = None,
) -> list[dict]:
    try:
        from llama_index.core import SimpleDirectoryReader
        from llama_index.core.node_parser import SentenceSplitter
    except ImportError:
        raise ImportError(
            "pip install llama-index-core llama-index-readers-file"
        )

    pdf_path = Path(pdf_dir)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    pdf_files = sorted(pdf_path.glob("*.pdf"))
    if file_filter:
        pdf_files = [f for f in pdf_files if f.name in file_filter]

    if not pdf_files:
        raise ValueError(f"No PDF files found in {pdf_dir}")

    log.info(f"Loading {len(pdf_files)} PDFs from {pdf_dir}")

    reader = SimpleDirectoryReader(
        input_files=[str(f) for f in pdf_files],
        filename_as_id=True,
    )
    documents = reader.load_data()
    log.info(f"Loaded {len(documents)} pages total")

    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    nodes = splitter.get_nodes_from_documents(documents)
    log.info(f"Split into {len(nodes)} chunks")

    raw_chunks = []
    chunk_counters: dict[str, int] = {}

    for node in nodes:
        source_file = Path(
            node.metadata.get("file_name") or
            node.metadata.get("file_path") or
            "unknown.pdf"
        ).name

        page_no = node.metadata.get("page_label") or node.metadata.get("page")
        try:
            page_no = int(page_no)
        except (TypeError, ValueError):
            page_no = None

        chunk_idx = chunk_counters.get(source_file, 0)
        chunk_counters[source_file] = chunk_idx + 1

        body = node.get_content().strip()
        if not body or len(body) < 30:
            continue

        title = _infer_section_title(body)
        retrieval_text = _prepend_title(title, body)

        hash_input = f"{source_file}:{chunk_idx}:{body[:80]}"
        doc_id = "pdf_" + hashlib.sha256(hash_input.encode()).hexdigest()[:8]

        raw_chunks.append({
            "doc_id": doc_id,
            "source_file": source_file,
            "page_no": page_no,
            "chunk_id": chunk_idx,
            "parent_section": title,
            "chunk_text": body,
            "retrieval_text": retrieval_text,
            "_category_hint": _file_category_hint(source_file),
        })

    log.info(f"Produced {len(raw_chunks)} valid chunks from {len(pdf_files)} PDFs")
    return raw_chunks