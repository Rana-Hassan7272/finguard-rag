"""
retrieval/build_vector_index.py
================================
Builds the FAISS vector index from the deduplicated corpus.

Responsibilities:
  - Load corpus.jsonl (built by build_corpus.py)
  - Embed QUESTION FIELD ONLY (not answer — cleaner retrieval signal)
  - L2-normalize all vectors
  - Build FAISS IndexFlatIP (exact cosine on normalized vecs)
  - Save versioned index, id_map, and index_manifest
  - Startup safety check: version_tag in config must match manifest

Run:
  python retrieval/build_vector_index.py
  python retrieval/build_vector_index.py --force   (overwrite existing index)
"""

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_vector_index")

CONFIG_PATH = Path("retrieval/configs/retrieval_config.yaml")


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Version safety helpers
# ---------------------------------------------------------------------------

def resolve_artifact_path(template: str, version: str) -> Path:
    """Replace {version} placeholder in artifact path template."""
    return Path(template.replace("{version}", version))


def check_existing_index(manifest_path: Path, version_tag: str, force: bool) -> bool:
    """
    Return True if it is safe to proceed with building.
    Warns if an index already exists for this version.
    Aborts (returns False) if index exists and --force not set.
    """
    if manifest_path.exists():
        if force:
            log.warning(
                f"Index already exists for version '{version_tag}'. "
                f"--force flag set — overwriting."
            )
            return True
        else:
            log.error(
                f"Index already exists for version '{version_tag}': {manifest_path}\n"
                f"To overwrite, run with --force.\n"
                f"To build a new version, increment version_tag in retrieval_config.yaml."
            )
            return False
    return True


# ---------------------------------------------------------------------------
# Corpus loader
# ---------------------------------------------------------------------------

def load_corpus(corpus_path: Path) -> list[dict]:
    if not corpus_path.exists():
        log.error(
            f"Corpus file not found: {corpus_path}\n"
            f"Run build_corpus.py first."
        )
        sys.exit(1)

    docs = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))

    log.info(f"Loaded {len(docs)} documents from corpus")
    return docs


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_questions(
    docs: list[dict],
    model_id: str,
    batch_size: int,
    use_gpu: bool,
) -> np.ndarray:
    """
    Embed the 'question' field of each document.

    We embed questions ONLY (not answers) because:
    - At query time, the user submits a question
    - Matching question-to-question maximizes retrieval precision
    - Answer text dilutes the embedding signal for retrieval
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        log.error("Run: pip install sentence-transformers")
        sys.exit(1)

    import torch
    device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    log.info(f"Loading embedder: {model_id} | device={device}")
    model = SentenceTransformer(model_id, device=device)

    questions = [doc["question"] for doc in docs]
    log.info(f"Embedding {len(questions)} questions (batch_size={batch_size})...")

    t0 = time.time()
    embeddings = model.encode(
        questions,
        batch_size=batch_size,
        normalize_embeddings=True,   # L2 normalize — required for IndexFlatIP = cosine
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    elapsed = time.time() - t0

    embeddings = embeddings.astype(np.float32)

    log.info(
        f"Embedding complete: shape={embeddings.shape} | "
        f"dtype={embeddings.dtype} | elapsed={elapsed:.1f}s"
    )

    # Sanity check: confirm vectors are normalized (norms should be ~1.0)
    norms = np.linalg.norm(embeddings, axis=1)
    log.info(
        f"Norm check: min={norms.min():.4f}, max={norms.max():.4f}, "
        f"mean={norms.mean():.4f} (expected ~1.0)"
    )

    return embeddings


# ---------------------------------------------------------------------------
# FAISS index builder
# ---------------------------------------------------------------------------

def build_faiss_index(embeddings: np.ndarray, index_type: str, dim: int):
    """
    Build FAISS index.

    IndexFlatIP: exact inner product search.
    On L2-normalized vectors, inner product == cosine similarity.
    No approximation — perfect recall, suitable for corpus < 100k docs.
    """
    try:
        import faiss
    except ImportError:
        log.error("Run: pip install faiss-cpu")
        sys.exit(1)

    log.info(f"Building FAISS index: type={index_type}, dim={dim}")

    if index_type == "IndexFlatIP":
        index = faiss.IndexFlatIP(dim)
    else:
        log.error(f"Unsupported index_type: {index_type}. Only IndexFlatIP supported currently.")
        sys.exit(1)

    t0 = time.time()
    index.add(embeddings)
    log.info(
        f"FAISS index built: ntotal={index.ntotal} vectors | elapsed={time.time()-t0:.2f}s"
    )

    # Sanity: quick self-retrieval test
    test_emb = embeddings[:1]
    scores, indices = index.search(test_emb, k=1)
    if indices[0][0] != 0:
        log.warning("Self-retrieval test FAILED — index may have issues!")
    else:
        log.info(f"Self-retrieval test PASSED (score={scores[0][0]:.4f})")

    return index


# ---------------------------------------------------------------------------
# Corpus hash (for manifest — detects corpus drift)
# ---------------------------------------------------------------------------

def compute_corpus_hash(docs: list[dict]) -> str:
    """SHA256 of all doc_id:corpus_hash pairs — stable fingerprint of corpus content."""
    combined = "|".join(
        f"{doc['doc_id']}:{doc.get('corpus_hash', doc['question'][:20])}"
        for doc in docs
    )
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_vector_index(args: argparse.Namespace) -> None:
    cfg = load_config()

    version_tag = cfg["embedding_model"]["version_tag"]
    model_id = cfg["embedding_model"]["local_path"] or cfg["embedding_model"]["hub_id"]
    batch_size = cfg["embedding_model"]["encode_batch_size"]
    use_gpu = cfg["embedding_model"]["use_gpu_if_available"]
    index_type = cfg["faiss"]["index_type"]
    dim = cfg["faiss"]["dimension"]

    # Resolve versioned artifact paths
    index_path = resolve_artifact_path(cfg["artifacts"]["faiss_index"], version_tag)
    id_map_path = resolve_artifact_path(cfg["artifacts"]["faiss_id_map"], version_tag)
    manifest_path = resolve_artifact_path(cfg["artifacts"]["faiss_manifest"], version_tag)

    # Safety: do not overwrite existing index unless --force
    if not check_existing_index(manifest_path, version_tag, force=args.force):
        sys.exit(1)

    # Ensure output directory exists
    index_path.parent.mkdir(parents=True, exist_ok=True)

    # Load corpus
    corpus_path = Path(cfg["artifacts"]["corpus_jsonl"])
    docs = load_corpus(corpus_path)

    if len(docs) == 0:
        log.error("Corpus is empty — nothing to index. Run build_corpus.py first.")
        sys.exit(1)

    # Embed questions
    embeddings = embed_questions(docs, model_id, batch_size, use_gpu)

    # Verify dim matches config
    actual_dim = embeddings.shape[1]
    if actual_dim != dim:
        log.error(
            f"Embedding dim mismatch: config says {dim}, model produced {actual_dim}. "
            f"Update faiss.dimension in retrieval_config.yaml."
        )
        sys.exit(1)

    # Build FAISS index
    index = build_faiss_index(embeddings, index_type, dim)

    # Build id_map: FAISS row index -> doc_id
    # This is how we go from FAISS result index back to our doc
    id_map = {str(faiss_row): doc["doc_id"] for faiss_row, doc in enumerate(docs)}

    # Compute corpus fingerprint for manifest
    corpus_hash = compute_corpus_hash(docs)

    # Manifest
    manifest = {
        "embedder_hub_id": cfg["embedding_model"]["hub_id"],
        "embedder_local_path": cfg["embedding_model"]["local_path"],
        "embedder_version_tag": version_tag,
        "faiss_index_type": index_type,
        "embedding_dim": dim,
        "normalize_embeddings": cfg["embedding_model"]["normalize_embeddings"],
        "corpus_doc_count": len(docs),
        "corpus_hash": corpus_hash,
        "faiss_ntotal": index.ntotal,
        "build_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "artifact_paths": {
            "faiss_index": str(index_path),
            "id_map": str(id_map_path),
            "manifest": str(manifest_path),
        },
    }

    # -- Write artifacts -----------------------------------------------------
    import faiss

    log.info(f"Saving FAISS index → {index_path}")
    faiss.write_index(index, str(index_path))

    log.info(f"Saving id_map → {id_map_path}")
    with open(id_map_path, "w", encoding="utf-8") as f:
        json.dump(id_map, f, indent=2)

    log.info(f"Saving manifest → {manifest_path}")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    log.info("=== Vector Index Build Complete ===")
    log.info(f"  Version tag     : {version_tag}")
    log.info(f"  Vectors indexed : {index.ntotal}")
    log.info(f"  Corpus hash     : {corpus_hash}")
    log.info(f"  FAISS index     : {index_path}")
    log.info(f"  ID map          : {id_map_path}")
    log.info(f"  Manifest        : {manifest_path}")


# ---------------------------------------------------------------------------
# Startup version safety check (used by retriever at runtime)
# ---------------------------------------------------------------------------

def verify_index_version(cfg: dict) -> bool:
    """
    Called at retriever startup — compares config version_tag with manifest.

    Returns True if safe to proceed.
    Logs an error and returns False if mismatch detected.

    Usage in retriever:
        from retrieval.build_vector_index import verify_index_version
        if not verify_index_version(cfg):
            raise RuntimeError("Index version mismatch — rebuild index before serving")
    """
    version_tag = cfg["embedding_model"]["version_tag"]
    manifest_path = resolve_artifact_path(cfg["artifacts"]["faiss_manifest"], version_tag)

    if not manifest_path.exists():
        log.error(
            f"Index manifest not found for version '{version_tag}': {manifest_path}\n"
            f"Run build_vector_index.py to build the index."
        )
        return False

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    manifest_version = manifest.get("embedder_version_tag")
    if manifest_version != version_tag:
        log.error(
            f"VERSION MISMATCH DETECTED:\n"
            f"  Config version_tag : {version_tag}\n"
            f"  Manifest version   : {manifest_version}\n"
            f"Refusing to serve. Rebuild index with: python retrieval/build_vector_index.py"
        )
        return False

    log.info(
        f"Index version check PASSED: version='{version_tag}' | "
        f"docs={manifest['corpus_doc_count']} | "
        f"built={manifest['build_timestamp']}"
    )
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build FAISS vector index from deduplicated corpus"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing index for this version tag (default: abort if exists)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_vector_index(args)