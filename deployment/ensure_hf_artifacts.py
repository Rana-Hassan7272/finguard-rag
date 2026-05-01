"""
Production helper: fetch retrieval/artifacts from a Hugging Face *Dataset* when
files are missing (e.g. FAISS indexes excluded from Git on Spaces).

Dataset layout must mirror repo paths, e.g.:
  retrieval/artifacts/corpus.jsonl
  retrieval/artifacts/faiss_embedder_v1_hardneg_mnrl.index
  retrieval/artifacts/pdf_faiss.index
  ...

Configure:
  - Secret / env: HF_ARTIFACTS_DATASET (repo id, e.g. hassan7272/finguard-artifacts)
  - Optional: HF_ARTIFACTS_REVISION (branch / commit hash)
  - Or YAML: deployment.hf_artifacts_dataset
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def list_required_artifact_rel_paths(cfg: dict) -> list[Path]:
    """Paths relative to repo root required for pipeline + dual PDF path."""
    art = cfg.get("artifacts") or {}
    v = (cfg.get("embedding_model") or {}).get("version_tag", "")
    keys = [
        "corpus_jsonl",
        "docstore_json",
        "corpus_stats_json",
        "faiss_index",
        "faiss_id_map",
        "faiss_manifest",
        "bm25_tokens",
        "bm25_manifest",
    ]
    out: list[Path] = []
    for key in keys:
        if key not in art:
            continue
        out.append(Path(str(art[key]).replace("{version}", v)))
    if cfg.get("phase6_7", {}).get("enable_dual_retriever", True):
        out.extend(
            [
                Path("retrieval/artifacts/pdf_faiss.index"),
                Path("retrieval/artifacts/pdf_faiss_meta.pkl"),
                Path("retrieval/artifacts/pdf_bm25.pkl"),
            ]
        )
    # de-dup preserve order
    seen: set[str] = set()
    unique: list[Path] = []
    for p in out:
        s = p.as_posix()
        if s not in seen:
            seen.add(s)
            unique.append(p)
    return unique


def ensure_hf_artifacts(repo_root: Path, cfg: dict, logger: Any = log) -> None:
    """
    If any required artifact is missing, download from HF Dataset `HF_ARTIFACTS_DATASET`.
    Raises FileNotFoundError with a clear message if files are missing and no dataset is set.
    """
    root = repo_root.resolve()
    required = list_required_artifact_rel_paths(cfg)
    missing = [p for p in required if not (root / p).is_file()]

    if not missing:
        logger.info("Retrieval artifacts: all required files present.")
        return

    ds = (
        os.environ.get("HF_ARTIFACTS_DATASET")
        or os.environ.get("HF_ARTIFACTS_REPO")
    )
    if not ds:
        dep = cfg.get("deployment") or {}
        ds = dep.get("hf_artifacts_dataset") or dep.get("artifacts_dataset")

    if not ds:
        sample = "\n".join(f"  - {m.as_posix()}" for m in missing[:20])
        extra = "\n  ..." if len(missing) > 20 else ""
        raise FileNotFoundError(
            f"Missing {len(missing)} artifact file(s). Git does not store FAISS indexes.\n"
            f"Either:\n"
            f"  1) Set Space secret HF_ARTIFACTS_DATASET=<org>/<dataset> with files at paths like "
            f"'retrieval/artifacts/…', or\n"
            f"  2) Use Git Xet (HF docs) to store binaries in the Space repo.\n\n"
            f"Missing:\n{sample}{extra}"
        )

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError("Install huggingface_hub (pip install huggingface_hub)") from e

    dep = cfg.get("deployment") or {}
    rev = os.environ.get("HF_ARTIFACTS_REVISION") or dep.get("artifacts_revision")

    logger.info(
        "Downloading %s missing artifact(s) from HF dataset %s …",
        len(missing),
        ds,
    )
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    for rel in missing:
        fn = rel.as_posix()
        hf_hub_download(
            repo_id=ds,
            filename=fn,
            repo_type="dataset",
            revision=rev if rev else None,
            local_dir=str(root),
            token=token,
        )
        logger.info("  fetched %s", fn)

    still = [p for p in required if not (root / p).is_file()]
    if still:
        raise RuntimeError(f"After download, still missing: {[p.as_posix() for p in still]}")
