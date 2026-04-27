from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import yaml


class GateFailReason(str, Enum):
    NO_DOCS = "no_reranked_docs"
    SCORE_BELOW_THRESHOLD = "score_below_threshold"
    EMPTY_DOC_TEXT = "empty_doc_text"


@dataclass
class GateDecision:
    passed: bool
    top_score: float
    threshold: float
    reason: Optional[GateFailReason]
    message: str
    llm_allowed: bool

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "top_score": self.top_score,
            "threshold": self.threshold,
            "reason": self.reason.value if self.reason else None,
            "message": self.message,
            "llm_allowed": self.llm_allowed,
        }


INSUFFICIENT_INFO_RESPONSES = {
    "roman_urdu": "Mujhe is sawal ka jawab nahi mila. Meherbani kar ke aur detail mein poochein.",
    "urdu": "مجھے اس سوال کا مناسب جواب نہیں ملا۔ براہ کرم مزید تفصیل کے ساتھ پوچھیں۔",
    "english": "I don't have enough information to answer this question accurately.",
}


def _load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class ConfidenceGate:
    def __init__(self, config_path: str = "retrieval/configs/retrieval_config.yaml"):
        cfg = _load_config(config_path)
        gate_cfg = cfg.get("reranker", {})
        self.threshold: float = gate_cfg.get("confidence_threshold", 0.40)
        self.fallback_message: str = gate_cfg.get(
            "fallback_message",
            INSUFFICIENT_INFO_RESPONSES["english"],
        )

    def check(
        self,
        reranked_docs: list,
        language: str = "english",
        threshold_override: Optional[float] = None,
    ) -> GateDecision:
        threshold = threshold_override if threshold_override is not None else self.threshold

        if not reranked_docs:
            return GateDecision(
                passed=False,
                top_score=0.0,
                threshold=threshold,
                reason=GateFailReason.NO_DOCS,
                message=INSUFFICIENT_INFO_RESPONSES.get(language, self.fallback_message),
                llm_allowed=False,
            )

        top_doc = reranked_docs[0]
        top_score = (
            top_doc.reranker_score
            if hasattr(top_doc, "reranker_score")
            else top_doc.get("reranker_score", 0.0)
        )

        doc_text = (
            top_doc.doc_text
            if hasattr(top_doc, "doc_text")
            else top_doc.get("doc_text", "")
        )
        if not doc_text or not doc_text.strip():
            return GateDecision(
                passed=False,
                top_score=float(top_score),
                threshold=threshold,
                reason=GateFailReason.EMPTY_DOC_TEXT,
                message=INSUFFICIENT_INFO_RESPONSES.get(language, self.fallback_message),
                llm_allowed=False,
            )

        if float(top_score) < threshold:
            return GateDecision(
                passed=False,
                top_score=float(top_score),
                threshold=threshold,
                reason=GateFailReason.SCORE_BELOW_THRESHOLD,
                message=INSUFFICIENT_INFO_RESPONSES.get(language, self.fallback_message),
                llm_allowed=False,
            )

        return GateDecision(
            passed=True,
            top_score=float(top_score),
            threshold=threshold,
            reason=None,
            message="",
            llm_allowed=True,
        )