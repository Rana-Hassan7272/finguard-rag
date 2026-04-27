"""
ingestion/infer_category.py

Assigns a category label to PDF chunks using keyword mapping.
Works on both filename hints and chunk text content.
Returns the winning category and confidence, or None if no match.
"""

import re
from dataclasses import dataclass
from typing import Optional


CATEGORY_KEYWORD_MAP: dict[str, list[tuple[str, float]]] = {
    "islamic_finance": [
        ("zakat", 1.0), ("zakah", 1.0), ("nisab", 1.0),
        ("riba", 1.0), ("sood", 0.9), ("halal", 0.8), ("haram", 0.8),
        ("murabaha", 1.0), ("musharakah", 1.0), ("ijarah", 1.0),
        ("takaful", 1.0), ("sukuk", 1.0), ("waqf", 0.9),
        ("shariah", 0.9), ("sharia", 0.9), ("islamic banking", 1.0),
        ("islamic finance", 1.0), ("interest free", 0.9),
        ("profit sharing", 0.8), ("fitrana", 0.9), ("ushr", 0.9),
        ("jurisprudence", 0.8), ("fiqh", 0.9),
        ("زکوٰۃ", 1.0), ("سود", 1.0), ("حلال", 0.8),
    ],
    "banking": [
        ("account opening", 0.9), ("current account", 0.9),
        ("savings account", 0.9), ("bank account", 0.8),
        ("kyc", 0.8), ("cnic", 0.7), ("cheque", 0.8),
        ("mobile banking", 0.9), ("internet banking", 0.9),
        ("atm", 0.7), ("debit card", 0.8), ("bank statement", 0.8),
        ("consumer banking", 0.9), ("retail banking", 0.8),
        ("banking policy", 0.9), ("sbp", 0.8),
        ("allied bank", 0.9), ("askari bank", 0.9),
        ("nbp", 0.8), ("hbl", 0.7), ("ubl", 0.7),
        ("asaan karobar", 1.0), ("karobar act", 1.0),
    ],
    "digital_finance": [
        ("easypaisa", 1.0), ("jazzcash", 1.0), ("jazz cash", 1.0),
        ("raast", 0.9), ("ibft", 0.9), ("mobile wallet", 0.9),
        ("digital payment", 0.9), ("digital wallet", 0.9),
        ("virtual asset", 0.9), ("cryptocurrency", 0.9),
        ("crypto", 0.8), ("blockchain", 0.8), ("token", 0.6),
        ("virtual assets ordinance", 1.0), ("virtual assets act", 1.0),
        ("send money", 0.7), ("payment gateway", 0.8),
    ],
    "financial_education": [
        ("income tax", 1.0), ("withholding tax", 1.0), ("tax return", 1.0),
        ("fbr", 1.0), ("federal board of revenue", 1.0),
        ("ntn", 0.9), ("filer", 0.9), ("non-filer", 0.9),
        ("tax deduction", 0.9), ("sales tax", 0.9), ("gst", 0.8),
        ("tax exemption", 0.9), ("tax filing", 0.9), ("iris", 0.8),
    ],
    "investment": [
        ("mutual fund", 1.0), ("stock exchange", 1.0), ("psx", 1.0),
        ("secp", 0.9), ("national savings", 0.9), ("prize bond", 1.0),
        ("treasury bill", 0.9), ("t-bill", 0.9), ("pib", 0.8),
        ("investment", 0.6), ("portfolio", 0.7), ("dividend", 0.8),
        ("capital gain", 0.9), ("equity", 0.7), ("behbood", 0.9),
        ("pensioner", 0.8), ("rda", 0.8), ("roshan digital", 0.9),
        ("pakistan investment", 0.9), ("stock market", 0.9),
    ],
    "insurance": [
        ("insurance", 0.9), ("takaful", 0.9), ("life insurance", 1.0),
        ("health insurance", 1.0), ("vehicle insurance", 1.0),
        ("premium", 0.6), ("claim", 0.6), ("policyholder", 0.8),
        ("insurer", 0.8), ("coverage", 0.7),
    ],
    "personal_finance": [
        ("remittance", 1.0), ("foreign remittance", 1.0),
        ("overseas pakistani", 0.9), ("swift", 0.9), ("iban", 0.9),
        ("western union", 1.0), ("moneygram", 1.0),
        ("roshan digital account", 1.0), ("nrp", 0.8),
        ("exchange rate", 0.8), ("forex", 0.8),
    ],
}

SATURATION_SCORE = 2.5


@dataclass
class CategoryInferenceResult:
    category: Optional[str]
    confidence: float
    matched_keywords: list[str]
    all_scores: dict[str, float]


def infer_category(
    text: str,
    filename_hint: Optional[str] = None,
) -> CategoryInferenceResult:
    combined = text.lower()
    if filename_hint:
        combined = filename_hint.lower() + " " + combined

    scores: dict[str, float] = {}
    matches: dict[str, list[str]] = {}

    for category, kw_list in CATEGORY_KEYWORD_MAP.items():
        score = 0.0
        matched = []
        sorted_kws = sorted(kw_list, key=lambda x: len(x[0]), reverse=True)

        for keyword, weight in sorted_kws:
            kl = keyword.lower()
            if " " in kl:
                if kl in combined:
                    score += weight
                    matched.append(keyword)
            else:
                is_urdu = any(0x0600 <= ord(c) <= 0x06FF for c in keyword)
                if is_urdu:
                    if keyword in text:
                        score += weight
                        matched.append(keyword)
                else:
                    pattern = r"(?<![a-z])" + re.escape(kl) + r"(?![a-z])"
                    if re.search(pattern, combined):
                        score += weight
                        matched.append(keyword)

        scores[category] = round(score, 4)
        matches[category] = matched

    if not scores or max(scores.values()) == 0.0:
        return CategoryInferenceResult(None, 0.0, [], scores)

    best = max(scores, key=lambda c: scores[c])
    best_score = scores[best]
    confidence = round(min(1.0, best_score / SATURATION_SCORE), 4)

    return CategoryInferenceResult(
        category=best,
        confidence=confidence,
        matched_keywords=matches[best],
        all_scores=scores,
    )


def infer_category_batch(
    chunks: list[dict],
    text_field: str = "retrieval_text",
) -> list[dict]:
    for chunk in chunks:
        result = infer_category(
            text=chunk.get(text_field, ""),
            filename_hint=chunk.get("source_file"),
        )
        chunk["category"] = result.category or "general"
        chunk["category_confidence"] = result.confidence
        chunk["category_matched"] = result.matched_keywords
    return chunks