import re
import unicodedata
import yaml
from pathlib import Path

CONFIG_PATH = Path("retrieval/configs/retrieval_config.yaml")

LEGAL_KEYWORDS = {
    "law", "act", "regulation", "section", "sbp", "secp", "ordinance", "clause",
    "statute", "schedule", "circular", "directive", "rule", "provision", "subsection",
    "qanoon", "dafa", "zabita", "ain", "hukm",
}

ROMAN_URDU_LEGAL_KEYWORDS = {
    "qanoon", "dafa", "zabita", "ain", "hukm", "shart", "shaq",
}

URDU_LEGAL_KEYWORDS = {
    "قانون", "دفعہ", "ضابطہ", "آئین", "حکم", "شرط", "شق", "ایکٹ", "ریگولیشن",
}

REPEATED_CHAR_PATTERN = re.compile(r"(.)\1{2,}")
WHITESPACE_PATTERN = re.compile(r"\s+")
PUNCTUATION_NOISE_PATTERN = re.compile(r"[^\w\s\u0600-\u06FF?]")

URDU_UNICODE_VARIANTS = {
    "\u06CC": "\u06CC",
    "\u0649": "\u06CC",
    "\u06C1": "\u06C1",
    "\u06BE": "\u06BE",
    "\u0643": "\u06A9",
    "\u064A": "\u06CC",
}


def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def detect_language(text: str) -> str:
    urdu_chars = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
    total_chars = len([c for c in text if not c.isspace()])
    if total_chars == 0:
        return "english"
    urdu_ratio = urdu_chars / total_chars
    if urdu_ratio > 0.4:
        return "urdu"
    roman_urdu_markers = {
        "hai", "hain", "kya", "kyaa", "nahi", "nahin", "aur", "ka", "ki", "ke",
        "mein", "se", "ko", "ne", "par", "wala", "wali", "kaise", "kitna",
        "zakat", "riba", "halal", "haram", "loan", "qarz",
    }
    words = set(text.lower().split())
    if words & roman_urdu_markers:
        return "roman_urdu"
    return "english"


def normalize_urdu_unicode(text: str) -> str:
    result = []
    for ch in text:
        result.append(URDU_UNICODE_VARIANTS.get(ch, ch))
    return "".join(result)


def normalize_query(text: str) -> str:
    has_urdu = any("\u0600" <= c <= "\u06FF" for c in text)
    if has_urdu:
        text = normalize_urdu_unicode(text)
        text = unicodedata.normalize("NFC", text)
    else:
        text = text.lower()
        text = REPEATED_CHAR_PATTERN.sub(lambda m: m.group(1) * 2, text)
    text = PUNCTUATION_NOISE_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


def detect_query_intent(text: str, language: str) -> str:
    words = set(text.lower().split())
    if language == "urdu":
        if words & URDU_LEGAL_KEYWORDS:
            return "legal"
    elif language == "roman_urdu":
        if words & ROMAN_URDU_LEGAL_KEYWORDS:
            return "legal"
        if words & LEGAL_KEYWORDS:
            return "legal"
    else:
        if words & LEGAL_KEYWORDS:
            return "legal"
    return "practical"


def get_source_routing_weights(query_intent: str, config: dict) -> tuple[float, float]:
    routing = config.get("source_routing", {})
    if query_intent == "legal":
        weights = routing.get("legal_query", {"qa": 0.3, "pdf": 0.7})
    elif query_intent == "practical":
        weights = routing.get("practical_query", {"qa": 0.7, "pdf": 0.3})
    else:
        weights = routing.get("default", {"qa": 0.5, "pdf": 0.5})
    return weights["qa"], weights["pdf"]


def route_query(raw_query: str) -> dict:
    config = load_config()
    normalized = normalize_query(raw_query)
    language = detect_language(normalized)
    intent = detect_query_intent(normalized, language)
    qa_weight, pdf_weight = get_source_routing_weights(intent, config)
    return {
        "raw_query": raw_query,
        "normalized_query": normalized,
        "language": language,
        "query_intent": intent,
        "qa_weight": qa_weight,
        "pdf_weight": pdf_weight,
    }