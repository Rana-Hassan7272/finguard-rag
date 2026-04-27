"""
retrieval/normalization.py
===========================
Query normalization for the retrieval pipeline.

Responsibilities:
  - Lowercase Latin characters
  - Trim/collapse whitespace
  - Strip punctuation that interferes with retrieval
  - Light Roman Urdu normalization (repeated char collapse, common substitutions)
  - Urdu Unicode normalization (multiple encodings of same letter → canonical)

Design principles:
  - Deterministic — same input always produces same output
  - Light-touch — preserve domain terms; do not over-normalize
  - No external dependencies — pure Python only
  - Fast — runs in microseconds per query

Usage:
    from retrieval.normalization import normalize_query

    clean = normalize_query("kyaaaa zakat ka hisab ???")
    # → "kya zakat ka hisab"

    clean = normalize_query("What  is  Murabaha ?")
    # → "what is murabaha"
"""

import re
import unicodedata
from typing import Optional


# ---------------------------------------------------------------------------
# Urdu Unicode normalization map
# ---------------------------------------------------------------------------
# Some Urdu characters have multiple Unicode encodings due to historical
# inconsistencies in Arabic/Urdu fonts and input methods.
# Map variant forms to canonical forms.

URDU_CHAR_MAP = {
    # Hamza variants
    "\u0624": "\u0648",   # و with hamza → plain و
    "\u0626": "\u06CC",   # ی with hamza → plain ی

    # Alef variants → plain alef
    "\u0622": "\u0627",   # آ → ا
    "\u0623": "\u0627",   # أ → ا
    "\u0625": "\u0627",   # إ → ا
    "\u0671": "\u0627",   # ٱ → ا

    # Kaf variants
    "\u06A9": "\u06A9",   # ک → ک (already canonical, keep)
    "\u0643": "\u06A9",   # ك (Arabic kaf) → ک (Urdu kaf)

    # Ya variants
    "\u0649": "\u06CC",   # ى (Arabic alef maqsura) → ی (Urdu ya)
    "\u064A": "\u06CC",   # ي (Arabic ya) → ی (Urdu ya)

    # He variants
    "\u0647": "\u06C1",   # ه (Arabic he) → ہ (Urdu he) — context-dependent,
                           # acceptable approximation for retrieval

    # Trailing spaces / zero-width chars
    "\u200B": "",          # zero-width space
    "\u200C": "",          # zero-width non-joiner
    "\u200D": "",          # zero-width joiner
    "\uFEFF": "",          # BOM
}

# Urdu diacritics to strip (not meaningful for retrieval matching)
URDU_DIACRITICS = re.compile(
    "[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]"
)


def normalize_urdu_unicode(text: str) -> str:
    """
    Normalize Urdu script to canonical Unicode forms.
    - Map variant characters to canonical equivalents
    - Strip Urdu diacritics (harakat — not used in standard writing)
    - Apply NFC normalization
    """
    # Apply character-level map
    normalized = "".join(URDU_CHAR_MAP.get(ch, ch) for ch in text)

    # Strip diacritics
    normalized = URDU_DIACRITICS.sub("", normalized)

    # NFC normalization
    normalized = unicodedata.normalize("NFC", normalized)

    return normalized


# ---------------------------------------------------------------------------
# Roman Urdu normalization
# ---------------------------------------------------------------------------

# Repeated character collapse: "kyaaaa" → "kyaa", "naiiin" → "naii"
# Keep up to 2 repetitions (handles doubled chars that are meaningful in Urdu)
REPEATED_CHAR_RE = re.compile(r"(.)\1{2,}")  # 3+ same chars → collapse to 2


def collapse_repeated_chars(text: str) -> str:
    """
    Collapse 3+ repeated Latin characters to 2.
    Handles emphatic spelling: "kyaaaa" → "kyaa", "bahuttt" → "bahutt"
    """
    return REPEATED_CHAR_RE.sub(r"\1\1", text)


# Common Roman Urdu spelling variants → canonical form
# Keep this minimal and evidence-based — only add if you observe real variants
ROMAN_URDU_SUBSTITUTIONS = {
    # Common alternate spellings of function words
    "kya": "kya",
    "kyaa": "kya",
    "kaise": "kaise",
    "kaisay": "kaise",
    "kaisa": "kaise",    # gender variant flattened for retrieval
    "hai": "hai",
    "hain": "hain",
    "hy": "hai",         # SMS-style shorthand
    "hei": "hai",
    "ka": "ka",
    "ki": "ki",
    "ke": "ke",
    "ko": "ko",
    "se": "se",
    "mein": "mein",
    "main": "mein",      # common alternate spelling
    "men": "mein",
    "me": "mein",
    "aur": "aur",
    "or": "aur",         # "or" often means "aur" in Roman Urdu context
    "nahi": "nahi",
    "nahin": "nahi",
    "nhn": "nahi",
    # Finance-specific Roman Urdu
    "hisab": "hisab",
    "hisaab": "hisab",
    "paise": "paise",
    "paisa": "paise",
    "paisay": "paise",
    "krna": "karna",
    "krein": "karein",
    "kren": "karein",
}


def apply_roman_urdu_substitutions(tokens: list[str]) -> list[str]:
    """Apply canonical substitutions to Roman Urdu tokens."""
    return [ROMAN_URDU_SUBSTITUTIONS.get(tok, tok) for tok in tokens]


# ---------------------------------------------------------------------------
# Core normalization pipeline
# ---------------------------------------------------------------------------

def normalize_query(
    text: str,
    apply_urdu_unicode: bool = True,
    apply_roman_urdu: bool = True,
    lowercase_latin: bool = True,
) -> str:
    """
    Main normalization entrypoint for all query types.

    Pipeline (in order):
      1. Strip leading/trailing whitespace
      2. Urdu Unicode normalization (variant chars → canonical)
      3. Urdu diacritics removal
      4. Lowercase Latin characters (Urdu script is not case-sensitive)
      5. Punctuation cleanup (remove query-terminal punctuation)
      6. Repeated character collapse (Roman Urdu emphasis: kyaaaa → kyaa)
      7. Roman Urdu token substitutions
      8. Whitespace normalization (collapse multiple spaces)

    Parameters
    ----------
    text               : raw query string (any language/script)
    apply_urdu_unicode : run Urdu char normalization (default: True)
    apply_roman_urdu   : run Roman Urdu substitution map (default: True)
    lowercase_latin    : lowercase Latin chars (default: True)

    Returns
    -------
    Normalized query string — same script, lighter form
    """
    if not text:
        return ""

    # Step 1: Strip
    text = text.strip()

    # Step 2+3: Urdu Unicode normalization
    if apply_urdu_unicode:
        text = normalize_urdu_unicode(text)

    # Step 4: Lowercase Latin only
    # We can't simply .lower() the whole string — that would corrupt Urdu chars
    if lowercase_latin:
        text = "".join(ch.lower() if ch.isascii() else ch for ch in text)

    # Step 5: Remove trailing query punctuation (?, !, ۔, ؟, ،)
    # Keep internal punctuation (e.g. "Meezan Bank, HBL" — the comma is informative)
    text = re.sub(r"[?!۔؟،,\.]+\s*$", "", text).strip()

    # Step 6: Repeated char collapse (Latin only — Urdu chars handled separately)
    # Apply only to Latin portions to avoid corrupting Urdu script
    parts = []
    for segment in re.split(r"(\s+)", text):
        if segment.isascii():
            segment = collapse_repeated_chars(segment)
        parts.append(segment)
    text = "".join(parts)

    # Step 7: Roman Urdu token substitutions
    if apply_roman_urdu:
        tokens = text.split()
        tokens = apply_roman_urdu_substitutions(tokens)
        text = " ".join(tokens)

    # Step 8: Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ---------------------------------------------------------------------------
# Batch normalization
# ---------------------------------------------------------------------------

def normalize_queries(texts: list[str], **kwargs) -> list[str]:
    """Normalize a list of queries. Kwargs passed to normalize_query."""
    return [normalize_query(t, **kwargs) for t in texts]


MINIMAL_EN_STOPWORDS = {
    "the", "a", "an", "is", "are", "am", "to", "of", "in", "on", "for",
    "and", "or", "with", "from", "at", "by", "as", "this", "that",
}


def tokenize(
    text: str,
    tokenizer_mode: str = "whitespace",
    stopword_mode: str = "minimal",
) -> list[str]:
    """
    Tokenize a normalized query/document for BM25.
    Keeps behavior deterministic and aligned between indexing and retrieval.
    """
    normalized = normalize_query(text)
    if not normalized:
        return []

    if tokenizer_mode == "whitespace":
        tokens = normalized.split()
    else:
        # Fallback tokenizer: keep Urdu + Latin + digits.
        tokens = re.findall(r"[\u0600-\u06FFA-Za-z0-9]+", normalized)

    if stopword_mode == "minimal":
        tokens = [t for t in tokens if t not in MINIMAL_EN_STOPWORDS]
    elif stopword_mode == "none":
        pass
    else:
        # Unknown stopword mode -> fail open without dropping tokens.
        tokens = tokens

    return tokens


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_cases = [
        # (input, expected_output)
        ("kyaaaa zakat ka hisab ???", "kya zakat ka hisab"),
        ("What  is  Murabaha ?", "what is murabaha"),
        ("  easypaisa se paise  kaise   bhejein  ", "easypaisa se paise kaise bhejein"),
        ("loan ki qist kitni hogi!", "loan ki qist kitni hogi"),
        ("meezan bank main account kholna hai hy", "meezan bank mein account kholna hai hai"),
        ("زکوٰۃ کا حساب کیسے کریں؟", "زکوٰۃ کا حساب کیسے کریں"),
        ("ZAKAT KA HISAB", "zakat ka hisab"),
        ("", ""),
        ("riba kya hein???", "riba kya hain"),
        ("savings pe profit milta hein ya nhn", "savings pe profit milta hain ya nahi"),
    ]

    print(f"\n{'Input':<50} {'Output'}")
    print("-" * 90)
    all_passed = True
    for inp, expected in test_cases:
        result = normalize_query(inp)
        ok = "✅" if result == expected else "❌"
        if result != expected:
            all_passed = False
        print(f"{repr(inp):<50} {repr(result)}  {ok}")
        if result != expected:
            print(f"  expected: {repr(expected)}")

    print(f"\n{'All passed ✅' if all_passed else 'Some failed ❌'}")