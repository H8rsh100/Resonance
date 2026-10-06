"""Devanagari-aware text normalisation and script/language detection."""
from __future__ import annotations

import re
import unicodedata

_DEVANAGARI = re.compile(r"[\u0900-\u097F]")
_LATIN = re.compile(r"[A-Za-z]")
_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\ufeff]")
_PUNCT = re.compile(r"[^\w\s\u0900-\u097F]", re.UNICODE)
_WS = re.compile(r"\s+")
_REPEAT_CHAR = re.compile(r"(.)\1{2,}")

# Romanised-Hindi markers: high-frequency function words that do not appear in
# ordinary English telecom or banking scripts.
_HINGLISH_MARKERS = {
    "hai", "nahi", "nahin", "kya", "kar", "karo", "karke", "aap", "aapka",
    "mera", "meri", "tum", "tumhe", "abhi", "sir", "ji", "mat", "dena",
    "kripya", "paisa", "paise", "bhej", "bhejo", "rupaye", "mujhe", "tumhe",
}


def normalize(text: str) -> str:
    """NFC-normalise, strip zero-width chars and collapse repeated punctuation."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = _ZERO_WIDTH.sub("", text)
    text = text.replace("।", " ").replace("॥", " ")
    text = _REPEAT_CHAR.sub(r"\1\1", text)
    text = _PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip().lower()


def script_of(text: str) -> str:
    """Return the dominant script tag for a piece of text."""
    dev = len(_DEVANAGARI.findall(text))
    lat = len(_LATIN.findall(text))
    if dev == 0 and lat == 0:
        return "unknown"
    return "devanagari" if dev >= lat else "latin"


def detect_language(text: str) -> str:
    """Coarse EN / HI / MR / Hinglish detection from script plus function words.

    Hindi and Marathi are not separable from orthography alone (they share the
    Devanagari block and script), so the Hinglish lexicon is the tie-breaker:
    Marathi loan verbs such as 'kar', 'karo' are absent from the Hindi set and
    Marathi-specific forms ('aah', 'tumcha') are not in it either.
    """
    text = normalize(text)
    script = script_of(text)
    if script == "latin":
        tokens = set(re.findall(r"[a-z]+", text))
        overlap = len(tokens & _HINGLISH_MARKERS)
        return "hinglish" if overlap >= 3 else "en"

    dev_chars = _DEVANAGARI.findall(text)
    if not dev_chars:
        return "unknown"
    blob = "".join(dev_chars)
    # Marathi and Hindi share the Devanagari block, so script alone cannot separate
    # them. Score the two marker sets and take the stronger one.
    mr_markers = ("आहे", "आहेत", "दाखल", "काटू", "बोलतो", "तुमच्यावर", "तुमची",
                  "तुम्ही", "जमा करा", "पाठवा", "मिळतील", "करू", "सांगू", "माफत",
                  "पूर्ण करा", "सहकार्य", "खोली", "कोर्टात", "वेळ")
    hi_markers = ("है", "हैं", "कीजिए", "करना", "भेजिए", "दिया", "गया", "गए",
                  "काटिए", "बताइए", "कर दें", "भेज दें", "होना", "साथ लाइए",
                  "खारेज़ी", "अदालत में", "समय नहीं")
    mr_score = sum(2 if len(m) > 4 else 1 for m in mr_markers if m in blob)
    hi_score = sum(2 if len(m) > 4 else 1 for m in hi_markers if m in blob)
    if mr_score == hi_score == 0:
        return "hi"
    return "mr" if mr_score > hi_score else "hi"


def snippet(text: str, window: int = 90) -> str:
    text = " ".join(text.split())
    return text if len(text) <= window else text[:window].rstrip() + "..."