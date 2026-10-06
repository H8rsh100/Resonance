"""Rule-based scam-indicator extraction.

The classifier gives a probability; this module gives the *evidence*. It is the
interpretability layer described in the proposed methodology (Section D/E) and is
what turns a score into an actionable citizen warning.

Five indicator families, each with English, Hindi, Marathi and romanised-Hinglish
cues. Phrases are matched on normalised text with word-boundary regexes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import INDICATORS
from .preprocess import normalize

LEXICON: dict[str, list[str]] = {
    "authority_impersonation": [
        # English
        r"cbi", r"central bureau of investigation", r"\bed\b", r"enforcement directorate",
        r"customs (?:officer|officials|department)", r"income tax (?:department|officer)",
        r"cyber ?cell", r"crime branch", r"police (?:station|department|officer)",
        r"rbi", r"reserve bank", r"narcotics control", r"special cell",
        r"calling (?:from|on behalf of) (?:the )?(?:cbi|ed|customs|police)",
        r"government (?:officer|official|department)",
        # Hindi
        r"सीबीआई", r"प्रवर्तन निदेशालय", r"ईडी", r"सीमा शुल्क", r"आयकर विभाग",
        r"साइबर सेल", r"अपराध शाखा", r"अपने केंद्रीय बैंक", r"सरकारी अधिकारी",
        # Marathi
        r"सीबीआय मधून", r"सीमा शुल्क", r"गुन्हे शाखा", r"सरकारी अधिकारी",
        # Hinglish
        r"\bsehakari\b", r"sarkari officer", r"cbi se", r"\bed se\b",
    ],
    "legal_threat": [
        r"\bfir\b", r"case number", r"\bsummons?\b", r"\bwarrant\b", r"arrest warrant",
        r"money laundering", r"\bsection 420\b", r"\b420\b", r"\bipc\b", r"it act",
        r"criminal case", r"ncbi", r"interpol", r"\bprosecution\b",
        r"you are (?:under|on) (?:investigation|inquiry)", r"charges? (?:have been )?filed",
        r"penalty notice", r"show cause notice", r"notice under section",
        # Hindi
        r"प्राथमिकी", r"केस\s*संख्या", r"मुकदमा", r"गिरफ्तारी वारंट", r"मनी लॉन्ड्रिंग",
        r"धोखाधड़ी का मामला", r"खारेज़ी", r"अदालत", r"अभियुक्त",
        # Marathi
        r"फिरोदी", r"खोली", r"गुन्हा दाखल", r"पोलिस वारंट", r"कोर्ट",
        # Hinglish
        r"mONEY LAUNDERING case", r"under investigation", r"legal notice",
    ],
    "urgency": [
        r"do not (?:disconnect|hang up|end the call)", r"don'?t (?:disconnect|hang up)",
        r"do not tell anyone", r"within (\d+) hours?", r"right now", r"immediately",
        r"this call is being recorded", r"call is being recorded", r"stay on the line",
        r"before (?:they|i) (?:trace|arrest|catch)", r"last chance", r"final warning",
        r"act (?:now|immediately)", r"time is running out",
        # Hindi
        r"फोन\s*मत\s*काट", r"कॉल मत काटिए", r"अभी", r"तुरंत", r"इसी समय",
        r"दर्ज कर (?:ले|रहे) (?:है|जा)", r"कॉल रिकॉर्ड हो रही है", r"समय नहीं है",
        # Marathi
        r"फोन काटू नका", r"लगेच", r"आत्ताच", r"नोंदवली जात आहे", r"वेळ नाही",
        # Hinglish
        r"phone mat kat", r"abhi abhi", r"jaldi se", r"turant",
    ],
    "isolation_secrecy": [
        r"do not tell (?:your )?(?:family|wife|husband|parents|spouse|anyone)",
        r"don'?t tell (?:your )?(?:family|wife|husband|parents|spouse|anyone)",
        r"keep (?:this|it) (?:call|confidential|secret|between us)",
        r"stay away from", r"do not speak to", r"this is confidential",
        r"you are (?:in|under) (?:trouble|danger)", r"you cannot sleep",
        r"you must comply", r"not your fault", r"you are being framed",
        # Hindi
        r"अपनी (?:पत्नी|पति|परिवार|माता|पिता) को (?:मत|न) बत",
        r"किसी को (?:मत|न) बत", r"यह बात गोपनीय", r"किसी को पता नहीं",
        r"आप फंसे हुए हैं", r"आप मुसीबत में हैं",
        # Marathi
        r"कुटुंबाला सांगू नका", r"कोणाला सांगू नका", r"गोपनीय",
        r"तुम्ही अडचणीत आहात", r"कोणाचा फोन घ्या",
    ],
    "financial_coercion": [
        r"transfer (?:the )?(?:money|funds|amount)", r"deposit (?:the )?money",
        r"upi (?:id|handle)", r"qr code", r"scan (?:this|the) (?:qr|code)",
        r"gift card", r"crypto ?(?:currency|wallet)", r"usdt", r"bitcoin",
        r"demanding ?draft", r"pay (?:the )?(?:fine|penalty|amount|charge)s?",
        r"release (?:your|him|her|the) (?:funds|money)",
        r"to (?:avoid|prevent) arrest", r"otherwise you will be arrested",
        r"security deposit", r"registration (?:fee|charge)", r"clearance certificate",
        # Hindi
        r"पैसे (?:भेज|ट्रांसफर|जमा)", r"रुपये (?:भेज|जमा)", r"यूपीआई",
        r"क्यूआर कोड", r"गिफ्ट कार्ड", r"ज़राना भर", r"जमानत",
        # Marathi
        r"पैसे (?:पाठव|ट्रान्सफर)", r"युपीआय", r"क्युआर कोड", r"इतर",
        # Hinglish
        r"paisa (?:bhej|bhejo|bhejo abhi)", r"amount transfer", r"UPI id de",
        r"QR se scan",
    ],
}

_COMPILED = {
    fam: [re.compile(p) for p in pats] for fam, pats in LEXICON.items()
}


@dataclass
class IndicatorHit:
    family: str
    evidence: str
    weight: int = 1


@dataclass
class IndicatorReport:
    hits: list[IndicatorHit] = field(default_factory=list)

    @property
    def families(self) -> list[str]:
        seen: list[str] = []
        for h in self.hits:
            if h.family not in seen:
                seen.append(h.family)
        return seen

    @property
    def score(self) -> int:
        return sum(h.weight for h in self.hits)

    def as_dict(self) -> dict:
        out: dict[str, list[str]] = {}
        for h in self.hits:
            out.setdefault(h.family, [])
            if h.evidence not in out[h.family]:
                out[h.family].append(h.evidence)
        return out


def _evidence(text: str, match: re.Match, pad: int = 34) -> str:
    start = max(0, match.start() - pad)
    end = min(len(text), match.end() + pad)
    return ("..." if start > 0 else "") + text[start:end].strip() + ("..." if end < len(text) else "")


def detect(text: str) -> IndicatorReport:
    """Extract scam indicators with surrounding evidence snippets."""
    norm = normalize(text)
    report = IndicatorReport()
    for family in INDICATORS:
        for pattern in _COMPILED[family]:
            m = pattern.search(norm)
            if m:
                weight = 2 if len(m.group(0).split()) > 1 else 1
                report.hits.append(IndicatorHit(family, _evidence(norm, m, weight), weight))
    return report