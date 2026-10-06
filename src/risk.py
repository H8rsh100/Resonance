"""Risk mapping and citizen-facing advisory text.

Owner: member 46. Kept separate from the classifier so the advisory language and
the thresholds can change without retraining. Guidance is available in English,
Hindi and Marathi to match the multilingual scope of the project.
"""
from __future__ import annotations

from .config import CYBER_HELPLINE, NCRB_URL, RISK_THRESHOLDS

FAMILY_LABELS = {
    "authority_impersonation": {
        "en": "Authority impersonation",
        "hi": "सरकारी अधिकारी का प्रतिरूपण",
        "mr": "सरकारी अधिकाऱ्याचा नक्कल",
    },
    "legal_threat": {
        "en": "Legal or police threat",
        "hi": "कानूनी या पुलिस धमकी",
        "mr": "कायदेशीर किंवा पोलिस धमकी",
    },
    "urgency": {
        "en": "Artificial urgency",
        "hi": "जल्दबाजी का दबाव",
        "mr": "घाईचा दबाव",
    },
    "isolation_secrecy": {
        "en": "Demands secrecy / isolation",
        "hi": "रहस्य और अकेले रहने का दबाव",
        "mr": "गोपनीयता आणि एकटे बसण्याचा दबाव",
    },
    "financial_coercion": {
        "en": "Pressure to pay or transfer money",
        "hi": "पैसे भेजने का दबाव",
        "mr": "पैसे पाठवण्याचा दबाव",
    },
}

GUIDANCE = {
    "HIGH": {
        "en": [
            "End the call now. Do not stay on the line, even if told to.",
            "Do not transfer money, scan any QR code, or share an OTP / PIN / CVV.",
            "No real government agency asks for payment over a phone call to avoid arrest.",
            "Call back on the number printed on your bank card to verify independently.",
            "Report at cybercrime.gov.in or call the Cyber Crime Helpline 1930.",
        ],
        "hi": [
            "अभी कॉल समाप्त करें। कहने पर भी लाइन पर न रुकें।",
            "पैसे ट्रांसफर न करें, कोई QR कोड स्कैन न करें, OTP / पिन / CVV साझा न करें।",
            "गिरफ्तारी से बचने के लिए कोई सरकारी कार्यालय फोन पर पैसे नहीं मांगता।",
            "अपने बैंक कार्ड पर छपे नंबर से खुद कॉल करके जांच करें।",
            "cybercrime.gov.in पर शिकायत करें या साइबर क्राइम हेल्पलाइन 1930 पर कॉल करें।",
        ],
        "mr": [
            "आत्ता कॉल संपवा. सांगितले असेल तरीही लाइनवर राहू नका.",
            "पैसे ट्रान्सफर करू नका, QR कोड स्कॅन करू नका, OTP / पिन / CVV शेअर करू नका.",
            "अटक टाळण्यासाठी कोणतीही सरकारी कार्यालय फोनवर पैसे मागत नाही.",
            "बँक कार्डवर छापलेल्या नंबरवरून स्वतः फोन करून तपासा.",
            "cybercrime.gov.in वर तक्रार नोंदवा किंवा 1930 या सायबर क्राइम हेल्पलाइनवर फोन करा.",
        ],
    },
    "MEDIUM": {
        "en": [
            "Treat this call as suspicious. Do not act on any payment request.",
            "Genuine banks and agencies never demand an OTP over a call.",
            "Hang up and verify through the official app or a number you already trust.",
            "If anyone asked you to stay on the line, that alone is a strong scam signal.",
        ],
        "hi": [
            "इस कॉल को संदिग्ध मानें। किसी भी भुगतान अनुरोध पर कार्रवाई न करें।",
            "असली बैंक या कार्यालय फोन पर OTP नहीं मांगते।",
            "कॉल काटें और आधिकारिक ऐप या भरोसेमंद नंबर से जांच करें।",
            "यदि कोई लाइन पर बने रहने को कहे, तो यह अपने आप में बड़ा संकेत है।",
        ],
        "mr": [
            "या कॉलला संशयास्पद माना. कोणत्याही पेमेंट विनंतीवर कृती करू नका.",
            "खरे बँक किंवा कार्यालय फोनवर OTP मागत नाही.",
            "कॉल काटा आणि अधिकृत अॅप किंवा विश्वासार्ह नंबरवरून तपासा.",
            "कोणी लाइनवर राहायला सांगितले असेल, तर हेच मोठा संकेत आहे.",
        ],
    },
    "LOW": {
        "en": [
            "No strong scam indicators were detected in this text.",
            "Still never share an OTP or PIN, and never transfer money on a cold call.",
            "If the caller claims to be from a bank or the government, hang up and call the official number yourself.",
        ],
        "hi": [
            "इस टेक्स्ट में कोई मजबूत संदिग्ध संकेत नहीं मिले।",
            "फिर भी OTP या पिन साझा न करें, और अचानक आई कॉल पर पैसे न भेजें।",
            "यदि कॉल करने वाला बैंक या सरकार बताए, तो कॉल काटें और खुद आधिकारिक नंबर डालें।",
        ],
        "mr": [
            "या मजकुरात मजबूत संशयास्पद संकेत आढळले नाहीत.",
            "तरीही OTP किंवा पिन शेअर करू नका, आणि अचानक फोनवर पैसे पाठवू नका.",
            "कॉल करणारा बँक किंवा सरकार असल्यास कॉल काटा आणि स्वतः अधिकृत नंबर वापरा.",
        ],
    },
}


def risk_level(prob: float) -> str:
    if prob >= RISK_THRESHOLDS["medium"]:
        return "HIGH"
    if prob >= RISK_THRESHOLDS["low"]:
        return "MEDIUM"
    return "LOW"


def label_for(family: str, lang: str) -> str:
    return FAMILY_LABELS.get(family, {}).get(lang, family)


def guidance(level: str, lang: str) -> list[str]:
    return GUIDANCE.get(level, GUIDANCE["LOW"]).get(lang, GUIDANCE["LOW"]["en"])


def report_channels() -> tuple[str, str]:
    return NCRB_URL, CYBER_HELPLINE