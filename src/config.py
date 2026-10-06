"""Central configuration for the Resonance digital-arrest scam detection system."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_EXTERNAL = ROOT / "data" / "external"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

for _p in (DATA_RAW, DATA_PROCESSED, DATA_EXTERNAL, ARTIFACTS, REPORTS, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

SEED = 42

LABEL_SCAM = 1
LABEL_LEGIT = 0
LABEL_NAMES = {LABEL_LEGIT: "legitimate", LABEL_SCAM: "digital_arrest_scam"}

LANGUAGES = ("en", "hi", "mr", "hinglish")

# Five indicator families from the proposed methodology (Section B/C).
INDICATORS = (
    "authority_impersonation",
    "legal_threat",
    "urgency",
    "isolation_secrecy",
    "financial_coercion",
)

# Risk thresholds tuned in Phase 1. High recall on scam, very low false positives,
# because the hackathon brief flags false-positive rate as the citizen-facing risk.
RISK_THRESHOLDS = {"low": 0.35, "medium": 0.65}

EXTERNAL_DATASET = "dbarbedillo/SMS_Spam_Multilingual_Collection_Dataset"
NCRB_URL = "https://cybercrime.gov.in/"
CYBER_HELPLINE = "1930"