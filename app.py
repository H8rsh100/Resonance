"""Citizen Fraud Shield - Streamlit demo for digital-arrest scam detection.

Run:  streamlit run app.py

Owner: member 46 (app shell + advisory), built on member 36's text branch and
member 34's indicator lexicons.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import REPORTS  # noqa: E402
from src.indicators import detect  # noqa: E402
from src.models.text_model import TextScamClassifier  # noqa: E402
from src.preprocess import detect_language  # noqa: E402
from src.risk import guidance, label_for, report_channels, risk_level  # noqa: E402

st.set_page_config(page_title="Citizen Fraud Shield", page_icon="🛡️", layout="wide")

SAMPLES = {
    "Digital arrest call (EN)": "Hello sir, this is Senior Inspector Rao calling from the Central Bureau of Investigation. A money laundering case has been registered against your name. You must deposit the amount today. Do not disconnect, this call is being recorded.",
    "Digital arrest call (HI)": "नमस्ते, मैं सीबीआई से बोल रहा हूँ, आपके खिलाफ मनी लॉन्ड्रिंग का मामला दर्ज किया गया है। फोन मत काटिए, कॉल रिकॉर्ड हो रही है। आपको तुरंत पैसे भेजने होंगे।",
    "Digital arrest call (MR)": "नमस्कार, मी सीबीआय मधून बोलतो, तुमच्यावर पैसेाचा गुन्हा दाखल आहे. फोन काटू नका, कॉल नोंदवली जात आहे. आत्ताच पैसे पाठवा.",
    "Family impersonation (EN)": "Papa, papa I am in trouble, please send 50000 immediately. Do not tell mummy about this, please.",
    "Genuine bank call (EN)": "Hello, this is HDFC Bank calling from your registered number. A transaction of 2500 rupees was declined at the merchant. Do you want to update your limit?",
    "Genuine bank call (HI)": "नमस्ते, मैं एचडीएफसी बैंक से बोल रहा हूँ, आपका कार्ड अस्थायी रूप से ब्लॉक कर दिया गया है, कृपया अपना ओटीपी बताइए।",
    "Genuine utility call (MR)": "नमस्कार, हे महाराष्ट्र राज्य विद्युत वितरण कंपनीचा कॉल आहे, तुमचे बिल 1250 रुपये आहे, 20 तारीखपर्यंत जमा करा.",
}

COLORS = {"HIGH": "#c0392b", "MEDIUM": "#e67e22", "LOW": "#1e8449"}


@st.cache_resource
def get_model():
    try:
        return TextScamClassifier.load()
    except Exception:
        return None


def render_verdict(text: str, clf: TextScamClassifier) -> None:
    prob = float(clf.proba([text])[0])
    level = risk_level(prob)
    lang = detect_language(text)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Scam risk score", f"{prob:.0%}")
    c2.metric("Risk level", level)
    c3.metric("Language detected", {"hinglish": "Hinglish", "unknown": "n/a"}.get(lang, lang.upper()))
    c4.metric("Operating threshold", f"{clf.threshold:.2f}")

    st.markdown(
        f"<div style='padding:14px 18px;border-radius:10px;background:{COLORS[level]};color:white'>"
        f"<b style='font-size:19px'>{level} RISK</b><br/>"
        f"<span style='font-size:14px'>"
        f"{'Likely digital-arrest or coercion scam. Follow the guidance below.' if level == 'HIGH' else 'Some scam indicators present. Treat with caution.' if level == 'MEDIUM' else 'No strong scam indicators detected in this text.'}"
        f"</span></div>",
        unsafe_allow_html=True,
    )

    rep = detect(text)
    left, right = st.columns([3, 2])

    with left:
        st.subheader("Detected scam indicators")
        if not rep.hits:
            st.success("No indicator families matched. The classifier score above is the only signal.")
        else:
            for family in rep.families:
                ev = [h.evidence for h in rep.hits if h.family == family][:2]
                st.markdown(f"**{label_for(family, lang)}**")
                for e in ev:
                    st.code(e.strip(), language=None)
        if rep.hits:
            st.caption(f"Evidence is taken directly from the submitted text so the verdict can be audited.")

    with right:
        st.subheader(f"What to do now ({lang.upper()})")
        for line in guidance(level, lang):
            st.markdown(f"- {line}")
        url, helpline = report_channels()
        st.markdown(f"**[Report at the National Cyber Crime Portal]({url})**  ·  Helpline `{helpline}`")


def main() -> None:
    st.title("Citizen Fraud Shield")
    st.caption("AI for Digital Public Safety - detecting coercion-based scam calls in English, Hindi, Marathi and Hinglish.")

    clf = get_model()
    if clf is None:
        st.error("Model artifact missing. Run `python -m src.data.build_dataset`, then `python -m src.evaluate`.")
        st.stop()

    tabs = st.tabs(["Check a call", "Batch check", "About the model"])

    with tabs[0]:
        st.markdown("Paste what the caller said, or load an example. Nothing is stored or transmitted.")
        keys = list(SAMPLES)
        pick = st.selectbox("Load an example", ["(none)"] + keys)
        default = SAMPLES[pick] if pick != "(none)" else ""
        text = st.text_area("Call transcript", value=default, height=150,
                            placeholder="Type or paste the call transcript here...")
        if st.button("Analyse", type="primary") and text.strip():
            render_verdict(text, clf)
        elif st.button("Analyse") and not text.strip():
            st.warning("Enter some text first.")

    with tabs[1]:
        st.caption("One interaction per line. Useful for triaging a shared call log.")
        batch = st.text_area("Transcripts (one per line)", height=180)
        if st.button("Analyse batch") and batch.strip():
            lines = [l.strip() for l in batch.splitlines() if l.strip()]
            probs = clf.proba(lines)
            st.dataframe(
                {"#": range(1, len(lines) + 1),
                 "risk": [risk_level(p) for p in probs],
                 "score": [f"{p:.0%}" for p in probs],
                 "indicators": [", ".join(detect(l).families) or "-" for l in lines],
                 "transcript": [l[:90] for l in lines]},
                use_container_width=True, hide_index=True)
            risky = sum(1 for p in probs if p >= clf.threshold)
            st.info(f"{risky} of {len(lines)} interactions flagged at the current threshold.")

    with tabs[2]:
        st.markdown(
            """
**How the verdict is produced**

1. **Normalisation** - unicode folding, zero-width character stripping, and script
   detection so Devanagari and Hinglish are handled separately from English.
2. **Text branch** - TF-IDF over word 1-2 grams plus character 3-5 grams, fed to a
   calibrated logistic regression trained with grouped 5-fold cross-validation.
   The character n-grams carry most of the robustness to typos and code-mixing.
3. **Indicator layer** - a rule engine over five scam indicator families (authority
   impersonation, legal threat, urgency, isolation/secrecy, financial coercion) in
   English, Hindi, Marathi and romanised Hinglish. It supplies the quoted evidence
   behind the warning.
4. **Risk mapping** - the probability is mapped to LOW / MEDIUM / HIGH at thresholds
   tuned to keep the false-positive rate at or below 10%, then mapped to localised
   safety guidance.

**What this is not**

- Text only. The speech branch (audio upload, ASR, voice-spoofing cues) and the
  multimodal fusion layer are Phase 2.
- The training corpus is built from documented advisory patterns, so it covers known
  script families rather than every adversarial phrasing.
- It is an advisory aid, not an official verification channel. A bank or government
  department will never ask for an OTP over a call.
""")
        results = REPORTS / "RESULTS.md"
        if results.exists():
            st.download_button("Download full evaluation report", results.read_text(encoding="utf-8"),
                               file_name="RESULTS.md", mime="text/markdown")


if __name__ == "__main__":
    main()