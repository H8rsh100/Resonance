# Resonance — Citizen Fraud Shield

**AI for Digital Public Safety: detecting digital-arrest and coercion-based scam calls**

Built for **ET AI Hackathon 2026, Problem Statement 6** (*AI for Digital Public Safety:
Defeating Counterfeiting, Fraud & Digital Arrest Scams*), tracking the *Digital Arrest
Scam Detection & Alerting* and *Citizen Fraud Shield* tracks.

India recorded 1.14 M cybercrime complaints in 2023 (up 60% YoY) and the Home Ministry
attributes over ₹1,776 crore in losses to "digital arrest" impersonation scams in the first
nine months of 2024. This project targets the preventive gap: not recovering money after
a complaint, but warning a citizen **before** a transfer happens.

---

## What it does

Paste (or upload) what a caller said. You get a risk verdict, the **quoted evidence**
behind it, and localised safety guidance — in English, Hindi, Marathi or Hinglish.

| Input | Output |
|---|---|
| Call transcript / message text | LOW / MEDIUM / HIGH risk, probability, indicator families with evidence snippets, advisory in the input language, NCRB portal + 1930 helpline |
| Batch list (one per line) | Risk table for triaging a shared call log |

Privacy model: analysis is local and stateless. Nothing is uploaded or stored, and the
system only ever analyses interactions the user chooses to submit.

---

## Phase 1 results

Full detail with figures: **[`reports/RESULTS.md`](reports/RESULTS.md)**

**In-domain, 5-fold `StratifiedGroupKFold` grouped by base call** (368 samples,
92 base calls, 4 languages). Grouping means augmented variants of one utterance can
never straddle the train/test boundary — verified leakage = 0.

| Accuracy | Precision | Recall | F1 | AUC | **FPR** | n |
|---|---|---|---|---|---|---|
| 0.905 | 0.928 | 0.906 | 0.916 | 0.973 | **0.096** | 368 |

**Cross-domain generalisation** on untouched public data
(`dbarbedillo/SMS_Spam_Multilingual_Collection_Dataset`, en/hi/mr, split by source row so
translations of the same SMS never cross the boundary):

| Training regime | F1 | AUC | FPR |
|---|---|---|---|
| digital-arrest + public-train (multi-domain) | **0.947** | **0.983** | **0.022** |
| digital-arrest only (documented failure) | 0.387 | 0.360 | 0.523 |

That second row is the honest result: a model trained only on short templated call
transcripts does **not** transfer to a different scam genre. Training on a mixed corpus
recovers it. This is the headline evidence for keeping a real, multi-genre training set,
and it is the top Phase-2 task.

Per-language F1: en 0.955 · mr 0.885 · hinglish 0.933 · hi 0.811. Coarse language
identification accuracy 91.8%.

### Ablation

| Configuration | F1 | AUC | FPR |
|---|---|---|---|
| lexicon indicator rules only (no ML) | 0.949 | 0.955 | 0.026 |
| word + char n-grams, unweighted LR | 0.941 | 0.980 | 0.077 |
| char 3–5 gram only | 0.939 | 0.985 | 0.096 |
| word + char, calibrated LR (ours) | 0.916 | 0.973 | 0.096 |
| word 1–2 gram only | 0.916 | 0.966 | 0.083 |

Read the top row carefully: the lexicon scores highly **because it was written from the
same advisory patterns as the corpus**, so it is circular as a comparison. Its real value
is auditability, not raw accuracy. The learned model is what generalises across domains.

---

## Architecture

```
                 ┌──────────────────┐
 user transcript│  Normalisation   │  unicode fold · zero-width strip
 or audio  ────▶│  + script/lang   │  EN / HI / MR / Hinglish
 (Phase 2)      └────────┬─────────┘
                          │
        ┌─────────────────┴──────────────────┐
        │                                    │
┌───────▼────────┐              ┌────────────▼───────────┐
│  TEXT BRANCH   │              │   SPEECH BRANCH        │
│  TF-IDF word   │              │   ASR → transcript     │
│  + char n-gram │              │   acoustic features    │
│  → calibrated  │              │   stress / spoof cues  │
│    LR          │              │   (Phase 2 · member 39) │
└───────┬────────┘              └────────────┬───────────┘
        │                                    │
        └──────────────┬─────────────────────┘
                       ▼
            ┌─────────────────────┐
            │  FUSION  (Phase 2)  │  member 46
            │  late fusion of     │
            │  branch scores      │
            └──────────┬──────────┘
                       ▼
            ┌─────────────────────┐
            │  INDICATOR LAYER    │  5 scam families, quoted
            │  (explainability)   │  evidence, EN/HI/MR
            └──────────┬──────────┘
                       ▼
            ┌─────────────────────┐
            │  RISK ENGINE        │  LOW / MEDIUM / HIGH at
            │  + ADVISORY         │  FPR-constrained threshold
            └─────────────────────┘  member 46
```

The classifier supplies the score; the indicator layer supplies the *reason*. A citizen
warning that cannot be audited is not usable in a public-safety context.

---

## Quickstart

**Windows (PowerShell 5.1 — note: no `&&` support)** — one command does everything:

```powershell
powershell -ExecutionPolicy Bypass -File .\bootstrap.ps1
```

Or run the steps separately:

```powershell
python -m src.data.build_dataset
python -m src.data.fetch_public       # ~30s, needs internet
python -m src.evaluate
streamlit run app.py                   # then open http://localhost:8501
```

**macOS / Linux / bash:**

```bash
./bootstrap.sh
```

Everything except `artifacts/` and `data/processed|external/` is committed, so a clean
clone reproduces every number above.

---

## Repository layout

```
app.py                       Streamlit Citizen Fraud Shield demo
src/
  config.py                  paths, labels, indicator names, thresholds
  preprocess.py              normalisation, script detection, EN/HI/MR/Hinglish
  indicators.py              5-family scam indicator lexicon + evidence snippets
  risk.py                    risk mapping + EN/HI/MR advisory text
  data/build_dataset.py      digital-arrest corpus (EN/HI/MR/Hinglish)
  data/fetch_public.py       public cross-domain corpus, grouped split
  features/text_features.py  TF-IDF word + char vectorisers
  models/text_model.py       classifier, calibration, threshold tuning
  evaluate.py                grouped CV, ablation, cross-domain tests, figures
reports/RESULTS.md           full evaluation report
reports/figures/             confusion matrix, ROC, ablation, per-language
docs/                        architecture, work plan, demo script
```

---

## Honest limitations

- The digital-arrest corpus is built from documented advisory patterns. It covers known
  script families, not every adversarial phrasing a real attacker would use.
- Grouped CV removes leakage, but the effective sample size is 92 base calls, not 368 rows.
- Text branch only. Audio input, ASR and multimodal fusion are Phase 2.
- This is an advisory aid, never a substitute for official verification. No bank or
  government department asks for an OTP over a phone call.

---

## Team and work split

| Member | Roll | Owns | Phase 1 status |
|---|---|---|---|
| Preeti Dudam | 34 | Data + literature: corpus construction, indicator lexicons, public dataset ingestion, annotation guide, lit review | **done** |
| Harsh Gade | 36 | NLP branch + MLOps: features, classifier, evaluation, cross-domain tests, repo hygiene, IndicBERT fine-tune | **done** |
| Yash Gaikwad | 39 | Speech branch: ASR, acoustic feature extraction, stress / voice-spoofing cues, audio tab | Phase 2 |
| Rutuja Ghare | 46 | Fusion + risk engine + UX: risk mapping, localised advisory, Streamlit shell, architecture diagram, IEEE report + deck | **done** |

See [`docs/WORKPLAN.md`](docs/WORKPLAN.md) for the Phase 2 sprint plan and
[`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md) for the checkpoint demo script.