# Phase 2 work plan — 14 days

Goal: turn the Phase 1 vertical slice into a hackathon submission with a working
prototype, architecture diagram, presentation deck and demo video.

Judging weights to optimise against: Innovation 25% · Business Impact 25% ·
Technical Excellence 20% · Scalability 15% · UX 15%.

## Ownership

| Owner | Module | Files |
|---|---|---|
| 34 Preeti | Data + literature | `src/data/`, indicator lexicons, annotation guide, IEEE lit review |
| 36 Harsh | NLP branch + MLOps | `src/models/`, `src/evaluate.py`, `src/features/` |
| 39 Yash | Speech branch | `src/features/speech_features.py`, `src/models/speech_model.py` |
| 46 Rutuja | Fusion + risk + UX | `src/fusion.py`, `src/risk.py`, `app.py`, `docs/ARCHITECTURE.md`, deck |

Each member owns one runnable module. Every commit should keep `python -m src.evaluate`
and `streamlit run app.py` working on `main`.

## Sprint

### Days 2–4 — parallel build

- **34**: scale the corpus from 92 base calls toward ~1.5k, prioritising the weak spots
  the Phase 1 report exposes (Hindi recall 0.750; per-call *sequence* labels, not just
  per-message). Add a `call_session` id so multi-turn escalation is modelled.
- **36**: fine-tune `ai4bharat/IndicBERT` behind the same interface as
  `TextScamClassifier` so it is a drop-in swap. Keep the TF-IDF model as the fallback
  and report both. Add counterfactual/adversarial test samples (spelled-out numerals,
  inserted filler, polite phrasing) — attackers avoid lexical triggers.
- **39**: audio path. Whisper/IndicASR for the transcript, `librosa` MFCC + prosody
  features (speaking rate, jitter, energy variance) for vocal stress, and an AI-voice
  / deepfake indicator. Deliver a real `speech_model.py` that accepts a wav upload.
- **46**: multi-turn session view in the app, and advisory copy review with a native
  speaker per language.

### Day 5 — integration

- **46**: `src/fusion.py`, late fusion of branch scores with a documented weighting.
- **36**: full ablation — text-only vs speech-only vs fused — plus threshold selection
  for the fused operating point.
- **34**: refresh the cross-domain suite with a second public source so generalisation
  is not measured against a single corpus.

### Days 6–8 — differentiators

- **34 + 46**: RAG advisory layer grounded in MHA / NCRB / I4C advisories, so the answer
  cites a real advisory rather than a hardcoded string. Cheap win on Innovation + Impact.
- **39**: fake-government-portal / spoofed-link detector as a second input channel.
- **36**: reporting package generator — a structured incident summary that maps to the
  NCRB complaint format (Business Impact).

### Days 9–10 — submission artefacts

- **46**: `docs/ARCHITECTURE.md`, 8-slide deck.
- **34**: IEEE report — abstract, related work (the Phase 1 literature review), method,
  experiments, limitations, references.
- **36**: reproduce every figure in the report from `python -m src.evaluate`.

### Days 11–12 — demo and rehearsal

- **36**: record the 3-minute demo video.
- **39**: 5-minute mock Q&A. Expect: *"how do you know your own dataset is not the
  answer?"* → the cross-domain table answers it. *"what is your false-positive cost?"* →
  the FPR-constrained threshold. *"why not just use an LLM?"* → latency, cost, auditability.

## Cut list, in order

1. Speech fusion (keep the audio tab, drop the fusion contribution)
2. RAG advisory layer
3. Reporting package generator
4. Counterfactual adversarial suite
5. Multi-language advisory beyond EN/HI/MR

## Explicitly out of scope

The problem statement also lists counterfeit-currency computer vision, fraud network
graph intelligence and geospatial crime mapping. Do **not** start them. Claim one track
and execute it well — three shallow tracks score worse than one complete one.