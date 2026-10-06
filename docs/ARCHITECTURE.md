# Architecture

## Design constraints

1. **Explainability is a requirement, not a nicety.** A citizen is being told that
   people trying to extort them are on the line. The system must be able to show which
   sentences drove the verdict. Hence the indicator layer is a first-class component,
   not a debug view.
2. **False positives are the expensive error.** A false positive costs trust and trains
   the citizen to dismiss real warnings. The operating threshold is therefore chosen to
   maximise F1 *subject to* FPR ≤ 10%, not to maximise accuracy.
3. **Privacy-aware by construction.** Only interactions the user voluntarily submits are
   analysed. No call interception, no network-level monitoring, no retention of
   personally identifiable information.
4. **Branches stay swappable.** Text and speech branches expose the same
   `fit` / `proba` / `tune_threshold` interface so fusion does not care what is behind
   them. That is what lets the TF-IDF model be replaced by fine-tuned IndicBERT, or by the
   speech branch, without touching the fusion or the app.

## Components

### 1. Input and preprocessing — `src/preprocess.py`

Unicode NFC folding, zero-width character removal, punctuation normalisation. Detects the
script (Devanagari vs Latin) and then the language: English, Hindi, Marathi or Hinglish.
Hindi and Marathi are not separable from orthography alone, so two marker sets are scored
and the stronger one wins. Hinglish is detected from romanised function-word frequency.

### 2. Text branch — `src/features/`, `src/models/text_model.py`

TF-IDF over word 1–2 grams and character 3–5 grams (`char_wb`), unioned, into a logistic
regression wrapped in `CalibratedClassifierCV`. The character n-grams are what carry
robustness to typos, romanisation and code-mixing — visible in the ablation table.

Class weighting is `balanced` because the corpus is not 50/50.

### 3. Speech branch — Phase 2, member 39

Audio upload → ASR produces a transcript that joins the text branch; the raw audio is
analysed independently for prosodic stress features (speaking rate, jitter, energy
variance) and synthetic-voice cues. This is the complementary evidence the methodology
argues is unavailable from text alone, particularly for family-impersonation calls that
are lexically innocuous.

### 4. Fusion — Phase 2, member 46

Late fusion of branch scores into a single risk probability. Chosen over early fusion
because the branches are independently useful, independently testable, and may be
unavailable at inference time — a missing audio branch degrades confidence rather than
breaking the pipeline.

### 5. Indicator layer — `src/indicators.py`

Five families from the proposed methodology: authority impersonation, legal threat,
urgency, isolation/secrecy, financial coercion. Regex rules over English, Hindi, Marathi
and romanised Hinglish, each returning a snippet of surrounding text as evidence.

### 6. Risk engine and advisory — `src/risk.py`

Maps probability to LOW / MEDIUM / HIGH, attaches localised guidance, and returns the
reporting channels (NCRB portal, 1930 helpline).

### 7. Application — `app.py`

Streamlit. Single-interaction check with example loader, batch triage view, and a panel
documenting what the system does and does not do.

## Data flow

```
transcript ──▶ preprocess ──▶ [text branch score] ─┐
                   │                                ├─▶ fusion ─▶ risk level
                   └────────▶ [indicator families] ┘                    │
                                                                    ▼
                                                      evidence + advisory (EN/HI/MR)
```

## Evaluation strategy

- **Grouped** validation: splits are by `group_id` (base call), never by row, because
  augmented variants of one utterance would otherwise leak across the boundary.
- **Out-of-fold predictions** for all reported in-domain metrics, so the operating point
  is not defined by one lucky fold.
- **Cross-domain** testing on public data with a row-grouped split, because translations
  of one SMS are the same message and would leak if split naively.
- Two regimes are reported for the public data: multi-domain training (generalisation)
  and digital-arrest-only training (a documented negative result).