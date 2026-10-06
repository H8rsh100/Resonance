# Checkpoint demo script — 6 minutes

Audience: course professor (and later, hackathon judges). Goal is to show a **working**
end-to-end vertical slice with honest numbers, not to claim completeness.

## 0:00 — the problem, in one number

> "India lost over ₹1,776 crore to digital-arrest impersonation scams in the first nine
> months of 2024. The money is gone by the time a complaint is filed. So we built a system
> that warns a citizen *before* the transfer happens."

Then show the methodology claim: **no public dataset exists for Indian digital-arrest
interactions.** We checked HuggingFace. That is the research gap our literature review
identified, and it is why we built a corpus.

## 0:45 — show the app, scam case first

Load the **Digital arrest call (EN)** sample → Analyse.

Point at, in order:
1. **HIGH RISK**, ~99% score
2. Three indicator families with the *quoted evidence* — authority impersonation,
   legal threat, urgency
3. The guidance panel, and the NCRB portal + 1930 helpline

The line that matters for a professor: **every claim is auditable.** The system does not
just output a number, it shows the sentences that produced it.

## 1:45 — the false-positive control

Load **Genuine bank call (EN)** → LOW risk. Say the number out loud.

> "That second case is the one that decides whether anyone keeps this tool. A citizen-facing
> detector that cries wolf gets ignored. So our operating threshold is not 'highest F1' —
> it is the highest F1 subject to a false-positive rate at or below 10%."

## 2:30 — languages

Run **Digital arrest call (HI)**, then **(MR)**, then the Hinglish family-impersonation
sample. Same pipeline, same interface, guidance switches to the input language.

> "Marathi and Hindi share the Devanagari block, so script detection alone cannot separate
> them. We score two marker sets instead. Overall language identification accuracy is 91.8%."

## 3:30 — the numbers, and the honest caveat

Open `reports/RESULTS.md`.

| Point | Say |
|---|---|
| Validation | 5-fold `StratifiedGroupKFold` **grouped by base call** — augmented variants of one utterance never straddle the split. Verified leakage = 0. |
| Headline | F1 0.916, AUC 0.973, FPR 0.096 on in-domain out-of-fold predictions |
| Cross-domain | F1 0.947, AUC 0.983, FPR 0.022 on untouched public en/hi/mr data |
| Ablation | char n-grams carry the robustness; the lexicon beats the model in-domain only because it was written from the same advisories — circular, and we say so |

Then the honesty beat, unprompted — this is what buys credibility:

> "We also recorded a negative result. A model trained only on our template corpus gets
> AUC 0.360 on the public corpus — worse than random. Training on a mixed corpus fixes it.
> That is why the multi-genre corpus is the top task for next week, and it is the single
> number a judge should hold us to."

## 5:00 — what each of us owns

| Member | Owns | Status |
|---|---|---|
| 34 Preeti | data + literature, indicator lexicons | done and demoed |
| 36 Harsh (me) | NLP branch, classifier, evaluation, repo | done and demoed |
| 39 Yash | speech branch — ASR, acoustic features, voice-spoof cues | next sprint |
| 46 Rutuja | risk engine, localised advisory, app, architecture | done and demoed |

## 5:30 — what is next, concretely

- Scale the corpus to ~1.5k with **multi-turn session** labels, so escalation over a
  three-day call is modelled rather than single messages
- Fine-tune IndicBERT behind the same interface as a drop-in swap
- Add the speech branch and late fusion, then re-run the ablation text vs speech vs fused
- RAG advisory layer citing real MHA/NCRB advisories

## Likely questions

| Question | Answer |
|---|---|
| Isn't your own dataset the answer? | It is why we added the cross-domain table. The template-only model scores AUC 0.360 out of domain. |
| Why not just prompt an LLM? | Latency and cost per citizen query, plus no auditability — a citizen warning needs quoted evidence and a stable threshold. |
| Why 92 base calls? | Phase 1. Honest small-data start with grouped validation; scaling is scheduled. |
| What's the false-positive cost? | That is the constraint we optimise against, and the number we report first. |
| What about audio? | Phase 2, and it is the reason we chose fusion over a single model. |
| Privacy? | Stateless, local, only user-submitted interactions. No interception. |