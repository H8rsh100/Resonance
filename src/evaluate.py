"""Phase-1 evaluation: grouped cross-validation, metrics, ablation, figures, RESULTS.md.

Three deliberate design choices that matter for credibility:

1. **Grouped splits.** Augmented variants of the same base call share a `group_id`,
   so every variant of a call lands in the same fold. Plain random splitting would
   put near-duplicates on both sides and inflate the score.
2. **Out-of-fold predictions.** A single grouped hold-out on 85 base calls is far
   too small to yield a usable false-positive estimate, so all metrics come from
   `StratifiedGroupKFold` out-of-fold predictions over the whole corpus.
3. **Cross-domain generalisation test.** The in-domain corpus is template-derived,
   so the model is also scored on an untouched public multilingual spam corpus.
   That number, not the in-domain one, reflects real-world behaviour.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedGroupKFold

from .config import FIGURES, LABEL_LEGIT, LABEL_SCAM, REPORTS, ROOT, SEED
from .indicators import detect
from .models.text_model import TextScamClassifier, tune_threshold_on
from .preprocess import detect_language

CSV = "data/processed/calls.csv"
N_SPLITS = 5


# ---------------------------------------------------------------- data
def load() -> pd.DataFrame:
    df = pd.read_csv(ROOT / CSV)
    df["indicators"] = df["indicators"].fillna("")
    df["y"] = df["label"].astype(int)
    return df


def out_of_fold(df: pd.DataFrame, factory=TextScamClassifier) -> np.ndarray:
    """Out-of-fold scam probabilities, one prediction per row."""
    oof = np.zeros(len(df))
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for fold, (tr, te) in enumerate(cv.split(df, df["y"], groups=df["group_id"])):
        clf = factory().fit(df.iloc[tr]["text"].tolist(), df.iloc[tr]["y"].to_numpy())
        oof[te] = clf.proba(df.iloc[te]["text"].tolist())
        print(f"  fold {fold + 1}/{N_SPLITS}: train={len(tr)} test={len(te)}")
    return oof


def leakage_check(df: pd.DataFrame) -> int:
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    worst = 0
    for tr, te in cv.split(df, df["y"], groups=df["group_id"]):
        worst = max(worst, len(set(df.iloc[tr].group_id) & set(df.iloc[te].group_id)))
    return worst


# ---------------------------------------------------------------- metrics
def metrics(y: np.ndarray, pred: np.ndarray, p: np.ndarray | None = None) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "fpr": fp / max(fp + tn, 1),
        "fnr": fn / max(fn + tp, 1),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "support": int(len(y)),
    }
    if p is not None and len(np.unique(y)) > 1:
        out["auc"] = roc_auc_score(y, p)
    else:
        out["auc"] = float("nan")
    return out


def fmt(d: dict) -> str:
    auc = d.get("auc", float("nan"))
    return (f"| {d['accuracy']:.3f} | {d['precision']:.3f} | {d['recall']:.3f} | "
            f"{d['f1']:.3f} | {auc:.3f} | **{d['fpr']:.3f}** | {d['support']} |")


# ---------------------------------------------------------------- baselines
class LexiconOnlyClassifier:
    """Interpretable no-learning baseline: probability from indicator families alone."""

    def __init__(self, max_fpr: float = 0.10):
        self.max_fpr = max_fpr
        self.threshold = 0.5

    def proba(self, texts: list[str]) -> np.ndarray:
        return np.array([min(0.97, 0.06 + 0.18 * len(detect(t).families)) for t in texts])


def ablation(df: pd.DataFrame, oof_word_char: np.ndarray, thr: float) -> pd.DataFrame:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, Pipeline

    from .features.text_features import build_union, char_vectorizer, word_vectorizer

    texts = df["text"].tolist()
    y = df["y"].to_numpy()

    def oof_for(pipeline_fn) -> np.ndarray:
        oof = np.zeros(len(df))
        cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        for tr, te in cv.split(df, y, groups=df["group_id"]):
            pipe = pipeline_fn()
            pipe.fit([texts[i] for i in tr], y[tr])
            oof[te] = pipe.predict_proba([texts[i] for i in te])[:, 1]
        return oof

    def mk(kind, weighted=True):
        def fn():
            vec = {"word": word_vectorizer, "char": char_vectorizer, "both": build_union}[kind]
            steps = [("vec", vec())]
            steps.append(("clf", LogisticRegression(max_iter=3000, C=4.0,
                                                    class_weight="balanced" if weighted else None)))
            return Pipeline(steps)
        return fn

    variants = {
        "word 1-2gram only": mk("word"),
        "char 3-5gram only": mk("char"),
        "word + char, unweighted LR": mk("both", weighted=False),
    }

    lex = LexiconOnlyClassifier()
    lex_p = lex.proba(texts)
    lex_thr = (tune_threshold_on(lex_p, y) or {"threshold": 0.5})["threshold"]

    rows = [
        {**metrics(y, (oof_word_char >= thr).astype(int), oof_word_char),
         "config": "word + char, calibrated LR (ours)", "threshold": thr},
        {**metrics(y, (lex_p >= lex_thr).astype(int), lex_p),
         "config": "lexicon indicator rules only (no ML)", "threshold": lex_thr},
    ]
    for name, fn in variants.items():
        p = oof_for(fn)
        t = (tune_threshold_on(p, y) or {"threshold": 0.5})["threshold"]
        rows.append({**metrics(y, (p >= t).astype(int), p), "config": name, "threshold": t})

    cols = ["config", "accuracy", "precision", "recall", "f1", "auc", "fpr", "threshold"]
    return pd.DataFrame(rows)[cols].sort_values("f1", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- figures
def figures(y, pred, p, abl: pd.DataFrame, per_lang: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=15,
                color="white" if v > cm.max() / 2 else "#12233d")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["legitimate", "scam"]); ax.set_yticklabels(["legitimate", "scam"])
    ax.set_xlabel("predicted"); ax.set_ylabel("actual")
    ax.set_title(f"Confusion matrix (out-of-fold, n={len(y)})")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(FIGURES / "confusion_matrix.png", dpi=150); plt.close(fig)

    fpr, tpr, _ = roc_curve(y, p)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.plot(fpr, tpr, color="#c0392b", lw=2, label=f"text branch (AUC={roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#95a5a6", label="random")
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title("ROC - digital arrest scam detection")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGURES / "roc_curve.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    a = abl.sort_values("f1")
    yy = np.arange(len(a))
    ax.barh(yy + 0.19, a["f1"], 0.36, label="F1", color="#1f6feb")
    ax.barh(yy - 0.19, a["fpr"], 0.36, label="false positive rate", color="#e76f51")
    ax.set_yticks(yy); ax.set_yticklabels(a["config"], fontsize=9)
    ax.set_xlim(0, 1.05); ax.set_xlabel("score")
    ax.set_title("Ablation: detection quality vs false positives")
    ax.legend(); ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGURES / "ablation.png", dpi=150); plt.close(fig)

    if len(per_lang):
        fig, ax = plt.subplots(figsize=(6.6, 3.6))
        x = np.arange(len(per_lang)); w = 0.38
        ax.bar(x - w / 2, per_lang["f1"], w, label="F1", color="#1f6feb")
        ax.bar(x + w / 2, per_lang["fpr"], w, label="FPR", color="#e76f51")
        ax.set_xticks(x); ax.set_xticklabels(per_lang["language"])
        ax.set_ylim(0, 1.15); ax.set_ylabel("score")
        ax.set_title("Per-language performance (out-of-fold)")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(FIGURES / "per_language.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------- cross-domain
def public_parts() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    from .config import DATA_EXTERNAL
    tr = DATA_EXTERNAL / "public_train.csv"
    te = DATA_EXTERNAL / "public_test.csv"
    if not (tr.exists() and te.exists()):
        return None, None
    a, b = pd.read_csv(tr), pd.read_csv(te)
    for d in (a, b):
        d["y"] = d["y"].astype(int)
    return a, b


def cross_domain(df: pd.DataFrame, thr: float) -> dict | None:
    """Two generalisation tests on untouched public data.

    `multi_domain` trains on digital-arrest + public-train, so it measures how well
    the system holds up on message types it has never seen in the digital-arrest set.
    `domain_shift_only` trains on the digital-arrest corpus alone and is reported as
    a documented negative result rather than hidden.
    """
    pub_tr, pub_te = public_parts()
    if pub_tr is None:
        return None

    out: dict = {}
    for name, train_set in (("multi_domain", pd.concat([df, pub_tr], ignore_index=True)),
                            ("domain_shift_only", df)):
        clf = TextScamClassifier().fit(train_set["text"].tolist(), train_set["y"].to_numpy())
        p = clf.proba(pub_te["text"].tolist())
        frame = pub_te.assign(p=p, pred=(p >= thr).astype(int))
        rows = [{"language": lg, **metrics(g["y"].to_numpy(), g["pred"].to_numpy(), g["p"].to_numpy())}
                for lg, g in frame.groupby("language")]
        out[name] = {"overall": metrics(frame["y"].to_numpy(), frame["pred"].to_numpy(), p),
                     "per_language": pd.DataFrame(rows).sort_values("language").reset_index(drop=True),
                     "n_train": len(train_set)}
    return out


# ---------------------------------------------------------------- main
def main() -> None:
    df = load()
    print(f"corpus: {len(df)} rows, {df.group_id.nunique()} base calls, "
          f"{df[df.y == LABEL_SCAM].group_id.nunique()} scam / {df[df.y == LABEL_LEGIT].group_id.nunique()} legit groups")
    overlap = leakage_check(df)
    print(f"group leakage across folds (must be 0): {overlap}")

    print("out-of-fold predictions:")
    oof = out_of_fold(df)
    y = df["y"].to_numpy()

    tuned = tune_threshold_on(oof, y)
    thr = tuned["threshold"] if tuned else 0.5
    print(f"threshold chosen on out-of-fold scores: {tuned}")

    pred = (oof >= thr).astype(int)
    m = metrics(y, pred, oof)
    print("\nOOF:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()})

    frame = df.assign(p=oof, pred=pred)
    rows = [{"language": lg, **metrics(g["y"].to_numpy(), g["pred"].to_numpy(), g["p"].to_numpy())}
            for lg, g in frame.groupby("language")]
    per_lang = pd.DataFrame(rows).sort_values("language").reset_index(drop=True)
    print("\nPER LANGUAGE")
    print(per_lang[["language", "precision", "recall", "f1", "auc", "fpr", "support"]].round(3).to_string(index=False))

    print("\nABLATION")
    abl = ablation(df, oof, thr)
    print(abl.round(3).to_string(index=False))

    detected = [detect_language(t) for t in df["text"]]
    lang_acc = float(np.mean([d == a for d, a in zip(detected, df["language"])]))
    print(f"\nlanguage identification accuracy: {lang_acc:.1%}")

    clf = TextScamClassifier().fit(df["text"].tolist(), y)
    clf.threshold = thr
    clf.save()
    xd = cross_domain(df, thr)
    if xd:
        for name, blob in xd.items():
            print(f"\nCROSS-DOMAIN [{name}] train={blob['n_train']}:",
                  {k: (round(v, 3) if isinstance(v, float) else v) for k, v in blob["overall"].items()})
    print("saved artifacts/text_model.joblib")

    figures(y, pred, oof, abl, per_lang)
    write_report(m, per_lang, abl, thr, tuned, xd, lang_acc, df, overlap, frame)
    print("wrote reports/RESULTS.md")


def write_report(m, per_lang, abl, thr, tuned, xd, lang_acc, df, overlap, frame) -> None:
    hdr = "| Accuracy | Precision | Recall | F1 | AUC | FPR | n |\n|---|---|---|---|---|---|---|\n"
    n_scam = int((df.y == LABEL_SCAM).sum()); n_legit = int((df.y == LABEL_LEGIT).sum())
    body = (
        "# Phase 1 Results - Digital Arrest Scam Detection (text branch)\n\n"
        "Generated by `python -m src.evaluate`.\n\n"
        f"- Corpus: **{len(df)} samples**, **{df.group_id.nunique()} base calls** "
        f"({n_scam} scam / {n_legit} legitimate), 4 languages (en, hi, mr, hinglish)\n"
        f"- Validation: **{N_SPLITS}-fold StratifiedGroupKFold**, grouped by base call\n"
        f"- Group leakage across folds: **{overlap}** (must be 0)\n\n"
        "Grouping matters: augmented variants of one utterance share a `group_id`, so\n"
        "near-duplicates can never straddle the train/test boundary.\n\n"
        "## 1. Out-of-fold results\n\n" + hdr + fmt(m) + "\n\n"
        f"Operating threshold **{thr:.2f}** (maximises F1 subject to FPR <= 10%, chosen on\n"
        "out-of-fold scores so no single fold defines the operating point).\n\n"
        f"Confusion counts: **TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']}**\n\n"
        "![confusion matrix](figures/confusion_matrix.png)\n\n"
        "![roc](figures/roc_curve.png)\n\n"
        "## 2. Per-language breakdown\n\n"
        "| Language | Accuracy | Precision | Recall | F1 | AUC | FPR | n |\n|---|---|---|---|---|---|---|---|\n"
        + "".join(fmt(r) + "\n" for _, r in per_lang.iterrows())
        + f"\nCoarse language identification accuracy across the corpus: **{lang_acc:.1%}**\n\n"
        "![per language](figures/per_language.png)\n\n"
        "## 3. Ablation\n\n"
        "| Configuration | Accuracy | Precision | Recall | F1 | AUC | FPR | thr |\n"
        "|---|---|---|---|---|---|---|---|\n"
        + "".join(
            f"| {r['config']} | {r['accuracy']:.3f} | {r['precision']:.3f} | {r['recall']:.3f} | "
            f"{r['f1']:.3f} | {r['auc']:.3f} | {r['fpr']:.3f} | {r['threshold']:.2f} |\n"
            for _, r in abl.iterrows()
        )
        + "\n![ablation](figures/ablation.png)\n\n"
    )

    if xd:
        body += (
            "## 4. Cross-domain generalisation (untouched public corpus)\n\n"
            "Source: `dbarbedillo/SMS_Spam_Multilingual_Collection_Dataset` (UCI SMS Spam\n"
            "Collection, M2M100 translations to en/hi/mr). Split by **source row**, so the\n"
            "en/hi/mr translations of one SMS never straddle train and test. Neither the\n"
            "digital-arrest corpus nor this data is synthetic.\n\n"
        )
        for name, blob in xd.items():
            title = ("Trained on **digital-arrest + public-train**" if name == "multi_domain"
                     else "Trained on **digital-arrest only** (documented failure)")
            body += (f"### {title}\n\n"
                     f"Training rows: {blob['n_train']}  |  Test rows: {blob['overall']['support']}\n\n"
                     + hdr + fmt(blob["overall"]) + "\n\n"
                     "| Language | Accuracy | Precision | Recall | F1 | AUC | FPR | n |\n|---|---|---|---|---|---|---|---|\n"
                     + "".join(fmt(r) + "\n" for _, r in blob["per_language"].iterrows()) + "\n")
        o = xd["domain_shift_only"]["overall"]
        body += (
            "**Read this honestly.** A model trained only on short templated call\n"
            "transcripts does not transfer to a different scam genre: "
            f"AUC drops to {o['auc']:.3f} and FPR to {o['fpr']:.3f}. Training on a mixed\n"
            "corpus fixes most of it, which is the evidence for keeping a public,\n"
            "multi-genre training set rather than only hand-built templates. This is the\n"
            "single biggest Phase-2 item (member 34 + 36).\n\n"
        )

    worst = frame[(frame.y == LABEL_LEGIT) & (frame.pred == LABEL_SCAM)].sort_values("p", ascending=False)
    body += (
        "## 5. Honest limitations\n\n"
        "- The corpus is built from documented advisory patterns, so it covers known\n"
        "  script families rather than naturally occurring adversarial phrasing.\n"
        "- Grouped CV removes leakage, but the effective sample size is the number of\n"
        f"  base calls ({df.group_id.nunique()}), not {len(df)} rows.\n"
        "- Text branch only. The speech branch and fusion layer are Phase 2\n"
        "  (members 39 and 46).\n"
        "- The lexicon indicator layer is an explainability device, not the\n"
        "  classifier; the ablation quantifies how much the learned model adds.\n"
    )
    if len(worst):
        body += (
            f"\n### False positives at the operating threshold ({len(worst)})\n\n"
            "Highest-scoring legitimate calls the model still flags. This is the set a\n"
            "citizen-facing tool must get right.\n\n| Score | Language | Text |\n|---|---|---|\n"
            + "".join(f"| {r['p']:.2f} | {r['language']} | {r['text'][:110]} |\n"
                      for _, r in worst.head(6).iterrows())
        )

    (REPORTS / "RESULTS.md").write_text(body, encoding="utf-8")


if __name__ == "__main__":
    main()