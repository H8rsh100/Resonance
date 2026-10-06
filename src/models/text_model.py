"""Scam-risk classifier for the text branch, with false-positive-aware thresholding.

Phase-1 model: TF-IDF (word + char n-grams) -> calibrated logistic regression.
The interface is deliberately kept identical to the Phase-2 transformer so that
`ai4bharat/IndicBERT` fine-tuning is a drop-in swap (owned by member 36).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix

from ..config import ARTIFACTS, RISK_THRESHOLDS
from ..features.text_features import build_union
from ..preprocess import normalize


def tune_threshold_on(p: np.ndarray, y: np.ndarray, max_fpr: float = 0.10) -> dict | None:
    """Pick the threshold maximising F1 subject to FPR <= max_fpr.

    The hackathon brief flags false-positive rate as the metric that matters for a
    citizen-facing tool: an unnecessary warning on a legitimate call erodes trust
    and teaches people to ignore the real ones. Returns None if no grid point
    satisfies the constraint.
    """
    best: dict | None = None
    for t in np.linspace(0.05, 0.95, 91):
        pred = (p >= t).astype(int)
        tn, fp, _, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        fpr = fp / max(fp + tn, 1)
        if fpr > max_fpr:
            continue
        _, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
        if best is None or f1 > best["f1"]:
            best = {"threshold": float(t), "f1": float(f1), "recall": float(r), "fpr": float(fpr)}
    return best


@dataclass
class TextScamClassifier:
    threshold: float = RISK_THRESHOLDS["medium"]
    max_fpr: float = 0.10
    fitted: bool = False

    # ---------------- training ----------------
    def fit(self, texts: list[str], y: np.ndarray) -> "TextScamClassifier":
        self.vectorizer = build_union()
        X = self.vectorizer.fit_transform([normalize(t) for t in texts])
        base = LogisticRegression(max_iter=3000, C=4.0, class_weight="balanced")
        self.model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        self.model.fit(X, y)
        self.fitted = True
        return self

    # ---------------- inference ----------------
    def proba(self, texts: list[str]) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("classifier is not fitted")
        X = self.vectorizer.transform([normalize(t) for t in texts])
        return self.model.predict_proba(X)[:, 1]

    def predict(self, texts: list[str]) -> np.ndarray:
        return (self.proba(texts) >= self.threshold).astype(int)

    # ---------------- threshold control ----------------
    def tune_threshold(self, texts: list[str], y: np.ndarray) -> dict:
        best = tune_threshold_on(self.proba(texts), y, self.max_fpr)
        if best:
            self.threshold = best["threshold"]
        return best or {}

    # ---------------- persistence ----------------
    def save(self, name: str = "text_model") -> Path:
        import joblib
        path = ARTIFACTS / f"{name}.joblib"
        joblib.dump(
            {"vectorizer": self.vectorizer, "model": self.model, "threshold": self.threshold},
            path,
        )
        return path

    @classmethod
    def load(cls, name: str = "text_model") -> "TextScamClassifier":
        import joblib
        blob = joblib.load(ARTIFACTS / f"{name}.joblib")
        obj = cls(threshold=blob["threshold"])
        obj.vectorizer, obj.model, obj.fitted = blob["vectorizer"], blob["model"], True
        return obj


def risk_level(prob: float) -> str:
    if prob >= RISK_THRESHOLDS["medium"]:
        return "HIGH"
    if prob >= RISK_THRESHOLDS["low"]:
        return "MEDIUM"
    return "LOW"


__all__ = ["TextScamClassifier", "risk_level", "tune_threshold_on"]