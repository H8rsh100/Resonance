"""Vectorisers for the text branch of the multimodal pipeline."""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from ..preprocess import normalize


def word_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        preprocessor=normalize,
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        strip_accents="unicode",
    )


def char_vectorizer() -> TfidfVectorizer:
    """Character n-grams recover robustness to typos, romanisation and code-mixing."""
    return TfidfVectorizer(
        preprocessor=normalize,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
        strip_accents="unicode",
    )


def build_union() -> FeatureUnion:
    return FeatureUnion([("word", word_vectorizer()), ("char", char_vectorizer())])