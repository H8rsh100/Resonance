"""Downloads the public cross-domain evaluation corpus (English / Hindi / Marathi).

Owner: member 34 (data). Writes `data/external/cross_domain.csv`, which
`src/evaluate.py` consumes for the generalisation test.

Source: `dbarbedillo/SMS_Spam_Multilingual_Collection_Dataset` - the UCI SMS Spam
Collection machine-translated into 21 languages with M2M100 (en, hi, mr included).
It is deliberately *never* used for training, only for held-out generalisation.
"""
from __future__ import annotations

import io
import urllib.request

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from ..config import DATA_EXTERNAL, EXTERNAL_DATASET, SEED

URL = "https://huggingface.co/datasets/dbarbedillo/SMS_Spam_Multilingual_Collection_Dataset/resolve/main/data-augmented.csv"
KEEP_LANGS = ("en", "hi", "mr")
PER_CLASS = 600
PUBLIC_TEST_FRAC = 0.25


def download() -> pd.DataFrame:
    with urllib.request.urlopen(URL, timeout=120) as r:
        raw = r.read()
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except UnicodeDecodeError:
            continue
    raise RuntimeError("could not decode cross-domain corpus")


def main() -> None:
    df = download()
    print("columns:", list(df.columns)[:6], "... rows:", len(df))

    text_col = next((c for c in ("text", "message", "Message", "body") if c in df.columns), None)
    lab_col = next((c for c in ("labels", "label", "class", "spam", "type") if c in df.columns), None)
    if not text_col or not lab_col:
        raise RuntimeError(f"expected text and label columns, found {list(df.columns)}")

    # The file is wide: one English `text` column plus `text_<lang>` translations,
    # so each language is extracted as its own narrow frame. `row` is kept as the
    # group id because the translations of one SMS share a row, and letting them
    # straddle a train/test split would leak the same message across both sides.
    col_for = {"en": text_col, "hi": "text_hi", "mr": "text_mr"}
    raw = df[lab_col].astype(str).str.lower().str.strip()
    y = raw.map({"spam": 1, "ham": 0, "1": 1, "0": 0, "true": 1, "false": 0})
    frames = []
    for lg in KEEP_LANGS:
        col = col_for.get(lg)
        if col is None or col not in df.columns:
            print(f"  {lg}: column missing, skipped")
            continue
        sub = pd.DataFrame({"row": df.index, "text": df[col].astype(str), "y": y}).dropna()
        sub = sub[(sub["text"].str.strip().str.len() > 10) & sub["y"].isin([0, 1])]
        sub["y"] = sub["y"].astype(int)
        for yy in (1, 0):
            take_n = PER_CLASS if yy == 1 else min(400, int((sub.y == 0).sum()))
            part = sub[sub.y == yy].head(take_n)
            frames.append(pd.DataFrame({
                "text": part["text"].to_numpy(),
                "group_id": "pub_" + part["row"].astype(str),
                "label": yy,
                "y": yy,
                "language": lg,
                "source": EXTERNAL_DATASET,
            }))
            print(f"  {lg:3s} {'scam' if yy else 'legit':5s} {len(part):5d}")

    res = pd.concat(frames, ignore_index=True)

    # Group-aware split: all translations of one SMS stay on the same side.
    gss = GroupShuffleSplit(n_splits=1, test_size=PUBLIC_TEST_FRAC, random_state=SEED)
    tr_idx, te_idx = next(gss.split(res, res["y"], groups=res["group_id"]))
    train_part, test_part = res.iloc[sorted(tr_idx)], res.iloc[sorted(te_idx)]
    for part, fname in ((train_part, "public_train.csv"), (test_part, "public_test.csv")):
        part.to_csv(DATA_EXTERNAL / fname, index=False, encoding="utf-8")
        print(f"wrote {DATA_EXTERNAL / fname}  rows={len(part)}  scam={int(part.y.sum())}")
    leak = len(set(train_part.group_id) & set(test_part.group_id))
    print(f"group leakage public train/test (must be 0): {leak}")


if __name__ == "__main__":
    main()