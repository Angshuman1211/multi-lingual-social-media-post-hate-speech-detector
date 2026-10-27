"""Download the four corpora and pool them into one labelled table.

Run: python -m src.data_prep
"""

import sys

import pandas as pd
from datasets import load_dataset, concatenate_datasets
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_PROC,
    DATASETS,
    MIN_NATIVE_SCRIPT,
    SCRIPT_RANGES,
    SEED,
)
from src.preprocess import clean_text


def _all_splits(hf_id: str) -> pd.DataFrame:
    """Pool every split a corpus ships. Two of the four give only `train`, so I
    re-split everything myself to keep the proportions comparable."""
    ds = load_dataset(hf_id)
    parts = [ds[s] for s in ds.keys()]
    return concatenate_datasets(parts).to_pandas()


def load_english() -> pd.DataFrame:
    df = _all_splits(DATASETS["en"]["hf_id"])
    # 0 hate, 1 offensive, 2 neither -> toxic covers the first two.
    df = df.rename(columns={"tweet": "text", "class": "orig"})
    df["label"] = (df["orig"] != 2).astype(int)
    df["label_en3"] = df["orig"]
    df["lang"] = "en"
    return df[["text", "label", "label_en3", "lang"]]


def load_english_mhs() -> pd.DataFrame:
    """Berkeley's Measuring Hate Speech corpus.

    Rows are annotator judgements, not comments, so I average the continuous
    hate score per comment first. The authors treat score > 0.5 as hate speech
    and I keep that cut rather than inventing my own.
    """
    df = _all_splits(DATASETS["en_mhs"]["hf_id"])
    agg = (
        df.groupby("comment_id")
        .agg(text=("text", "first"), score=("hate_speech_score", "mean"))
        .reset_index()
    )
    agg["label"] = (agg["score"] >= 0.5).astype(int)
    agg["lang"] = "en"
    agg["label_en3"] = -1
    return agg[["text", "label", "label_en3", "lang"]]


def load_hindi() -> pd.DataFrame:
    df = _all_splits(DATASETS["hi"]["hf_id"])
    df = df.rename(columns={"rawtweets": "text", "binary class": "orig"})
    df["label"] = (df["orig"].astype(str).str.upper() == "HOF").astype(int)
    df["lang"] = "hi"
    df["label_en3"] = -1
    return df[["text", "label", "label_en3", "lang"]]


def load_bengali() -> pd.DataFrame:
    df = _all_splits(DATASETS["bn"]["hf_id"])
    df = df.rename(columns={"sentence": "text", "hate": "orig"})
    df["label"] = df["orig"].astype(int)
    df["lang"] = "bn"
    df["label_en3"] = -1
    return df[["text", "label", "label_en3", "lang"]]


def load_tamil() -> pd.DataFrame:
    """Tamil YouTube comments from the DravidianCodeMix shared task.

    The corpus is 81% romanised, and script-based routing cannot recognise those
    rows as Tamil, so I keep only the rows actually written in Tamil script. That
    leaves 7,357 comments at 19.6% offensive, which is a far better base rate than
    the Telugu corpus this replaced.
    """
    ds = load_dataset(DATASETS["ta"]["hf_id"], "tamil")
    names = ds["train"].features["label"].names
    df = concatenate_datasets([ds[s] for s in ds.keys()]).to_pandas()
    df["lname"] = df["label"].map(lambda i: names[i])
    # `not-Tamil` is a language-identification bucket, not a hate label.
    df = df[~df["lname"].str.lower().str.startswith("not-")]
    df["label"] = (df["lname"] != "Not_offensive").astype(int)

    lo, hi = SCRIPT_RANGES["ta"][0]

    def native_share(t):
        letters = [c for c in str(t) if c.isalpha()]
        if not letters:
            return 0.0
        return sum(1 for c in letters if lo <= ord(c) <= hi) / len(letters)

    before = len(df)
    df = df[df["text"].map(native_share) > MIN_NATIVE_SCRIPT]
    print(f"[ta] kept {len(df)} of {before} rows written in Tamil script "
          f"({len(df) / before:.0%}); the rest are romanised code-mixed")
    df["lang"] = "ta"
    df["label_en3"] = -1
    return df[["text", "label", "label_en3", "lang"]]


LOADERS = {
    "en": load_english,
    "en_mhs": load_english_mhs,
    "hi": load_hindi,
    "bn": load_bengali,
    "ta": load_tamil,
}


def build() -> pd.DataFrame:
    frames = []
    for src, fn in LOADERS.items():
        print(f"[{src}] downloading {DATASETS[src]['hf_id']} ...", flush=True)
        df = fn()
        before = len(df)
        df["text"] = df["text"].map(clean_text)
        # Drop empties and near-empties, then exact duplicates within a source.
        df = df[df["text"].str.len() >= 5]
        df = df.drop_duplicates(subset=["text"])
        df["source"] = src
        print(
            f"[{src}] {before} rows -> {len(df)} after cleaning "
            f"({before - len(df)} empty/duplicate), toxic={df['label'].mean():.1%}"
        )
        frames.append(df)
    pooled = pd.concat(frames, ignore_index=True)
    # The two English corpora overlap a little, and a text appearing in both
    # splits would leak straight from train into test.
    n = len(pooled)
    pooled = pooled.drop_duplicates(subset=["text"])
    print(f"\npooled {n} -> {len(pooled)} after cross-source dedup")
    return pooled


def split(df: pd.DataFrame) -> pd.DataFrame:
    """Stratified 70/15/15 on language and label jointly, so every language keeps
    its toxic ratio in all three splits. Hindi and Tamil are small enough that
    an unstratified split would skew their base rates."""
    strat = df["lang"] + "_" + df["label"].astype(str)
    train, rest = train_test_split(
        df, test_size=0.30, random_state=SEED, stratify=strat
    )
    strat_rest = rest["lang"] + "_" + rest["label"].astype(str)
    val, test = train_test_split(
        rest, test_size=0.50, random_state=SEED, stratify=strat_rest
    )
    train, val, test = train.copy(), val.copy(), test.copy()
    train["split"], val["split"], test["split"] = "train", "val", "test"
    return pd.concat([train, val, test], ignore_index=True)


def main() -> int:
    df = split(build())
    out = DATA_PROC / "hate_speech_multilingual.parquet"
    df.to_parquet(out, index=False)

    print(f"\nwrote {out}  ({len(df):,} rows)")
    summary = (
        df.groupby(["lang", "split"])
        .agg(n=("label", "size"), toxic=("label", "mean"))
        .reset_index()
        .pivot(index="lang", columns="split", values=["n", "toxic"])
    )
    print(summary.to_string(float_format=lambda v: f"{v:.3f}"))
    print("\nby source:")
    print(
        df.groupby("source")
        .agg(n=("label", "size"), toxic=("label", "mean"))
        .to_string(float_format=lambda v: f"{v:.3f}")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
