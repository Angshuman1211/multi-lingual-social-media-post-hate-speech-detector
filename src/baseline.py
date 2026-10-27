"""TF-IDF + linear classifier baseline.

Trains one model per language, one joint model over all four, and an English
three-way model that keeps Davidson's hate / offensive distinction.

Run: python -m src.baseline
"""

import json
import sys

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from src import evaluate as ev
from src.config import DATA_PROC, LABELS_EN3, MODELS, RESULTS, SEED


def make_pipeline(max_word=60_000, max_char=120_000) -> Pipeline:
    """Word n-grams carry the slurs, character n-grams carry the morphology.

    Bengali and Tamil are agglutinative enough that a word-level vocabulary
    misses most inflected forms, and char_wb also absorbs the spelling noise
    that people use to dodge filters.
    """
    feats = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=max_word,
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 5),
                    min_df=3,
                    max_features=max_char,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    clf = LogisticRegression(
        max_iter=2000,
        C=4.0,
        # Without this the minority class collapses on the skewed languages.
        class_weight="balanced",
        random_state=SEED,
    )
    return Pipeline([("feats", feats), ("clf", clf)])


def fit_eval(train, val, test, name, label_col="label"):
    pipe = make_pipeline()
    pipe.fit(train["text"], train[label_col])
    joblib.dump(pipe, MODELS / f"baseline_{name}.joblib")
    return pipe, pipe.predict_proba(val["text"])[:, 1], pipe.predict_proba(test["text"])[:, 1]


def main() -> int:
    df = pd.read_parquet(DATA_PROC / "hate_speech_multilingual.parquet")
    train = df[df.split == "train"]
    val = df[df.split == "val"]
    test = df[df.split == "test"]
    results = {}
    thresholds = {}

    # 1. One joint model over all four languages, with a threshold per language
    #    tuned on validation.
    print("=== joint multilingual baseline ===", flush=True)
    _, vprob, tprob = fit_eval(train, val, test, "joint")
    thr = {
        lang: ev.tune_threshold(
            val.loc[val.lang == lang, "label"], vprob[(val.lang == lang).to_numpy()]
        )
        for lang in sorted(df["lang"].unique())
    }
    thresholds["joint"] = thr
    pred_default = (tprob >= 0.5).astype(int)
    pred_tuned = (
        tprob >= test["lang"].map(thr).to_numpy()
    ).astype(int)
    results["joint_default_threshold"] = ev.per_language(test, pred_default, tprob)
    results["joint"] = ev.per_language(test, pred_tuned, tprob)
    print("at default 0.5:")
    print(ev.report_table(results["joint_default_threshold"]))
    print(f"\nat tuned thresholds {thr}:")
    print(ev.report_table(results["joint"]))

    # 2. One model per language.
    print("\n=== per-language baselines (tuned thresholds) ===", flush=True)
    per_lang = {}
    thresholds["per_language"] = {}
    for lang in sorted(df["lang"].unique()):
        tr, va, te = (x[x.lang == lang] for x in (train, val, test))
        _, vp, tp = fit_eval(tr, va, te, lang)
        t = ev.tune_threshold(va["label"], vp)
        thresholds["per_language"][lang] = t
        pred_l = (tp >= t).astype(int)
        per_lang[lang] = ev.binary_metrics(te["label"], pred_l, tp)
        per_lang[lang]["threshold"] = t
        per_lang[lang]["confusion"] = ev.cm(te["label"], pred_l)
    results["per_language"] = per_lang
    print(ev.report_table(per_lang))

    # 3. English three-way, the distinction the binary head discards.
    print("\n=== English 3-class (hate / offensive / neither) ===", flush=True)
    # Only Davidson carries the three-way label, so the Berkeley rows sit out.
    tr_en = train[(train.lang == "en") & (train.label_en3 >= 0)]
    te_en = test[(test.lang == "en") & (test.label_en3 >= 0)]
    pipe3 = make_pipeline()
    pipe3.fit(tr_en["text"], tr_en["label_en3"])
    pred3 = pipe3.predict(te_en["text"])
    joblib.dump(pipe3, MODELS / "baseline_en3.joblib")
    rep3 = ev.text_report(te_en["label_en3"], pred3, list(LABELS_EN3.values()))
    print(rep3)
    results["en3"] = {
        "report": rep3,
        "confusion": ev.cm(te_en["label_en3"], pred3),
    }

    results["thresholds"] = thresholds
    with open(RESULTS / "baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(MODELS / "baseline_thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"\nwrote {RESULTS / 'baseline_results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
