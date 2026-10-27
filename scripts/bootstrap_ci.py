"""Bootstrap confidence intervals on the baseline vs MuRIL gap, per language.

Hindi has 735 test rows and MuRIL scores 0.029 below the baseline there. That is
well inside the range where a point estimate means nothing, so I resample the
test set and report an interval instead of asserting a winner.

Run: python scripts/bootstrap_ci.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import DATA_PROC, LANGUAGES, MODELS, RESULTS  # noqa: E402
from src.predict import get_detector  # noqa: E402

N_BOOT = 2000
SEED = 42


def main() -> int:
    df = pd.read_parquet(DATA_PROC / "hate_speech_multilingual.parquet")
    test = df[df.split == "test"].reset_index(drop=True)

    det = get_detector()
    if not det.muril_available():
        print("MuRIL is not trained yet.")
        return 1

    with open(MODELS / "baseline_thresholds.json") as f:
        base_thr = json.load(f)["per_language"]
    with open(MODELS / "muril_all" / "thresholds.json") as f:
        muril_thr = json.load(f)

    rng = np.random.default_rng(SEED)
    rows = []
    for lang in ["en", "hi", "bn", "ta"]:
        part = test[test.lang == lang].reset_index(drop=True)
        y = part["label"].to_numpy()

        pipe = joblib.load(MODELS / f"baseline_{lang}.joblib")
        pb = pipe.predict_proba(part["text"].tolist())[:, 1]
        pm = det._score_muril(part["text"].tolist())

        base_pred = (pb >= base_thr[lang]).astype(int)
        muril_pred = (pm >= muril_thr.get(lang, 0.5)).astype(int)

        def macro(yy, pp):
            return f1_score(yy, pp, average="macro", zero_division=0)

        obs = macro(y, muril_pred) - macro(y, base_pred)
        diffs = np.empty(N_BOOT)
        n = len(y)
        for b in range(N_BOOT):
            idx = rng.integers(0, n, n)
            if y[idx].sum() in (0, len(idx)):
                diffs[b] = 0.0
                continue
            diffs[b] = macro(y[idx], muril_pred[idx]) - macro(y[idx], base_pred[idx])
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        # Share of resamples where MuRIL is ahead.
        p_better = float((diffs > 0).mean())
        rows.append({
            "language": LANGUAGES[lang], "n": n, "observed_delta": obs,
            "ci_low": lo, "ci_high": hi, "p_muril_better": p_better,
            "significant": bool(lo > 0 or hi < 0),
        })
        print(f"{LANGUAGES[lang]:<9} n={n:>5}  delta={obs:+.3f}  "
              f"95% CI [{lo:+.3f}, {hi:+.3f}]  "
              f"MuRIL ahead in {p_better:.0%} of resamples  "
              f"{'SIGNIFICANT' if (lo > 0 or hi < 0) else 'not significant'}")

    with open(RESULTS / "bootstrap_ci.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {RESULTS / 'bootstrap_ci.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
