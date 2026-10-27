"""Metric helpers shared by the baseline and the transformer."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def binary_metrics(y_true, y_pred, y_prob=None) -> dict:
    """Macro F1 leads because three of the four languages are imbalanced and
    accuracy flatters the majority class everywhere except Hindi."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1], average="binary", pos_label=1, zero_division=0
    )
    out = {
        "n": int(len(y_true)),
        "n_toxic": int(y_true.sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "toxic_precision": float(p),
        "toxic_recall": float(r),
        "toxic_f1": float(f1),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    return out


def per_language(df, y_pred, y_prob=None) -> dict:
    """Break every metric down by language. A single pooled number would be
    dominated by English purely on row count."""
    res = {}
    y_pred = np.asarray(y_pred)
    y_prob = None if y_prob is None else np.asarray(y_prob)
    for lang in sorted(df["lang"].unique()):
        m = (df["lang"] == lang).to_numpy()
        res[lang] = binary_metrics(
            df.loc[m, "label"].to_numpy(),
            y_pred[m],
            None if y_prob is None else y_prob[m],
        )
    res["overall"] = binary_metrics(df["label"].to_numpy(), y_pred, y_prob)
    # Mean over languages, so Tamil counts as much as English.
    res["macro_over_languages"] = float(
        np.mean([res[l]["macro_f1"] for l in sorted(df["lang"].unique())])
    )
    return res


def report_table(res: dict, langs=("en", "hi", "bn", "ta")) -> str:
    head = f"{'lang':<8}{'n':>7}{'toxic':>7}{'macroF1':>9}{'tox_P':>8}{'tox_R':>8}{'tox_F1':>8}{'AUC':>8}"
    lines = [head, "-" * len(head)]
    for l in list(langs) + ["overall"]:
        if l not in res:
            continue
        m = res[l]
        auc = f"{m['roc_auc']:.3f}" if "roc_auc" in m else "   -  "
        lines.append(
            f"{l:<8}{m['n']:>7}{m['n_toxic']:>7}{m['macro_f1']:>9.3f}"
            f"{m['toxic_precision']:>8.3f}{m['toxic_recall']:>8.3f}{m['toxic_f1']:>8.3f}{auc:>8}"
        )
    return "\n".join(lines)


def tune_threshold(y_true, y_prob, objective="macro_f1") -> float:
    """Pick the decision threshold on validation data instead of assuming 0.5.

    Base rates differ a lot by language, so argmax is rarely the right cut.
    I tune per language and never touch test.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if len(np.unique(y_true)) < 2:
        return 0.5
    best_t, best_s = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        pred = (y_prob >= t).astype(int)
        s = (
            f1_score(y_true, pred, average="macro", zero_division=0)
            if objective == "macro_f1"
            else f1_score(y_true, pred, pos_label=1, zero_division=0)
        )
        if s > best_s:
            best_t, best_s = float(t), float(s)
    return best_t


def text_report(y_true, y_pred, target_names) -> str:
    return classification_report(
        y_true, y_pred, target_names=target_names, zero_division=0, digits=3
    )


def cm(y_true, y_pred) -> list:
    # Pin the labels so a split that predicts one class still returns 2x2.
    return confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
