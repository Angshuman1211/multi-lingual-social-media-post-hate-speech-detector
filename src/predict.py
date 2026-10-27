"""Inference wrapper the web interface talks to.

Holds the baseline pipelines, the fine-tuned MuRIL model if it has been trained,
the tuned decision thresholds, and one explanation method that works for both
kinds of model.
"""

import json
from functools import lru_cache

import joblib
import numpy as np

from src.config import LABELS_EN3, LANGUAGES, MAX_LEN, MODELS
from src.preprocess import clean_text, detect_language


def _load_thresholds():
    path = MODELS / "baseline_thresholds.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"per_language": {}, "joint": {}}


class Detector:
    """Loads lazily so the app starts fast and still runs if MuRIL is absent."""

    def __init__(self):
        self._baselines = {}
        self._en3 = None
        self._muril = None
        self._muril_tok = None
        self._muril_thr = {}
        self.thresholds = _load_thresholds()

    # ---------- model loading ----------

    def baseline(self, lang: str):
        """Per-language model where I have one, joint model otherwise."""
        key = lang if (MODELS / f"baseline_{lang}.joblib").exists() else "joint"
        if key not in self._baselines:
            self._baselines[key] = joblib.load(MODELS / f"baseline_{key}.joblib")
        return self._baselines[key], key

    def en3(self):
        if self._en3 is None:
            p = MODELS / "baseline_en3.joblib"
            self._en3 = joblib.load(p) if p.exists() else False
        return self._en3

    def muril_available(self) -> bool:
        return (MODELS / "muril_all" / "config.json").exists()

    def muril(self):
        if self._muril is None:
            if not self.muril_available():
                self._muril = False
                return False, None
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            d = MODELS / "muril_all"
            self._muril_tok = AutoTokenizer.from_pretrained(str(d))
            # low_cpu_mem_usage builds the model on the meta device first and
            # materialises it as the checkpoint streams in. Under Streamlit that
            # left the weights on meta and every prediction died on .cpu(), so I
            # allocate real storage up front.
            self._muril = AutoModelForSequenceClassification.from_pretrained(
                str(d), low_cpu_mem_usage=False
            )
            self._muril.eval()
            self._device = torch.device(
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
            self._muril.to(self._device)
            stranded = [
                n for n, p in self._muril.named_parameters() if p.device.type == "meta"
            ]
            if stranded:
                raise RuntimeError(
                    f"{len(stranded)} MuRIL parameters are still on the meta device "
                    f"(first: {stranded[:3]}). The checkpoint at {d} did not load."
                )
            thr = d / "thresholds.json"
            if thr.exists():
                with open(thr) as f:
                    self._muril_thr = json.load(f)
        if self._muril is False:
            return False, None
        return self._muril, self._muril_tok

    # ---------- scoring ----------

    def _score_baseline(self, texts, lang):
        pipe, used = self.baseline(lang)
        probs = pipe.predict_proba(list(texts))[:, 1]
        return probs, used

    def _score_muril(self, texts):
        import torch

        model, tok = self.muril()
        if model is False:
            return None
        out = []
        for i in range(0, len(texts), 32):
            enc = tok(
                list(texts[i : i + 32]),
                truncation=True,
                max_length=MAX_LEN,
                padding=True,
                return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                logits = model(**enc).logits
            out.append(torch.softmax(logits.float(), -1)[:, 1].cpu().numpy())
        return np.concatenate(out)

    def threshold(self, lang: str, model: str) -> float:
        if model == "muril":
            return float(self._muril_thr.get(lang, 0.5))
        return float(self.thresholds.get("per_language", {}).get(lang, 0.5))

    def predict(self, text: str, model: str = "baseline", lang: str = None) -> dict:
        raw = text
        cleaned = clean_text(text)
        lang = lang or detect_language(cleaned)

        if model == "muril" and self.muril_available():
            prob = float(self._score_muril([cleaned])[0])
            used = "MuRIL (fine-tuned)"
        else:
            p, key = self._score_baseline([cleaned], lang)
            prob = float(p[0])
            used = f"TF-IDF + LogReg ({key})"
            model = "baseline"

        thr = self.threshold(lang, model)
        out = {
            "text": raw,
            "cleaned": cleaned,
            "language": lang,
            "language_name": LANGUAGES.get(lang, lang),
            "probability": prob,
            "threshold": thr,
            "label": int(prob >= thr),
            "label_name": "Toxic" if prob >= thr else "Clean",
            "model": used,
        }

        # English is the only corpus that separates hate from merely offensive.
        if lang == "en" and out["label"] == 1 and self.en3() is not False:
            p3 = self.en3().predict_proba([cleaned])[0]
            out["en3"] = {LABELS_EN3[i]: float(p3[i]) for i in range(len(p3))}
            out["en3_top"] = LABELS_EN3[int(np.argmax(p3))]
        return out

    def predict_batch(self, texts, model: str = "baseline"):
        cleaned = [clean_text(t) for t in texts]
        langs = [detect_language(c) for c in cleaned]
        probs = np.zeros(len(cleaned))

        if model == "muril" and self.muril_available():
            probs = self._score_muril(cleaned)
            used = "MuRIL (fine-tuned)"
        else:
            # Route each language to its own baseline.
            for lang in set(langs):
                idx = [i for i, l in enumerate(langs) if l == lang]
                p, _ = self._score_baseline([cleaned[i] for i in idx], lang)
                probs[idx] = p
            used = "TF-IDF + LogReg"
            model = "baseline"

        thr = np.array([self.threshold(l, model) for l in langs])
        return {
            "text": list(texts),
            "language": langs,
            "probability": probs,
            "label": (probs >= thr).astype(int),
            "model": used,
        }

    # ---------- explanation ----------

    def explain(self, text: str, model: str = "baseline", top_k: int = 8) -> list:
        """Leave-one-token-out occlusion.

        I drop each whitespace token in turn and measure how far the toxic
        probability falls. Positive values mean the token pushed the text toward
        toxic. The same routine works for the linear model and the transformer,
        so the interface explains both the same way.
        """
        cleaned = clean_text(text)
        tokens = cleaned.split()
        if not tokens:
            return []
        # Cap the work so a pasted paragraph cannot stall the interface.
        if len(tokens) > 60:
            tokens = tokens[:60]
        variants = [cleaned] + [
            " ".join(tokens[:i] + tokens[i + 1 :]) for i in range(len(tokens))
        ]
        lang = detect_language(cleaned)

        if model == "muril" and self.muril_available():
            probs = self._score_muril(variants)
        else:
            probs, _ = self._score_baseline(variants, lang)

        base = probs[0]
        contrib = [(tokens[i], float(base - probs[i + 1])) for i in range(len(tokens))]
        contrib.sort(key=lambda kv: abs(kv[1]), reverse=True)
        return contrib[:top_k]


@lru_cache(maxsize=1)
def get_detector() -> Detector:
    return Detector()
