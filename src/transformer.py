"""Fine-tune MuRIL for multilingual toxic / clean classification.

Run:  python -m src.transformer                      # all four languages
      python -m src.transformer --exclude-lang te    # zero-shot transfer run

I wrote the loop by hand rather than using Trainer so the class weighting, the
MPS placement and the per-language validation are all visible in one place.
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src import evaluate as ev
from src.config import DATA_PROC, MAX_LEN, MODELS, RESULTS, SEED, TRANSFORMER_MODEL


class TextData(Dataset):
    def __init__(self, df):
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], self.labels[i]


def make_collate(tokenizer):
    def collate(batch):
        texts, labels = zip(*batch)
        enc = tokenizer(
            list(texts),
            truncation=True,
            max_length=MAX_LEN,
            padding=True,
            # MPS recompiles its graph for every new input shape, and free
            # dynamic padding gave a different length almost every batch, which
            # cost 14 minutes over the first 100 steps. Bucketing to multiples of
            # 32 leaves four possible shapes instead of 128.
            pad_to_multiple_of=32,
            return_tensors="pt",
        )
        enc["labels"] = torch.tensor(labels, dtype=torch.long)
        return enc

    return collate


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs = []
    for batch in loader:
        labels = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        probs.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--exclude-lang", default=None,
                    help="hold a language out of training for a zero-shot transfer test")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="subsample every split, for smoke tests only")
    ap.add_argument("--train-embeddings", action="store_true",
                    help="also fine-tune the word embedding matrix (2.5x slower)")
    args = ap.parse_args()

    tag = args.tag or (f"no_{args.exclude_lang}" if args.exclude_lang else "all")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device} | tag: {tag}", flush=True)

    df = pd.read_parquet(DATA_PROC / "hate_speech_multilingual.parquet")
    train = df[df.split == "train"].copy()
    val = df[df.split == "val"].copy()
    test = df[df.split == "test"].copy()

    if args.limit:

        def _sub(x):
            idx = []
            for _, g in x.groupby("lang"):
                idx.extend(g.sample(min(len(g), args.limit), random_state=SEED).index)
            return x.loc[idx]

        train, val, test = (_sub(x) for x in (train, val, test))
    if args.exclude_lang:
        train = train[train.lang != args.exclude_lang]
        val = val[val.lang != args.exclude_lang]
    print(f"train rows: {len(train):,}  ({dict(train.lang.value_counts())})", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        TRANSFORMER_MODEL, num_labels=2
    ).to(device)

    # MuRIL's multilingual vocab is 197k, so the word embedding matrix alone is
    # 151M of its 238M parameters. Freezing it cuts the step cost by 2.5x on MPS
    # and there is little to gain from re-learning embeddings on 75k examples.
    if not args.train_embeddings:
        model.bert.embeddings.word_embeddings.weight.requires_grad_(False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {trainable / 1e6:.0f}M of "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.0f}M", flush=True)

    collate = make_collate(tokenizer)
    dl_train = DataLoader(TextData(train), batch_size=args.batch_size, shuffle=True,
                          collate_fn=collate)
    dl_val = DataLoader(TextData(val), batch_size=64, collate_fn=collate)
    dl_test = DataLoader(TextData(test), batch_size=64, collate_fn=collate)

    # Inverse-frequency weights, so the toxic class is not drowned out.
    counts = train["label"].value_counts().sort_index().to_numpy()
    weights = torch.tensor(counts.sum() / (2.0 * counts), dtype=torch.float32).to(device)
    print(f"class weights: {weights.tolist()}", flush=True)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01
    )
    total_steps = len(dl_train) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optim, max_lr=args.lr, total_steps=total_steps, pct_start=0.1, anneal_strategy="linear"
    )

    best_score, history = -1.0, []
    out_dir = MODELS / f"muril_{tag}"
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        for step, batch in enumerate(dl_train, 1):
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = loss_fn(model(**batch).logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
            running += loss.item()
            if step % 100 == 0:
                rate = step / (time.time() - t0)
                eta = (len(dl_train) - step) / max(rate, 1e-9) / 60
                print(f"  ep{epoch} {step}/{len(dl_train)} loss={running/step:.4f} "
                      f"{rate:.1f} it/s eta {eta:.1f}m", flush=True)

        vprob = predict(model, dl_val, device)
        vres = ev.per_language(val, (vprob >= 0.5).astype(int), vprob)
        score = vres["macro_over_languages"]
        history.append({"epoch": epoch, "val": vres})
        print(f"\n[epoch {epoch}] val macro-F1 over languages = {score:.4f}")
        print(ev.report_table(vres), flush=True)
        if score > best_score:
            best_score = score
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            print(f"  saved -> {out_dir}", flush=True)

    # Reload the best checkpoint before touching test.
    model = AutoModelForSequenceClassification.from_pretrained(out_dir).to(device)
    vprob = predict(model, dl_val, device)
    tprob = predict(model, dl_test, device)

    thr = {}
    for lang in sorted(val["lang"].unique()):
        m = (val.lang == lang).to_numpy()
        thr[lang] = ev.tune_threshold(val.loc[m, "label"], vprob[m])
    # A held-out language has no validation data, so it keeps the default.
    for lang in sorted(test["lang"].unique()):
        thr.setdefault(lang, 0.5)

    pred_default = (tprob >= 0.5).astype(int)
    pred_tuned = (tprob >= test["lang"].map(thr).to_numpy()).astype(int)

    res = {
        "tag": tag,
        "args": vars(args),
        "val_history": history,
        "thresholds": thr,
        "test_default_threshold": ev.per_language(test, pred_default, tprob),
        "test_tuned_threshold": ev.per_language(test, pred_tuned, tprob),
    }
    for lang in sorted(test["lang"].unique()):
        m = (test.lang == lang).to_numpy()
        res["test_tuned_threshold"][lang]["confusion"] = ev.cm(
            test.loc[m, "label"], pred_tuned[m]
        )

    print("\n=== TEST at default 0.5 ===")
    print(ev.report_table(res["test_default_threshold"]))
    print(f"\n=== TEST at tuned thresholds {thr} ===")
    print(ev.report_table(res["test_tuned_threshold"]))

    with open(RESULTS / f"muril_{tag}_results.json", "w") as f:
        json.dump(res, f, indent=2)
    with open(out_dir / "thresholds.json", "w") as f:
        json.dump(thr, f, indent=2)
    print(f"\nwrote {RESULTS / f'muril_{tag}_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
