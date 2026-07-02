#!/usr/bin/env python3
"""Kaggle 'Tweet Sentiment Extraction': given a tweet + its sentiment, extract the PHRASE (span) that expresses that
sentiment. Metric = word-level Jaccard between predicted and gold span (the competition metric). This is pure SPAN
EXTRACTION -- the same shape as our SQuAD reader -- so we solve it two ways:

  1. NO-TRAIN baseline (runtime): return the whole tweet (the known strong floor -- neutral gold ~= whole text, and
     short tweets overlap a lot). Quantifies the floor with zero training.
  2. TRAINED reader (thinking/squad_reader.Reader): frame it as QA -- 'question' = the sentiment word, passage =
     the tweet, label = the selected_text span. Trained ONCE on the public train split, shipped as weights; not an
     LLM. (Train on GPU via a launcher -- local MPS is too slow for the LSTM.)

  python -m thinking.tweet_sent --baseline
  python -m thinking.tweet_sent --selftest
  python -m thinking.tweet_sent --train --epochs 4 --save /tmp/tse.pt     # (GPU)
"""
import argparse
import csv
import sys

import numpy as np

from thinking import squadqa as Q
from thinking import squad_reader as R

DATA = "kaggle_data/tweet_sent/tweet_dataset.csv"
SENTS = {"positive", "negative", "neutral"}


def jaccard(a, b):
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0


def load(path=DATA):
    """Rows with a gold span and a {positive,negative,neutral} sentiment -> ex {q:[sentiment], p:tweet toks, s,e, golds}."""
    out = []
    with open(path) as f:
        for row in csv.DictReader(f):
            sent = (row.get("new_sentiment") or "").strip().lower()
            text, sel = row.get("text", ""), (row.get("selected_text") or "").strip()
            if sent not in SENTS or not sel or not text.strip():
                continue
            offs = R.tok_offsets(text)
            cstart = text.find(sel)
            if cstart < 0:
                continue
            sp = R._span_to_tokens(offs, cstart, sel)
            if sp is None:
                continue
            out.append({"q": [sent], "p": [t for t, _ in offs], "s": sp[0], "e": sp[1], "golds": [sel]})
    return out


def split(exs, frac=0.85, seed=0):
    rng = np.random.default_rng(seed); idx = rng.permutation(len(exs)); k = int(len(exs) * frac)
    return [exs[i] for i in idx[:k]], [exs[i] for i in idx[k:]]


def baseline_whole(exs):
    return np.mean([jaccard(" ".join(e["p"]), e["golds"][0]) for e in exs])


def evaluate_reader(model, exs, stoi, dev, n=None):
    sub = exs[:n] if n else exs
    return float(np.mean([jaccard(R.predict(model, e, stoi, dev), e["golds"][0]) for e in sub]))


def selftest():
    import torch
    exs = load()[:400]
    assert exs, "TSE data not found/empty -- did the kaggle download run?"
    stoi, emb = R.load_glove(100, R._vocab_words([exs]))
    model = R.Reader(emb, hidden=64)
    R.train(model, list(exs[:300]), stoi, torch.device("cpu"), epochs=12, bs=16, lr=2e-3)
    j = evaluate_reader(model, exs[:300], stoi, torch.device("cpu"))
    print(f"tweet_sent selftest: overfit-train Jaccard {j:.3f} (reader fits the sentiment-span task)")
    assert j > 0.55, f"reader not learning the span task: {j}"
    print("tweet_sent selftest OK (span-extraction reader transfers to Tweet Sentiment Extraction)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true"); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--epochs", type=int, default=4); ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dim", type=int, default=100); ap.add_argument("--save", default="/tmp/tse.pt")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    exs = load(); tr, de = split(exs)
    print(f"Tweet Sentiment Extraction: {len(exs)} labeled | train {len(tr)} / dev {len(de)}")
    if a.baseline:
        print(f"  NO-TRAIN baseline (predict whole tweet): dev Jaccard {baseline_whole(de):.3f}")
    if a.train:
        import torch
        from device import get_device
        dev = get_device()
        stoi, emb = R.load_glove(a.dim, R._vocab_words([tr, de]))
        print(f"  train reader: vocab {len(stoi)} | device {dev}", flush=True)
        model = R.Reader(emb, a.hidden).to(dev)
        R.train(model, tr, stoi, dev, epochs=a.epochs)
        torch.save({"state": model.state_dict(), "stoi": stoi, "emb_shape": emb.shape, "hidden": a.hidden}, a.save)
        print(f"  TRAINED reader: dev Jaccard {evaluate_reader(model, de, stoi, dev):.3f} "
              f"(baseline {baseline_whole(de):.3f})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
