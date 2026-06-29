#!/usr/bin/env python3
"""ONE non-LLM span reader trained on MULTIPLE public span-extraction tasks at once, so a single shipped weight
generalizes across the task FAMILY -- the no-LLM realization of 'one base weight, don't train per dataset'.

The base SQuAD-only reader did NOT transfer to Tweet Sentiment Extraction (zero-shot Jaccard 0.177 < 0.395 baseline)
because TSE is a different task TYPE (the 'question' is a sentiment label, never seen). Fix: train the SAME reader
(thinking/squad_reader.Reader) on SQuAD (wh-question -> answer span) AND TSE (sentiment label -> phrase span)
together. It learns BOTH instruction types -> one weight handles QA and sentiment-span. Still not an LLM.

  python -m thinking.multitask --selftest
  python -m thinking.multitask --train --squad-n 30000 --epochs 4 --save /tmp/mt.pt   # (GPU)
"""
import argparse
import sys

import numpy as np
import torch

from device import get_device
from thinking import squad_reader as R
from thinking import tweet_sent as T


def build(squad_n, dim=100):
    sq = R.read_spans(R.TRAIN, n=squad_n)
    sqdev = R.read_spans(R.DEV)
    tse = T.load(); tse_tr, tse_de = T.split(tse)
    train = sq + tse_tr
    stoi, emb = R.load_glove(dim, R._vocab_words([train, sqdev, tse_de]))
    return train, sqdev, tse_de, stoi, emb, len(sq), len(tse_tr)


def eval_both(model, sqdev, tse_de, stoi, dev, sqn=2000, tsen=2000):
    em, f1 = R.evaluate(model, sqdev, stoi, dev, n=sqn)
    j = T.evaluate_reader(model, tse_de, stoi, dev, n=tsen)
    return f1, j


def selftest():
    dev = torch.device("cpu")
    sq = R.read_spans(R.DEV, n=200)                          # use dev as a stand-in corpus for the smoke test
    tse = T.load()[:200]
    train = sq[:150] + tse[:150]
    stoi, emb = R.load_glove(100, R._vocab_words([train, sq, tse]))
    model = R.Reader(emb, hidden=64).to(dev)
    R.train(model, list(train), stoi, dev, epochs=10, bs=16, lr=2e-3)
    f1 = R.evaluate(model, sq[:150], stoi, dev)[1]
    j = T.evaluate_reader(model, tse[:150], stoi, dev)
    print(f"multitask selftest: one reader -> SQuAD F1 {f1:.3f} | TSE Jaccard {j:.3f} (both learned by ONE weight)")
    assert f1 > 0.4 and j > 0.5, f"multitask reader not learning both: f1={f1} j={j}"
    print("multitask selftest OK (single non-LLM weight handles QA + sentiment-span)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true"); ap.add_argument("--train", action="store_true")
    ap.add_argument("--squad-n", type=int, default=30000); ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--dim", type=int, default=100); ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--save", default="/tmp/mt.pt")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.train:
        dev = get_device()
        train, sqdev, tse_de, stoi, emb, nsq, ntse = build(a.squad_n, a.dim)
        print(f"MULTITASK train: SQuAD {nsq} + TSE {ntse} = {len(train)} | vocab {len(stoi)} | device {dev}", flush=True)
        model = R.Reader(emb, a.hidden).to(dev)
        R.train(model, train, stoi, dev, epochs=a.epochs)
        torch.save({"state": model.state_dict(), "stoi": stoi, "emb_shape": emb.shape, "hidden": a.hidden}, a.save)
        f1, j = eval_both(model, sqdev, tse_de, stoi, dev, sqn=0, tsen=0)
        print(f"ONE WEIGHT -> SQuAD dev F1 {f1:.3f} | TSE dev Jaccard {j:.3f} "
              f"(SQuAD-only base was: SQuAD 0.690 / TSE zero-shot 0.177; TSE baseline 0.395)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
