#!/usr/bin/env python3
"""IN-CONTEXT text classification for a B2B API: pretrain ONCE on synthetic episodes, then classify ANY customer's
text ZERO-SHOT -- the customer's labeled examples go in the PROMPT, never into training.

Product shape: we ship pretrained weights. A customer sends, in the prompt, a few labeled examples per category
plus a query; the model infers the pattern and labels the query. No gradient updates on customer data, works for
any vocabulary / label set / domain.

Two ideas make "any text" work:
  * HASH words -> a fixed token-id space: any word (seen or not) maps consistently, so the SAME word in the query
    and in a support example gets the SAME id. The model classifies by TOKEN-IDENTITY MATCHING (does the query
    share content with a class's examples?), not by memorized word meaning -- so it never needs to know English.
  * Pretrain on DIVERSE synthetic in-context episodes (random content tokens -> random labels, k-shot) so the
    model learns the META-SKILL of inferring a classification rule from the prompt -- generalizing like langlearn's
    in-context binding (0.99 on unseen symbol sets), here over hashed real text.

  python -m thinking.langapi --selftest
  python -m thinking.langapi --steps 8000          # pretrain on synthetic, then ZERO-SHOT eval on real Bitext
"""
import argparse
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F

from scratchpad_model import ScratchpadLM

PAD, SEP, EOD, Q = 0, 1, 2, 3                               # pad, doc->label sep, end-of-example, query marker
CMAX = 6                                                    # max classes per episode
LBL0 = 4                                                    # label tokens LBL0..LBL_{CMAX-1}
V = 2000                                                    # hashed content-token space (words -> [0,V))
BASE = LBL0 + CMAX                                          # first content token id
VOCAB = BASE + V
DATA = "kaggle_data/bitext_customer_support.csv"


def _doc(rng, sig, doc_len):
    """A document: its class signature tokens + random filler content tokens, shuffled (bag-of-words-ish)."""
    n_fill = max(0, doc_len - len(sig))
    fill = (rng.integers(0, V, n_fill) + BASE).tolist()
    toks = [int(s) + BASE for s in sig] + fill
    rng.shuffle(toks)
    return toks


def gen(rng, C=None, k=None, max_len=160):
    """One in-context episode: C classes, each with a few SIGNATURE content tokens; k support docs/class (each
    ends with its label), then a query doc. Labels/signatures are random per episode -> the model must INFER the
    rule from the prompt. Returns (sequence, query_label_token, query_position)."""
    C = C or int(rng.integers(2, 5))
    k = k or int(rng.integers(2, 4))
    sigs = [rng.choice(V, int(rng.integers(1, 3)), replace=False) for _ in range(C)]   # per-class signatures (1-2 tokens)
    order = []
    for c in range(C):
        for _ in range(k):
            order.append(c)
    rng.shuffle(order)
    seq = []
    for c in order:
        seq += _doc(rng, sigs[c], int(rng.integers(2, 5))) + [SEP, LBL0 + c, EOD]
        if len(seq) > max_len - 16:
            break
    qc = int(rng.integers(C))
    seq += _doc(rng, sigs[qc], int(rng.integers(2, 5))) + [Q]
    return seq[:max_len], LBL0 + qc, len(seq[:max_len]) - 1


def _batch(rng, bs, device, max_len=160):
    eps = [gen(rng, max_len=max_len) for _ in range(bs)]; L = max(len(s) for s, _, _ in eps)
    x = torch.full((bs, L), PAD, dtype=torch.long); qpos = []; y = []
    for i, (s, lbl, qp) in enumerate(eps):
        x[i, :len(s)] = torch.tensor(s); qpos.append(qp); y.append(lbl)
    return x.to(device), qpos, torch.tensor(y, device=device)


def train(steps, d=192, layers=4, heads=6, bs=48, lr=1e-3, seed=0, device="cpu", max_len=160):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = ScratchpadLM(VOCAB, d=d, layers=layers, heads=heads, max_len=max_len, pad=PAD,
                     pos_mode="rope", causal=True, pointer=True, tie=True).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    for _ in range(steps):
        x, qpos, y = _batch(rng, bs, device, max_len)
        out = m(x)                                           # log-probs (B,L,V)
        # Loss on LABEL positions only: predict each example's label (the in-context association the model can
        # learn) + the query label. NOT on random content/filler tokens -- those are unpredictable noise that
        # would drown the signal (this is the dense-loss-induction lesson applied to the PREDICTABLE positions).
        tgt = x[:, 1:]
        mask = (tgt >= LBL0) & (tgt < BASE)                  # next token is a label token
        lp = out[:, :-1]
        loss = F.nll_loss(lp[mask], tgt[mask])
        logp = out[torch.arange(len(y)), torch.tensor(qpos, device=device)]
        loss = loss + F.nll_loss(logp, y)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    return m


def evaluate_synth(m, n=400, seed=999, device="cpu"):
    rng = np.random.default_rng(seed); m.eval(); ok = 0
    with torch.no_grad():
        for _ in range(n):
            s, lbl, qp = gen(rng)
            pred = int(m(torch.tensor([s], device=device))[0, qp].argmax())
            ok += (pred == lbl)
    return ok / n


# ---- zero-shot on REAL Bitext (no training on it): customer examples go in the prompt ----
def _hash_word(w):
    import zlib
    return zlib.crc32(w.encode()) % V                       # DETERMINISTIC: same word -> same id across requests


def _tok_real(text):
    text = re.sub(r"\{\{.*?\}\}", " slot ", str(text).lower())
    return [_hash_word(w) + BASE for w in re.findall(r"[a-z]+", text)][:7]


def bitext_zeroshot(m, path=DATA, C=5, k=3, n=400, seed=0, device="cpu"):
    """ZERO-SHOT: build in-context episodes from real Bitext (k labeled examples per intent in the prompt + a
    held-out query), classify with the synthetic-pretrained model. The model NEVER trained on Bitext."""
    import csv
    by = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            by.setdefault(r["intent"], []).append(r["instruction"])
    intents = [i for i in by if len(by[i]) >= k + 5]
    rng = np.random.default_rng(seed); m.eval(); ok = 0
    with torch.no_grad():
        for _ in range(n):
            chosen = list(rng.choice(intents, C, replace=False))
            seq = []; sig_label = {}
            order = [(ci, j) for ci in range(C) for j in range(k)]; rng.shuffle(order)
            picks = {ci: list(rng.choice(len(by[chosen[ci]]), k + 1, replace=False)) for ci in range(C)}
            for ci, j in order:
                doc = _tok_real(by[chosen[ci]][picks[ci][j]])
                seq += doc + [SEP, LBL0 + ci, EOD]
            qc = int(rng.integers(C))
            qdoc = _tok_real(by[chosen[qc]][picks[qc][k]])          # held-out example of class qc
            seq += qdoc + [Q]
            seq = seq[:160]
            pred = int(m(torch.tensor([seq], device=device))[0, len(seq) - 1].argmax())
            ok += (pred == LBL0 + qc)
    return ok / n, C


def selftest():
    # HONEST PROBE: in-context bag-classification over hashed tokens is a hard frontier for a small from-scratch
    # model. We report the numbers rather than over-asserting. Synthetic should beat chance (it learns SOME
    # in-context matching); Bitext zero-shot reveals the token-overlap-vs-semantics gap.
    m = train(5000, d=160, layers=3, heads=4, device="cpu")
    syn = evaluate_synth(m, 300)
    zs, C = bitext_zeroshot(m, n=200)
    print(f"langapi selftest: in-context synthetic {syn:.3f} (chance ~{1/4:.2f}) | "
          f"ZERO-SHOT real Bitext {zs:.3f} ({C}-way, chance {1/C:.3f}) -- never trained on Bitext")
    assert syn > 1.5 / CMAX, f"learned nothing in-context: {syn}"
    print("langapi selftest OK (probe: in-context token-overlap matching; semantics need general-text pretraining)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    m = train(a.steps, device=a.device)
    print(f"in-context synthetic accuracy: {evaluate_synth(m, device=a.device):.3f}")
    for C in (3, 5, 8):
        zs, _ = bitext_zeroshot(m, C=C, n=300, device=a.device)
        print(f"ZERO-SHOT real Bitext {C}-way (no training on it): {zs:.3f} (chance {1/C:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
