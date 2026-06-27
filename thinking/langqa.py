#!/usr/bin/env python3
"""Customer-support intent QA with a langtext-style looped encoder (real Kaggle data, verified on labels).

Applies the langtext machinery to a REAL dataset (Bitext customer support, 27K question->intent->response rows):
  * a LOOPED weight-tied block with INPUT INJECTION encodes the question ("think" T steps).
  * a learned per-token SALIENCE gate distinguishes content words from function words WITHOUT a hardcoded
    stoplist -- the model learns that function words (the/i/a, in every intent) carry no signal and downweights
    them, while content words (cancel/refund/invoice) are upweighted because they predict the intent. This is the
    recurrence/content idea from langtext/langrole, learned end-to-end and INSPECTABLE (we print the ranking).
  * the intent LABEL is the verifier (held-out accuracy); a correct intent retrieves that intent's templated
    response.

No hardcoded word lists, no pretrained model -- vocab and structure are learned from the data.

  python -m thinking.langqa --selftest
  python -m thinking.langqa --steps 4000 --data kaggle_data/bitext_customer_support.csv
"""
import argparse
import re
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from thinking.reasoner import Block

PAD, UNK = 0, 1
DATA = "kaggle_data/bitext_customer_support.csv"


def tokenize(text):
    text = re.sub(r"\{\{.*?\}\}", " <slot> ", str(text).lower())
    return re.findall(r"<slot>|[a-z]+", text)


def load(path, seed=0, max_len=40):
    import csv
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append((tokenize(r["instruction"]), r["intent"], r["response"]))
    rng = np.random.default_rng(seed); idx = rng.permutation(len(rows))
    cut = int(0.9 * len(rows))
    tr_i, te_i = idx[:cut], idx[cut:]
    # vocab + intents from TRAIN only
    from collections import Counter
    wc = Counter(w for i in tr_i for w in rows[i][0])
    itos = ["<pad>", "<unk>"] + [w for w, c in wc.most_common() if c >= 2]
    stoi = {w: i for i, w in enumerate(itos)}
    intents = sorted({rows[i][1] for i in tr_i})
    i2id = {t: k for k, t in enumerate(intents)}
    # a templated response per intent = its most common response in train
    rc = {t: Counter() for t in intents}
    for i in tr_i:
        rc[rows[i][1]][rows[i][2]] += 1
    template = {t: rc[t].most_common(1)[0][0] for t in intents}

    def enc(toks):
        ids = [stoi.get(w, UNK) for w in toks][:max_len]
        return ids + [PAD] * (max_len - len(ids))

    def pack(ii):
        X = torch.tensor([enc(rows[i][0]) for i in ii])
        y = torch.tensor([i2id[rows[i][1]] for i in ii])
        return X, y

    Xtr, ytr = pack(tr_i); Xte, yte = pack(te_i)
    return Xtr, ytr, Xte, yte, itos, intents, template


class IntentNet(nn.Module):
    def __init__(self, vocab, n_intent, d=160, h=4, max_len=40):
        super().__init__()
        self.tok = nn.Embedding(vocab, d, padding_idx=PAD)
        self.pos = nn.Embedding(max_len, d)
        self.gate = nn.Linear(d, 1)                          # learned per-token salience (content vs function)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.block = Block(d, h)                             # ONE tied block, reused every loop
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_intent))

    def load_block(self, block_state):
        """Transplant a PRETRAINED looped Block (the reasoning core) from langtext. Vocab-agnostic: it operates on
        d-dim vectors, so only the Block transfers; token embeddings + intent head are learned on QA."""
        self.block.load_state_dict(block_state)

    def forward(self, ids, T=4, return_sal=False):
        B, L = ids.shape
        kpm = torch.cat([ids == PAD, torch.zeros(B, 1, dtype=torch.bool, device=ids.device)], 1)
        e = self.tok(ids)
        sal = torch.sigmoid(self.gate(e))                    # (B,L,1) -- learned, no hardcoded stoplist
        x = e * sal + self.pos(torch.arange(L, device=ids.device))[None]
        x = torch.cat([x, self.cls.expand(B, -1, -1)], 1)
        h = torch.zeros_like(x)
        for _ in range(max(1, T)):
            h = self.block(h + x, kpm)                        # INPUT INJECTION + weight-tied recurrence
        logits = self.head(h[:, -1])
        return (logits, sal) if return_sal else logits


def train(Xtr, ytr, n_intent, vocab, steps=4000, d=160, h=4, bs=128, lr=1e-3, seed=0, device="cpu",
          block_state=None):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = IntentNet(vocab, n_intent, d=d, h=h, max_len=Xtr.shape[1]).to(device)
    if block_state is not None:
        m.load_block(block_state)                            # warm-start the looped reasoner from pretrained langtext
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    for _ in range(steps):
        bi = rng.integers(0, len(Xtr), bs)
        xb = Xtr[bi].to(device); yb = ytr[bi].to(device)
        loss = F.cross_entropy(m(xb), yb)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    return m


def pretrain_block(steps=5000, d=160, h=4, device="cpu"):
    """Pretrain langtext's looped reasoner on its SYNTHETIC multi-hop walk; return its Block weights to transplant.
    d/h must match the QA encoder so the Block transfers."""
    from thinking import langtext
    m = langtext.train(steps, lo=3, hi=8, mu=5.0, sigma=2.0, d=d, h=h, device=device)
    return m.block.state_dict()


def transfer_experiment(data=DATA, pretrain_steps=5000, ft_steps=1500, sizes=(150, 400, 1000, 3000),
                        d=160, h=4, device="cpu", seed=0):
    """Does pretraining the looped reasoner on synthetic chains help REAL customer-support QA? Compare scratch vs
    pretrained-Block init across train sizes (transfer should help most when target data is scarce)."""
    Xtr, ytr, Xte, yte, itos, intents, _ = load(data, seed=seed, max_len=40)
    print(f"pretraining langtext looped reasoner ({pretrain_steps} steps) ...", flush=True)
    block_state = pretrain_block(steps=pretrain_steps, d=d, h=h, device=device)
    rng = np.random.default_rng(seed)
    print(f"{'train_size':>10} | {'scratch':>8} | {'pretrained':>10} | {'gain':>6}")
    for sz in sizes:
        sub = rng.integers(0, len(Xtr), min(sz, len(Xtr)))
        Xs, ys = Xtr[sub], ytr[sub]
        ms = train(Xs, ys, len(intents), len(itos), steps=ft_steps, d=d, h=h, seed=seed, device=device)
        mp = train(Xs, ys, len(intents), len(itos), steps=ft_steps, d=d, h=h, seed=seed, device=device,
                   block_state=block_state)
        a_s = accuracy(ms, Xte, yte, device=device); a_p = accuracy(mp, Xte, yte, device=device)
        print(f"{sz:>10} | {a_s:>8.3f} | {a_p:>10.3f} | {a_p - a_s:>+6.3f}", flush=True)
    return 0


@torch.no_grad()
def accuracy(m, X, y, device="cpu", bs=512):
    m.eval(); ok = 0
    for i in range(0, len(X), bs):
        pred = m(X[i:i + bs].to(device)).argmax(-1).cpu()
        ok += int((pred == y[i:i + bs]).sum())
    return ok / len(X)


@torch.no_grad()
def salience_ranking(m, itos, device="cpu"):
    """Each token's learned salience (content words high, function words low) -- demonstrates content extraction."""
    m.eval()
    ids = torch.arange(len(itos), device=device)[None]
    e = m.tok(ids)
    sal = torch.sigmoid(m.gate(e))[0, :, 0].cpu().numpy()
    order = np.argsort(-sal)
    top = [(itos[i], round(float(sal[i]), 3)) for i in order[:15] if itos[i] not in ("<pad>", "<unk>")]
    bot = [(itos[i], round(float(sal[i]), 3)) for i in order[::-1][:15] if itos[i] not in ("<pad>", "<unk>")]
    return top, bot


def selftest():
    Xtr, ytr, Xte, yte, itos, intents, template = load(DATA, max_len=40)
    # small/fast: subsample train
    sub = np.random.default_rng(0).integers(0, len(Xtr), 4000)
    m = train(Xtr[sub], ytr[sub], len(intents), len(itos), steps=1500, d=96, h=4, device="cpu")
    acc = accuracy(m, Xte, yte)
    chance = 1.0 / len(intents)
    print(f"langqa selftest: held-out intent accuracy {acc:.3f} (chance {chance:.3f}, {len(intents)} intents)")
    top, _ = salience_ranking(m, itos)
    print("  top learned-salience tokens (content):", [w for w, _ in top[:10]])
    assert acc > 0.5, f"intent classifier failed: {acc}"
    print("langqa selftest OK (langtext-style looped encoder classifies real customer-support intents, "
          "verified on labels; content words learned via salience, no hardcoded stoplist)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--transfer", action="store_true", help="pretrain langtext -> transfer Block -> QA (vs scratch)")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--d", type=int, default=160)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.transfer:
        return transfer_experiment(data=a.data, d=a.d, device=a.device)
    Xtr, ytr, Xte, yte, itos, intents, template = load(a.data, max_len=40)
    print(f"loaded {len(Xtr)} train / {len(Xte)} test; vocab {len(itos)}; intents {len(intents)}")
    m = train(Xtr, ytr, len(intents), len(itos), steps=a.steps, d=a.d, device=a.device)
    acc = accuracy(m, Xte, yte, device=a.device)
    print(f"held-out intent accuracy: {acc:.3f} (chance {1/len(intents):.3f})")
    top, bot = salience_ranking(m, itos, device=a.device)
    print("learned content words (high salience):", [w for w, _ in top])
    print("learned function words (low salience):", [w for w, _ in bot])
    # demo: a few held-out questions -> predicted intent -> templated response
    m.eval()
    with torch.no_grad():
        for i in range(3):
            pred = int(m(Xte[i:i + 1].to(a.device)).argmax(-1))
            print(f"\n  Q tokens -> intent: {intents[pred]}\n  templated response: {template[intents[pred]][:120]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
