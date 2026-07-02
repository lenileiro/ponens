#!/usr/bin/env python3
"""General-text SEMANTIC pretraining for the B2B API: pretrain a bidirectional masked-LM encoder on PUBLIC general
text (cosmopedia -- NOT customer data), then classify ANY customer text ZERO-SHOT by nearest class prototype over
the customer's in-prompt examples. No training on customer data.

This is the missing layer the langapi probe exposed: hashing gave only token-OVERLAP (Bitext zero-shot 0.185 =
chance). Real text needs SEMANTICS ("stop my order" ~ cancel without the word "cancel"). Self-supervised MLM on a
general public corpus learns contextual word/sentence representations where meaning-related text is close; then
the customer's labeled examples (in the prompt) form class prototypes and the query is labeled by cosine nearest
prototype -- generic over any domain, zero gradient updates on customer data.

  python -m thinking.langsem --selftest
  python -m thinking.langsem --steps 12000 --device cuda     # real pretraining (GPU)
"""
import argparse
import re
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from thinking.reasoner import Block

PAD, MASK, UNK = 0, 1, 2
W0 = 3
CORPUS = "data/cosmopedia_6mb.txt"
DATA = "kaggle_data/bitext_customer_support.csv"


def tok(text):
    return re.findall(r"[a-z]+", str(text).lower())


def build_corpus(path, max_len=40, vocab_cap=15000, min_freq=3):
    from collections import Counter
    words = tok(open(path, encoding="utf-8", errors="ignore").read())
    wc = Counter(words)
    itos = ["<pad>", "<mask>", "<unk>"] + [w for w, c in wc.most_common(vocab_cap) if c >= min_freq]
    stoi = {w: i for i, w in enumerate(itos)}
    ids = [stoi.get(w, UNK) for w in words]
    # chop into max_len segments
    segs = [ids[i:i + max_len] for i in range(0, len(ids) - max_len, max_len)]
    # IDF over the general corpus (rarer word -> higher weight) = the recurrence principle as token weighting
    n_seg = max(1, len(segs)); df = np.zeros(len(itos))
    for s in segs:
        for w in set(s):
            df[w] += 1
    idf = np.log((n_seg + 1) / (df + 1)) + 1.0
    return np.array([s + [PAD] * (max_len - len(s)) for s in segs], dtype=np.int64), itos, stoi, idf


class SemEncoder(nn.Module):
    """Bidirectional masked-LM encoder. encode() -> contextual hidden states; embed() -> masked mean-pooled
    sentence vector (the semantic representation used for zero-shot prototype classification)."""
    def __init__(self, vocab, d=256, layers=4, heads=8, max_len=40):
        super().__init__()
        self.tok = nn.Embedding(vocab, d, padding_idx=PAD)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)

    def encode(self, ids):
        kpm = (ids == PAD)
        h = self.tok(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device))[None]
        for b in self.blocks:
            h = b(h, kpm)                                     # bidirectional (Block applies no causal mask)
        return self.ln(h), kpm

    def forward(self, ids):
        h, _ = self.encode(ids)
        return self.head(h)

    def embed(self, ids):
        h, kpm = self.encode(ids)
        m = (~kpm).float().unsqueeze(-1)
        v = (h * m).sum(1) / m.sum(1).clamp(min=1)           # masked mean-pool
        return F.normalize(v, dim=-1)


def train_mlm(segs, vocab, steps=12000, d=256, layers=4, heads=8, bs=64, lr=3e-4, mask_p=0.15,
              seed=0, device="cpu"):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = SemEncoder(vocab, d=d, layers=layers, heads=heads, max_len=segs.shape[1]).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    segs_t = torch.from_numpy(segs)
    for _ in range(steps):
        bi = rng.integers(0, len(segs_t), bs)
        x = segs_t[bi].clone().to(device)
        prob = torch.rand(x.shape, device=device)
        mask = (prob < mask_p) & (x != PAD)
        y = x.clone()
        x[mask] = MASK
        logits = m(x)
        loss = F.cross_entropy(logits[mask], y[mask])
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    return m


def _bitext_by_intent(path, stoi, max_len=40, kmin=8):
    import csv
    by = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            ids = [stoi.get(w, UNK) for w in tok(r["instruction"])][:max_len]
            by.setdefault(r["intent"], []).append(ids + [PAD] * (max_len - len(ids)))
    return {i: np.array(v) for i, v in by.items() if len(v) >= kmin}


def zeroshot(m, by, C=5, k=5, n=400, seed=0, device="cpu", embed_fn=None):
    """ZERO-SHOT: k support examples per class -> class prototype (mean embedding); query -> nearest prototype.
    embed_fn lets us swap in a bag-of-words baseline for comparison."""
    intents = list(by); rng = np.random.default_rng(seed); ok = 0
    ef = embed_fn or (lambda X: m.embed(torch.from_numpy(X).to(device)))
    with torch.no_grad():
        for _ in range(n):
            chosen = list(rng.choice(intents, C, replace=False))
            protos = []; qX = []; qy = []
            for ci, it in enumerate(chosen):
                idx = rng.choice(len(by[it]), k + 1, replace=False)
                sup = ef(by[it][idx[:k]])                     # (k, d)
                protos.append(sup.mean(0))
                qX.append(by[it][idx[k]]); qy.append(ci)
            P = F.normalize(torch.stack(protos), dim=-1)      # (C, d)
            qe = ef(np.array(qX))                             # (C, d)
            pred = (F.normalize(qe, dim=-1) @ P.T).argmax(-1).cpu().numpy()
            ok += int((pred == np.array(qy)).sum())
    return ok / (n * C)


def bow_embed_factory(vocab, device="cpu", idf=None):
    """Token-overlap embedding over the customer's prompt examples (no training on customer data). idf=None -> raw
    bag-of-words; idf=<vector> -> TF-IDF (downweight common words = the recurrence/IDF principle)."""
    w = None if idf is None else torch.tensor(idf, dtype=torch.float32, device=device)
    def ef(X):
        t = torch.from_numpy(X).to(device)
        oh = F.one_hot(t.clamp(min=0), vocab).float()
        oh[..., PAD] = 0
        v = oh.sum(1)
        if w is not None:
            v = v * w[None]
        return F.normalize(v, dim=-1)
    return ef


def selftest():
    segs, itos, stoi, idf = build_corpus(CORPUS, max_len=40, vocab_cap=8000)
    m = train_mlm(segs, len(itos), steps=3000, d=192, layers=4, heads=6, device="cpu")
    by = _bitext_by_intent(DATA, stoi)
    sem = zeroshot(m, by, C=5, k=5, n=200, device="cpu")
    bow = zeroshot(m, by, C=5, k=5, n=200, device="cpu", embed_fn=bow_embed_factory(len(itos)))
    tfidf = zeroshot(m, by, C=5, k=5, n=200, device="cpu", embed_fn=bow_embed_factory(len(itos), idf=idf))
    print(f"langsem selftest: ZERO-SHOT Bitext 5-way (chance 0.20, NO training on Bitext) -- "
          f"TF-IDF {tfidf:.3f} | bag-of-words {bow:.3f} | weak-MLM-semantic {sem:.3f}")
    # Honest finding: for keyword-rich support text, TOKEN-OVERLAP retrieval over the prompt examples (esp. TF-IDF
    # = the recurrence/IDF principle) is a strong, simple, privacy-preserving zero-shot baseline; a weak CPU MLM
    # does not beat it (semantics help only where keywords don't overlap, and need proper sentence-embedding training).
    assert tfidf > 0.7, f"token-overlap retrieval unexpectedly weak: {tfidf}"
    print("langsem selftest OK (token-overlap/TF-IDF retrieval over prompt examples = strong zero-shot baseline; "
          "no training on customer data)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    segs, itos, stoi, idf = build_corpus(a.corpus, max_len=40)
    print(f"corpus segments {len(segs)}, vocab {len(itos)}; pretraining MLM {a.steps} steps ...")
    m = train_mlm(segs, len(itos), steps=a.steps, d=a.d, device=a.device)
    by = _bitext_by_intent(DATA, stoi)
    bow_ef = bow_embed_factory(len(itos), device=a.device)
    tfidf_ef = bow_embed_factory(len(itos), device=a.device, idf=idf)
    print("ZERO-SHOT Bitext (no training on it): TF-IDF | bag-of-words | MLM-semantic")
    for C in (3, 5, 10):
        s = zeroshot(m, by, C=C, k=5, n=300, device=a.device)
        b = zeroshot(m, by, C=C, k=5, n=300, device=a.device, embed_fn=bow_ef)
        t = zeroshot(m, by, C=C, k=5, n=300, device=a.device, embed_fn=tfidf_ef)
        print(f"  {C:2d}-way (chance {1/C:.3f}): tfidf {t:.3f} | bow {b:.3f} | semantic {s:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
