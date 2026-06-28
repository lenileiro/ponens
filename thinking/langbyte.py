#!/usr/bin/env python3
"""Language-agnostic SEMANTIC encoder for the B2B API: a BYTE-LEVEL (no word vocab, any script) sentence encoder
trained CONTRASTIVELY (SimCSE) on general PUBLIC text (cosmopedia -- not customer data), evaluated ZERO-SHOT on a
SEMANTIC task (emotions) where the char-n-gram baseline is at chance.

Why this exists: the char-n-gram retrieval (thinking/langzero) is strong on surface/keyword tasks (support 0.92,
language-ID 0.88) but ~chance on emotion (0.42 3-way) -- emotion needs MEANING, not surface overlap. To capture
meaning while staying language-agnostic and never training on customer data, we learn sentence embeddings from
bytes via SimCSE (dropout makes two views of the same sentence a positive pair; other sentences are negatives).
Emotions is the clean test: char-n-grams = chance there, so any gain is real semantics, not keyword leakage.

  python -m thinking.langbyte --selftest
  python -m thinking.langbyte --steps 40000 --device cuda    # real contrastive pretraining (GPU)
"""
import argparse
import csv
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import thinking.langzero as Z

PAD = 256
VOCAB = 257                                                # 256 byte values + PAD
CORPUS = "data/cosmopedia_6mb.txt,data/tinystories_8mb.txt"   # mix: textbook + emotional story text
EMO = "kaggle_data/emotions.csv"


def encode_bytes(text, max_len=96):
    b = list(str(text).encode("utf-8"))[:max_len]
    return b + [PAD] * (max_len - len(b))


class ByteEncoder(nn.Module):
    """Bidirectional byte-level transformer -> mean-pooled, projected sentence embedding. Dropout drives SimCSE."""
    def __init__(self, d=256, layers=4, heads=8, max_len=96, drop=0.1):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, d, padding_idx=PAD)
        self.pos = nn.Embedding(max_len, d)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=drop,
                                                                batch_first=True, activation="gelu",
                                                                norm_first=True) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.proj = nn.Linear(d, d)

    def forward(self, ids):
        kpm = (ids == PAD)
        h = self.tok(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device))[None]
        for layer in self.layers:
            h = layer(h, src_key_padding_mask=kpm)
        h = self.ln(h)
        m = (~kpm).float().unsqueeze(-1)
        v = (h * m).sum(1) / m.sum(1).clamp(min=1)          # masked mean-pool
        return F.normalize(self.proj(v), dim=-1)


def load_sentences(path, max_len=96, min_chars=20):
    """path may be comma-separated to MIX corpora (e.g. cosmopedia + tinystories -- the latter carries more
    emotional language, which textbook-style cosmopedia lacks)."""
    sents = []
    for p in str(path).split(","):
        txt = open(p.strip(), encoding="utf-8", errors="ignore").read()
        sents += [s.strip() for s in txt.replace("\n", " ").split(".") if len(s.strip()) >= min_chars]
    return [encode_bytes(s, max_len) for s in sents]


def train(steps, sents, d=256, layers=4, heads=8, bs=64, lr=3e-4, temp=0.05, seed=0, device="cpu", max_len=96):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = ByteEncoder(d=d, layers=layers, heads=heads, max_len=max_len).to(device); m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    S = torch.tensor(sents)
    for _ in range(steps):
        bi = rng.integers(0, len(S), bs)
        x = S[bi].to(device)
        z1 = m(x); z2 = m(x)                                # two dropout views -> positive pair
        sim = (z1 @ z2.T) / temp                            # in-batch negatives
        lbl = torch.arange(len(x), device=device)
        loss = 0.5 * (F.cross_entropy(sim, lbl) + F.cross_entropy(sim.T, lbl))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    m.eval(); return m


def by_text(path, textcol, labelcol, kmin=8, cap=300):
    by = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            by.setdefault(str(r[labelcol]), []).append(r[textcol])
    return {k: v[:cap] for k, v in by.items() if len(v) >= kmin}


def _neural_embed(m, device, max_len=96):
    def ef(texts):
        ids = torch.tensor([encode_bytes(t, max_len) for t in texts], device=device)
        with torch.no_grad():
            return m(ids).cpu().numpy()
    return ef


def _charngram_embed(texts):
    return np.stack([Z._dense(Z.ngram_counts(t)) / (np.linalg.norm(Z._dense(Z.ngram_counts(t))) + 1e-9)
                     for t in texts])


def zeroshot(by, embed_fn, C=3, k=5, n=300, seed=0):
    intents = list(by); rng = np.random.default_rng(seed); ok = 0
    for _ in range(n):
        chosen = list(rng.choice(intents, C, replace=False))
        protos = []; qX = []; qy = []
        for ci, it in enumerate(chosen):
            idx = rng.choice(len(by[it]), k + 1, replace=False)
            E = embed_fn([by[it][j] for j in idx[:k]])
            protos.append(E.mean(0)); qX.append(by[it][idx[k]]); qy.append(ci)
        P = np.stack(protos); P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
        Q = embed_fn(qX); Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9)
        ok += int(((Q @ P.T).argmax(1) == np.array(qy)).sum())
    return ok / (n * C)


def selftest():
    sents = load_sentences(CORPUS)[:20000]
    m = train(3000, sents, d=128, layers=2, heads=4, bs=64, device="cpu")
    by = by_text(EMO, "text", "label")
    neural = zeroshot(by, _neural_embed(m, "cpu"), C=3, k=5, n=150)
    char = zeroshot(by, _charngram_embed, C=3, k=5, n=150)
    print(f"langbyte selftest: EMOTIONS 3-way zero-shot (chance 0.333) -- "
          f"byte-SimCSE-semantic {neural:.3f} | char-n-gram {char:.3f}")
    # CPU smoke = pipeline check ONLY. A d128/2-layer/3k-step byte encoder is far too small to learn semantics
    # (here ~chance/below) -- contrastive semantic learning needs scale + an emotion-bearing corpus -> the GPU run.
    assert 0.0 <= neural <= 1.0, "broken pipeline"
    print("langbyte selftest OK (byte-level SimCSE pipeline runs end-to-end; semantic gain requires the GPU run)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    sents = load_sentences(a.corpus)
    print(f"corpus sentences {len(sents)}; SimCSE contrastive pretraining {a.steps} steps on bytes ...")
    m = train(a.steps, sents, d=a.d, layers=a.layers, device=a.device)
    by = by_text(EMO, "text", "label")
    nef = _neural_embed(m, a.device)
    print("EMOTIONS zero-shot (no training on it): byte-SimCSE-semantic vs char-n-gram")
    for C in (3, 6):
        s = zeroshot(by, nef, C=C, k=5, n=300)
        c = zeroshot(by, _charngram_embed, C=C, k=5, n=300)
        print(f"  {C}-way (chance {1/C:.3f}): semantic {s:.3f} | char-n-gram {c:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
