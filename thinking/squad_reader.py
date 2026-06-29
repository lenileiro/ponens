#!/usr/bin/env python3
"""Stage 2a: a from-scratch NON-LLM reading-comprehension reader (DrQA-style: GloVe + BiLSTM + attention + learned
start/end span pointers). Trained ONCE on public SQuAD-train, shipped as weights; inference does no training. This is
NOT an LLM -- no internet-scale pretraining, just GloVe word vectors + a small task-trained encoder (the pre-2018
state of the art, ~0.70-0.80 F1).

Our rule pipeline is the SCAFFOLD around it (the frontier lesson 'retrieval beats scale'): at inference we can PRUNE
the passage to the top-k sentences with our retriever before the reader runs (fewer tokens, higher precision).

  python -m thinking.squad_reader --selftest                       # tiny CPU: pipeline learns (overfits) -> OK
  python -m thinking.squad_reader --train --n 40000 --epochs 3 --save /tmp/reader.pt
  python -m thinking.squad_reader --eval --save /tmp/reader.pt
"""
import argparse
import json
import re
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from device import get_device
from thinking import squadqa as Q

TRAIN = "/tmp/squad/train-v1.1.json"
DEV = "/tmp/squad/dev-v1.1.json"
PAD, UNK = 0, 1


def tok_offsets(text):
    return [(m.group(), m.start()) for m in re.finditer(r"\w+|[^\w\s]", text)]


def _span_to_tokens(offs, a_start, a_text):
    """Map a gold answer char span to (start_tok, end_tok) over our tokenization; None if it can't be aligned."""
    a_end = a_start + len(a_text)
    si = ei = None
    for i, (t, off) in enumerate(offs):
        if off <= a_start < off + len(t):
            si = i
        if off < a_end <= off + len(t):
            ei = i
    if si is None:                                           # fall back to nearest-covering tokens
        si = next((i for i, (t, off) in enumerate(offs) if off >= a_start), None)
    if ei is None:
        ei = next((i for i in range(len(offs) - 1, -1, -1) if offs[i][1] < a_end), si)
    if si is None or ei is None or ei < si:
        return None
    return si, ei


def read_spans(path, n=None):
    data = json.load(open(path))["data"]
    out = []
    for art in data:
        for para in art["paragraphs"]:
            ctx = para["context"]; offs = tok_offsets(ctx); ptoks = [t for t, _ in offs]
            for qa in para["qas"]:
                golds = [a["text"] for a in qa["answers"]]
                a = qa["answers"][0]
                sp = _span_to_tokens(offs, a["answer_start"], a["text"])
                if sp is None:
                    continue
                out.append({"q": Q.toks(qa["question"]), "p": ptoks, "s": sp[0], "e": sp[1], "golds": golds})
                if n and len(out) >= n:
                    return out
    return out


def load_glove(dim, vocab_words):
    """GloVe vectors for the words we need (+ pad/unk). Returns (stoi, embedding matrix)."""
    import gensim.downloader as api
    kv = api.load(f"glove-wiki-gigaword-{dim}")
    stoi = {"<pad>": PAD, "<unk>": UNK}
    vecs = [np.zeros(dim), np.zeros(dim)]
    for wd in vocab_words:
        if wd in kv.key_to_index and wd not in stoi:
            stoi[wd] = len(stoi); vecs.append(kv[wd])
    return stoi, np.asarray(vecs, dtype=np.float32)


def _enc(toks, stoi):
    return [stoi.get(t.lower(), UNK) for t in toks]


class Reader(nn.Module):
    """DrQA-lite: aligned-question embedding + exact-match feature -> passage BiLSTM; question BiLSTM -> self-attn
    vector; bilinear start/end pointers over passage tokens."""
    def __init__(self, emb, hidden=128):
        super().__init__()
        V, D = emb.shape
        self.emb = nn.Embedding(V, D, padding_idx=PAD)
        self.emb.weight.data.copy_(torch.from_numpy(emb)); self.emb.weight.requires_grad_(False)
        self.align = nn.Linear(D, D)
        self.p_rnn = nn.LSTM(2 * D + 1, hidden, batch_first=True, bidirectional=True)
        self.q_rnn = nn.LSTM(D, hidden, batch_first=True, bidirectional=True)
        self.q_self = nn.Linear(2 * hidden, 1)
        self.start = nn.Bilinear(2 * hidden, 2 * hidden, 1)
        self.end = nn.Bilinear(2 * hidden, 2 * hidden, 1)

    def forward(self, p, q, pmask, qmask, ematch):
        pe, qe = self.emb(p), self.emb(q)                                  # (B,Tp,D),(B,Tq,D)
        # aligned question embedding: soft attention from each passage word to question words (DrQA f_align)
        scores = torch.relu(self.align(pe)) @ torch.relu(self.align(qe)).transpose(1, 2)  # (B,Tp,Tq)
        scores = scores.masked_fill(~qmask[:, None, :], -1e4)
        aligned = torch.softmax(scores, -1) @ qe                          # (B,Tp,D)
        pin = torch.cat([pe, aligned, ematch[..., None]], -1)
        ph, _ = self.p_rnn(pin)                                           # (B,Tp,2h)
        qh, _ = self.q_rnn(qe)                                            # (B,Tq,2h)
        qa = self.q_self(qh).squeeze(-1).masked_fill(~qmask, -1e4)
        qvec = (torch.softmax(qa, -1)[..., None] * qh).sum(1)            # (B,2h)
        B, Tp, H = ph.shape
        qx = qvec[:, None, :].expand(B, Tp, H).contiguous()
        s = self.start(ph.contiguous(), qx).squeeze(-1).masked_fill(~pmask, -1e4)
        e = self.end(ph.contiguous(), qx).squeeze(-1).masked_fill(~pmask, -1e4)
        return s, e


def _batch(exs, stoi, dev):
    P = [_enc(e["p"], stoi) for e in exs]; Qq = [_enc(e["q"], stoi) for e in exs]
    Tp, Tq = max(len(x) for x in P), max(len(x) for x in Qq)
    p = torch.full((len(exs), Tp), PAD, dtype=torch.long); q = torch.full((len(exs), Tq), PAD, dtype=torch.long)
    em = torch.zeros(len(exs), Tp); pm = torch.zeros(len(exs), Tp, dtype=torch.bool); qm = torch.zeros(len(exs), Tq, dtype=torch.bool)
    for i, e in enumerate(exs):
        p[i, :len(P[i])] = torch.tensor(P[i]); q[i, :len(Qq[i])] = torch.tensor(Qq[i])
        pm[i, :len(P[i])] = True; qm[i, :len(Qq[i])] = True
        qwords = set(t.lower() for t in e["q"])
        for j, t in enumerate(e["p"]):
            if t.lower() in qwords:
                em[i, j] = 1.0
    s = torch.tensor([e["s"] for e in exs]); en = torch.tensor([e["e"] for e in exs])
    return (p.to(dev), q.to(dev), pm.to(dev), qm.to(dev), em.to(dev)), (s.to(dev), en.to(dev))


def train(model, data, stoi, dev, epochs=3, bs=32, lr=1e-3):
    opt = torch.optim.Adam([w for w in model.parameters() if w.requires_grad], lr=lr)
    rng = np.random.default_rng(0)
    import time
    nb = max(1, len(data) // bs)
    for ep in range(epochs):
        rng.shuffle(data); tot = 0.0; t0 = time.time()
        for bi, i in enumerate(range(0, len(data), bs)):
            chunk = data[i:i + bs]
            X, (gs, ge) = _batch(chunk, stoi, dev)
            s, e = model(*X)
            loss = F.cross_entropy(s, gs) + F.cross_entropy(e, ge)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
            if bi % 100 == 0:
                print(f"  ep{ep + 1} batch {bi}/{nb} loss {loss.item():.2f} ({(time.time() - t0) / (bi + 1) * 1000:.0f} ms/b)", flush=True)
        print(f"  epoch {ep + 1}/{epochs} loss {tot / nb:.3f} ({time.time() - t0:.0f}s)", flush=True)
    return model


@torch.no_grad()
def predict(model, ex, stoi, dev, max_len=15):
    X, _ = _batch([{**ex, "s": 0, "e": 0}], stoi, dev)
    s, e = model(*X)
    s, e = torch.log_softmax(s, -1)[0], torch.log_softmax(e, -1)[0]
    T = len(ex["p"])
    best, bi, bj = -1e9, 0, 0
    sv = s.cpu().numpy(); ev = e.cpu().numpy()
    for i in range(T):
        for j in range(i, min(i + max_len, T)):
            if sv[i] + ev[j] > best:
                best, bi, bj = sv[i] + ev[j], i, j
    return " ".join(ex["p"][bi:bj + 1])


@torch.no_grad()
def evaluate(model, exs, stoi, dev, n=None):
    model.eval(); sub = exs[:n] if n else exs; em = f1 = 0
    for e in sub:
        pred = predict(model, e, stoi, dev)
        em += max(int(Q._norm(pred) == Q._norm(g)) for g in e["golds"])
        f1 += max(Q._f1(pred, g) for g in e["golds"])
    return em / len(sub), f1 / len(sub)


def _vocab_words(exlists):
    s = set()
    for exs in exlists:
        for e in exs:
            for t in e["q"] + e["p"]:
                s.add(t.lower())
    return s


def selftest():
    dev = torch.device("cpu")
    exs = read_spans(DEV, n=64)
    stoi, emb = load_glove(100, _vocab_words([exs]))
    data = [e for e in exs if all(t.lower() in stoi or True for t in e["p"])][:48]
    model = Reader(emb, hidden=64).to(dev)
    train(model, list(data), stoi, dev, epochs=8, bs=16, lr=2e-3)
    em, f1 = evaluate(model, data, stoi, dev)                 # can it FIT the training set? -> pipeline correct
    print(f"squad_reader selftest: overfit-train F1 {f1:.3f} (architecture + span data pipeline learn)")
    assert f1 > 0.4, f"reader pipeline not learning: {f1}"
    print("squad_reader selftest OK (DrQA-style non-LLM reader: GloVe + BiLSTM + attention + span pointers)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true"); ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--n", type=int, default=40000); ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--dim", type=int, default=100); ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--evaln", type=int, default=1500, help="eval on a dev subset for fast feedback (0=full)")
    ap.add_argument("--save", default="/tmp/reader.pt")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    dev = get_device()
    if a.train:
        tr = read_spans(TRAIN, n=a.n); de = read_spans(DEV)
        stoi, emb = load_glove(a.dim, _vocab_words([tr, de]))
        print(f"train {len(tr)} (span-aligned) | dev {len(de)} | vocab {len(stoi)} | device {dev}", flush=True)
        model = Reader(emb, a.hidden).to(dev)
        train(model, tr, stoi, dev, epochs=a.epochs)
        torch.save({"state": model.state_dict(), "stoi": stoi, "emb_shape": emb.shape, "hidden": a.hidden}, a.save)
        em, f1 = evaluate(model, de, stoi, dev, n=a.evaln or None)
        print(f"SQuAD dev (n={a.evaln or len(de)}) -- reader: EM {em:.3f} | F1 {f1:.3f}", flush=True)
    if a.eval and not a.train:
        ck = torch.load(a.save, map_location=dev)
        de = read_spans(DEV); stoi = ck["stoi"]
        model = Reader(np.zeros(ck["emb_shape"], dtype=np.float32), ck["hidden"]).to(dev)
        model.load_state_dict(ck["state"])
        em, f1 = evaluate(model, de, stoi, dev, n=a.evaln or None)
        print(f"SQuAD dev (n={a.evaln or len(de)}) -- reader: EM {em:.3f} | F1 {f1:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
