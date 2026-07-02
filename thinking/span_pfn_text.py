#!/usr/bin/env python3
"""In-context span PFN on REAL text (GloVe tokens) -- bridging the synthetic proof-of-concept to real words. The
task is GENUINELY in-context: two modes on real tweets that the SUPPORT must disambiguate --
  - 'whole'  (neutral tweets): the answer span is the WHOLE tweet;
  - 'phrase' (positive/negative tweets): the answer span is just the sentiment-bearing PHRASE.
The same tweet could be either, so the model can't guess from the query alone -- it must infer the mode from the K
support examples' answer flags and apply it. No per-task training; one meta-trained weight; not an LLM.

Ablation (zero the support flags) should collapse Jaccard -> proves it USES the in-context support.

  python -m thinking.span_pfn_text --selftest
  python -m thinking.span_pfn_text --train --steps 4000
"""
import argparse
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from thinking import squad_reader as R
from thinking import tweet_sent as T

try:
    from device import get_device
except Exception:
    def get_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

L = 36            # max tokens per tweet (pad/truncate)
K = 4             # support examples per episode


def build_pools():
    """TSE tweets grouped into the two in-context MODES, each as (token-ids-placeholder-text, in-span flags)."""
    exs = T.load()
    whole, phrase = [], []                                    # 'whole' = neutral (span=all), 'phrase' = pos/neg
    for e in exs:
        toks = e["p"][:L]
        if not toks:
            continue
        gold = set(e["golds"][0].lower().split())
        fl = [1 if toks[i].lower() in gold else 0 for i in range(len(toks))]
        if sum(fl) == 0:
            continue
        sent = e["q"][0]
        (whole if sent == "neutral" else phrase).append((toks, fl))
    return {"whole": whole, "phrase": phrase}


def _vocab(pools):
    s = set()
    for v in pools.values():
        for toks, _ in v:
            for t in toks:
                s.add(t.lower())
    return s


class TextPFN(nn.Module):
    def __init__(self, emb, d=128, layers=3, heads=4):
        super().__init__()
        V, D = emb.shape
        self.emb = nn.Embedding(V, D, padding_idx=R.PAD)
        self.emb.weight.data.copy_(torch.from_numpy(emb)); self.emb.weight.requires_grad_(False)
        self.proj = nn.Linear(D, d)
        self.flag = nn.Embedding(3, d)                        # 0 support-out, 1 support-in, 2 query-unknown
        self.pos = nn.Embedding(L, d)
        self.seg = nn.Embedding(K + 1, d)
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, batch_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.out = nn.Linear(d, 1)

    def forward(self, toks, flags, segs, poss, padmask):
        h = self.proj(self.emb(toks)) + self.flag(flags) + self.pos(poss) + self.seg(segs)
        h = self.enc(h, src_key_padding_mask=padmask)
        return self.out(h).squeeze(-1)


def _episode(rng, pools, stoi, mask_support=False):
    mode = rng.choice(["whole", "phrase"])
    return _episode_pool(rng, pools[mode], stoi, mask_support)


def _episode_pool(rng, pool, stoi, mask_support=False):
    idx = rng.choice(len(pool), size=K + 1, replace=False)
    T_ = (K + 1) * L
    toks = np.full(T_, R.PAD, int); flags = np.zeros(T_, int); segs = np.zeros(T_, int)
    poss = np.zeros(T_, int); pad = np.ones(T_, bool); y = np.zeros(L); ymask = np.zeros(L)
    for e, ix in enumerate(idx):
        tks, fl = pool[ix]; n = len(tks); sl = e * L
        for i in range(n):
            toks[sl + i] = stoi.get(tks[i].lower(), R.UNK); segs[sl + i] = e; poss[sl + i] = i; pad[sl + i] = False
            flags[sl + i] = (2 if e == K else (0 if mask_support else fl[i]))
        if e == K:
            y[:n] = fl; ymask[:n] = 1
    return toks, flags, segs, poss, pad, y, ymask


def _batch(rng, pools, stoi, bs, dev, mask_support=False):
    cols = [list(_episode(rng, pools, stoi, mask_support)) for _ in range(bs)]
    arrs = [np.stack([c[i] for c in cols]) for i in range(7)]
    t = lambda a, f=torch.long: torch.tensor(a, dtype=f).to(dev)
    toks, flags, segs, poss = (t(arrs[i]) for i in range(4))
    pad = torch.tensor(arrs[4], dtype=torch.bool).to(dev)
    y = torch.tensor(arrs[5], dtype=torch.float32).to(dev); ym = torch.tensor(arrs[6], dtype=torch.float32).to(dev)
    return toks, flags, segs, poss, pad, y, ym


def _query_logits(model, toks, flags, segs, poss, pad):
    z = model(toks, flags, segs, poss, pad)
    return z[:, K * L:(K + 1) * L]


def train(model, pools, stoi, dev, steps=4000, bs=64, lr=1e-3, seed=0):
    rng = np.random.default_rng(seed); opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    for st in range(steps):
        toks, flags, segs, poss, pad, y, ym = _batch(rng, pools, stoi, bs, dev)
        z = _query_logits(model, toks, flags, segs, poss, pad)
        loss = (F.binary_cross_entropy_with_logits(z, y, reduction="none") * ym).sum() / ym.sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); opt.step()
        if st % 500 == 0:
            print(f"  step {st} loss {loss.item():.3f}", flush=True)
    return model


@torch.no_grad()
def evaluate(model, pools, stoi, dev, n=600, seed=99, mask_support=False):
    rng = np.random.default_rng(seed); js = []
    for _ in range(n):
        toks, flags, segs, poss, pad, y, ym = _batch(rng, pools, stoi, 1, dev, mask_support=mask_support)
        z = _query_logits(model, toks, flags, segs, poss, pad)[0]
        m = ym[0].bool(); pred = (z[m] > 0).float(); gold = y[0][m]
        # word-level Jaccard over the query's tokens (in-span sets)
        ps, gs = set(np.where(pred.cpu().numpy() > 0)[0]), set(np.where(gold.cpu().numpy() > 0)[0])
        js.append(len(ps & gs) / len(ps | gs) if (ps | gs) else 1.0)
    return float(np.mean(js))


# ---- MULTI-TASK: many genuinely-distinct REAL extraction tasks; the support must reveal WHICH one ----
# These are STRUCTURAL/SEMANTIC ground-truth generators (not hand-matched word lists): the MODEL never sees the
# rule -- it must infer the task from the K support examples' flags and apply it to the query. Mirrors how the
# synthetic prior plants spans; here the spans are real tweet phenomena.
TASKS = ["phrase", "whole", "number", "allcaps", "after_at", "after_hash"]


def _surface_flags(toks):
    """Per-token flags for the surface/positional tasks over a real tweet."""
    out = {"number": [], "allcaps": [], "after_at": [], "after_hash": []}
    for i, t in enumerate(toks):
        prev = toks[i - 1] if i > 0 else ""
        out["number"].append(1 if any(c.isdigit() for c in t) else 0)
        out["allcaps"].append(1 if (t.isalpha() and t.isupper() and len(t) >= 2) else 0)
        out["after_at"].append(1 if prev == "@" else 0)        # the @mention handle
        out["after_hash"].append(1 if prev == "#" else 0)      # the #hashtag word
    return out


def build_task_pools():
    exs = T.load()
    pools = {k: [] for k in TASKS}
    for e in exs:
        toks = e["p"][:L]
        if not toks:
            continue
        gold = set(e["golds"][0].lower().split())
        fl = [1 if toks[i].lower() in gold else 0 for i in range(len(toks))]
        if sum(fl):
            (pools["whole"] if e["q"][0] == "neutral" else pools["phrase"]).append((toks, fl))
        for k, v in _surface_flags(toks).items():
            if sum(v):
                pools[k].append((toks, v))
    return pools


def _mt_batch(rng, pools, stoi, bs, dev, mask_support=False, task=None, tasks=None):
    pool_tasks = tasks or TASKS
    cols = []
    for _ in range(bs):
        tk = task or pool_tasks[rng.integers(len(pool_tasks))]
        cols.append(list(_episode_pool(rng, pools[tk], stoi, mask_support)))
    arrs = [np.stack([c[i] for c in cols]) for i in range(7)]
    t = lambda a, f=torch.long: torch.tensor(a, dtype=f).to(dev)
    toks, flags, segs, poss = (t(arrs[i]) for i in range(4))
    pad = torch.tensor(arrs[4], dtype=torch.bool).to(dev)
    y = torch.tensor(arrs[5], dtype=torch.float32).to(dev); ym = torch.tensor(arrs[6], dtype=torch.float32).to(dev)
    return toks, flags, segs, poss, pad, y, ym


def train_mt(model, pools, stoi, dev, steps=6000, bs=64, lr=1e-3, seed=0, tasks=None):
    rng = np.random.default_rng(seed); opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    for st in range(steps):
        toks, flags, segs, poss, pad, y, ym = _mt_batch(rng, pools, stoi, bs, dev, tasks=tasks)
        z = _query_logits(model, toks, flags, segs, poss, pad)
        loss = (F.binary_cross_entropy_with_logits(z, y, reduction="none") * ym).sum() / ym.sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); opt.step()
        if st % 500 == 0:
            print(f"  step {st} loss {loss.item():.3f}", flush=True)
    return model


@torch.no_grad()
def eval_mt(model, pools, stoi, dev, n=300, seed=99, mask_support=False, task=None):
    rng = np.random.default_rng(seed); js = []
    for _ in range(n):
        toks, flags, segs, poss, pad, y, ym = _mt_batch(rng, pools, stoi, 1, dev, mask_support=mask_support, task=task)
        z = _query_logits(model, toks, flags, segs, poss, pad)[0]
        m = ym[0].bool(); pred = (z[m] > 0).float(); gold = y[0][m]
        ps, gs = set(np.where(pred.cpu().numpy() > 0)[0]), set(np.where(gold.cpu().numpy() > 0)[0])
        js.append(len(ps & gs) / len(ps | gs) if (ps | gs) else 1.0)
    return float(np.mean(js))


def selftest():
    pools = build_pools()
    assert pools["whole"] and pools["phrase"], "TSE pools empty"
    stoi, emb = R.load_glove(100, _vocab(pools))
    dev = torch.device("cpu")
    tr = {k: v[:600] for k, v in pools.items()}
    model = TextPFN(emb, d=96, layers=2, heads=4).to(dev)
    train(model, tr, stoi, dev, steps=1500, bs=32)
    jac = evaluate(model, tr, stoi, dev, n=300)
    mjac = evaluate(model, tr, stoi, dev, n=300, mask_support=True)
    print(f"span_pfn_text selftest: in-context Jaccard {jac:.3f} | support-ablated {mjac:.3f} "
          f"(modes whole/phrase on REAL tweets)")
    assert jac > 0.5, f"real-text PFN not learning in-context: {jac}"
    assert jac - mjac > 0.05, f"support not used (mode not inferred in-context): {jac} vs {mjac}"
    print("span_pfn_text selftest OK (in-context span extraction on REAL text -- mode inferred from support, no train)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true"); ap.add_argument("--train", action="store_true")
    ap.add_argument("--multitask", action="store_true")
    ap.add_argument("--heldout-all", action="store_true")
    ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--d", type=int, default=128)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.heldout_all:
        # GENERALIZATION test: meta-train on all tasks EXCEPT one, then test in-context on the UNSEEN task --
        # the task is defined to the model ONLY by its support examples (the "define-extraction-by-examples" claim).
        pools = build_task_pools(); stoi, emb = R.load_glove(100, _vocab(pools)); dev = get_device()
        tr = {k: v[:int(len(v) * .8)] for k, v in pools.items()}
        te = {k: v[int(len(v) * .8):] for k, v in pools.items()}
        # query-visible-trigger tasks (after_at/after_hash) are solvable without support -> not a real generalization
        # test; hold out only the genuinely support-dependent tasks.
        heldouts = ["phrase", "whole", "number", "allcaps"]
        print("pool sizes: " + " ".join(f"{k}={len(v)}" for k, v in pools.items()) + f" | dev {dev}", flush=True)
        print("\nHELD-OUT-TASK generalization (train on 5 tasks, in-context on the UNSEEN 6th):", flush=True)
        for ho in heldouts:
            train_tasks = [t for t in TASKS if t != ho]
            model = TextPFN(emb, d=a.d, layers=3, heads=4).to(dev)
            train_mt(model, tr, stoi, dev, steps=a.steps, tasks=train_tasks, seed=0)
            j = eval_mt(model, te, stoi, dev, task=ho); ja = eval_mt(model, te, stoi, dev, task=ho, mask_support=True)
            print(f"  HELD-OUT {ho:9s}: in-context {j:.3f} | support-ablated {ja:.3f}  "
                  f"(trained on {','.join(train_tasks)})", flush=True)
        return 0
    if a.multitask:
        pools = build_task_pools(); stoi, emb = R.load_glove(100, _vocab(pools)); dev = get_device()
        tr = {k: v[:int(len(v) * .8)] for k, v in pools.items()}
        te = {k: v[int(len(v) * .8):] for k, v in pools.items()}
        print("pool sizes: " + " ".join(f"{k}={len(v)}" for k, v in pools.items()) + f" | dev {dev}", flush=True)
        model = TextPFN(emb, d=a.d, layers=3, heads=4).to(dev)
        train_mt(model, tr, stoi, dev, steps=a.steps)
        print("\nMULTI-TASK in-context PFN on REAL text (held-out tweets, per task):", flush=True)
        overall, overall_abl = [], []
        for tk in TASKS:
            j = eval_mt(model, te, stoi, dev, task=tk); ja = eval_mt(model, te, stoi, dev, task=tk, mask_support=True)
            overall.append(j); overall_abl.append(ja)
            print(f"  {tk:11s} in-context {j:.3f} | support-ablated {ja:.3f}", flush=True)
        print(f"  {'MEAN':11s} in-context {np.mean(overall):.3f} | support-ablated {np.mean(overall_abl):.3f}", flush=True)
        return 0
    if a.train:
        pools = build_pools(); stoi, emb = R.load_glove(100, _vocab(pools)); dev = get_device()
        # held-out split: train episodes from first 80% of each pool, eval from the rest
        tr = {k: v[:int(len(v) * .8)] for k, v in pools.items()}
        te = {k: v[int(len(v) * .8):] for k, v in pools.items()}
        print(f"TextPFN: whole {len(pools['whole'])} / phrase {len(pools['phrase'])} | vocab {len(stoi)} | dev {dev}", flush=True)
        model = TextPFN(emb, d=a.d, layers=3, heads=4).to(dev)
        train(model, tr, stoi, dev, steps=a.steps)
        jac = evaluate(model, te, stoi, dev); mjac = evaluate(model, te, stoi, dev, mask_support=True)
        print(f"REAL-TEXT in-context PFN (held-out tweets): Jaccard {jac:.3f} | support-ablated {mjac:.3f} "
              f"(whole-tweet baseline ~0.40)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
