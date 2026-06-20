#!/usr/bin/env python3
"""lang_bridge -- learn the mapping between natural language and LOTA (thinking/lang.py).

This is the PAYOFF that turns LOTA into the road to C2: instead of learning English from scratch
(which hit the memorization wall -- C2_ROADMAP.md), the model learns a *mapping* in two directions,
each verifiable by the LOTA engine:

    PARSE   : English  -> LOTA        (understanding)
    GENERATE: LOTA     -> English     (realization)

One ScratchpadLM (pointer head, so it can COPY content symbols across the boundary -- decisive for
held-out content, as we learned) trained bidirectionally with a <parse>/<gen> direction tag.

The data is a SMALL COMPOSITIONAL grammar: aligned (English, LOTA) pairs over kinds/verbs/quantifiers/
negation/modality/events. The held-out split reserves unseen (quantifier x kind x verb) COMBINATIONS,
so passing requires generalizing the mapping compositionally -- not memorizing pairs.

Gates (lead-measured, on the held-out split):
  PARSE   : well-formed (lang.check) AND canonical-LOTA exact match AND semantic-equiv (same truth as
            gold across random worlds, via lang.evaluate).
  GENERATE: round-trip faithful -- parse the generated English back and recover the gold LOTA.

  python -m thinking.lang_bridge --selftest
  python -m thinking.lang_bridge --steps 4000 --out /tmp/bridge.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scratchpad_model import ScratchpadLM  # noqa: E402
import thinking.lang as L  # noqa: E402  (read/write/check/desugar/evaluate/World -- the verifier)


# ===================================================================================================
# 1. LEXICON + COMPOSITIONAL (English <-> LOTA) GENERATOR
# ===================================================================================================
# kind symbol -> (singular, plural).  The LOTA symbol is the singular form (the canonical lemma).
KINDS = {
    "dog": ("dog", "dogs"), "cat": ("cat", "cats"), "bird": ("bird", "birds"),
    "fish": ("fish", "fish"), "horse": ("horse", "horses"), "duck": ("duck", "ducks"),
    "frog": ("frog", "frogs"), "cow": ("cow", "cows"), "sheep": ("sheep", "sheep"),
    "owl": ("owl", "owls"),
}
# intransitive ability verb -> base form (LOTA symbol == base).
ABIL = {"fly": "fly", "swim": "swim", "run": "run", "sing": "sing", "jump": "jump", "hide": "hide"}
# transitive verb -> (base, past).
TRANS = {"chase": ("chase", "chased"), "see": ("see", "saw"), "like": ("like", "liked"),
         "follow": ("follow", "followed"), "eat": ("eat", "ate"), "watch": ("watch", "watched")}


def _art(word):
    return "an" if word[0] in "aeiou" else "a"


def render(sig):
    """Render one signature to its aligned (english, lota). Factored out so a relabeled signature can
    be RE-rendered on the fly (the relabeling lever that forces copying instead of memorization)."""
    t = sig[0]
    if t in ("univ", "nouniv", "exist"):
        _, k, v = sig
        sg, pl = KINDS[k]
        if t == "univ":
            return f"every {sg} can {v}", f"(forall (?x {k}) (can ({v} :agent ?x)))"
        if t == "nouniv":
            return f"no {sg} can {v}", f"(forall (?x {k}) (not (can ({v} :agent ?x))))"
        return f"some {pl} can {v}", f"(exists (?x {k}) (can ({v} :agent ?x)))"
    _, k1, k2, v = sig                               # transitive past event
    sg1, sg2 = KINDS[k1][0], KINDS[k2][0]
    base, past = TRANS[v]
    return (f"{_art(sg1)} {sg1} {past} {_art(sg2)} {sg2}",
            f"(event {base} :agent ({_art(sg1)} {k1}) :patient ({_art(sg2)} {k2}) :tense past)")


def gen_pairs():
    """Enumerate aligned (english, lota, signature) examples. `sig` keys the holdout split."""
    pairs = []
    for k in KINDS:
        for v in ABIL:
            for t in ("univ", "nouniv", "exist"):
                sig = (t, k, v)
                nl, lota = render(sig)
                pairs.append(dict(nl=nl, lota=lota, sig=sig))
    for k1 in KINDS:
        for k2 in KINDS:
            if k1 == k2:
                continue
            for v in TRANS:
                sig = ("trans", k1, k2, v)
                nl, lota = render(sig)
                pairs.append(dict(nl=nl, lota=lota, sig=sig))
    return pairs


def lemma_pools(pairs):
    """The kind/abil/trans lemmas that actually occur in `pairs` (used as the relabel vocabulary, so
    relabeling NEVER introduces a held-out word into training)."""
    kinds, abil, trans = set(), set(), set()
    for p in pairs:
        s = p["sig"]
        if s[0] in ("univ", "nouniv", "exist"):
            kinds.add(s[1]); abil.add(s[2])
        else:
            kinds.add(s[1]); kinds.add(s[2]); trans.add(s[3])
    return dict(kind=sorted(kinds), abil=sorted(abil), trans=sorted(trans))


def relabel_sig(sig, pools, rng):
    """Swap the content lemmas of a signature for random same-type lemmas from `pools` (train lemmas).
    The structure is preserved; only the content changes -> the model cannot memorize a specific
    word->symbol association and must COPY the content across, which then transfers to NOVEL words."""
    t = sig[0]
    pick = lambda key: pools[key][rng.integers(len(pools[key]))]
    if t in ("univ", "nouniv", "exist"):
        return (t, pick("kind"), pick("abil"))
    k1 = pick("kind"); k2 = pick("kind")
    while k2 == k1 and len(pools["kind"]) > 1:
        k2 = pick("kind")
    return ("trans", k1, k2, pick("trans"))


def split_pairs(pairs, held_frac=0.2, seed=0):
    """COMBO holdout: unseen COMBINATIONS of seen words (compositional recombination)."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(pairs))
    rng.shuffle(idx)
    cut = int(len(pairs) * (1.0 - held_frac))
    return [pairs[i] for i in idx[:cut]], [pairs[i] for i in idx[cut:]]


def content_words(sig):
    if sig[0] in ("univ", "nouniv", "exist"):
        return {sig[1], sig[2]}                 # kind, verb
    if sig[0] == "trans":
        return {sig[1], sig[2], sig[3]}         # kind1, kind2, transverb
    return set()


def split_lexical(pairs, seed=0, n_kinds=2, n_abil=1, n_trans=1):
    """LEXICAL holdout: hold out entire WORDS -- they NEVER appear in training, in any template. Tests
    whether the model can map a word it has never been trained on (the pointer-copy / morphology limit:
    a novel KIND whose singular surface == its LOTA symbol can be copied; plural/past morphology cannot)."""
    rng = np.random.default_rng(seed)
    hk = set(rng.choice(list(KINDS), n_kinds, replace=False).tolist())
    ha = set(rng.choice(list(ABIL), n_abil, replace=False).tolist())
    ht = set(rng.choice(list(TRANS), n_trans, replace=False).tolist())
    hw = hk | ha | ht
    train = [p for p in pairs if not (content_words(p["sig"]) & hw)]
    test = [p for p in pairs if content_words(p["sig"]) & hw]
    return train, test, dict(kinds=sorted(hk), abil=sorted(ha), trans=sorted(ht))


def split_structural(pairs, held_family="nouniv"):
    """STRUCTURAL holdout: hold out a whole SENTENCE STRUCTURE (a template family). Tests systematicity
    -- generalizing to an unseen construction. (In this grammar each structure carries some unique
    tokens, so this also probes the unseen-token floor; documented in the report.)"""
    train = [p for p in pairs if p["sig"][0] != held_family]
    test = [p for p in pairs if p["sig"][0] == held_family]
    return train, test


# ===================================================================================================
# 2. VOCAB + ENCODING (shared word/symbol vocab; <parse>/<gen> direction tags)
# ===================================================================================================
SPECIAL = ["<pad>", "<bos>", "<sep>", "<eos>", "<parse>", "<gen>"]

# Tokenization mode. 'word' = whitespace/symbol tokens (novel words are unseen atoms -> uncopyable).
# 'char' = characters (a novel word decomposes into SEEN characters -> copyable). The whole point of
# the char rebuild is to crack novel-WORD generalization, which word-level cannot (diagnosed root cause).
TOKMODE = {"mode": "word"}


def nl_toks(s):
    return list(s) if TOKMODE["mode"] == "char" else s.split()


def lota_toks(s):
    return list(s) if TOKMODE["mode"] == "char" else L.tokenize(s)


def detok(tokens):
    """Reconstruct a string from decoded tokens (chars join tight; words join with spaces)."""
    return "".join(tokens) if TOKMODE["mode"] == "char" else " ".join(tokens)


class Vocab:
    def __init__(self, itos):
        self.itos = itos
        self.stoi = {t: i for i, t in enumerate(itos)}
        self.pad = 0

    @classmethod
    def build(cls, pairs):
        toks = set()
        for p in pairs:
            toks.update(nl_toks(p["nl"]))
            toks.update(lota_toks(p["lota"]))
        return cls(SPECIAL + sorted(toks))

    def __len__(self):
        return len(self.itos)

    def enc(self, toks):
        return [self.stoi[t] for t in toks]


def example_tokens(pair, direction):
    """direction 'parse': NL -> LOTA ; 'gen': LOTA -> NL. Returns (tokens, write_start)."""
    src = nl_toks(pair["nl"]) if direction == "parse" else lota_toks(pair["lota"])
    tgt = lota_toks(pair["lota"]) if direction == "parse" else nl_toks(pair["nl"])
    tag = "<parse>" if direction == "parse" else "<gen>"
    seq = ["<bos>", tag] + src + ["<sep>"] + tgt + ["<eos>"]
    write_start = len(["<bos>", tag] + src + ["<sep>"])
    return seq, write_start


# ===================================================================================================
# 3. MODEL + TRAIN
# ===================================================================================================
def build_model(vocab_size, max_len, d=256, layers=4, heads=8):
    assert (d // heads) % 2 == 0
    return ScratchpadLM(vocab=vocab_size, d=d, layers=layers, heads=heads, max_len=max_len,
                        pad=0, pos_mode="rope", tie=True, pointer=True, causal=True)


def make_batch(rng, vocab, pairs, device, batch, block, relabel_p=0.0, pools=None):
    seqs, masks, csrcs = [], [], []
    tries = 0
    while len(seqs) < batch and tries < batch * 20:
        tries += 1
        p = pairs[rng.integers(len(pairs))]
        if relabel_p > 0.0 and pools and rng.random() < relabel_p:   # relabel -> re-render
            sig = relabel_sig(p["sig"], pools, rng)
            nl, lota = render(sig)
            p = dict(nl=nl, lota=lota, sig=sig)
        direction = "parse" if rng.random() < 0.5 else "gen"
        seq, ws = example_tokens(p, direction)
        if len(seq) > block:
            continue
        ids = vocab.enc(seq)
        m = [0] * len(ids)
        for i in range(ws, len(ids)):
            m[i] = 1
        # copy-source: for each TARGET position, the first SOURCE position (before <sep>) holding the
        # same token id -> the supervision signal for the pointer copy-head.
        first = {}
        for i in range(ws):
            first.setdefault(ids[i], i)
        csrc = [-1] * len(ids)
        for i in range(ws, len(ids)):
            if ids[i] in first:
                csrc[i] = first[ids[i]]
        seqs.append(ids); masks.append(m); csrcs.append(csrc)
    if not seqs:
        return None
    Ln = max(len(s) for s in seqs)
    ids_b = torch.full((len(seqs), Ln), vocab.pad, dtype=torch.long)
    msk_b = torch.zeros((len(seqs), Ln), dtype=torch.float)
    csrc_b = torch.full((len(seqs), Ln), -1, dtype=torch.long)
    for r, (s, m, c) in enumerate(zip(seqs, masks, csrcs)):
        ids_b[r, :len(s)] = torch.tensor(s)
        msk_b[r, :len(m)] = torch.tensor(m, dtype=torch.float)
        csrc_b[r, :len(c)] = torch.tensor(c, dtype=torch.long)
    return ids_b.to(device), msk_b.to(device), csrc_b.to(device)


def train(model, vocab, pairs, steps, device, batch=32, block=64, lr=3e-3, seed=0, log_every=0,
          relabel_p=0.0, aux_w=0.0, pools=None):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    model.to(device).train()
    for step in range(steps):
        b = make_batch(rng, vocab, pairs, device, batch, block, relabel_p=relabel_p, pools=pools)
        if b is None:
            continue
        ids, mask, csrc = b
        out = model(ids)
        logits = out[:, :-1]
        tgt = ids[:, 1:]
        sel = mask[:, 1:] > 0
        loss = F.cross_entropy(logits[sel], tgt[sel])
        if aux_w > 0.0 and getattr(model, "pointer", False):   # supervise the copy head
            a0 = model.blocks[-1]._attn[:, 0]                   # (B,L,L) copy attention
            q = a0[:, :-1, :]
            src = csrc[:, 1:]
            sup = (src >= 0) & (mask[:, 1:] > 0)
            if sup.any():
                loss = loss + aux_w * F.nll_loss(torch.log(q[sup] + 1e-9), src[sup])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f"  step {step:5d}  loss {loss.item():.4f}", flush=True)
    return model


# ===================================================================================================
# 4. DECODE + GATES
# ===================================================================================================
@torch.no_grad()
def decode(model, vocab, pair, direction, device, block=64, max_new=None):
    if max_new is None:
        max_new = 110 if TOKMODE["mode"] == "char" else 48
    src = nl_toks(pair["nl"]) if direction == "parse" else lota_toks(pair["lota"])
    tag = "<parse>" if direction == "parse" else "<gen>"
    ids = vocab.enc(["<bos>", tag] + src + ["<sep>"])
    eos = vocab.stoi["<eos>"]
    out = []
    model.eval()
    for _ in range(max_new):
        ctx = torch.tensor([ids[-block:]], device=device)
        nxt = int(model(ctx)[0, -1].argmax())
        if nxt == eos:
            break
        ids.append(nxt); out.append(vocab.itos[nxt])
    return out


def _canon(lota_str):
    """Canonical LOTA string (parse + re-print); None if unparseable."""
    try:
        return L.write(L.read(lota_str))
    except Exception:
        return None


def _random_worlds(n=8, seed=0):
    """DISCRIMINATING random worlds for the semantic-equivalence gate. Each world has MULTIPLE
    entities of EVERY kind (so universals/existentials are non-vacuous and actually separate different
    forms -- a sparse world made non-equivalent forms look equal, a hollow metric we explicitly avoid)
    with abilities randomized per entity, plus several random events."""
    rng = np.random.default_rng(seed)
    ksyms, verbs, tverbs = list(KINDS), list(ABIL), list(TRANS)
    worlds = []
    for _ in range(n):
        atoms, ents = [], []
        for k in ksyms:
            for j in range(2):                         # 2 instances/kind -> non-vacuous quantifiers
                e = f"{k}{j}"; ents.append(e); atoms.append((k, (e,)))
                for v in verbs:
                    if rng.random() < 0.5:
                        atoms.append((v, (e,)))
        for ei in range(4):                            # several events with random agent/patient
            a, b = ents[rng.integers(len(ents))], ents[rng.integers(len(ents))]
            tv = tverbs[rng.integers(len(tverbs))]
            ev = f"ev{ei}"
            atoms += [(tv, (ev,)), ("agent", (ev, a)), ("patient", (ev, b))]
        worlds.append(L.World(atoms=atoms))
    return worlds


def _semeq(lota_a, lota_b, worlds):
    """True iff the two forms evaluate to the same truth in every world (best-effort; skips on error)."""
    ok = checked = 0
    for w in worlds:
        try:
            checked += 1
            if L.evaluate(L.desugar(L.read(lota_a)), w) == L.evaluate(L.desugar(L.read(lota_b)), w):
                ok += 1
        except Exception:
            return None
    return checked > 0 and ok == checked


def eval_parse(model, vocab, test, device, block):
    wf = exact = sem = 0
    worlds = _random_worlds(6, seed=1)
    samples = []
    for p in test:
        pred = detok(decode(model, vocab, p, "parse", device, block))
        canon = _canon(pred)
        wellformed = canon is not None and L.check(L.read(pred)) == []
        gold = _canon(p["lota"])
        em = wellformed and canon == gold
        se = bool(em) or (wellformed and _semeq(pred, p["lota"], worlds) is True)
        wf += int(wellformed); exact += int(em); sem += int(se)
        samples.append((p["nl"], pred, p["lota"], em))
    n = max(1, len(test))
    return dict(n=len(test), well_formed=wf / n, exact_match=exact / n, semantic=sem / n), samples


def eval_gen(model, vocab, test, device, block):
    """Round-trip faithfulness: generate English from LOTA, parse it back, recover the gold LOTA."""
    exact = faithful = 0
    samples = []
    for p in test:
        gen = detok(decode(model, vocab, p, "gen", device, block))
        em = gen == p["nl"]
        back = detok(decode(model, vocab, {"nl": gen, "lota": p["lota"]}, "parse", device, block))
        rt = _canon(back) == _canon(p["lota"]) and _canon(p["lota"]) is not None
        exact += int(em); faithful += int(rt)
        samples.append((p["lota"], gen, p["nl"], em, rt))
    n = max(1, len(test))
    return dict(n=len(test), exact_match=exact / n, roundtrip_faithful=faithful / n), samples


# ===================================================================================================
# 5. SELFTEST + RUN
# ===================================================================================================
def selftest():
    pairs = gen_pairs()
    assert len(pairs) > 100, ("too few pairs", len(pairs))
    # every generated LOTA must be well-formed + parseable + checkable
    for p in pairs[:50]:
        node = L.read(p["lota"])
        assert L.check(node) == [], ("bad gold LOTA", p["lota"], L.check(node))
    train_p, test_p = split_pairs(pairs, held_frac=0.2, seed=0)
    assert test_p and train_p
    # signatures in test must be disjoint from train (held-out COMBINATIONS)
    tr_sig = {tuple(p["sig"]) for p in train_p}
    te_sig = {tuple(p["sig"]) for p in test_p}
    assert tr_sig.isdisjoint(te_sig), "train/test signature leak"

    vocab = Vocab.build(pairs)
    # encoding round-trips
    seq, ws = example_tokens(train_p[0], "parse")
    assert vocab.enc(seq) and ws < len(seq)

    # tiny train lifts loss + produces at least one well-formed parse
    dev = "cpu"
    torch.set_num_threads(2)
    m = build_model(len(vocab), max_len=64, d=64, layers=2, heads=4)
    train(m, vocab, train_p, steps=120, device=dev, batch=16, block=64, lr=3e-3, seed=0)
    pr, _ = eval_parse(m, vocab, test_p[:8], dev, 64)
    assert pr["well_formed"] > 0.0, "no well-formed parse after tiny train"
    # the verifier gates work end to end
    assert _canon("(forall (?x dog) (can (run :agent ?x)))") is not None
    print("lang_bridge selftest OK")


def run(steps, out_path, seed=0, d=256, layers=4, heads=8, block=64, device="cpu", verbose=True,
        regime="combo", relabel_p=0.0, aux_w=0.0):
    pairs = gen_pairs()
    if regime == "lexical":
        train_p, test_p, hw = split_lexical(pairs, seed=seed)
        info = f"unseen WORDS {hw}"
    elif regime == "structural":
        train_p, test_p = split_structural(pairs, held_family="nouniv")
        info = "unseen STRUCTURE ('no K can V' / negative-universal held out entirely)"
    else:
        regime = "combo"
        train_p, test_p = split_pairs(pairs, held_frac=0.2, seed=seed)
        info = "unseen COMBINATIONS of seen words"
    vocab = Vocab.build(pairs)
    if verbose:
        print(f"[regime: {regime} | tok: {TOKMODE['mode']}] {info}", flush=True)
        print(f"pairs: {len(pairs)} (train {len(train_p)} / held-out {len(test_p)}) | "
              f"vocab {len(vocab)} | device {device} | steps {steps}", flush=True)
    if not test_p:
        print("  (empty held-out set; skipping)"); return dict(regime=regime, n_test=0)
    pools = lemma_pools(train_p)                      # relabel only among TRAIN lemmas (no leak)
    m = build_model(len(vocab), max_len=block, d=d, layers=layers, heads=heads)
    train(m, vocab, train_p, steps=steps, device=device, batch=32, block=block, lr=3e-3,
          seed=seed, log_every=(max(1, steps // 8) if verbose else 0),
          relabel_p=relabel_p, aux_w=aux_w, pools=pools)
    pr, psamp = eval_parse(m, vocab, test_p, device, block)
    gr, gsamp = eval_gen(m, vocab, test_p, device, block)
    if verbose:
        print("\n== held-out PARSE (English -> LOTA) ==")
        print(f"  well-formed {pr['well_formed']:.3f} | exact-match {pr['exact_match']:.3f} | "
              f"semantic-equiv {pr['semantic']:.3f}  (n={pr['n']})")
        for nl, pred, gold, em in psamp[:5]:
            print(f"   [{'OK ' if em else 'xx '}] {nl!r} -> {pred}")
        misses = [(nl, pred, gold) for nl, pred, gold, em in psamp if not em]
        if misses:
            print(f"  -- {len(misses)} parse misses (non-exact); first few: --")
            for nl, pred, gold in misses[:5]:
                print(f"     {nl!r}\n       got  {pred}\n       gold {gold}")
        print("\n== held-out GENERATE (LOTA -> English) ==")
        print(f"  exact-match {gr['exact_match']:.3f} | round-trip-faithful "
              f"{gr['roundtrip_faithful']:.3f}  (n={gr['n']})")
        for lota, gen, gold, em, rt in gsamp[:5]:
            print(f"   [{'OK ' if rt else 'xx '}] {lota} -> {gen!r}")
    result = dict(steps=steps, seed=seed, device=device, regime=regime,
                  model=dict(d=d, layers=layers, heads=heads, vocab=len(vocab)),
                  n_train=len(train_p), n_test=len(test_p), parse=pr, generate=gr)
    if out_path:
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        if verbose:
            print(f"\nwrote {out_path}")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--regime", default="combo",
                    choices=["combo", "lexical", "structural", "all"])
    ap.add_argument("--relabel", type=float, default=0.0,
                    help="prob of relabeling content lemmas during training (forces copy -> the lever "
                         "for NOVEL-WORD generalization)")
    ap.add_argument("--aux", type=float, default=0.0, help="weight on the copy-head supervision loss")
    ap.add_argument("--char", action="store_true",
                    help="CHARACTER-level tokenization (novel words compose from seen chars -> "
                         "copyable; the diagnosed fix for novel-WORD generalization)")
    args = ap.parse_args(argv)
    if args.char:
        TOKMODE["mode"] = "char"
        if args.block < 128:
            args.block = 160
    if args.selftest:
        selftest(); return 0
    regimes = ["combo", "lexical", "structural"] if args.regime == "all" else [args.regime]
    results = {}
    for r in regimes:
        print(f"\n{'='*70}\nREGIME = {r}  (relabel={args.relabel} aux={args.aux})\n{'='*70}")
        results[r] = run(args.steps, None, seed=args.seed, d=args.d, layers=args.layers,
                         heads=args.heads, block=args.block, device=args.device, regime=r,
                         relabel_p=args.relabel, aux_w=args.aux)
    print(f"\n{'='*70}\nGENERALIZATION MAP (held-out, lead-measured)\n{'='*70}")
    print(f"{'regime':>12} | {'n':>4} | {'parse exact':>11} | {'parse wf':>8} | "
          f"{'gen exact':>9} | {'roundtrip':>9}")
    for r in regimes:
        x = results[r]
        if x.get("n_test"):
            print(f"{r:>12} | {x['n_test']:>4} | {x['parse']['exact_match']:>11.3f} | "
                  f"{x['parse']['well_formed']:>8.3f} | {x['generate']['exact_match']:>9.3f} | "
                  f"{x['generate']['roundtrip_faithful']:>9.3f}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
