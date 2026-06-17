"""English COMPREHENSION exam: read a tiny self-made corpus, then PASS a fresh
held-out cloze exam BY COMPREHENSION (bidirectional masked fill), not by
left-to-right next-token prediction.

Self-contained and fresh. Reuses ONE engine -- ScratchpadLM configured as a
BIDIRECTIONAL masked encoder (causal=False -> full attention; the causal mask in
scratchpad_model.CausalBlock.forward is applied only when use_causal is True). The
model READS by masked-LM (mask ~15% of positions, predict the originals at masked
positions only), then ANSWERS a cloze item by placing <mask> in the blank, running
the bidirectional forward, and scoring each option's log-prob AT the mask position.

The exam items are HELD-OUT instances of the SAME grammar/vocab patterns the corpus
teaches (subject-verb agreement, a/an articles, plurals, prepositions, grounded
animal/color/object vocab). Passing therefore requires generalizing the rule, not
recalling a memorized sentence -- exam sentences are disjoint from corpus sentences.

THE FER COMPARISON: the project bet is that a FER-style FACTORED relational
representation generalizes from FEWER examples. "FER-on" here = ScratchpadLM with
arch="relational" (Dual-Attention: half the heads read LEARNED token-identity
symbols as values -- the Edge-Transformer / relational-bottleneck family in
scratchpad_model.py). "FER-off" = arch="standard" (ordinary content attention). We
train both at several corpus sizes and compare held-out exam accuracy.

CPU/MPS only -- small model, trains in minutes. No GPU, no RunPod, no external data.

Run:
  python -m thinking.exam --selftest
  python -m thinking.exam --steps 1500 --seed 0
  python -m thinking.exam --fer --steps 1500 --seed 0   # also run the FER table
"""
import argparse
import json
import math
import re

import numpy as np
import torch
import torch.nn.functional as F

from scratchpad_model import ScratchpadLM

PAD = "<pad>"
MASK = "<mask>"


# --------------------------------------------------------------------------- #
# 1. Vocab
# --------------------------------------------------------------------------- #
class WordVocab:
    """Word-level vocab. index 0=<pad>, 1=<mask>, then corpus words (sorted).
    Own whitespace/punctuation splitter. JSON-serializable via itos."""

    _TOKEN = re.compile(r"[A-Za-z]+|[.,!?;:]")

    def __init__(self, itos):
        assert itos[0] == PAD and itos[1] == MASK, "itos must start with <pad>,<mask>"
        self.itos = list(itos)
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.pad = 0
        self.mask = 1

    @classmethod
    def build(cls, sentences):
        words = set()
        for s in sentences:
            words.update(cls.tokenize(s))
        itos = [PAD, MASK] + sorted(words)
        return cls(itos)

    @staticmethod
    def tokenize(s):
        return cls_tokenize(s)

    def encode(self, s):
        return [self.stoi[w] for w in self.tokenize(s) if w in self.stoi]

    def decode(self, ids):
        out = []
        for i in ids:
            w = self.itos[i]
            if w in (PAD,):
                continue
            out.append(w)
        # join words with spaces, but no space before punctuation
        s = ""
        for w in out:
            if re.fullmatch(r"[.,!?;:]", w):
                s += w
            else:
                s += (" " if s else "") + w
        return s

    def has(self, w):
        return w in self.stoi

    def __len__(self):
        return len(self.itos)


def cls_tokenize(s):
    return WordVocab._TOKEN.findall(s.lower())


# --------------------------------------------------------------------------- #
# 2. Corpus -- tiny self-made grounded English teaching LEARNABLE patterns.
# --------------------------------------------------------------------------- #
# Grounded vocab. The corpus and the exam are kept genuinely DISJOINT (verified
# at runtime): the exam reuses the same WORDS and the same RULES but in
# combinations / frames that NEVER appear verbatim in training, so passing
# requires generalizing the rule (compose a known word's number with a known
# verb's agreement; apply a known a/an or plural in a novel frame), not recall.
ANIMALS_SG = ["cat", "dog", "bird", "fish", "horse", "cow", "duck", "fox",
              "frog", "bee", "owl", "pig", "hen", "goat", "rat", "bat"]
COLORS = ["red", "blue", "green", "black", "white", "brown", "yellow", "gray"]
OBJECTS_SG = ["ball", "cup", "box", "hat", "book", "car", "key", "pen",
              "shoe", "rock", "leaf", "cake", "drum", "kite", "lamp", "nest"]
PLACES = ["mat", "hill", "tree", "barn", "pond", "field", "roof", "bed"]
PREPS = ["on", "in", "under", "near", "by"]
VERBS = ["sits", "runs", "jumps", "sleeps", "eats", "walks", "swims", "hops"]
# words beginning with a vowel sound -> take "an"
VOWEL_NOUNS = ["apple", "egg", "owl", "ant", "ox", "ear", "elf", "igloo"]
CONS_NOUNS = ["dog", "cat", "ball", "cup", "hat", "book", "car", "key"]

# ----- HELD-OUT splits (the exam tests exactly these; the corpus must NOT) -----
# Agreement: these (subject, verb) PAIRS are never co-occurred in training. The
# subject's number is taught with OTHER verbs; the verb's agreement with OTHER
# subjects -> the exam pair is a novel composition.
HELDOUT_AGREE = [("cat", "sits"), ("dog", "runs"), ("bird", "jumps"),
                 ("fish", "swims"), ("frog", "hops"), ("owl", "sleeps")]
# Article: training teaches a/an in the "i see / she has" frames; the exam uses
# a DISTINCT frame ("look at ...") that never appears in training -> the article
# choice for the noun must transfer to a novel frame.
ARTICLE_TRAIN_FRAMES = ["i see", "she has", "we want", "he found"]
ARTICLE_EXAM_FRAME = "look at"
# Plural: training counts with "one/two/many"; the exam counts with "three",
# a number word that only ever appears in the exam frame.
PLURAL_EXAM_NUMBER = "three"
# Vocab (color slot): these (color, object) pairs are held out of the
# color-predication and color-NP frames in training.
HELDOUT_COLOR = [("red", "ball"), ("blue", "cup"), ("green", "box"), ("black", "hat")]


def _plural(w):
    if w.endswith(("s", "x", "z", "ch", "sh")):
        return w + "es"
    if w.endswith("y") and w[-2] not in "aeiou":
        return w[:-1] + "ies"
    if w == "leaf":
        return "leaves"
    return w + "s"


def _verb_plural(v):
    # singular verb (3rd person -s) -> bare plural verb. drill agreement.
    table = {"sits": "sit", "runs": "run", "jumps": "jump", "sleeps": "sleep",
             "eats": "eat", "walks": "walk", "swims": "swim", "hops": "hop"}
    return table[v]


def build_corpus(seed, n=None):
    """Tiny grounded English. Returns a list of sentences (lowercased w/ a period).
    If n is given, subsample to n sentences (deterministically) for the
    sample-efficiency sweep. Every sentence instantiates a learnable pattern."""
    rng = np.random.default_rng(seed)
    sents = []
    heldout_agree = set(HELDOUT_AGREE)
    heldout_color = set(HELDOUT_COLOR)

    # -- subject-verb agreement: "the cat sits" / "the cats sit"
    #    SKIP held-out (subject, verb) pairs entirely; every subject still sees
    #    other verbs and every verb still sees other subjects.
    for a in ANIMALS_SG:
        for v in VERBS:
            if (a, v) in heldout_agree:
                continue
            sents.append(f"the {a} {v} .")
            sents.append(f"the {_plural(a)} {_verb_plural(v)} .")

    # -- articles: a/an across several TRAIN frames (never the exam frame's
    #    a/an slot). The exam frame ("look at") IS exposed -- but only with the
    #    definite article -- so its embedding is trained while the a/an decision
    #    in that frame stays held out.
    for fr in ARTICLE_TRAIN_FRAMES:
        for w in CONS_NOUNS:
            sents.append(f"{fr} a {w} .")
        for w in VOWEL_NOUNS:
            sents.append(f"{fr} an {w} .")
    for w in CONS_NOUNS + VOWEL_NOUNS:
        sents.append(f"{ARTICLE_EXAM_FRAME} the {w} .")

    # -- plurals: counting drills "one cat" / "two cats" / "many cats".
    #    The exam counts with "three". To give "three" a trained embedding we
    #    DO use it in training -- but only with nouns that are NOT in the exam's
    #    plural set, so the exam (three, exam-noun) combos remain held out while
    #    the rule "three + plural" is learnable from the other nouns.
    plural_exam_nouns = {"cat", "dog", "ball", "cup", "box", "fox", "leaf", "key"}
    for w in ANIMALS_SG + OBJECTS_SG:
        sents.append(f"one {w} .")
        sents.append(f"two {_plural(w)} .")
        sents.append(f"many {_plural(w)} .")
        if w not in plural_exam_nouns:
            sents.append(f"{PLURAL_EXAM_NUMBER} {_plural(w)} .")

    # -- prepositions + grounded color/object: "the red ball is on the mat"
    #    SKIP held-out (color, object) pairs.
    for c in COLORS:
        for o in OBJECTS_SG:
            if (c, o) in heldout_color:
                continue
            p = PREPS[rng.integers(len(PREPS))]
            pl = PLACES[rng.integers(len(PLACES))]
            sents.append(f"the {c} {o} is {p} the {pl} .")

    # -- color predication: "the cat is black" (teaches color words in a slot)
    for a in ANIMALS_SG:
        c = COLORS[rng.integers(len(COLORS))]
        sents.append(f"the {a} is {c} .")

    # dedup, deterministic shuffle
    sents = sorted(set(sents))
    rng2 = np.random.default_rng(seed + 1)
    rng2.shuffle(sents)
    if n is not None:
        sents = sents[:n]
    return sents


# --------------------------------------------------------------------------- #
# 3. Exam -- HELD-OUT cloze items testing the SAME patterns.
# --------------------------------------------------------------------------- #
def build_exam(seed):
    """Cloze items {left, right, options, answer, cat}. Each is a HELD-OUT
    instance of a trained pattern -- the exact (frame, word) combination never
    occurs in build_corpus (verified disjoint by main, overlap must be 0).
    Options are single words; chance = 1/len(options)."""
    rng = np.random.default_rng(seed + 100)
    items = []

    # --- agreement (held-out subject+verb pairs) ---
    # "the cat ___" -> the singular verb form ; "the cats ___" -> the plural.
    # The (cat, sits) pair was withheld from training, so a correct answer means
    # composing cat's number (seen with other verbs) with the agreement of this
    # verb (seen with other subjects).
    for a, vs in HELDOUT_AGREE:
        vp = _verb_plural(vs)
        items.append(dict(cat="agreement",
                          left=f"the {a}", right=".",
                          options=[vs, vp], answer=vs))
        items.append(dict(cat="agreement",
                          left=f"the {_plural(a)}", right=".",
                          options=[vs, vp], answer=vp))

    # --- articles a/an in a NOVEL frame ("look at ...") ---
    art = [("dog", "a"), ("apple", "an"), ("egg", "an"), ("cup", "a"),
           ("owl", "an"), ("ball", "a"), ("ant", "an"), ("hat", "a")]
    for w, ans in art:
        items.append(dict(cat="article",
                          left=ARTICLE_EXAM_FRAME, right=f"{w} .",
                          options=["a", "an"], answer=ans))

    # --- plurals in a NOVEL counting frame ("three ___") ---
    plur = ["cat", "dog", "ball", "cup", "box", "fox", "leaf", "key"]
    for w in plur:
        items.append(dict(cat="plural",
                          left=PLURAL_EXAM_NUMBER, right=".",
                          options=[w, _plural(w)], answer=_plural(w)))

    # --- vocab: held-out (color, object) color-slot fills ---
    # "the ___ ball is on the mat" -- the right answer is the color; the
    # distractor is a non-color noun. These (color, object) pairs were withheld.
    for c, o in HELDOUT_COLOR:
        distract = rng.choice([x for x in OBJECTS_SG if x != o] +
                              [x for x in ANIMALS_SG])
        items.append(dict(cat="vocab",
                          left="the", right=f"{o} is on the mat .",
                          options=[c, distract], answer=c))

    rng.shuffle(items)
    return items


# --------------------------------------------------------------------------- #
# 4. masked-LM reading
# --------------------------------------------------------------------------- #
def _batch(seqs, pad):
    L = max(len(s) for s in seqs)
    out = torch.full((len(seqs), L), pad, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out


def train_mlm(model, vocab, sentences, steps, mask_p=0.15, seed=0, lr=3e-3,
              batch=32, device="cpu", log_every=0):
    """Masked-LM reading. Mask ~mask_p of (non-pad) positions with <mask>,
    bidirectional forward, cross-entropy ONLY at the masked positions vs the
    original tokens. Always mask at least one position per sentence."""
    rng = np.random.default_rng(seed)
    enc = [vocab.encode(s) for s in sentences]
    enc = [e for e in enc if len(e) >= 2]
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for step in range(steps):
        idx = rng.integers(0, len(enc), size=min(batch, len(enc)))
        seqs = [list(enc[i]) for i in idx]
        masked, targets = [], []
        for s in seqs:
            ms = list(s)
            tg = [-100] * len(s)
            # choose positions to mask
            n_mask = max(1, int(round(len(s) * mask_p)))
            pos = rng.choice(len(s), size=n_mask, replace=False)
            for p in pos:
                tg[p] = s[p]
                ms[p] = vocab.mask
            masked.append(ms)
            targets.append(tg)
        ids = _batch(masked, vocab.pad).to(device)
        tgt = _batch_targets(targets).to(device)

        logits = model(ids)  # (B,L,V) bidirectional (model.causal=False)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               tgt.reshape(-1), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f"  step {step:4d}  mlm_loss {loss.item():.4f}")
    return model


def _batch_targets(targets):
    L = max(len(t) for t in targets)
    out = torch.full((len(targets), L), -100, dtype=torch.long)
    for i, t in enumerate(targets):
        out[i, : len(t)] = torch.tensor(t, dtype=torch.long)
    return out


# --------------------------------------------------------------------------- #
# 5. answer_cloze -- COMPREHENSION (bidirectional masked fill)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _mask_logprobs(model, vocab, left, right, device="cpu"):
    """Return log-softmax distribution AT the single mask position for
    left + [<mask>] + right, using the bidirectional forward."""
    lids = vocab.encode(left)
    rids = vocab.encode(right)
    ids = lids + [vocab.mask] + rids
    mpos = len(lids)
    t = torch.tensor([ids], dtype=torch.long, device=device)
    logits = model(t)  # bidirectional
    return F.log_softmax(logits[0, mpos], dim=-1)


@torch.no_grad()
def answer_cloze(model, vocab, item, device="cpu"):
    """Bidirectional masked fill: place <mask> in the blank, score each option
    word's log-prob at the mask position, return the argmax option."""
    model.eval()
    lp = _mask_logprobs(model, vocab, item["left"], item["right"], device)
    best, best_lp = None, -1e30
    for opt in item["options"]:
        oid = vocab.stoi.get(opt, None)
        score = lp[oid].item() if oid is not None else -1e30
        if score > best_lp:
            best_lp, best = score, opt
    return best


@torch.no_grad()
def answer_cloze_causal(model, vocab, item, device="cpu"):
    """CONTRAST baseline: same weights, but CAUSAL forward + left-to-right option
    scoring. Score = sum of log p(token | preceding) over left + option (no
    right context, because a causal LM cannot see the future). This is
    next-token prediction, NOT comprehension."""
    model.eval()
    best, best_lp = None, -1e30
    for opt in item["options"]:
        oid = vocab.stoi.get(opt, None)
        if oid is None:
            continue
        ids = vocab.encode(item["left"]) + [oid]
        if len(ids) < 2:
            # nothing to condition on; fall back to scoring option's marginal at pos0
            t = torch.tensor([[ids[-1]]], dtype=torch.long, device=device)
            logits = model(t, causal=True)
            score = F.log_softmax(logits[0, 0], -1)[oid].item()
        else:
            t = torch.tensor([ids], dtype=torch.long, device=device)
            logits = model(t, causal=True)  # causal forward on SAME weights
            lp = F.log_softmax(logits, dim=-1)
            # score every predicted token from position 0.. predicting ids[1..]
            score = 0.0
            for p in range(len(ids) - 1):
                score += lp[0, p, ids[p + 1]].item()
        if score > best_lp:
            best_lp, best = score, opt
    return best


# --------------------------------------------------------------------------- #
# 6. exam_accuracy
# --------------------------------------------------------------------------- #
def exam_accuracy(model, vocab, items, device="cpu", answer_fn=answer_cloze):
    correct = 0
    cats = {}
    for it in items:
        pred = answer_fn(model, vocab, it, device)
        ok = pred == it["answer"]
        correct += ok
        c = it.get("cat", "all")
        cats.setdefault(c, [0, 0])
        cats[c][0] += ok
        cats[c][1] += 1
    acc = correct / len(items)
    chance = float(np.mean([1.0 / len(it["options"]) for it in items]))
    per_cat = {c: round(n / d, 3) for c, (n, d) in sorted(cats.items())}
    return dict(acc=round(acc, 4), chance=round(chance, 4),
                passed=acc >= 0.80, n=len(items), per_cat=per_cat)


# --------------------------------------------------------------------------- #
# model factory
# --------------------------------------------------------------------------- #
def make_model(vocab_size, max_len, arch="standard", d=128, layers=3, heads=4):
    assert (d // heads) % 2 == 0, "RoPE needs even head dim"
    return ScratchpadLM(vocab=vocab_size, d=d, layers=layers, heads=heads,
                        max_len=max_len, pad=0, arch=arch, pos_mode="rope",
                        tie=True, pointer=False, causal=False)


def _max_len(vocab, sentences, items):
    m = 4
    for s in sentences:
        m = max(m, len(vocab.encode(s)))
    for it in items:
        m = max(m, len(vocab.encode(it["left"])) + 1 + len(vocab.encode(it["right"])))
    return m + 4


# --------------------------------------------------------------------------- #
# save / load
# --------------------------------------------------------------------------- #
def save(path, model, vocab):
    blob = dict(config=model.config,
                itos=vocab.itos,
                state_dict={k: v.cpu() for k, v in model.state_dict().items()})
    torch.save(blob, path)


def load(path, device="cpu"):
    blob = torch.load(path, map_location=device, weights_only=False)
    vocab = WordVocab(blob["itos"])
    model = ScratchpadLM(**blob["config"])
    model.load_state_dict(blob["state_dict"])
    model.to(device).eval()
    return model, vocab


# --------------------------------------------------------------------------- #
# selftest
# --------------------------------------------------------------------------- #
def selftest():
    import tempfile, os
    device = "cpu"
    torch.manual_seed(0)

    # vocab round-trip
    sents = ["the cat sits .", "the cats sit .", "i see an apple ."]
    vocab = WordVocab.build(sents)
    # round-trip on token ids (decode normalizes spacing before punctuation)
    s = "the cat sits ."
    ids = vocab.encode(s)
    assert vocab.encode(vocab.decode(ids)) == ids, "vocab round-trip failed"
    assert vocab.decode(ids) == "the cat sits.", "decode spacing unexpected"

    # trivially-learnable mini-pattern: "the X big" / "the X small" with a fixed
    # cue word -- ~300 steps must lift a held cloze above chance (0.5).
    mini = []
    for x in ["a", "b", "c", "d", "e", "f"]:
        mini.append(f"the {x} big stop .")
        mini.append(f"the {x} small go .")
    mvocab = WordVocab.build(mini)
    mlen = _max_len(mvocab, mini, [])
    model = make_model(len(mvocab), mlen, arch="standard", d=32, layers=1, heads=2)
    train_mlm(model, mvocab, mini, steps=300, seed=0, lr=5e-3, batch=12, device=device)
    # held cloze: "big" -> "stop", "small" -> "go"
    items = [dict(cat="m", left="the a big", right=".", options=["stop", "go"], answer="stop"),
             dict(cat="m", left="the a small", right=".", options=["stop", "go"], answer="go")]
    res = exam_accuracy(model, mvocab, items, device)
    assert res["acc"] > res["chance"], f"selftest pattern not learned: {res}"

    # save/load identical prediction
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "m.pt")
        save(p, model, mvocab)
        m2, v2 = load(p, device)
        a1 = answer_cloze(model, mvocab, items[0], device)
        a2 = answer_cloze(m2, v2, items[0], device)
        assert a1 == a2, "save/load prediction mismatch"

    print("exam selftest OK")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def _verify_disjoint(corpus, items, vocab):
    cset = set(corpus)
    overlap = 0
    for it in items:
        full = (it["left"] + " " + it["answer"] + " " + it["right"]).strip()
        # normalize spacing the same way decode does
        norm = vocab.decode(vocab.encode(full))
        if norm in {vocab.decode(vocab.encode(c)) for c in cset}:
            overlap += 1
    return overlap


def run_once(steps, seed, d, layers, heads, arch, out_path=None, device="cpu",
             verbose=True):
    corpus = build_corpus(seed)
    items = build_exam(seed)
    vocab = WordVocab.build(corpus + [it["left"] + " " + it["right"] + " " +
                                      " ".join(it["options"]) for it in items])
    mlen = _max_len(vocab, corpus, items)

    overlap = _verify_disjoint(corpus, items, vocab)
    if verbose:
        print(f"corpus sentences: {len(corpus)}  vocab: {len(vocab)}  max_len: {mlen}")
        print(f"exam items: {len(items)}  held-out overlap with corpus: {overlap}")

    model = make_model(len(vocab), mlen, arch=arch, d=d, layers=layers, heads=heads)
    train_mlm(model, vocab, corpus, steps=steps, seed=seed, device=device,
              log_every=(steps // 5 if verbose else 0))

    res = exam_accuracy(model, vocab, items, device, answer_cloze)
    res_causal = exam_accuracy(model, vocab, items, device, answer_cloze_causal)

    if verbose:
        print(f"\n== held-out exam (arch={arch}) ==")
        print(f"comprehension (bidirectional masked fill): acc={res['acc']}  "
              f"chance={res['chance']}  passed={res['passed']}")
        print(f"  per-category: {res['per_cat']}")
        print(f"causal next-token baseline (same weights):  acc={res_causal['acc']}  "
              f"chance={res_causal['chance']}")
        print(f"  per-category: {res_causal['per_cat']}")

    if out_path:
        with open(out_path, "w") as f:
            json.dump(dict(arch=arch, steps=steps, seed=seed,
                           corpus=len(corpus), comprehension=res,
                           causal_baseline=res_causal, overlap=overlap), f, indent=2)
        if verbose:
            print(f"wrote {out_path}")
    return res, res_causal


def run_fer_sweep(steps, seed, d, layers, heads, device="cpu",
                  sizes=(50, 100, 200, 400)):
    """Sample-efficiency: FER-on (arch=relational) vs FER-off (arch=standard) at
    several corpus sizes; report held-out exam accuracy for each. Held-out exam
    and vocab are fixed across sizes (full vocab) so only corpus size varies."""
    full = build_corpus(seed)
    items = build_exam(seed)
    vocab = WordVocab.build(full + [it["left"] + " " + it["right"] + " " +
                                    " ".join(it["options"]) for it in items])
    mlen = _max_len(vocab, full, items)
    print(f"\n== FER sample-efficiency sweep (full corpus={len(full)}) ==")
    print(f"{'size':>6} | {'FER-off(std)':>13} | {'FER-on(rel)':>12}")
    print("-" * 40)
    table = {}
    for n in sizes:
        if n > len(full):
            continue
        corpus = full[:n]
        accs = {}
        for arch in ("standard", "relational"):
            m = make_model(len(vocab), mlen, arch=arch, d=d, layers=layers, heads=heads)
            train_mlm(m, vocab, corpus, steps=steps, seed=seed, device=device)
            r = exam_accuracy(m, vocab, items, device, answer_cloze)
            accs[arch] = r["acc"]
        table[n] = accs
        print(f"{n:>6} | {accs['standard']:>13.3f} | {accs['relational']:>12.3f}")
    # verdict
    def first_pass(arch):
        for n in sorted(table):
            if table[n][arch] >= 0.80:
                return n
        return None
    fp_off, fp_on = first_pass("standard"), first_pass("relational")
    print("-" * 40)
    print(f"first passing size: FER-off={fp_off}  FER-on={fp_on}")
    if fp_on is not None and (fp_off is None or fp_on < fp_off):
        verdict = f"YES -- FER-on reaches >=0.80 at fewer examples ({fp_on} vs {fp_off})"
    elif fp_off is not None and (fp_on is None or fp_off < fp_on):
        verdict = f"NO -- FER-off reaches >=0.80 at fewer examples ({fp_off} vs {fp_on})"
    elif fp_on is not None and fp_on == fp_off:
        verdict = f"INCONCLUSIVE -- both first pass at size {fp_on}"
    else:
        verdict = "INCONCLUSIVE -- neither reached 0.80 in this sweep"
    print("verdict:", verdict)
    return table, verdict


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--arch", default="standard",
                    choices=["standard", "relational", "abstractor"])
    ap.add_argument("--fer", action="store_true",
                    help="also run the FER-on vs FER-off sample-efficiency sweep")
    ap.add_argument("--fer-steps", type=int, default=1200)
    ap.add_argument("--out", default=None, help="write result JSON here")
    args = ap.parse_args(argv)

    if args.selftest:
        selftest()
        return 0

    device = "cpu"  # CPU-safe / deterministic; small enough for minutes
    run_once(args.steps, args.seed, args.d, args.layers, args.heads, args.arch,
             out_path=args.out, device=device)
    if args.fer:
        run_fer_sweep(args.fer_steps, args.seed, args.d, args.layers, args.heads,
                      device=device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
