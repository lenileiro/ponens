"""C2-style REAL-English reading comprehension eval -- honest baseline.

For the FIRST time we administer a NON-TEMPLATED English comprehension exam built
from REAL public-domain prose already in the repo (data/cosmopedia_6mb.txt, a
complex textbook-style corpus; optionally data/books/alice.txt). C2-style means the
hard end of CEFR: vocabulary-in-context, discourse cohesion, and coherence. Expect
LOW / near-chance scores from a tiny from-scratch model -- that is the POINT. This
harness exists to measure honestly where the model stands, not to inflate it.

CONTAMINATION RULE (enforced): the source text is split character-wise into a
disjoint TRAIN portion and a HELD-OUT portion. The LM is trained ONLY on TRAIN.
EVERY eval item is constructed from HELD-OUT sentences the model never saw during
training. A runtime check asserts no held-out eval sentence appears in the TRAIN
text (and the splits are byte-disjoint by construction). No external/exam material.

ITEM TYPES (each: stem + 4 options + answer index; chance = 0.25), auto-built from
held-out passages:
  1. LEXICAL CLOZE   -- blank one content word (len>=4) in a held-out sentence;
                        distractors are real corpus words of the same POS-ish band
                        and similar frequency. Tests vocabulary-in-context.
  2. GAPPED-SENTENCE -- remove one sentence from a held-out multi-sentence passage;
                        pick the true sentence vs 3 sentences from OTHER held-out
                        passages. Tests discourse cohesion.
  3. NEXT-SENTENCE   -- given a held-out passage prefix, pick the true continuation
                        vs 3 distractors from elsewhere. Tests coherence.

ANSWERING MECHANISMS (reported separately):
  - UNIGRAM/frequency baseline: a sanity floor that uses ONLY corpus token
    frequencies (no learned model).
  - TRAINED LM: a small from-scratch ScratchpadLM trained on TRAIN text. We score
    each option with a CAUSAL forward (left-to-right total log-prob of the option
    text given its context) and pick the argmax. Word-level vocab capped to the
    most frequent words; rare words map to <unk>.

CPU only, deterministic. Run:
  .venv/bin/python -m thinking.c2_eval --selftest
  .venv/bin/python -m thinking.c2_eval --steps 2000 --out /tmp/c2_baseline.json
"""
import argparse
import json
import math
import re
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

from scratchpad_model import ScratchpadLM

PAD = "<pad>"
UNK = "<unk>"
BOS = "<bos>"

_TOKEN = re.compile(r"[a-z]+|[0-9]+|[.,!?;:'\"()-]")
# rough stopword set so lexical-cloze blanks land on CONTENT words
_STOP = set(
    "the a an and or but if then else of to in on at by for with from into over "
    "under is are was were be been being am do does did have has had will would "
    "shall should can could may might must this that these those it its it's he "
    "she they them his her their our your my we you i as so not no nor too very "
    "such than which who whom whose what when where why how all any both each few "
    "more most other some only own same up out off down about above after again "
    "against because before below between during further here once there while".split()
)


# --------------------------------------------------------------------------- #
# tokenization helpers
# --------------------------------------------------------------------------- #
def tokenize(s):
    return _TOKEN.findall(s.lower())


def split_sentences(text):
    """Split lowercased prose into sentences on . ! ? terminators. Keep the
    terminator. Filter to clean, well-formed sentences (alpha-dominant, sane len)."""
    text = re.sub(r"\s+", " ", text.strip())
    raw = re.split(r"(?<=[.!?])\s+", text)
    out = []
    for s in raw:
        s = s.strip()
        toks = tokenize(s)
        if not (6 <= len(toks) <= 40):
            continue
        if not s[-1:] in ".!?":
            continue
        # mostly real words
        alpha = sum(1 for t in toks if t.isalpha())
        if alpha < 0.7 * len(toks):
            continue
        out.append(s)
    return out


# --------------------------------------------------------------------------- #
# Vocab (word-level, frequency-capped, <unk> for the long tail)
# --------------------------------------------------------------------------- #
class WordVocab:
    """index 0=<pad>, 1=<unk>, 2=<bos>, then the top-`cap` most frequent words."""

    def __init__(self, itos):
        assert itos[:3] == [PAD, UNK, BOS]
        self.itos = list(itos)
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.pad, self.unk, self.bos = 0, 1, 2

    @classmethod
    def build(cls, text, cap=8000):
        cnt = Counter(tokenize(text))
        common = [w for w, _ in cnt.most_common(cap)]
        return cls([PAD, UNK, BOS] + common)

    def encode(self, s, bos=False):
        ids = [self.stoi.get(w, self.unk) for w in tokenize(s)]
        return ([self.bos] + ids) if bos else ids

    def __len__(self):
        return len(self.itos)


# --------------------------------------------------------------------------- #
# Contamination split: TRAIN text vs HELD-OUT sentences (byte-disjoint)
# --------------------------------------------------------------------------- #
def load_source(paths):
    chunks = []
    for p in paths:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            chunks.append(f.read().lower())
    return "\n".join(chunks)


def contamination_split(text, held_frac=0.15, seed=0):
    """Split the source into a TRAIN string and a list of HELD-OUT sentences.
    The split is by a contiguous TAIL slice of the raw character stream, so the
    TRAIN string and the held-out region are byte-disjoint. Eval items are built
    ONLY from held-out sentences; a later check asserts none appear in TRAIN."""
    n = len(text)
    cut = int(n * (1.0 - held_frac))
    train_text = text[:cut]
    held_text = text[cut:]
    held_sents = split_sentences(held_text)
    return train_text, held_sents


# --------------------------------------------------------------------------- #
# Item construction (all from HELD-OUT sentences)
# --------------------------------------------------------------------------- #
def _freq_band(word, freq, nbands=6):
    """Bucket a word by log-frequency so distractors are drawn from the same band."""
    f = freq.get(word, 1)
    return min(nbands - 1, int(math.log10(f + 1) * 2))


def build_lexical_cloze(held_sents, freq, vocab, n_items, rng):
    """Blank ONE content word (alpha, len>=4, in-vocab, not a stopword); options =
    true word + 3 distractors from the same frequency band that are NOT in the
    sentence. The model picks which word fills the blank."""
    # index candidate distractor words per band
    band_words = {}
    for w, f in freq.items():
        if w.isalpha() and len(w) >= 4 and w not in _STOP and w in vocab.stoi:
            band_words.setdefault(_freq_band(w, freq), []).append(w)
    items = []
    order = list(range(len(held_sents)))
    rng.shuffle(order)
    for si in order:
        if len(items) >= n_items:
            break
        toks = tokenize(held_sents[si])
        cand_pos = [i for i, t in enumerate(toks)
                    if t.isalpha() and len(t) >= 4 and t not in _STOP
                    and t in vocab.stoi]
        if not cand_pos:
            continue
        pos = cand_pos[rng.integers(len(cand_pos))]
        true = toks[pos]
        band = _freq_band(true, freq)
        pool = [w for w in band_words.get(band, [])
                if w != true and w not in toks]
        if len(pool) < 3:
            continue
        distract = list(rng.choice(pool, size=3, replace=False))
        options = distract + [true]
        rng.shuffle(options)
        left = " ".join(toks[:pos])
        right = " ".join(toks[pos + 1:])
        items.append(dict(type="lexical_cloze", left=left, right=right,
                          options=options, answer=options.index(true)))
    return items


def build_gapped_sentence(held_passages, n_items, rng):
    """From a held-out passage (>=4 sentences) remove one INTERIOR sentence; options
    = true sentence + 3 distractor sentences pulled from OTHER held-out passages."""
    all_sents = [s for p in held_passages for s in p]
    items = []
    order = list(range(len(held_passages)))
    rng.shuffle(order)
    for pi in order:
        if len(items) >= n_items:
            break
        p = held_passages[pi]
        if len(p) < 4:
            continue
        gap = 1 + int(rng.integers(len(p) - 2))  # interior sentence
        true = p[gap]
        # distractors from sentences not in this passage
        pool = [s for s in all_sents if s not in p]
        if len(pool) < 3:
            continue
        distract = list(rng.choice(pool, size=3, replace=False))
        options = distract + [true]
        rng.shuffle(options)
        before = " ".join(p[:gap])
        after = " ".join(p[gap + 1:])
        items.append(dict(type="gapped_sentence", left=before, right=after,
                          options=options, answer=options.index(true)))
    return items


def build_next_sentence(held_passages, n_items, rng):
    """Given a held-out passage PREFIX, options = true next sentence + 3 distractors
    from other passages. Pick the real continuation."""
    all_sents = [s for p in held_passages for s in p]
    items = []
    order = list(range(len(held_passages)))
    rng.shuffle(order)
    for pi in order:
        if len(items) >= n_items:
            break
        p = held_passages[pi]
        if len(p) < 3:
            continue
        k = 1 + int(rng.integers(len(p) - 1))  # prefix length, true next = p[k]
        true = p[k]
        pool = [s for s in all_sents if s not in p]
        if len(pool) < 3:
            continue
        distract = list(rng.choice(pool, size=3, replace=False))
        options = distract + [true]
        rng.shuffle(options)
        prefix = " ".join(p[:k])
        items.append(dict(type="next_sentence", left=prefix, right="",
                          options=options, answer=options.index(true)))
    return items


def group_passages(held_sents, per=5):
    """Chunk held-out sentences into contiguous passages of `per` sentences."""
    return [held_sents[i:i + per] for i in range(0, len(held_sents) - per + 1, per)]


def build_items(held_sents, freq, vocab, n_each, seed):
    rng = np.random.default_rng(seed + 7)
    passages = group_passages(held_sents, per=5)
    lex = build_lexical_cloze(held_sents, freq, vocab, n_each, rng)
    gap = build_gapped_sentence(passages, n_each, rng)
    nxt = build_next_sentence(passages, n_each, rng)
    return {"lexical_cloze": lex, "gapped_sentence": gap, "next_sentence": nxt}


# --------------------------------------------------------------------------- #
# Answering: UNIGRAM baseline
# --------------------------------------------------------------------------- #
def answer_unigram(item, freq):
    """Frequency floor. For cloze: pick the option word with the highest corpus
    frequency. For sentence-choice items: pick the option whose words have the
    highest mean log-frequency. Uses ONLY corpus counts, no learned model."""
    best, best_s = 0, -1e30
    for i, opt in enumerate(item["options"]):
        toks = tokenize(opt)
        if not toks:
            s = -1e30
        else:
            s = float(np.mean([math.log(freq.get(t, 1) + 1) for t in toks]))
        if s > best_s:
            best_s, best = s, i
    return best


# --------------------------------------------------------------------------- #
# Answering: TRAINED LM (causal option scoring)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _seq_logprob(model, vocab, context_ids, option_ids, device, max_len):
    """Total log p(option_ids | context_ids) under a causal forward. Scores only
    the option tokens (length-normalized comparison handled by caller)."""
    if not option_ids:
        return -1e30
    ids = ([vocab.bos] + context_ids + option_ids)[-max_len:]
    # how many of the trailing tokens are the option (after truncation)
    n_opt = min(len(option_ids), len(ids) - 1)
    t = torch.tensor([ids], dtype=torch.long, device=device)
    logits = model(t, causal=True)               # (1, L, V)
    lp = F.log_softmax(logits[0], dim=-1)
    total = 0.0
    L = len(ids)
    for j in range(L - n_opt, L):                # predict ids[j] from ids[:j]
        total += lp[j - 1, ids[j]].item()
    return total / n_opt                          # length-normalized


@torch.no_grad()
def answer_lm(model, vocab, item, device, max_len):
    """Score each option with the trained causal LM and pick argmax. Context =
    left stem; for cloze the option is the blanked word followed by the right
    side (so right context informs scoring only via the option's own tokens, as a
    causal LM cannot see the future -- this is honest left-to-right scoring)."""
    model.eval()
    ctx = vocab.encode(item["left"]) if item["left"] else []
    best, best_s = 0, -1e30
    for i, opt in enumerate(item["options"]):
        oid = vocab.encode(opt)
        s = _seq_logprob(model, vocab, ctx, oid, device, max_len)
        if s > best_s:
            best_s, best = s, i
    return best


# --------------------------------------------------------------------------- #
# Training the small LM on TRAIN text (causal next-token)
# --------------------------------------------------------------------------- #
def train_lm(model, vocab, train_text, steps, seq_len=128, batch=16, lr=3e-3,
             seed=0, device="cpu", log_every=0):
    """Plain causal next-token LM training over a flat token stream of TRAIN text."""
    rng = np.random.default_rng(seed)
    stream = vocab.encode(train_text)
    stream = torch.tensor(stream, dtype=torch.long)
    n = stream.numel()
    assert n > seq_len + 1, "train stream too short"
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for step in range(steps):
        starts = rng.integers(0, n - seq_len - 1, size=batch)
        xb = torch.stack([stream[s:s + seq_len] for s in starts]).to(device)
        yb = torch.stack([stream[s + 1:s + seq_len + 1] for s in starts]).to(device)
        logits = model(xb, causal=True)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               yb.reshape(-1), ignore_index=vocab.pad)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f"  step {step:4d}  lm_loss {loss.item():.4f}")
    return model


# --------------------------------------------------------------------------- #
# model factory
# --------------------------------------------------------------------------- #
def make_model(vocab_size, max_len, d=256, layers=4, heads=8):
    assert (d // heads) % 2 == 0, "RoPE needs even head dim"
    return ScratchpadLM(vocab=vocab_size, d=d, layers=layers, heads=heads,
                        max_len=max_len, pad=0, pos_mode="rope",
                        tie=True, pointer=False, causal=True)


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def score(items_by_type, answer_fn):
    """answer_fn(item) -> chosen option index. Returns per-type dict with acc,
    chance, n_items."""
    out = {}
    for typ, items in items_by_type.items():
        if not items:
            out[typ] = dict(acc=0.0, chance=0.25, n_items=0)
            continue
        correct = sum(1 for it in items if answer_fn(it) == it["answer"])
        chance = float(np.mean([1.0 / len(it["options"]) for it in items]))
        out[typ] = dict(acc=round(correct / len(items), 4),
                        chance=round(chance, 4), n_items=len(items))
    return out


# --------------------------------------------------------------------------- #
# contamination verification
# --------------------------------------------------------------------------- #
def verify_disjoint(train_text, items_by_type):
    """Assert NO eval item's source sentence appears in the TRAIN text. For cloze,
    the source sentence is left+answer+right; for sentence items it is the answer
    sentence and the surrounding context. Returns count of leaks (must be 0)."""
    leaks = 0
    for typ, items in items_by_type.items():
        for it in items:
            ans = it["options"][it["answer"]]
            if typ == "lexical_cloze":
                full = " ".join(x for x in [it["left"], ans, it["right"]] if x)
            else:
                full = ans
            # collapse whitespace; the held-out region is a byte-disjoint tail of
            # the source, so a true sentence cannot also be in train_text unless
            # the corpus repeats it verbatim -- which we still catch here.
            probe = re.sub(r"\s+", " ", full).strip()
            if len(probe) >= 25 and probe in train_text:
                leaks += 1
    return leaks


# --------------------------------------------------------------------------- #
# selftest
# --------------------------------------------------------------------------- #
def selftest():
    import tempfile, os
    device = "cpu"
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    # synthetic mini "prose": repeated learnable sentences
    base = ("the quick scientist measured the bright signal carefully today . "
            "a curious student studied the ancient manuscript with great patience . "
            "the river flowed gently past the silent village every morning . "
            "engineers designed the powerful engine inside the modern factory . "
            "the teacher explained the difficult concept to the eager class . ")
    text = base * 60
    vocab = WordVocab.build(text, cap=200)
    # vocab round-trip on encode
    ids = vocab.encode("the bright signal")
    assert vocab.stoi[vocab.itos[ids[0]]] == ids[0], "vocab round-trip failed"

    train_text, held = contamination_split(text, held_frac=0.2, seed=0)
    held = held if held else split_sentences(base)
    freq = Counter(tokenize(train_text))
    items = build_items(held, freq, vocab, n_each=4, seed=0)
    # item construction validity: 4 options, answer in range
    for typ, its in items.items():
        for it in its:
            assert len(it["options"]) == 4, f"{typ}: not 4 options"
            assert 0 <= it["answer"] < 4, f"{typ}: answer out of range"
            assert it["options"][it["answer"]] is not None

    # model forward shape
    mlen = 64
    model = make_model(len(vocab), mlen, d=32, layers=2, heads=2)
    t = torch.tensor([vocab.encode("the quick scientist", bos=True)], dtype=torch.long)
    logits = model(t, causal=True)
    assert logits.shape == (1, t.shape[1], len(vocab)), f"bad shape {logits.shape}"

    # a few train steps reduce loss
    before = _quick_loss(model, vocab, train_text, mlen)
    train_lm(model, vocab, train_text, steps=60, seq_len=32, batch=8,
             seed=0, device=device)
    after = _quick_loss(model, vocab, train_text, mlen)
    assert after < before, f"loss did not drop: {before:.3f} -> {after:.3f}"

    # save/load identical prediction
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "m.pt")
        torch.save(dict(config=model.config, itos=vocab.itos,
                        state_dict={k: v.cpu() for k, v in model.state_dict().items()}), p)
        blob = torch.load(p, map_location=device, weights_only=False)
        v2 = WordVocab(blob["itos"])
        m2 = ScratchpadLM(**blob["config"])
        m2.load_state_dict(blob["state_dict"])
        m2.to(device).eval()
        anyit = items["lexical_cloze"][0] if items["lexical_cloze"] else \
            items["next_sentence"][0]
        a1 = answer_lm(model, vocab, anyit, device, mlen)
        a2 = answer_lm(m2, v2, anyit, device, mlen)
        assert a1 == a2, "save/load prediction mismatch"

    # contamination check is sane on synthetic data
    _ = verify_disjoint(train_text, items)

    print("c2_eval selftest OK")


@torch.no_grad()
def _quick_loss(model, vocab, text, seq_len):
    model.eval()
    stream = torch.tensor(vocab.encode(text), dtype=torch.long)
    n = min(seq_len, stream.numel() - 1)
    x = stream[:n].unsqueeze(0)
    y = stream[1:n + 1].unsqueeze(0)
    logits = model(x, causal=True)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                           y.reshape(-1), ignore_index=vocab.pad).item()


# --------------------------------------------------------------------------- #
# real run
# --------------------------------------------------------------------------- #
def run(steps, seed, out_path, paths, n_each=60, d=256, layers=4, heads=8,
        seq_len=128, vocab_cap=8000, max_len=160, device="cpu", verbose=True,
        batch=16, lr=3e-3, extra_train_paths=None, log_every=None):
    # EVAL source (the C2 held-out items) comes ONLY from `paths` (cosmopedia): split off a
    # disjoint held-out tail, build items from it. EXTRA corpora (extra_train_paths) are appended
    # to TRAIN text only -- more language signal for the from-scratch LM, never to held-out.
    # verify_disjoint still guards that no eval sentence leaked into the (now larger) train text.
    text = load_source(paths)
    train_text, held = contamination_split(text, held_frac=0.15, seed=seed)
    extra_chars = 0
    if extra_train_paths:
        extra = load_source(list(extra_train_paths))
        extra_chars = len(extra)
        train_text = train_text + "\n" + extra
    vocab = WordVocab.build(train_text, cap=vocab_cap)
    freq = Counter(tokenize(train_text))

    # DEDUP held-out vs TRAIN: the corpus repeats sentences verbatim (boilerplate), and extra
    # corpora can overlap, so some "held" sentences actually occur in TRAIN. Those are not valid
    # held-out items -- drop them BEFORE building items so the eval pool is clean by construction.
    def _norm(s):
        return re.sub(r"\s+", " ", s).strip()
    train_sent_set = set(_norm(s) for s in split_sentences(train_text))
    n_held_before = len(held)
    held = [s for s in held if _norm(s) not in train_sent_set]
    deduped = n_held_before - len(held)

    items = build_items(held, freq, vocab, n_each=n_each, seed=seed)

    # DEFENSIVE: also drop any individual item whose source sentence still appears in TRAIN (catches
    # substring/overlap cases the sentence-set dedup misses) so the run never aborts on contamination.
    def _item_leaks(it, typ):
        ans = it["options"][it["answer"]]
        full = (" ".join(x for x in [it["left"], ans, it["right"]] if x)
                if typ == "lexical_cloze" else ans)
        probe = _norm(full)
        return len(probe) >= 25 and probe in train_text
    dropped_items = 0
    for typ in list(items):
        kept = [it for it in items[typ] if not _item_leaks(it, typ)]
        dropped_items += len(items[typ]) - len(kept)
        items[typ] = kept

    leaks = verify_disjoint(train_text, items)
    assert leaks == 0, f"CONTAMINATION: {leaks} eval sentences found in TRAIN text"

    if verbose:
        ntr = len(vocab.encode(train_text))
        print(f"source chars: {len(text):,}  train chars: {len(train_text):,} "
              f"(+{extra_chars:,} extra)  held-out sentences: {len(held):,}")
        print(f"train tokens: {ntr:,}  vocab: {len(vocab):,}  device: {device}  "
              f"model d{d}/L{layers}/h{heads} batch{batch} seq{seq_len}")
        print(f"items: " + ", ".join(f"{k}={len(v)}" for k, v in items.items()))
        print(f"dedup: dropped {deduped} held sentences + {dropped_items} items that occur in TRAIN")
        print(f"contamination check: {leaks} eval sentences in TRAIN (must be 0)")

    model = make_model(len(vocab), max_len, d=d, layers=layers, heads=heads)
    train_lm(model, vocab, train_text, steps=steps, seq_len=seq_len, batch=batch, lr=lr,
             seed=seed, device=device,
             log_every=(log_every if log_every is not None
                        else (max(1, steps // 8) if verbose else 0)))

    uni = score(items, lambda it: answer_unigram(it, freq))
    lm = score(items, lambda it: answer_lm(model, vocab, it, device, max_len))

    if verbose:
        print("\n== C2-style held-out reading eval (chance = 0.25) ==")
        print(f"{'item type':>18} | {'n':>4} | {'unigram':>8} | {'trained LM':>10} | chance")
        print("-" * 62)
        for typ in items:
            print(f"{typ:>18} | {uni[typ]['n_items']:>4} | "
                  f"{uni[typ]['acc']:>8.3f} | {lm[typ]['acc']:>10.3f} | "
                  f"{uni[typ]['chance']:.2f}")

    result = dict(
        source_paths=paths, extra_train_paths=list(extra_train_paths or []),
        steps=steps, seed=seed, device=device, batch=batch, seq_len=seq_len,
        model=dict(d=d, layers=layers, heads=heads, vocab=len(vocab)),
        contamination=dict(train_chars=len(train_text), extra_chars=extra_chars,
                           held_sentences=len(held),
                           leaks_into_train=leaks,
                           disjoint=(leaks == 0)),
        chance=0.25,
        unigram=uni, trained_lm=lm,
    )
    if out_path:
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        if verbose:
            print(f"\nwrote {out_path}")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-each", type=int, default=60)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--vocab-cap", type=int, default=8000)
    ap.add_argument("--alice", action="store_true",
                    help="also include data/books/alice.txt as source")
    ap.add_argument("--out", default=None, help="write result JSON here")
    ap.add_argument("--device", default="cpu", help="cpu | cuda (GPU scaling probe)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=160)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--log-every", type=int, default=None)
    ap.add_argument("--extra-data", action="store_true",
                    help="append cosmopedia_4mb + tinystories as EXTRA TRAIN text "
                         "(more language signal; eval items stay cosmopedia held-out only)")
    args = ap.parse_args(argv)

    if args.selftest:
        selftest()
        return 0

    paths = ["data/cosmopedia_6mb.txt"]
    if args.alice:
        paths.append("data/books/alice.txt")
    extra = None
    if args.extra_data:
        extra = ["data/cosmopedia_4mb.txt", "data/tinystories_8mb.txt",
                 "data/tinystories_4mb.txt"]
    run(args.steps, args.seed, args.out, paths, n_each=args.n_each,
        d=args.d, layers=args.layers, heads=args.heads,
        vocab_cap=args.vocab_cap, device=args.device, batch=args.batch,
        seq_len=args.seq_len, max_len=args.max_len, lr=args.lr,
        log_every=args.log_every, extra_train_paths=extra)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
