"""Mind-0: ONE model that READS, WRITES, and forms CONCEPTS of English, learned from a book.

A single model (Mind) holds the skills in shared weights:
  WRITE    causal next-token LM over the book's windows                -> generates English.
  READ     bidirectional masked-LM (predict masked tokens from both sides) -> comprehension.
  CONCEPT  a LatentConceptHead reads the LM's hidden states; trained with VICReg over two masked
           views (same window, different corruptions -> same concepts) + slot factorization
           -> disentangled, reusable concepts / word-meaning associations (the reasoning pillar).

UL2-style: the same ScratchpadLM toggles attention per forward (causal write / bidirectional
read), and a concept head sits on top of its representations. Reuses thinking/concepts.py
(LatentConceptHead, VICReg, factorization, FER scores) — the project's concept machinery.

NOTHING about English is hardcoded: vocabulary, function-word detection, cloze blanks/distractors,
and prompts all come from the BOOK's own statistics. Only constants are control tokens + knobs.

  python -m thinking.mind --selftest
  python -m thinking.mind --read data/books/alice.txt --steps 12000 --checkpoint /tmp/mind.pt
  python -m thinking.mind --read data/books/alice.txt --eval --checkpoint /tmp/mind.pt
"""
import argparse
import json
import os
import re
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scratchpad_model import ScratchpadLM
from device import get_device
from .concepts import (LatentConceptHead, latent_concept_vicreg_loss,
                       latent_concept_slot_factorization_loss, latent_concept_fer_loss,
                       latent_concept_fer_scores)

DEV = get_device()

PAD, MASK, BOS, EOS = "<pad>", "<mask>", "<bos>", "<eos>"
SPECIALS = (PAD, MASK, BOS, EOS)
_TOK = re.compile(r"[a-z]+|[0-9]+|[^\sa-z0-9]")   # structural tokenizer: words / numbers / punct
MIND_SELF_TEACH_SIGNALS = ("write", "read", "concept")
MIND_SELF_TEACH_SCORE_KEYS = {
    "write": "write_score",
    "read": "read_score",
    "concept": "concept_score",
}
MIND_SELF_TEACH_SIGNAL_OBJECTIVES = {
    "write": ("write_prob",),
    "read": ("read_prob",),
    "concept": ("concept_prob", "concept_w", "fer_w"),
}
MIND_SELF_TEACH_WEIGHT_KEYS = tuple(dict.fromkeys(
    key
    for keys in MIND_SELF_TEACH_SIGNAL_OBJECTIVES.values()
    for key in keys))


def tokenize(text):
    return _TOK.findall(text.lower())


class Vocab:
    def __init__(self, itos):
        self.itos = list(itos)
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        for name, attr in zip(SPECIALS, ("pad", "mask", "bos", "eos")):
            setattr(self, attr, self.stoi[name])

    @classmethod
    def from_tokens(cls, tokens):
        return cls(list(SPECIALS) + sorted(set(tokens)))

    def __len__(self):
        return len(self.itos)

    def enc(self, words):
        return [self.stoi[w] for w in words if w in self.stoi]


class Mind(nn.Module):
    """ScratchpadLM (write+read) + LatentConceptHead (concepts) in one model."""

    def __init__(self, vocab_size, d=256, layers=4, heads=8, max_len=48, pad=0,
                 concept_slots=6, concept_heads=4):
        super().__init__()
        if (d // heads) % 2 != 0:
            raise ValueError(f"rope needs even head dim: d//heads={d // heads}")
        self.lm = ScratchpadLM(vocab_size, d=d, layers=layers, heads=heads, max_len=max_len,
                               pad=pad, arch="standard", pos_mode="rope", tie=True,
                               pointer=False, loop=False, mem=False, causal=True)
        self.concept = LatentConceptHead(concept_slots, d, heads=concept_heads, mixer_layers=1)
        self.pad = pad
        self.max_len = max_len
        self.cfg = dict(vocab_size=vocab_size, d=d, layers=layers, heads=heads,
                        max_len=max_len, pad=pad, concept_slots=concept_slots,
                        concept_heads=concept_heads)

    def forward(self, ids, causal=None):
        return self.lm(ids, causal=causal)

    def concept_slots(self, ids, project=False):
        self.lm(ids, causal=False)                       # populate hidden states (bidirectional)
        return self.concept(self.lm._last_hidden, mask=ids.eq(self.pad), project=project)


def make_mind(vocab_size, d=256, layers=4, heads=8, max_len=48, pad=0, concept_slots=6,
              concept_heads=4, device=DEV):
    return Mind(vocab_size, d=d, layers=layers, heads=heads, max_len=max_len, pad=pad,
                concept_slots=concept_slots, concept_heads=concept_heads).to(device)


# --------------------------------------------------- data-driven book preparation

def read_windows(path, window=20, stride=14):
    toks = tokenize(open(path, encoding="utf-8").read())
    return toks, [toks[i:i + window] for i in range(0, max(1, len(toks) - window + 1), stride)]


def frequency_profile(tokens, function_frac=0.03):
    freq = Counter(tokens)
    ranked = [w for w, _ in freq.most_common()]
    n_func = max(5, int(round(function_frac * len(ranked))))
    return freq, set(ranked[:n_func])


def _maskable(window, function_set):
    return [i for i, w in enumerate(window) if w not in function_set]


def _band_distractors(gold, freq, function_set, rng, k, band=0.5):
    fg = freq[gold]
    lo, hi = fg * band, (fg / band if band else fg)
    pool = [w for w, f in freq.items()
            if w not in function_set and w != gold and lo <= f <= hi]
    if len(pool) < k:
        pool = [w for w in freq if w not in function_set and w != gold]
    rng.shuffle(pool)
    return pool[:k]


def make_exam(windows, freq, function_set, rng, n=40, n_options=3):
    items = []
    for wi in rng.permutation(len(windows)):
        if len(items) >= n:
            break
        w = windows[int(wi)]
        cand = _maskable(w, function_set)
        if not cand:
            continue
        p = int(rng.choice(cand))
        gold = w[p]
        distract = _band_distractors(gold, freq, function_set, rng, n_options - 1)
        if len(distract) < n_options - 1:
            continue
        opts = [gold] + distract
        order = rng.permutation(n_options)
        items.append({"left": w[:p], "right": w[p + 1:],
                      "options": [opts[int(j)] for j in order],
                      "answer": int(np.where(order == 0)[0][0]), "gold": gold,
                      "text": " ".join(w[:p] + ["___"] + w[p + 1:])})
    return items


def prepare(path, seed=0, window=20, stride=14, n_exam=40, hold_frac=0.1, function_frac=0.03):
    toks, wins = read_windows(path, window=window, stride=stride)
    freq, function_set = frequency_profile(toks, function_frac=function_frac)
    rng = np.random.default_rng(seed)
    cut = int(len(wins) * (1 - hold_frac))
    train_w, held_w = wins[:cut], wins[cut:]
    exam = make_exam(held_w, freq, function_set, rng, n=n_exam)
    return train_w, exam, Vocab.from_tokens(toks), freq, function_set


# ----------------------------------------------- train (WRITE + READ + CONCEPT, shared weights)

def _pad_batch(seqs, pad, device):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad, dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)
    return ids


def _masked_view(seqs, vocab, rng, mask_p, device, want_labels):
    ids = _pad_batch(seqs, vocab.pad, device)
    labels = torch.full_like(ids, -100) if want_labels else None
    for i, e in enumerate(seqs):
        k = max(1, int(round(mask_p * len(e))))
        for p in rng.choice(len(e), size=k, replace=False):
            if want_labels:
                labels[i, p] = ids[i, p]
            ids[i, p] = vocab.mask
    return (ids, labels) if want_labels else ids


def _loss_quality(loss):
    loss = float(loss)
    if not np.isfinite(loss):
        return 0.0
    return float(1.0 / (1.0 + max(0.0, loss)))


def _loss_report(losses):
    return {
        key: (float(value) if np.isfinite(float(value)) else None)
        for key, value in losses.items()
    }


def _sample_encoded(enc, rng, n):
    if not enc:
        return []
    n = min(int(n), len(enc))
    if n <= 0:
        return []
    return [enc[int(i)] for i in rng.permutation(len(enc))[:n]]


@torch.no_grad()
def _encoded_write_eval(model, vocab, enc, rng, device=DEV, n=128):
    model.eval()
    pick = _sample_encoded(enc, rng, n)
    if not pick:
        return {"skipped": True, "loss": 0.0, "score": 0.0,
                "token_accuracy": 0.0, "n": 0}
    ids = _pad_batch([[vocab.bos] + e + [vocab.eos] for e in pick],
                     vocab.pad, device)
    logits = model(ids)
    targets = ids[:, 1:]
    pred_logits = logits[:, :-1]
    loss = F.cross_entropy(pred_logits.reshape(-1, pred_logits.size(-1)),
                           targets.reshape(-1), ignore_index=vocab.pad)
    mask = targets.ne(vocab.pad)
    correct = pred_logits.argmax(-1).eq(targets).masked_select(mask)
    acc = float(correct.float().mean().item()) if correct.numel() else 0.0
    return {"skipped": False, "loss": float(loss.item()),
            "score": _loss_quality(loss.item()), "token_accuracy": acc,
            "n": len(pick)}


@torch.no_grad()
def _encoded_read_eval(model, vocab, enc, rng, mask_p, device=DEV, n=128):
    model.eval()
    pick = _sample_encoded(enc, rng, n)
    if not pick:
        return {"skipped": True, "loss": 0.0, "score": 0.0,
                "mask_accuracy": 0.0, "n": 0}
    ids, labels = _masked_view(pick, vocab, rng, mask_p, device, want_labels=True)
    logits = model(ids, causal=False)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                           labels.reshape(-1), ignore_index=-100)
    mask = labels.ne(-100)
    correct = logits.argmax(-1).eq(labels).masked_select(mask)
    acc = float(correct.float().mean().item()) if correct.numel() else 0.0
    return {"skipped": False, "loss": float(loss.item()),
            "score": acc, "mask_accuracy": acc, "n": len(pick)}


@torch.no_grad()
def _encoded_concept_eval(model, vocab, enc, rng, device=DEV, n=128):
    model.eval()
    pick = _sample_encoded(enc, rng, n)
    if not pick:
        return {"skipped": True, "fer_score": 0.0, "score": 0.0,
                "fragmentation": 0.0, "slot_correlation": 0.0,
                "slot_imbalance": 0.0, "n": 0}
    ids = _pad_batch(pick, vocab.pad, device)
    slots = model.concept_slots(ids, project=False)
    fer_score, comps = latent_concept_fer_scores(slots)
    fer_mean = float(fer_score.mean().item()) if fer_score.numel() else 0.0
    return {"skipped": False, "fer_score": fer_mean,
            "score": _loss_quality(fer_mean),
            "fragmentation": float(comps["fragmentation"].mean().item()),
            "slot_correlation": float(comps["slot_correlation"].mean().item()),
            "slot_imbalance": float(comps["slot_imbalance"].mean().item()),
            "n": len(pick)}


def mind_score_components(write_eval, read_eval, concept_eval):
    rows = {
        "write": write_eval,
        "read": read_eval,
        "concept": concept_eval,
    }
    scores = {}
    active = []
    for signal, row in rows.items():
        skipped = bool(row.get("skipped", False))
        scores[f"{signal}_skipped"] = skipped
        score = float(row.get("score", 0.0))
        scores[MIND_SELF_TEACH_SCORE_KEYS[signal]] = min(1.0, max(0.0, score))
        if not skipped:
            active.append(scores[MIND_SELF_TEACH_SCORE_KEYS[signal]])
    scores["score"] = float(np.mean(active)) if active else 0.0
    return scores


def mind_eval_bundle_from_encoded(model, vocab, enc, rng, mask_p, device=DEV, n=128):
    write_eval = _encoded_write_eval(model, vocab, enc, rng, device=device, n=n)
    read_eval = _encoded_read_eval(model, vocab, enc, rng, mask_p, device=device, n=n)
    concept_eval = _encoded_concept_eval(model, vocab, enc, rng, device=device, n=n)
    return {
        "write": write_eval,
        "read": read_eval,
        "concept": concept_eval,
        "score_components": mind_score_components(
            write_eval, read_eval, concept_eval),
    }


def mind_eval_bundle(model, vocab, windows, rng, mask_p=0.15, device=DEV, n=128):
    enc = [vocab.enc(w)[:model.max_len - 2] for w in windows]
    enc = [e for e in enc if len(e) >= 2]
    return mind_eval_bundle_from_encoded(
        model, vocab, enc, rng, mask_p, device=device, n=n)


def mind_task_probabilities(weights):
    raw = {
        "write": max(0.0, float(weights.get("write_prob", 0.0))),
        "read": max(0.0, float(weights.get("read_prob", 0.0))),
        "concept": max(0.0, float(weights.get("concept_prob", 0.0))),
    }
    total = sum(raw.values())
    if total <= 0.0:
        raise ValueError("mind task probabilities must have positive total mass")
    return {key: float(value / total) for key, value in raw.items()}


def mind_self_teach_weight_plan(score_components, budget=0.0):
    budget = float(budget)
    if budget < 0.0:
        raise ValueError("mind self-teach budget must be non-negative")
    extras = {key: 0.0 for key in MIND_SELF_TEACH_WEIGHT_KEYS}
    deficits = {}
    active = []
    for signal in MIND_SELF_TEACH_SIGNALS:
        if bool(score_components.get(f"{signal}_skipped", False)):
            continue
        score = float(score_components.get(
            MIND_SELF_TEACH_SCORE_KEYS[signal], 0.0))
        quality = min(1.0, max(0.0, score))
        deficit = max(0.0, 1.0 - quality)
        deficits[signal] = float(deficit)
        if deficit > 0.0:
            active.append((signal, deficit))
    total_deficit = sum(deficit for _signal, deficit in active)
    if budget > 0.0 and total_deficit > 0.0:
        for signal, deficit in active:
            signal_budget = budget * deficit / total_deficit
            objectives = MIND_SELF_TEACH_SIGNAL_OBJECTIVES[signal]
            objective_share = signal_budget / float(len(objectives))
            for key in objectives:
                extras[key] += objective_share
    top_signal = None
    if active:
        top_signal = max(active, key=lambda item: item[1])[0]
    return {
        "enabled": bool(budget > 0.0),
        "budget": float(budget),
        "total_deficit": float(total_deficit),
        "top_signal": top_signal,
        "signal_deficits": deficits,
        "active_signals": [signal for signal, _deficit in active],
        "weight_extras": {key: float(value) for key, value in extras.items()},
    }


def mind_self_teach_weight_maps(score_components=None, budget=0.0,
                                **base_weights):
    budget = float(budget)
    if budget < 0.0:
        raise ValueError("mind self-teach budget must be non-negative")
    missing = [
        key for key in MIND_SELF_TEACH_WEIGHT_KEYS
        if key not in base_weights]
    if missing:
        raise ValueError(
            "missing mind self-teach base weights: "
            + ", ".join(sorted(missing)))
    base = {
        key: float(base_weights[key])
        for key in MIND_SELF_TEACH_WEIGHT_KEYS}
    effective = dict(base)
    if budget == 0.0:
        return None, base, effective
    if score_components is None:
        raise ValueError(
            "mind self-teach needs score components when budget is positive")
    plan = mind_self_teach_weight_plan(score_components, budget=budget)
    extras = plan["weight_extras"]
    effective = {
        key: float(base[key]) + float(extras.get(key, 0.0))
        for key in MIND_SELF_TEACH_WEIGHT_KEYS}
    return plan, base, effective


def train(model, vocab, windows, steps=12000, batch=32, lr=1e-3, read_frac=0.4,
          concept_frac=0.2, concept_w=0.05, fer_w=1.0, mask_p=0.15, seed=0, device=DEV,
          log_every=0, self_teach_w=0.0, self_teach_eval_every=0,
          self_teach_eval_n=128):
    steps = int(steps)
    batch = int(batch)
    if steps <= 0:
        raise ValueError("mind training steps must be positive")
    if batch <= 0:
        raise ValueError("mind training batch must be positive")
    read_frac = float(read_frac)
    concept_frac = float(concept_frac)
    concept_w = float(concept_w)
    fer_w = float(fer_w)
    mask_p = float(mask_p)
    self_teach_w = float(self_teach_w)
    self_teach_eval_every = int(self_teach_eval_every)
    self_teach_eval_n = int(self_teach_eval_n)
    if read_frac < 0.0 or concept_frac < 0.0 or read_frac + concept_frac > 1.0:
        raise ValueError("mind task fractions must be non-negative and sum to <= 1")
    if concept_w < 0.0 or fer_w < 0.0:
        raise ValueError("mind concept weights must be non-negative")
    if mask_p <= 0.0 or mask_p >= 1.0:
        raise ValueError("mind mask probability must be in (0, 1)")
    if self_teach_w < 0.0:
        raise ValueError("mind self-teach weight must be non-negative")
    if self_teach_eval_every < 0:
        raise ValueError("mind self-teach eval interval must be non-negative")
    if self_teach_w > 0.0 and self_teach_eval_n <= 0:
        raise ValueError("mind self-teach eval count must be positive")
    max_len = model.max_len
    enc = [vocab.enc(w)[:max_len - 2] for w in windows]
    enc = [e for e in enc if len(e) >= 2]
    if not enc:
        raise ValueError("mind training needs at least one encoded window")
    rng = np.random.default_rng(seed)
    eval_rng = np.random.default_rng(seed + 1009)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    self_teach_base_weights = {
        "write_prob": float(1.0 - read_frac - concept_frac),
        "read_prob": float(read_frac),
        "concept_prob": float(concept_frac),
        "concept_w": float(concept_w),
        "fer_w": float(fer_w),
    }
    self_teach_effective_weights = dict(self_teach_base_weights)
    task_probs = mind_task_probabilities(self_teach_effective_weights)
    effective_concept_w = float(concept_w)
    effective_fer_w = float(fer_w)
    self_teach_plan = None
    self_teach_reports = []

    def update_self_teach_weights(step):
        nonlocal self_teach_effective_weights, task_probs
        nonlocal effective_concept_w, effective_fer_w, self_teach_plan
        was_training = bool(model.training)
        bundle = mind_eval_bundle_from_encoded(
            model, vocab, enc, eval_rng, mask_p, device=device,
            n=self_teach_eval_n)
        if was_training:
            model.train()
        plan, _base, effective = mind_self_teach_weight_maps(
            bundle["score_components"], budget=self_teach_w,
            **self_teach_base_weights)
        self_teach_effective_weights = effective
        task_probs = mind_task_probabilities(effective)
        effective_concept_w = float(effective["concept_w"])
        effective_fer_w = float(effective["fer_w"])
        self_teach_plan = plan | {
            "step": int(step),
            "score_components": bundle["score_components"],
            "task_probs": dict(task_probs),
            "effective_weights": dict(effective),
        }
        self_teach_reports.append(self_teach_plan)
        return self_teach_plan

    if self_teach_w > 0.0:
        update_self_teach_weights(0)
    loss_last = {"write": float("nan"), "read": float("nan"), "concept": float("nan")}
    loss_first = dict(loss_last)
    for st in range(steps):
        pick = [enc[int(i)] for i in rng.integers(0, len(enc), size=batch)]
        u = rng.random()
        if u < task_probs["read"]:                              # READ: bidirectional MLM
            ids, labels = _masked_view(pick, vocab, rng, mask_p, device, want_labels=True)
            logits = model(ids, causal=False)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   labels.reshape(-1), ignore_index=-100)
            key = "read"
        elif u < task_probs["read"] + task_probs["concept"]:    # CONCEPT: VICReg over two views
            a = _masked_view(pick, vocab, rng, mask_p, device, want_labels=False)
            b = _masked_view(pick, vocab, rng, mask_p, device, want_labels=False)
            sa = model.concept_slots(a, project=True)
            sb = model.concept_slots(b, project=True)
            # down-weighted: VICReg starts ~50x the LM losses; left unscaled it dominates the
            # shared trunk and wrecks read/write (verified: read 0.475 -> 0.40). concept_w keeps
            # concept gradients comparable to the LM objectives so concepts shape, not disrupt.
            # VICReg = invariance across views (concepts) ; factorization + FER = sharpen the
            # slots (reduce fragmentation/entanglement so concepts specialize, not stay uniform).
            loss = effective_concept_w * (
                latent_concept_vicreg_loss(sa, sb)
                + 0.1 * latent_concept_slot_factorization_loss(sa)
                + effective_fer_w * latent_concept_fer_loss(sa))
            key = "concept"
        else:                                                  # WRITE: causal next-token LM
            ids = _pad_batch([[vocab.bos] + e + [vocab.eos] for e in pick], vocab.pad, device)
            logits = model(ids)
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                                   ids[:, 1:].reshape(-1), ignore_index=vocab.pad)
            key = "write"
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_last[key] = float(loss.item())
        if loss_first[key] != loss_first[key]:
            loss_first[key] = loss_last[key]
        if (self_teach_w > 0.0 and self_teach_eval_every > 0
                and (st + 1) < steps
                and (st + 1) % self_teach_eval_every == 0):
            update_self_teach_weights(st + 1)
        if log_every and (st + 1) % log_every == 0:
            print(f"  mind {st + 1}/{steps} write {loss_last['write']:.3f} "
                  f"read {loss_last['read']:.3f} concept {loss_last['concept']:.3f} "
                  f"p(w/r/c) {task_probs['write']:.2f}/"
                  f"{task_probs['read']:.2f}/{task_probs['concept']:.2f}",
                  flush=True)
    final = _loss_report(loss_last) | {
        "task_probs": dict(task_probs),
        "self_teach_w": float(self_teach_w),
        "self_teach_plan": self_teach_plan,
        "self_teach_base_weights": dict(self_teach_base_weights),
        "self_teach_effective_weights": dict(self_teach_effective_weights),
        "self_teach_reports": list(self_teach_reports),
        "concept_w": float(effective_concept_w),
        "fer_w": float(effective_fer_w),
    }
    return {"first": _loss_report(loss_first), "final": final, "steps": int(steps),
            "self_teach_w": float(self_teach_w),
            "self_teach_plan": self_teach_plan,
            "self_teach_base_weights": dict(self_teach_base_weights),
            "self_teach_effective_weights": dict(self_teach_effective_weights),
            "self_teach_reports": list(self_teach_reports)}


# ------------------------------------------------------------ WRITE / READ / CONCEPT eval

@torch.no_grad()
def write(model, vocab, prompt_ids, max_new=60, temperature=0.8, top_k=40, rep_penalty=1.15,
          device=DEV):
    model.eval()
    max_len = model.max_len
    ids = [vocab.bos] + list(prompt_ids)
    start = len(ids)
    for _ in range(max_new):
        x = torch.tensor([ids[-max_len:]], dtype=torch.long, device=device)
        logits = model(x)[0, -1].float()
        for t in set(ids):
            logits[t] /= rep_penalty
        for sp in (vocab.pad, vocab.mask):
            logits[sp] = float("-inf")
        if temperature > 0:
            logits = logits / temperature
            if top_k:
                thr = torch.topk(logits, min(top_k, logits.numel())).values[-1]
                logits[logits < thr] = float("-inf")
            nxt = int(torch.multinomial(torch.softmax(logits, -1), 1))
        else:
            nxt = int(logits.argmax())
        if nxt == vocab.eos:
            break
        ids.append(nxt)
    return " ".join(vocab.itos[i] for i in ids[start:] if i != vocab.eos)


@torch.no_grad()
def answer_cloze(model, vocab, item, device=DEV):
    model.eval()
    left = vocab.enc(item["left"])
    seq = left + [vocab.mask] + vocab.enc(item["right"])
    x = torch.tensor([seq], dtype=torch.long, device=device)
    logp = F.log_softmax(model(x, causal=False)[0, len(left)].float(), dim=-1)
    scores = [logp[vocab.stoi[o]].item() if o in vocab.stoi else float("-inf")
              for o in item["options"]]
    return int(np.argmax(scores)), scores


def comprehension_accuracy(model, vocab, items, device=DEV):
    res = [int(answer_cloze(model, vocab, it, device=device)[0] == it["answer"]) for it in items]
    n = max(1, len(items))
    chance = float(np.mean([1.0 / len(it["options"]) for it in items])) if items else 0.0
    return {"accuracy": sum(res) / n, "chance": chance, "correct": sum(res), "n": len(items),
            "passed": bool(sum(res) / n >= 0.80)}


@torch.no_grad()
def concept_quality(model, vocab, windows, rng, device=DEV, n=128):
    """FER concept-quality on held-out windows: lower fragmentation/correlation = cleaner concepts."""
    model.eval()
    enc = [vocab.enc(w) for w in windows]
    enc = [e for e in enc if len(e) >= 2]
    pick = [enc[int(i)] for i in rng.permutation(len(enc))[:min(n, len(enc))]]
    ids = _pad_batch(pick, vocab.pad, device)
    slots = model.concept_slots(ids, project=False)
    fer_score, comps = latent_concept_fer_scores(slots)
    fer_mean = float(fer_score.mean()) if fer_score.numel() else 0.0
    return {"score": _loss_quality(fer_mean), "fer_score": fer_mean,
            "fragmentation": float(comps["fragmentation"].mean()),
            "slot_correlation": float(comps["slot_correlation"].mean()),
            "n": len(pick), "slots": slots.shape[1]}


def demo_prompts(windows, vocab, rng, k=3, plen=2):
    out = []
    for wi in rng.permutation(len(windows))[:k]:
        w = windows[int(wi)][:plen]
        out.append((" ".join(w), vocab.enc(w)))
    return out


# --------------------------------------------------------------------- save/load

def _torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def save_model(path, model, vocab, report=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": model.cfg,
                "itos": vocab.itos, "report": report or {}}, path)


def load_model(path, device=DEV):
    ck = _torch_load(path, device)
    vocab = Vocab(ck["itos"])
    model = Mind(**ck["config"]).to(device)
    model.load_state_dict(ck["state_dict"], strict=False)
    model.eval()
    return model, vocab, ck


# ----------------------------------------------------------------------- selftest

DEFAULT_BOOK = "data/books/alice.txt"


def selftest():
    device = "cpu"
    if not os.path.exists(DEFAULT_BOOK):
        print(f"mind selftest SKIP (no book at {DEFAULT_BOOK})")
        return
    toks = tokenize(open(DEFAULT_BOOK, encoding="utf-8").read())[:4000]
    open("/tmp/_mind_slice.txt", "w").write(" ".join(toks))
    train_w, exam, vocab, freq, fset = prepare("/tmp/_mind_slice.txt", n_exam=5, window=16,
                                               stride=10)
    assert train_w and exam and len(vocab) > len(SPECIALS)
    synthetic_scores = {
        "write_score": 0.1,
        "read_score": 0.7,
        "concept_score": 0.4,
    }
    self_teach_plan = mind_self_teach_weight_plan(
        synthetic_scores, budget=0.09)
    assert self_teach_plan["enabled"] is True
    assert self_teach_plan["top_signal"] == "write"
    assert self_teach_plan["weight_extras"]["write_prob"] > 0.0
    assert self_teach_plan["weight_extras"]["concept_prob"] > 0.0
    assert abs(sum(self_teach_plan["weight_extras"].values()) - 0.09) < 1e-6
    _plan, _base, effective = mind_self_teach_weight_maps(
        synthetic_scores, budget=0.09, write_prob=0.4, read_prob=0.4,
        concept_prob=0.2, concept_w=0.05, fer_w=1.0)
    probs = mind_task_probabilities(effective)
    assert abs(sum(probs.values()) - 1.0) < 1e-6
    model = make_mind(len(vocab), d=32, layers=1, heads=2, max_len=32, pad=vocab.pad,
                      concept_slots=4, concept_heads=2, device=device)
    rep = train(model, vocab, train_w, steps=300, batch=8, lr=1e-3, device=device)
    assert rep["final"]["write"] < rep["first"]["write"], rep
    assert rep["final"]["read"] < rep["first"]["read"], rep
    assert rep["final"]["concept"] < rep["first"]["concept"], rep
    w = write(model, vocab, vocab.enc(train_w[0][:2]), max_new=8, device=device)
    assert isinstance(w, str)
    pred, sc = answer_cloze(model, vocab, exam[0], device=device)
    assert 0 <= pred < len(exam[0]["options"]) and len(sc) == len(exam[0]["options"])
    cq = concept_quality(model, vocab, train_w, np.random.default_rng(0), device=device)
    assert cq["slots"] == 4
    assert 0.0 <= cq["score"] <= 1.0
    model_st = make_mind(len(vocab), d=32, layers=1, heads=2, max_len=32, pad=vocab.pad,
                         concept_slots=4, concept_heads=2, device=device)
    rep_st = train(
        model_st, vocab, train_w, steps=40, batch=8, lr=1e-3, device=device,
        read_frac=0.3, concept_frac=0.3, self_teach_w=0.05,
        self_teach_eval_every=10, self_teach_eval_n=12)
    assert rep_st["self_teach_w"] == 0.05
    assert rep_st["self_teach_plan"]["enabled"] is True
    assert rep_st["self_teach_reports"]
    assert rep_st["final"]["task_probs"]["write"] > 0.0
    assert rep_st["final"]["self_teach_effective_weights"]["concept_w"] >= 0.05
    path = "/tmp/_mind_selftest.pt"
    save_model(path, model, vocab, rep)
    m2, v2, _ = load_model(path, device=device)
    assert answer_cloze(m2, v2, exam[0], device=device)[0] == pred
    print("mind selftest OK")


# --------------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--read", default=DEFAULT_BOOK)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--n-exam", type=int, default=40)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--stride", type=int, default=14)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--read-frac", type=float, default=0.4)
    ap.add_argument("--concept-frac", type=float, default=0.2)
    ap.add_argument("--concept-w", type=float, default=0.05)
    ap.add_argument("--fer-w", type=float, default=1.0)
    ap.add_argument("--mask-p", type=float, default=0.15)
    ap.add_argument("--self-teach-w", type=float, default=0.0,
                    help=("budget for reallocating mind task/objective weight "
                          "from validation deficits"))
    ap.add_argument("--self-teach-eval-every", type=int, default=1000,
                    help="training steps between self-teach validation refreshes")
    ap.add_argument("--self-teach-eval-n", type=int, default=128,
                    help="encoded windows used for each self-teach validation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--concept-slots", type=int, default=6)
    ap.add_argument("--max-len", type=int, default=48)
    ap.add_argument("--log-every", type=int, default=3000)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    if args.selftest:
        selftest()
        return
    if args.steps <= 0 or args.batch <= 0:
        ap.error("--steps and --batch must be positive")
    if args.lr <= 0.0:
        ap.error("--lr must be positive")
    if args.read_frac < 0.0 or args.concept_frac < 0.0 or args.read_frac + args.concept_frac > 1.0:
        ap.error("--read-frac and --concept-frac must be non-negative and sum to <= 1")
    if args.concept_w < 0.0 or args.fer_w < 0.0:
        ap.error("--concept-w and --fer-w must be non-negative")
    if args.mask_p <= 0.0 or args.mask_p >= 1.0:
        ap.error("--mask-p must be in (0, 1)")
    if args.self_teach_w < 0.0:
        ap.error("--self-teach-w must be non-negative")
    if args.self_teach_eval_every < 0:
        ap.error("--self-teach-eval-every must be non-negative")
    if args.self_teach_w > 0.0 and args.self_teach_eval_n <= 0:
        ap.error("--self-teach-eval-n must be positive when self-teach is enabled")

    train_w, exam, vocab, freq, fset = prepare(
        args.read, seed=args.seed, window=args.window, stride=args.stride, n_exam=args.n_exam)
    rng = np.random.default_rng(args.seed)
    prompts = demo_prompts(train_w, vocab, rng)

    if args.eval:
        if not args.checkpoint or not os.path.exists(args.checkpoint):
            raise SystemExit("--eval requires an existing --checkpoint")
        model, vocab, _ = load_model(args.checkpoint, device=DEV)
        rep = {"read_comprehension": comprehension_accuracy(model, vocab, exam, device=DEV),
               "concept_quality": concept_quality(model, vocab, train_w, rng, device=DEV),
               "write_samples": {p: write(model, vocab, ids, device=DEV) for p, ids in prompts}}
        print(json.dumps(rep, indent=1), flush=True)
        return

    model = make_mind(len(vocab), d=args.d, layers=args.layers, heads=args.heads,
                      max_len=args.max_len, pad=vocab.pad, concept_slots=args.concept_slots,
                      device=DEV)
    tr = train(model, vocab, train_w, steps=args.steps, batch=args.batch, lr=args.lr,
               read_frac=args.read_frac, concept_frac=args.concept_frac,
               concept_w=args.concept_w, fer_w=args.fer_w, mask_p=args.mask_p,
               seed=args.seed, device=DEV, log_every=args.log_every,
               self_teach_w=args.self_teach_w,
               self_teach_eval_every=args.self_teach_eval_every,
               self_teach_eval_n=args.self_teach_eval_n)
    report = {"book": args.read, "train_windows": len(train_w), "vocab_size": len(vocab),
              "params": sum(p.numel() for p in model.parameters()), "train": tr,
              "read_comprehension": comprehension_accuracy(model, vocab, exam, device=DEV),
              "concept_quality": concept_quality(model, vocab, train_w, rng, device=DEV),
              "write_samples": {p: write(model, vocab, ids, device=DEV) for p, ids in prompts}}
    if args.checkpoint:
        save_model(args.checkpoint, model, vocab, report)
        report["checkpoint"] = args.checkpoint
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1), flush=True)


if __name__ == "__main__":
    main()
