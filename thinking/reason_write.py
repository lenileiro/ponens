#!/usr/bin/env python3
"""reason->write UNIFICATION: the verified reasoning core produces a CONTENT PLAN (an
engine-DERIVED, ordered set of true atoms about a subject), a small WRITER renders that
plan into a multi-sentence English paragraph, and the Datalog engine CHECKS faithfulness
(every generated sentence must parse back to an atom in the closure).

This is the bridge between two existing pieces in the repo:
  - reason_realtext.py : the REAL common-sense KB (ISA/PART_OF/HAS_PROP), its Datalog
    closure (the engine derives multi-hop facts like robin->animal, robin can move),
    its render templates and exact parse-back, and HELDOUT subjects (multi-hop derived
    facts never STATED in training). We import all of this read-only.
  - write_lm.py        : the char-level causal ScratchpadLM "free writer" -- locally
    fluent, globally INCOHERENT (loops "the story of the story of"). This is the
    baseline we must beat on coherence.

The thesis: CONTENT-PLANNED rendering is both COHERENT (distinct, well-formed, on-topic
sentences -- no loops) AND FAITHFUL (every sentence is engine-verified true). Free-writing
without a plan is neither. We measure this on HELD-OUT subjects (subjects whose derived
facts were never in the writer's training plans).

CPU only, small, deterministic. Does not modify any other file.

  python -m thinking.reason_write --selftest
  python -m thinking.reason_write --steps 3000 --out /tmp/reason_write.json
"""
import argparse
import json
import os
import random
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scratchpad_model import ScratchpadLM  # noqa: E402
from device import get_device  # noqa: E402
import thinking.reason_realtext as RT  # noqa: E402  (READ-ONLY reuse of the KB + helpers)


# ===================================================================================================
# 1. CONTENT PLAN: for a subject entity, collect the engine-DERIVED set of true atoms about it from
#    the closure (its isa chain + inherited properties + parts). These are facts the engine COMPOSED,
#    most of which were never directly stated as base facts -- they come from RT.CLOSURE, not RT.EDB.
# ===================================================================================================
def content_plan(subject):
    """Ordered list of engine-true atoms about `subject`, drawn from the deductive closure.

    Order = a sensible exposition order: identity (isa chain, specific->general), then the
    properties it inherits, then the parts of the most specific category it belongs to.
    Every atom is verified `in RT.CLOSURE` (engine-derived truth)."""
    plan = []
    seen = set()

    def add(atom):
        if atom in RT.CLOSURE and atom not in seen:
            seen.add(atom)
            plan.append(atom)

    # (a) isa chain: subject -> ... -> living_thing  (specific to general)
    isa_chain = [subject] + RT._ancestors(RT.EDB, "isa", subject)
    for anc in RT._ancestors(RT.EDB, "isa", subject):
        add(("isa", (subject, anc)))
    # (b) inherited properties: every prop the subject has via the closure (incl. inherited)
    all_props = sorted({p for (_a, p) in RT.HAS_PROP})
    for prop in all_props:
        add(("has_prop", (subject, prop)))
    # (c) parts: parts of any category in the subject's isa chain (e.g. a robin has a wing)
    #     part_of(part, category): emit when `category` is something the subject is-a.
    for cat in isa_chain:
        for (p, (x, y)) in sorted(RT.EDB):
            if p == "part_of" and y == cat:
                add(("part_of", (x, cat)))
    return plan


def planned_subjects():
    """Subjects that have a non-trivial content plan (>= 2 derived atoms). These are the
    entities we can write paragraphs about."""
    subs = []
    for s in sorted(RT.ISA_NODES):
        if len(content_plan(s)) >= 2:
            subs.append(s)
    return subs


# HELD-OUT split is by FACT, not by whole subject -- mirroring reason_realtext's HELDOUT design
# (reserved (subject, derived-atom) PAIRS). EVERY subject is seen during training (so the writer
# knows the subject token and its base identity), but the specific multi-hop DERIVED atoms in
# RT.HELDOUT are STRIPPED from that subject's training paragraph. At eval we request the FULL plan
# (including the withheld derived atoms), so the writer must render derived sentences it was never
# trained to produce for that subject -- the true reason->write composition test.
def is_heldout_atom(atom):
    pred, args_ = atom
    return (pred, args_) in RT.HELDOUT


def train_plan(subject):
    """Plan used for TRAINING this subject: full content plan MINUS held-out derived atoms."""
    return [a for a in content_plan(subject) if not is_heldout_atom(a)]


def eval_plan(subject):
    """Plan used for EVAL: the full content plan (includes held-out derived atoms)."""
    return content_plan(subject)


def subjects_with_heldout():
    """Subjects that have at least one held-out derived atom -- the meaningful eval set."""
    return [s for s in planned_subjects() if any(is_heldout_atom(a) for a in content_plan(s))]


# kept for API/back-compat in CLI reporting
def heldout_subjects():
    return subjects_with_heldout()


# Subjects ENTIRELY withheld from training (whole-subject holdout = the hard novel-entity test).
# Empty by default => fact-level holdout (every subject seen; only its derived atoms withheld).
HOLDOUT_SUBJECTS = set()


def train_subjects():
    """All planned subjects whose TRAIN plan is still non-trivial (>=2 atoms after stripping),
    excluding any whole-subject holdout (HOLDOUT_SUBJECTS) so those subjects are never seen."""
    return [s for s in planned_subjects()
            if len(train_plan(s)) >= 2 and s not in HOLDOUT_SUBJECTS]


# ===================================================================================================
# 2. RENDER target: a content plan -> a multi-sentence English paragraph, using reason_realtext's
#    OWN templated realizations (so we control the surface and can parse it back exactly). We use the
#    canonical deterministic SENT realizers (RT.SENT) so the paragraph is stable and parseable.
# ===================================================================================================
def atom_to_sentence(atom):
    """Render ONE atom to its canonical English sentence (token list), reusing reason_realtext."""
    pred, (a, b) = atom
    return RT.SENT[pred](a, b)  # e.g. ["a","robin","is","a","bird","."]


def plan_to_paragraph_tokens(plan):
    """Concatenate the per-atom canonical sentences into one paragraph token stream."""
    toks = []
    for atom in plan:
        toks += atom_to_sentence(atom)
    return toks


# ---- writer vocabulary: WORD-LEVEL over reason_realtext's structural + label vocab + control tokens.
# (Word-level keeps the writer small and makes the boundary between content tokens and the plan obvious.)
CTRL = ["<pad>", "<bos>", "<plan>", "<write>", "<eos>"]


def build_vocab():
    itos, stoi = RT.build_vocab()  # reason_realtext base vocab (struct + labels + plurals)
    # ScratchpadLM reserves index 0 as <pad>. RT's vocab already has <pad> at index 0.
    # Insert our remaining control tokens right after <pad>, keeping <pad>=0.
    assert itos[0] == "<pad>", "expected reason_realtext <pad> at index 0"
    extra = [t for t in CTRL if t != "<pad>" and t not in stoi]
    itos2 = [itos[0]] + extra + itos[1:]
    assert len(itos2) == len(set(itos2)), "duplicate vocab token"
    return itos2, {t: i for i, t in enumerate(itos2)}


# ---- the writer is supervised to map  <bos> <plan> [atom-keyword tokens] <write> [paragraph] <eos>.
# The plan is encoded as a compact keyword sequence (pred + the two label words per atom) so the
# writer learns to CONDITION its prose on the supplied content, not to free-associate.
PRED_KW = {"isa": "is", "part_of": "part", "has_prop": "can"}  # all in RT vocab


def plan_keywords(plan):
    toks = []
    for (pred, (a, b)) in plan:
        toks += [PRED_KW[pred], a, b]
    return toks


def build_example_tokens(plan):
    """Full supervised sequence for one subject's plan."""
    seq = ["<bos>", "<plan>"] + plan_keywords(plan) + ["<write>"]
    write_start = len(seq)
    seq = seq + plan_to_paragraph_tokens(plan) + ["<eos>"]
    return seq, write_start


# ---- entity pool for SUBJECT RELABELING (the core fix for plan->subject binding).
# Teacher forcing lets the writer cheat: it copies the subject from the PREVIOUS sentence of the
# paragraph instead of from the PLAN. At free-running decode the first sentence has no prior subject,
# so it emits a frequent default (e.g. "animal") and then faithfully repeats that WRONG subject for
# every later sentence -- exactly the 0/32 subject-fidelity failure we measured. Relabeling the
# subject token to a random entity (consistently in BOTH the plan and the paragraph) makes the cheat
# impossible: the relabeled token is unpredictable from the LM head, so the only way to render the
# first sentence's subject is to COPY it from the plan -- which is what the pointer head is for. This
# is the relabeling-equivariance lever that makes pointer copy generalize to held-out subjects.
ENTITY_POOL = sorted({
    s for s in planned_subjects()
} | {
    x for s in planned_subjects() for (p, (a, b)) in content_plan(s) if p == "isa" for x in (a, b)
})


def build_supervised_example(rng, subj, plan, relabel_p=0.0):
    """Return (seq_tokens, write_start, copy_src) for one subject.
    copy_src[p] = the FIRST plan-region position whose token equals seq[p] (a token the writer should
    COPY from the supplied plan rather than generate), or -1. Supervising the pointer copy-head toward
    copy_src teaches plan->paragraph copying (esp. the subject). With prob `relabel_p` the subject
    token is consistently replaced by a random other entity to FORCE plan-sourced rendering."""
    seq, write_start = build_example_tokens(plan)
    if relabel_p > 0.0 and rng.random() < relabel_p:
        present = set(seq)
        choices = [e for e in ENTITY_POOL if e != subj and e not in present]
        if choices:
            r = rng.choice(choices)
            seq = [r if t == subj else t for t in seq]
    # copy-source alignment: for each WRITE-region position, find its token in the plan region.
    plan_pos = {}
    for i in range(write_start):
        plan_pos.setdefault(seq[i], i)  # first occurrence
    copy_src = [-1] * len(seq)
    for p in range(write_start, len(seq)):
        if seq[p] in plan_pos:
            copy_src[p] = plan_pos[seq[p]]
    return seq, write_start, copy_src


# ===================================================================================================
# Model
# ===================================================================================================
def build_model(vocab_size, d=192, layers=4, heads=6, max_len=192, pointer=False):
    assert d % heads == 0 and (d // heads) % 2 == 0
    # pointer=True (+ required tie) lets the writer COPY subject/label words from the supplied
    # content plan instead of regenerating them -- decisive for faithful rendering of the plan
    # (reason_realtext uses the same pointer head to break the held-out copy floor). The free-writing
    # baseline keeps pointer=False (write_lm's plain causal decoder).
    return ScratchpadLM(vocab=vocab_size, d=d, layers=layers, heads=heads, max_len=max_len,
                        pos_mode="rope", causal=True, pointer=pointer, tie=True, loop=False)


def make_batch(rng, stoi, subjects, device, batch, block, relabel_p=0.0):
    fill = stoi["<pad>"]
    seqs, masks, csrcs = [], [], []
    tries = 0
    while len(seqs) < batch and tries < batch * 12:
        tries += 1
        subj = rng.choice(subjects)
        plan = train_plan(subj)  # held-out derived atoms stripped from training paragraph
        seq, write_start, copy_src = build_supervised_example(rng, subj, plan, relabel_p=relabel_p)
        if len(seq) > block:
            continue
        ids = [stoi[t] for t in seq]
        m = [0] * len(ids)
        for i in range(write_start, len(ids)):  # supervise the WRITE region (paragraph + <eos>)
            m[i] = 1
        seqs.append(ids)
        masks.append(m)
        csrcs.append(copy_src)
    if not seqs:
        return None
    L = max(len(s) for s in seqs)
    ids_b = torch.full((len(seqs), L), fill, dtype=torch.long)
    mask_b = torch.zeros((len(seqs), L), dtype=torch.float)
    csrc_b = torch.full((len(seqs), L), -1, dtype=torch.long)
    for r, (s, m, c) in enumerate(zip(seqs, masks, csrcs)):
        ids_b[r, :len(s)] = torch.tensor(s)
        mask_b[r, :len(m)] = torch.tensor(m, dtype=torch.float)
        csrc_b[r, :len(c)] = torch.tensor(c, dtype=torch.long)
    return ids_b.to(device), mask_b.to(device), csrc_b.to(device)


def loss_fn(model, ids, mask, copy_src=None, aux_w=0.0):
    out = model(ids)
    logits = out[:, :-1]
    target = ids[:, 1:]
    tgt_mask = (mask[:, 1:] > 0)
    sel_logits = logits[tgt_mask]
    sel_target = target[tgt_mask]
    main = out.sum() * 0.0 if sel_logits.numel() == 0 else F.cross_entropy(sel_logits, sel_target)
    # copy-head aux: supervise the pointer copy-head to attend from each write-region position to the
    # plan-region position holding the SAME token (esp. the subject) -> teaches plan->paragraph copy.
    if aux_w > 0.0 and copy_src is not None and getattr(model, "pointer", False):
        a0 = model.blocks[-1]._attn[:, 0]      # (B,L,L): a0[b,q,s] = copy attention query q -> source s
        q_attn = a0[:, :-1, :]                  # query q=p-1 predicts token at position p
        src = copy_src[:, 1:]                    # source position for the token predicted at each query
        sup = (src >= 0) & (mask[:, 1:] > 0)     # supervise only copyable tokens in the write region
        if sup.any():
            aux = F.nll_loss(torch.log(q_attn[sup] + 1e-9), src[sup])
            main = main + aux_w * aux
    return main


# ===================================================================================================
# Generation: condition on <bos><plan>[keywords]<write>, greedily decode the paragraph until <eos>.
# ===================================================================================================
@torch.no_grad()
def write_paragraph(model, stoi, itos, plan, device, block=192, max_new=120):
    model.eval()
    prompt = ["<bos>", "<plan>"] + plan_keywords(plan) + ["<write>"]
    ids = [stoi[t] for t in prompt]
    eos = stoi["<eos>"]
    out = []
    for _ in range(max_new):
        ctx = torch.tensor([ids[-block:]], device=device)
        nxt = int(model(ctx)[0, -1].argmax())
        if nxt == eos:
            break
        ids.append(nxt)
        out.append(itos[nxt])
    model.train()
    return out  # paragraph token list (no control tokens)


def split_sentences(tokens):
    """Split a paragraph token list into sentences on '.'; return list of token lists (no dot)."""
    sents, cur = [], []
    for t in tokens:
        if t == ".":
            if cur:
                sents.append(cur)
            cur = []
        else:
            cur.append(t)
    if cur:
        sents.append(cur)
    return sents


# ===================================================================================================
# 3. FAITHFULNESS: parse each generated sentence back to an atom; verify it is in the closure.
# ===================================================================================================
def faithfulness(paragraph_tokens):
    """Returns (n_verified, n_total, details). A sentence is faithful iff it parses to an atom
    (via reason_realtext's exact parser) that is in RT.CLOSURE (engine-true)."""
    sents = split_sentences(paragraph_tokens)
    n_ok, details = 0, []
    for s in sents:
        atom = RT.parse_sentence(s)  # s already has the trailing '.' stripped by split_sentences
        verified = atom is not None and atom in RT.CLOSURE
        n_ok += int(verified)
        details.append((s, atom, verified))
    return n_ok, len(sents), details


# ===================================================================================================
# 4. COHERENCE proxy (applies to BOTH content-planned and free-writing): fraction of sentences that
#    are (i) DISTINCT (no repetition loops), (ii) WELL-FORMED (parse to a known relation template),
#    (iii) ON-TOPIC for the REQUESTED subject. On-topic is STRICT: the sentence's subject term (the
#    `a` of its atom) must be the requested subject (or a part the subject has). A sentence that is
#    engine-true but about a DIFFERENT entity (e.g. free-writing's memorized "a bee ...") is NOT
#    on-topic for "ant" -- this is what makes the proxy actually subject-specific and exposes both
#    free-writing's wrong-subject output and any copy-the-neighbor error in the planned writer.
# ===================================================================================================
def topic_subjects(subject):
    """The set of valid SENTENCE-SUBJECT terms for a paragraph about `subject`: the subject itself,
    plus the part terms it legitimately owns (part_of(part, cat) where cat is in its isa chain)."""
    subs = {subject}
    for (p, (a, _b)) in content_plan(subject):
        if p == "part_of":
            subs.add(a)  # e.g. "a wing is part of a bird" -> "wing" is a valid sentence subject
    return subs


def coherence(paragraph_tokens, subject):
    sents = split_sentences(paragraph_tokens)
    if not sents:
        return 0, 0, []
    valid_subj = topic_subjects(subject)
    seen = set()
    n_ok, details = 0, []
    for s in sents:
        key = tuple(s)
        distinct = key not in seen
        seen.add(key)
        atom = RT.parse_sentence(s)  # well-formed == parses to a relation template
        wellformed = atom is not None
        on_topic = wellformed and atom[1][0] in valid_subj  # sentence's subject == requested subject
        ok = distinct and wellformed and on_topic
        n_ok += int(ok)
        details.append((s, distinct, wellformed, on_topic))
    return n_ok, len(sents), details


# ===================================================================================================
# Free-writing BASELINE (write_lm-style): a char/word causal LM with NO content plan. We train the
# SAME architecture purely as a next-token LM over the paragraphs of TRAIN subjects (no <plan>
# conditioning), then free-generate from "<bos>" on held-out subjects. As write_lm documents, this
# produces locally-plausible but globally incoherent text (loops / off-topic), which the coherence
# proxy exposes.
# ===================================================================================================
def make_freewrite_batch(rng, stoi, subjects, device, batch, block, relabel_p=0.0):
    # relabel_p is accepted for a uniform train_writer interface but unused: the free-writing
    # baseline has NO plan to copy from, so relabeling would only corrupt it. Kept as the honest
    # "no content plan" control.
    fill = stoi["<pad>"]
    seqs = []
    tries = 0
    while len(seqs) < batch and tries < batch * 12:
        tries += 1
        subj = rng.choice(subjects)
        plan = train_plan(subj)  # same training data as planned writer, minus the <plan> conditioning
        seq = ["<bos>"] + plan_to_paragraph_tokens(plan) + ["<eos>"]
        if len(seq) > block:
            continue
        seqs.append([stoi[t] for t in seq])
    if not seqs:
        return None
    L = max(len(s) for s in seqs)
    ids_b = torch.full((len(seqs), L), fill, dtype=torch.long)
    mask_b = torch.zeros((len(seqs), L), dtype=torch.float)
    for r, s in enumerate(seqs):
        ids_b[r, :len(s)] = torch.tensor(s)
        mask_b[r, 1:len(s)] = 1.0  # predict every token after <bos>
    return ids_b.to(device), mask_b.to(device)  # free writer: no copy_src



@torch.no_grad()
def free_write(model, stoi, itos, device, block=192, max_new=120):
    model.eval()
    ids = [stoi["<bos>"]]
    eos = stoi["<eos>"]
    out = []
    for _ in range(max_new):
        ctx = torch.tensor([ids[-block:]], device=device)
        nxt = int(model(ctx)[0, -1].argmax())
        if nxt == eos:
            break
        ids.append(nxt)
        out.append(itos[nxt])
    model.train()
    return out


def train_writer(make_batch_fn, seed, steps, lr, device, vocab,
                 dim, layers, heads, batch, block, label, pointer=False,
                 relabel_p=0.0, aux_w=0.0):
    itos, stoi = vocab
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = build_model(len(itos), d=dim, layers=layers, heads=heads, max_len=block,
                        pointer=pointer).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    model.train()
    subs = train_subjects()
    last = 0.0
    for step in range(steps):
        out = make_batch_fn(rng, stoi, subs, device, batch, block, relabel_p=relabel_p)
        if out is None:
            continue
        if len(out) == 2:
            ids, mask = out
            csrc = None
        else:
            ids, mask, csrc = out
        loss = loss_fn(model, ids, mask, copy_src=csrc, aux_w=aux_w)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        last = loss.item()
        if (step + 1) % 200 == 0:
            print(f"    [{label} seed{seed}] step {step+1}/{steps} loss {last:.4f}", flush=True)
    return model, last


# ===================================================================================================
# Selftest
# ===================================================================================================
def selftest():
    device = "cpu"
    torch.set_num_threads(2)
    vocab = build_vocab()
    itos, stoi = vocab

    # vocab round-trip
    sample = ["<bos>", "<plan>", "robin", "bird", "<write>", "is", "can", ".", "<eos>"]
    ids = [stoi[t] for t in sample]
    back = [itos[i] for i in ids]
    assert back == sample, ("vocab round-trip failed", back)

    # a non-trivial plan exists and renders + parses back to closure atoms
    plan = content_plan("robin")
    assert len(plan) >= 2, ("robin plan too small", plan)
    para = plan_to_paragraph_tokens(plan)
    nok, ntot, _ = faithfulness(para)
    assert nok == ntot and ntot >= 2, ("templated render not all engine-true", nok, ntot)

    # forward shape
    model = build_model(len(itos), d=64, layers=2, heads=4, max_len=96, pointer=True).to(device)
    x = torch.tensor([ids], device=device)
    logits = model(x)
    assert logits.shape == (1, len(ids), len(itos)), ("bad logits shape", tuple(logits.shape))

    # a few train steps lift a trivial content-plan rendering above chance (loss drops)
    rng = random.Random(0)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(40):
        out = make_batch(rng, stoi, train_subjects(), device, batch=16, block=96, relabel_p=0.5)
        ids_b, mask_b, csrc_b = out
        loss = loss_fn(model, ids_b, mask_b, copy_src=csrc_b, aux_w=1.0)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] - 0.3, ("train did not lift above chance", losses[0], losses[-1])

    # save -> load identical
    import tempfile
    path = os.path.join(tempfile.gettempdir(), "rw_selftest.pt")
    torch.save({"model": model.state_dict(), "config": model.config, "itos": itos}, path)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m2 = build_model(len(ck["itos"]), d=64, layers=2, heads=4, max_len=96, pointer=True)
    m2.load_state_dict(ck["model"])
    m2.eval(); model.eval()
    with torch.no_grad():
        a = model(x); b = m2(x)
    assert torch.allclose(a, b, atol=1e-5), "save/load mismatch"
    os.remove(path)

    print("reason_write selftest OK")


# ===================================================================================================
# Full run
# ===================================================================================================
def evaluate_planned(model, stoi, itos, subjects, device, block):
    """Content-planned writer: SUBJECT-FIDELITY (primary) + faithfulness + coherence on held-out."""
    f_ok = f_tot = c_ok = c_tot = named = 0
    samples = []
    for subj in subjects:
        plan = eval_plan(subj)  # FULL plan incl. held-out derived atoms the writer must compose
        para = write_paragraph(model, stoi, itos, plan, device, block=block)
        fo, ft, _ = faithfulness(para)
        co, ct, _ = coherence(para, subj)
        f_ok += fo; f_tot += ft; c_ok += co; c_tot += ct
        named += int(subj in para)  # PRIMARY: does the paragraph actually name the requested subject?
        samples.append((subj, " ".join(para)))
    fidelity = named / max(1, len(subjects))
    return fidelity, f_ok / max(1, f_tot), c_ok / max(1, c_tot), \
        (named, len(subjects), f_ok, f_tot, c_ok, c_tot), samples


def evaluate_free(model, stoi, itos, subjects, device, block):
    """Free-writing baseline: same coherence proxy (judged against each held-out subject's topic),
    plus faithfulness for completeness. Free-writing has no plan, so we generate one paragraph per
    subject from <bos> and score it against that subject."""
    f_ok = f_tot = c_ok = c_tot = named = 0
    samples = []
    for subj in subjects:
        para = free_write(model, stoi, itos, device, block=block)
        fo, ft, _ = faithfulness(para)
        co, ct, _ = coherence(para, subj)
        f_ok += fo; f_tot += ft; c_ok += co; c_tot += ct
        named += int(subj in para)
        samples.append((subj, " ".join(para)))
    fidelity = named / max(1, len(subjects))
    return fidelity, f_ok / max(1, f_tot), c_ok / max(1, c_tot), \
        (named, len(subjects), f_ok, f_tot, c_ok, c_tot), samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--block", type=int, default=192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--holdout", choices=["fact", "subject"], default="fact",
                    help="fact: withhold derived atoms (subject seen). subject: withhold WHOLE subjects "
                         "(novel-entity test) -- the writer must render a never-seen subject by copying "
                         "it from the plan.")
    ap.add_argument("--holdout-n", type=int, default=8, help="# whole subjects to hold out (subject mode)")
    ap.add_argument("--relabel", type=float, default=0.0,
                    help="prob of relabeling the subject token during training (forces plan-sourced "
                         "rendering; the lever for whole-subject generalization). 0 reproduces the "
                         "verified fact-level result.")
    ap.add_argument("--aux", type=float, default=0.0, help="weight on the copy-head supervision loss")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    device = get_device()
    if device == "mps":
        device = "cpu"  # this module is CPU-deterministic; avoid mps nondeterminism
    nt = os.environ.get("RW_THREADS")
    torch.set_num_threads(int(nt) if nt else max(1, os.cpu_count() or 1))

    vocab = build_vocab()
    itos, stoi = vocab

    # whole-subject holdout: pick N planned subjects to withhold ENTIRELY from training.
    global HOLDOUT_SUBJECTS
    if args.holdout == "subject":
        pool = [s for s in planned_subjects() if len(content_plan(s)) >= 3]
        HOLDOUT_SUBJECTS = set(random.Random(args.seed).sample(pool, min(args.holdout_n, len(pool))))

    tr = train_subjects()
    held = sorted(HOLDOUT_SUBJECTS) if args.holdout == "subject" else heldout_subjects()
    print(f"reason->write | vocab={len(itos)} | planned subjects={len(planned_subjects())} "
          f"(train={len(tr)}) | holdout={args.holdout} | eval subjects={len(held)} | "
          f"relabel={args.relabel} aux={args.aux} | device={device} | steps={args.steps}", flush=True)
    if args.holdout == "subject":
        print(f"  WHOLE-SUBJECT holdout (novel-entity test): {held} are NEVER seen in training; the "
              f"writer must render each by COPYING the subject from the supplied plan.", flush=True)
    else:
        print(f"  fact-level split: every subject is trained, but its HELD-OUT derived atoms "
              f"(RT.HELDOUT) are stripped from the training paragraph and only requested at eval.",
              flush=True)
    # show one content plan, marking which atoms are held-out (composed at eval, unseen in training)
    demo = held[0] if held else tr[0]
    print(f"  example content plan for '{demo}' ({len(content_plan(demo))} engine-derived atoms; "
          f"* = held-out, never rendered in training):", flush=True)
    for atom in content_plan(demo):
        mark = " *" if is_heldout_atom(atom) else "  "
        print(f"   {mark} {atom}  ->  {' '.join(atom_to_sentence(atom))}", flush=True)

    print("\n[A] training CONTENT-PLANNED writer (plan -> paragraph):", flush=True)
    planned_model, ploss = train_writer(make_batch, args.seed, args.steps, args.lr, device, vocab,
                                        args.dim, args.layers, args.heads, args.batch, args.block,
                                        "PLANNED", pointer=True, relabel_p=args.relabel, aux_w=args.aux)
    print("\n[B] training FREE-WRITING baseline (no plan, next-token LM):", flush=True)
    free_model, floss = train_writer(make_freewrite_batch, args.seed, args.steps, args.lr, device,
                                     vocab, args.dim, args.layers, args.heads, args.batch, args.block,
                                     "FREE")

    eval_subjects = held if held else tr
    pfid, pf, pc, pcounts, psamples = evaluate_planned(planned_model, stoi, itos, eval_subjects, device, args.block)
    ffid, ff, fc, fcounts, fsamples = evaluate_free(free_model, stoi, itos, eval_subjects, device, args.block)

    print("\n================= RESULTS (HELD-OUT subjects) =================", flush=True)
    print(f"CONTENT-PLANNED  SUBJECT-FIDELITY {pfid:.3f} ({pcounts[0]}/{pcounts[1]})  |  "
          f"faithfulness {pf:.3f} ({pcounts[2]}/{pcounts[3]})  "
          f"coherence {pc:.3f} ({pcounts[4]}/{pcounts[5]})", flush=True)
    print(f"FREE-WRITING     SUBJECT-FIDELITY {ffid:.3f} ({fcounts[0]}/{fcounts[1]})  |  "
          f"faithfulness {ff:.3f} ({fcounts[2]}/{fcounts[3]})  "
          f"coherence {fc:.3f} ({fcounts[4]}/{fcounts[5]})", flush=True)

    print("\n--- sample CONTENT-PLANNED paragraphs (verbatim greedy decode) ---", flush=True)
    for subj, para in psamples[:3]:
        print(f"  [{subj}] {para}", flush=True)
    print("\n--- sample FREE-WRITING paragraphs (verbatim greedy decode) ---", flush=True)
    for subj, para in fsamples[:3]:
        print(f"  [{subj}] {para}", flush=True)

    def _pack(fid, faith, coh, counts, samples, loss):
        # counts = (named, n_subj, f_ok, f_tot, c_ok, c_tot)
        return {"subject_fidelity": fid, "faithfulness": faith, "coherence": coh,
                "fidelity_counts": counts[:2], "faith_counts": counts[2:4], "coh_counts": counts[4:],
                "final_loss": loss,
                "samples": [{"subject": s, "paragraph": p} for s, p in samples]}

    results = {
        "steps": args.steps, "seed": args.seed, "device": device,
        "dim": args.dim, "layers": args.layers, "heads": args.heads,
        "holdout": args.holdout, "relabel": args.relabel, "aux": args.aux,
        "n_planned_subjects": len(planned_subjects()),
        "n_train_subjects": len(tr), "n_heldout_subjects": len(held),
        "heldout_subjects": held,
        "planned": _pack(pfid, pf, pc, pcounts, psamples, ploss),
        "free": _pack(ffid, ff, fc, fcounts, fsamples, floss),
    }
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nsaved results -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
