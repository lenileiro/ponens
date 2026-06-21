#!/usr/bin/env python3
"""wordnet_full -- TRAIN the comprehension backbone on ALL of WordNet, against the full brain.

Zero-shot MiniLM + brain already gives comprehension ~0.73 on a sampled taxonomy, but at the FULL
scale (19.8k candidate parents) retrieval is far harder. So we FINE-TUNE the embedder on all ~87.6k
WordNet (definition -> is-a parent) pairs with a contrastive objective, then comprehend against the
COMPLETE is-a brain (every synset's ancestors), with the brain selecting the provable parent.

  COMPREHEND: embed a concept's DEFINITION (fine-tuned) -> nearest candidate parent GLOSSES -> top-k;
    the BRAIN keeps the ones it can prove (ancestor in the full closure) and picks the most specific.
  The model PROPOSES (now WordNet-specialized); the BRAIN PROVES.

  python -m thinking.wordnet_full --selftest
  python -m thinking.wordnet_full --steps 4000 --device cuda --save runs/wn_full.pt
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ===================================================================================================
# Data: ALL WordNet concepts with a definition + an is-a parent; the full is-a brain (ancestors)
# ===================================================================================================
def load_all(max_concepts=None, seed=0):
    from nltk.corpus import wordnet as wn
    rows, gloss = [], {}
    for s in wn.all_synsets():
        d = s.definition()
        gloss[s.name()] = d or s.name().split(".")[0].replace("_", " ")
        if d and s.hypernyms():
            path = [x.name() for x in s.hypernym_paths()[0]]   # root..leaf
            rows.append({"name": s.name(), "def": d, "parent": path[-2],
                         "ancestors": set(path[:-1])})
    rng = np.random.default_rng(seed); rng.shuffle(rows)
    if max_concepts:
        rows = rows[:max_concepts]
    return rows, gloss


def split(rows, held=0.1, seed=0):
    idx = np.arange(len(rows)); np.random.default_rng(seed).shuffle(idx)
    cut = int(len(rows) * (1 - held))
    return [rows[i] for i in idx[:cut]], [rows[i] for i in idx[cut:]]


# ===================================================================================================
# Trainable embedder (fine-tuned MiniLM, mean-pooled)
# ===================================================================================================
class Embedder(nn.Module):
    def __init__(self, model_name=EMBED_MODEL):
        super().__init__()
        from transformers import AutoTokenizer, AutoModel
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, texts, device, max_len=48):
        enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True,
                       max_length=max_len).to(device)
        h = self.model(**enc).last_hidden_state
        m = enc.attention_mask.unsqueeze(-1).float()
        return F.normalize((h * m).sum(1) / m.sum(1).clamp(min=1), dim=-1)

    @torch.no_grad()
    def embed_all(self, texts, device, bs=256):
        self.eval()
        out = [self(texts[i:i + bs], device).cpu() for i in range(0, len(texts), bs)]
        return torch.cat(out)


def train(emb, tr, gloss, steps, device, batch=256, lr=2e-5, temp=0.05, seed=0, log=0):
    """Contrastive: a concept's DEFINITION embedding should match its is-a PARENT's gloss embedding
    (in-batch negatives, InfoNCE) -- specializes the embedder for WordNet def<->parent retrieval."""
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    opt = torch.optim.AdamW(emb.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    emb.to(device).train()
    n = len(tr)
    for step in range(steps):
        bc = [tr[i] for i in rng.integers(0, n, size=batch)]
        with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
            a = emb([c["def"] for c in bc], device)
            b = emb([gloss[c["parent"]] for c in bc], device)
            logits = a @ b.t() / temp
            lbl = torch.arange(len(bc), device=device)
            loss = 0.5 * (F.cross_entropy(logits, lbl) + F.cross_entropy(logits.t(), lbl))
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if log and (step % log == 0 or step == steps - 1):
            print(f"  step {step:5d}  loss {loss.item():.4f}", flush=True)


# ===================================================================================================
# Eval: comprehension (brain-assisted) + gate calibration, against the FULL is-a brain
# ===================================================================================================
@torch.no_grad()
def evaluate(emb, tr, te, gloss, device, topk=10, verbose=True, tag=""):
    cands = sorted({c["parent"] for c in tr + te})          # full candidate parent set
    ci = {p: i for i, p in enumerate(cands)}
    cand_emb = emb.embed_all([gloss[p] for p in cands], device)
    de = emb.embed_all([c["def"] for c in te], device)
    sims = de @ cand_emb.t()
    top1 = exact = assisted = recall_k = asserted = a_false = abst = a_wrong = nogate_false = 0
    for r, c in enumerate(te):
        order = sims[r].topk(min(topk, len(cands))).indices.tolist()
        names = [cands[j] for j in order]
        anc = c["ancestors"]
        top1_true = names[0] in anc
        nogate_false += int(not top1_true); top1 += int(top1_true)
        prov = [x for x in names if x in anc]               # BRAIN gate: ancestor in full closure
        recall_k += int(bool(prov))
        if prov:
            pick = prov[0]                                   # top-ranked provable (most confident true)
            assisted += 1; asserted += 1; a_false += int(pick not in anc)
            exact += int(pick == c["parent"])
        else:
            abst += 1; a_wrong += int(not top1_true)
    n = max(1, len(te))
    res = dict(tag=tag, n=len(te), n_cand=len(cands), top1=top1 / n, brain_assisted=assisted / n,
               recall_at_k=recall_k / n, exact=exact / max(1, asserted), coverage=asserted / n,
               asserted_false=a_false / max(1, asserted), abstain=abst / n,
               abstain_correct=a_wrong / max(1, abst), nogate_false=nogate_false / n)
    if verbose:
        print(f"\n== FULL-WORDNET comprehension + gate {tag} ({res['n']} held | {res['n_cand']} candidates) ==")
        print(f"  top-1 in-closure {res['top1']:.3f} -> BRAIN-ASSISTED {res['brain_assisted']:.3f} | "
              f"recall@{topk} {res['recall_at_k']:.3f} | exact-parent(asserted) {res['exact']:.3f}")
        print(f"  GATE: coverage {res['coverage']:.3f} | FALSE-assertion {res['asserted_false']:.3f} | "
              f"abstain {res['abstain']:.3f} (correct {res['abstain_correct']:.3f}) | "
              f"no-gate-false {res['nogate_false']:.3f}")
    return res


def run(steps=4000, max_concepts=None, seed=0, device=None, batch=256, save=None, out=None, verbose=True):
    device = device or pick_device()
    t = time.time()
    rows, gloss = load_all(max_concepts=max_concepts, seed=seed)
    tr, te = split(rows, 0.1, seed)
    emb = Embedder().to(device)                              # move BEFORE zero-shot eval (device-safe)
    if verbose:
        print(f"  FULL WordNet | {len(rows)} concepts (train {len(tr)}/held {len(te)}) | "
              f"candidates {len(set(c['parent'] for c in rows))} | device {device} | load {time.time()-t:.0f}s",
              flush=True)
        print("  -- zero-shot (before fine-tuning) --", flush=True)
        evaluate(emb, tr, te, gloss, device, verbose=verbose, tag="[zero-shot]")
    train(emb, tr, gloss, steps, device, batch=batch, seed=seed,
          log=(max(1, steps // 6) if verbose else 0))
    res = evaluate(emb, tr, te, gloss, device, verbose=verbose, tag="[fine-tuned]")
    if save:
        torch.save({"state": emb.model.state_dict(), "model": EMBED_MODEL}, save)
        if verbose:
            print(f"  saved fine-tuned embedder -> {save}", flush=True)
    if out:
        import json
        json.dump(res, open(out, "w"), indent=2)
        if verbose:
            print(f"  wrote {out}", flush=True)
    return res


# ===================================================================================================
# CHAT with the FINE-TUNED full-WordNet agent: comprehend (fine-tuned embedder + brain) -> write -> gate
# ===================================================================================================
def build_chat(embedder_path=None, max_concepts=None, device=None, gen=True, verbose=True):
    import thinking.meaning as M
    import thinking.backbone as BB
    device = device or pick_device()
    rows, gloss = load_all(max_concepts=max_concepts)
    ancestors = {r["name"]: r["ancestors"] for r in rows}
    cands = sorted({r["parent"] for r in rows})
    emb = Embedder().to(device)
    if embedder_path:
        emb.model.load_state_dict(torch.load(embedder_path, map_location=device)["state"])
        if verbose:
            print(f"  loaded FINE-TUNED embedder <- {embedder_path}", flush=True)
    emb.eval()
    if verbose:
        print(f"  embedding {len(cands)} candidate glosses...", flush=True)
    cand_emb = emb.embed_all([gloss[c] for c in cands], device)
    generator = BB.Generator(device=device) if gen else None
    return dict(emb=emb, cand_emb=cand_emb, cands=cands, gloss=gloss, ancestors=ancestors,
                device=device, gen=generator, M=M)


def _resolve(text, gloss):
    """Map a word/synset-name to a known WordNet concept (with a definition), else None (free text)."""
    from nltk.corpus import wordnet as wn
    key = text.strip().lower().replace(" ", "_")
    if key in gloss:
        return key
    try:
        cands = [s for s in wn.synsets(key) if s.name() in gloss]
    except Exception:
        cands = []
    if not cands:
        return None
    # prefer the EPONYMOUS sense -- a synset literally named for the word (e.g. whale.n.02) over senses
    # where the word is only a synonym (e.g. giant.n.04 where 'whale' = a huge person). Avoids WSD slips.
    named = [s for s in cands if s.name().split(".")[0] == key]
    return (named or cands)[0].name()


@torch.no_grad()
def chat_respond(agent, text, topk=10):
    M = agent["M"]
    known = _resolve(text, agent["gloss"])
    query = agent["gloss"][known] if known else text
    qe = agent["emb"].embed_all([query], agent["device"])
    order = (qe @ agent["cand_emb"].t())[0].topk(min(topk, len(agent["cands"]))).indices.tolist()
    cands = [agent["cands"][j] for j in order]
    if known is not None:                                   # BRAIN-ASSIST: keep provable ancestors
        anc = agent["ancestors"].get(known, set())
        prov = [x for x in cands if x in anc]
        pick, proven = (prov[0] if prov else cands[0]), bool(prov)
        lemma = M.parent_name_text(known)
    else:
        pick, proven, lemma = cands[0], False, "this"
    cat = M.parent_name_text(pick)
    written = agent["gen"].write(lemma, cat) if agent["gen"] else "(no generator)"
    anc_chain = ", ".join(M.parent_name_text(a) for a in list(agent["ancestors"].get(pick, []))[:8])
    if known is None:
        head = f"  (novel input -- can't kernel-verify; best guess) a {cat}"
    elif proven:
        head = f"  understood as : a {cat}   [VERIFIED -- {known} is-a {cat} in the WordNet brain]"
    else:
        head = f"  uncertain: best guess 'a {cat}', but the BRAIN CANNOT verify it for {known}"
    return "\n".join([head, f"  my definition: {written}",
                      f"  brain: {cat} is-a -> {anc_chain or '(root)'}"])


def _hyper_path(name):
    """Ordered leaf..root is-a path of a synset (the brain's is-a structure for explanations)."""
    from nltk.corpus import wordnet as wn
    try:
        return [x.name() for x in wn.synset(name).hypernym_paths()[0]][::-1]
    except Exception:
        return [name]


def relate(agent, x, y):
    """Answer 'is an X a Y?' AND EXPLAIN WHY, faithfully from the brain's is-a structure:
    YES -> the is-a chain (transitivity); NO -> the disjointness witness (separate branches)."""
    M = agent["M"]
    C = _resolve(x, agent["gloss"]) or x
    B = _resolve(y, agent["gloss"]) or y
    pc, pb = _hyper_path(C), _hyper_path(B)
    pcset, pbset = set(pc), set(pb)
    cn, bn = M.parent_name_text(C), M.parent_name_text(B)
    gen = agent.get("gen")

    def verbalize(verdict, facts, proof):
        """Render the brain's proof in natural language -- the LM phrases it, using ONLY these facts."""
        head = verdict[0].upper() + verdict[1:] + "."     # crisp verdict, always explicit
        if not gen:
            return f"  {head}\n  proof: {proof}"
        nl = gen.say(f"Explain in one or two clear, natural sentences why {verdict}, using ONLY these "
                     f"facts and adding nothing new: {facts}. Reply with only the explanation.")
        return f"  {head}\n  {nl}\n  (proof: {proof})"

    if B in pcset:                                           # YES: walk the is-a chain C -> ... -> B
        chain = pc[:pc.index(B) + 1]
        proof = " -> ".join(M.parent_name_text(z) for z in chain)
        facts = "; ".join(f"a {M.parent_name_text(chain[i])} is a kind of {M.parent_name_text(chain[i+1])}"
                          for i in range(len(chain) - 1))
        return verbalize(f"a {cn} is a {bn}", facts, proof)
    if C in pbset:
        return f"It's the other way around: a {bn} is a {cn}, not a {cn} is a {bn}."
    lca = next((z for z in pc if z in pbset), None)          # LOWEST common ancestor
    if lca is not None:
        cbranch = M.parent_name_text(pc[pc.index(lca) - 1] if pc.index(lca) > 0 else C)
        ln = M.parent_name_text(lca)
        facts = (f"a {cn} is a {cbranch}; both a {cbranch} and a {bn} are kinds of {ln}, but they are "
                 f"separate, mutually exclusive kinds, so nothing is both a {cbranch} and a {bn}")
        proof = f"{cn} is-a {cbranch}; {cbranch} disjoint {bn} (separate branches under {ln})"
        return verbalize(f"a {cn} is NOT a {bn}", facts, proof)
    return f"  the brain can't relate '{cn}' and '{bn}' (no shared is-a ancestry)."


_REL = __import__("re").compile(r"is\s+(?:an?\s+)?(.+?)\s+(?:an?\s+)(.+?)\??$", __import__("re").I)


def answer(agent, text):
    """Dispatch: 'is a X a Y?' -> relate+explain; otherwise comprehend+define."""
    m = _REL.match(text.strip())
    if m:
        return relate(agent, m.group(1).strip(), m.group(2).strip())
    return chat_respond(agent, text)


def chat(agent):
    print("\n== CHAT with the FINE-TUNED full-WordNet agent -- a word, a definition, or 'is a X a Y?'; "
          "blank to quit ==", flush=True)
    while True:
        try:
            line = input("you> ").strip()
        except EOFError:
            break
        if not line:
            break
        print(answer(agent, line), flush=True)


def selftest():
    torch.set_num_threads(2)
    r = run(steps=60, max_concepts=2000, seed=0, device="cpu", batch=64, verbose=False)
    assert r["n"] > 0 and r["asserted_false"] == 0.0, ("gate must assert no false is-a", r)
    print("wordnet_full selftest OK")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--max-concepts", type=int, default=None, help="cap concepts (default: ALL WordNet)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default=None)
    ap.add_argument("--save", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--load", default=None, help="load fine-tuned embedder ckpt (wn_full.pt) for chat")
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--ask", default=None)
    args = ap.parse_args(argv)
    if args.selftest:
        selftest(); return 0
    if args.chat or args.ask:                                # CHAT with the fine-tuned full-WordNet agent
        agent = build_chat(embedder_path=args.load, max_concepts=args.max_concepts, device=args.device)
        if args.ask:
            print(answer(agent, args.ask), flush=True)
        if args.chat or not args.ask:
            chat(agent)
        return 0
    run(steps=args.steps, max_concepts=args.max_concepts, seed=args.seed, batch=args.batch,
        device=args.device, save=args.save, out=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
