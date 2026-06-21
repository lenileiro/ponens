#!/usr/bin/env python3
"""backbone -- a PRETRAINED text backbone for the language front-end, with the BRAIN for reasoning +
verification. The from-scratch char encoder hit the open-vocab wall (comprehension ~0.57); a pretrained
model gives semantic understanding for free.

  COMPREHEND: embed a concept's DEFINITION and the candidate parents' DEFINITIONS with a pretrained LM
    (mean-pooled), match def<->def by cosine -> top-k; the BRAIN keeps the provable ones and picks the
    MOST SPECIFIC (kernel-proven). Zero-shot -- no from-scratch training. (>> char-CNN: 0.73 vs 0.57.)
  WRITE: prompt a pretrained generative LM to write a definition from the brain-verified facts; the
    BRAIN GATEKEEPS every is-a claim (only kernel-provable content survives).

The model PROPOSES (now with real language competence); the BRAIN PROVES.

  python -m thinking.backbone --selftest
  python -m thinking.backbone --per-pos 300 --model HuggingFaceTB/SmolLM2-135M-Instruct
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import thinking.meaning as M  # noqa: E402  (gather / build_brain / brain_ancestor_index / parent_name_text)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # retrieval-trained embedder (beats LM mean-pool)


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Backbone:
    """Wraps a pretrained LM for (a) sentence EMBEDDING (mean-pooled hidden states) and (b) generation."""
    def __init__(self, model_name=DEFAULT_MODEL, device=None, gen=False):
        from transformers import AutoTokenizer, AutoModel
        self.device = device or pick_device()
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModel.from_pretrained(model_name, dtype=torch.float32).to(self.device).eval()
        self.gen_lm = None
        if gen:
            from transformers import AutoModelForCausalLM
            self.gen_lm = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=torch.float32).to(self.device).eval()

    @torch.no_grad()
    def embed(self, texts, bs=64, max_len=64):
        out = []
        for i in range(0, len(texts), bs):
            enc = self.tok(texts[i:i + bs], return_tensors="pt", padding=True, truncation=True,
                           max_length=max_len).to(self.device)
            h = self.model(**enc).last_hidden_state
            m = enc.attention_mask.unsqueeze(-1).float()
            e = (h * m).sum(1) / m.sum(1).clamp(min=1)               # mean-pool over real tokens
            out.append(torch.nn.functional.normalize(e, dim=-1).cpu())
        return torch.cat(out)

    @torch.no_grad()
    def generate(self, prompt, max_new=40):
        enc = self.tok(prompt, return_tensors="pt").to(self.device)
        ids = self.gen_lm.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                   pad_token_id=self.tok.pad_token_id)
        return self.tok.decode(ids[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()


GEN_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"          # pretrained generative LM for WRITING


class Generator:
    """A pretrained instruct LM that WRITES a definition from the brain-verified facts (prompt-based).
    Fluent language for free; the brain still gatekeeps the claims."""
    def __init__(self, model_name=GEN_MODEL, device=None):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.device = device or pick_device()
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.lm = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).to(self.device).eval()

    @torch.no_grad()
    def write(self, lemma, parent, parts=None, max_new=48):
        facts = f"It is a kind of {parent}."
        if parts:
            facts += " It has: " + ", ".join(parts[:4]) + "."
        msg = [{"role": "user", "content": f"Write a concise one-sentence dictionary definition of "
                f"'{lemma}'. {facts} Reply with only the definition."}]
        prompt = self.tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        enc = self.tok(prompt, return_tensors="pt").to(self.device)
        ids = self.lm.generate(**enc, max_new_tokens=max_new, do_sample=False,
                               pad_token_id=self.tok.pad_token_id)
        return self.tok.decode(ids[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()


def parent_gloss(p):
    from nltk.corpus import wordnet as wn
    try:
        return wn.synset(p).definition() or M.parent_name_text(p)
    except Exception:
        return M.parent_name_text(p)


def comprehend_eval(bb, concepts, te, brain, banc, pnames, topk=10):
    """Brain-assisted comprehension via the pretrained backbone (def<->parent-gloss)."""
    cand_emb = bb.embed([parent_gloss(p) for p in pnames])
    held = [c for c in te if c["parent"]]
    def_emb = bb.embed([c["views"][0].replace("means ", "") for c in held])
    sims = def_emb @ cand_emb.t()
    top1 = exact = assisted = recall_k = 0
    picks = []
    for r, c in enumerate(held):
        order = sims[r].topk(min(topk, len(pnames))).indices.tolist()
        cands = [pnames[j] for j in order]
        C = c["name"]
        prov = [x for x in cands if brain._atom_term(("isa", (C, x)))[0]]
        pick = max(prov, key=lambda x: len(banc.get(x, ()))) if prov else cands[0]
        top1 += int(brain._atom_term(("isa", (C, cands[0])))[0])
        recall_k += int(bool(prov)); exact += int(pick == c["parent"])
        assisted += int(brain._atom_term(("isa", (C, pick)))[0])
        picks.append((C, c["parent"], pick))
    n = max(1, len(held))
    return dict(n=len(held), top1=top1 / n, brain_assisted=assisted / n, recall_at_k=recall_k / n,
                exact=exact / n), picks


def run(per_pos=200, seed=0, model=DEFAULT_MODEL, topk=10, device=None, gen=False, verbose=True,
        n_show=6):
    concepts, parent, _nt, relations = M.gather(per_pos=per_pos, seed=seed)
    tr, te = M.split(concepts, 0.25, seed)
    brain = M.build_brain(concepts, relations)
    banc = M.brain_ancestor_index(brain)
    pnames = sorted({c["parent"] for c in concepts if c["parent"]})
    bb = Backbone(model, device=device, gen=gen)
    if verbose:
        print(f"  PRETRAINED backbone {model} on {bb.device} | {len(concepts)} concepts | brain "
              f"{len(brain.known)} kernel-closed facts | candidates {len(pnames)}", flush=True)
    res, picks = comprehend_eval(bb, concepts, te, brain, banc, pnames, topk=topk)
    if verbose:
        print("\n== COMPREHENSION via PRETRAINED backbone (def<->def) + BRAIN-ASSIST (top-k, brain picks) ==")
        print(f"  {res['n']} held-out | top-1 proven {res['top1']:.3f} -> BRAIN-ASSISTED "
              f"{res['brain_assisted']:.3f} | recall@{topk} {res['recall_at_k']:.3f} | exact {res['exact']:.3f}")
        print("  (from-scratch char-CNN was: top-1 0.36 / brain-assisted 0.574)")
        for C, tp, pk in picks[:n_show]:
            mark = "OK" if pk == tp else ".."
            print(f"     [{mark}] {C:24s} true {M.parent_name_text(tp):16s} -> {M.parent_name_text(pk)}")
    return res


# ===================================================================================================
# Full pretrained-backbone AGENT: comprehend (MiniLM+brain) -> write (gen LM) -> brain-verify. Chat.
# ===================================================================================================
def build_agent(per_pos=200, seed=0, embed_model=DEFAULT_MODEL, gen_model=GEN_MODEL, device=None):
    concepts, parent, _nt, relations = M.gather(per_pos=per_pos, seed=seed)
    brain = M.build_brain(concepts, relations)
    banc = M.brain_ancestor_index(brain)
    pnames = sorted({c["parent"] for c in concepts if c["parent"]})
    bb = Backbone(embed_model, device=device)
    cand_emb = bb.embed([parent_gloss(p) for p in pnames])
    gen = Generator(gen_model, device=device)
    name2i = {c["name"]: i for i, c in enumerate(concepts)}
    return dict(concepts=concepts, relations=relations, brain=brain, banc=banc, pnames=pnames,
                bb=bb, cand_emb=cand_emb, gen=gen, name2i=name2i,
                dfacts=__import__("thinking.roundtrip", fromlist=["direct_facts"]).direct_facts(
                    concepts, relations))


def respond(agent, text, topk=10):
    """READ a word or a definition -> COMPREHEND (backbone+brain) -> WRITE (gen LM) -> brain-ground."""
    bb, brain, pnames = agent["bb"], agent["brain"], agent["pnames"]
    key = text.strip().lower()
    known = next((c for c in agent["concepts"]
                  if c["name"] == key or M.parent_name_text(c["name"]) == key), None)
    query = known["views"][0].replace("means ", "") if known else text
    sims = bb.embed([query])[0] @ agent["cand_emb"].t()
    cands = [pnames[j] for j in sims.topk(min(topk, len(pnames))).indices.tolist()]
    if known is not None:                                    # brain-assist: keep provable, most specific
        C = known["name"]
        prov = [x for x in cands if brain._atom_term(("isa", (C, x)))[0]]
        p_hat = max(prov, key=lambda x: len(agent["banc"].get(x, ()))) if prov else cands[0]
        parts = [M.parent_name_text(o) for (pr, o) in agent["dfacts"].get(C, [])][:4]
        lemma = M.parent_name_text(C)
    else:                                                    # novel description (can't brain-verify C)
        p_hat, parts, lemma = cands[0], [], "this"
    cat = M.parent_name_text(p_hat)
    written = agent["gen"].write(lemma, cat, parts)          # WRITE (pretrained LM, fluent)
    facts = brain_facts_about(brain, p_hat)
    out = [f"  understood as : a {cat}",
           f"  my definition: {written}",
           f"  brain proves about '{cat}': {facts}"]
    if known is not None:
        ok = brain._atom_term(("isa", (known["name"], p_hat)))[0]
        out.append(f"  [known: {known['name']}] is-a {cat} -> {'KERNEL-PROVEN' if ok else 'unproven'}")
    return "\n".join(out)


def brain_facts_about(brain, node, k=8):
    fs = [(p, a[1]) for (p, a) in brain.known if len(a) == 2 and a[0] == node][:k]
    return ", ".join(f"{p} {M.parent_name_text(o)}" for p, o in fs) or "(none yet)"


def chat(agent):
    print("\n== CHAT (pretrained backbone + brain) -- type a word or a definition; blank to quit ==",
          flush=True)
    while True:
        try:
            line = input("you> ").strip()
        except EOFError:
            break
        if not line:
            break
        print(respond(agent, line), flush=True)


def save_agent(agent, path):
    """A pretrained-backbone agent has NO trained weights -- just the brain data + model names."""
    import json
    json.dump({"concepts": [{k: c[k] for k in ("name", "pos", "syn", "views", "parent", "ancestors")}
                            for c in agent["concepts"]],
               "relations": {k: sorted(map(list, v)) for k, v in agent["relations"].items()},
               "embed_model": agent["bb"].model.name_or_path,
               "gen_model": agent["gen"].lm.name_or_path}, open(path, "w"))


def load_agent(path, device=None):
    import json
    d = json.load(open(path))
    concepts = d["concepts"]
    relations = {k: set(map(tuple, v)) for k, v in d["relations"].items()}
    brain = M.build_brain(concepts, relations)
    banc = M.brain_ancestor_index(brain)
    pnames = sorted({c["parent"] for c in concepts if c["parent"]})
    bb = Backbone(d["embed_model"], device=device)
    cand_emb = bb.embed([parent_gloss(p) for p in pnames])
    gen = Generator(d["gen_model"], device=device)
    return dict(concepts=concepts, relations=relations, brain=brain, banc=banc, pnames=pnames,
                bb=bb, cand_emb=cand_emb, gen=gen, name2i={c["name"]: i for i, c in enumerate(concepts)},
                dfacts=__import__("thinking.roundtrip", fromlist=["direct_facts"]).direct_facts(
                    concepts, relations))


def selftest():
    torch.set_num_threads(2)
    r = run(per_pos=30, seed=0, verbose=False)
    assert r["n"] > 0 and 0.0 <= r["brain_assisted"] <= 1.0, r
    print("backbone selftest OK")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--per-pos", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--device", default=None)
    ap.add_argument("--gen-model", default=GEN_MODEL)
    ap.add_argument("--chat", action="store_true", help="build the full agent and chat (comprehend+write)")
    ap.add_argument("--ask", default=None, help="one-shot prompt to the full agent")
    ap.add_argument("--save", default=None, help="save the agent (brain data + model names)")
    ap.add_argument("--load", default=None, help="load a saved agent")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest(); return 0
    if args.load or args.chat or args.ask:                   # full agent: comprehend + WRITE + chat
        agent = load_agent(args.load, device=args.device) if args.load else \
            build_agent(per_pos=args.per_pos, seed=args.seed, embed_model=args.model,
                        gen_model=args.gen_model, device=args.device)
        print(f"agent ready | {len(agent['concepts'])} concepts | brain {len(agent['brain'].known)} "
              f"kernel-closed facts | embed={args.model} gen={args.gen_model}", flush=True)
        if args.save:
            save_agent(agent, args.save); print(f"saved agent -> {args.save}", flush=True)
        if args.ask:
            print(respond(agent, args.ask), flush=True)
        if args.chat or not args.ask:
            chat(agent)
        return 0
    run(per_pos=args.per_pos, seed=args.seed, model=args.model, topk=args.topk, device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
