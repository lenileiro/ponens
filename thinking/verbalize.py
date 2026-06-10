"""VERBALIZER: expressive English output, learned OUTSIDE the task data.

Separation of concerns: the reasoner thinks in the canonical, checkable trace language (a proof
doesn't need style); the verbalizer renders the VERIFIED conclusion + derivation into fluent
English. Its linguistic competence comes from PRETRAINING on broad real English (CLOTH, the
encyclopedic corpus) -- the task pairs only teach it to ground that competence in a trace.

  phase A  byte-BPE LM pretraining on broad English        (fluency: outside the task data)
  phase B  finetune on (trace -> explanation) pairs        (grounding: small, task-specific)

Seed explanations are compositional realizations (multiple syntactic frames); enrich with
frontier paraphrases via `claude -p` (gen_corpus pattern) for full diversity:
  python -m thinking.verbalize --steps 4000 --out verbalizer.pt
  python -m thinking.verbalize --sample verbalizer.pt
"""
import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scratchpad_model import ScratchpadLM
from device import get_device

DEV = get_device()

# compositional explanation frames (seed surface; frontier paraphrase extends this)
REL_FRAMES = [
    "{x} is {z} 's {rel} . the chain is clear : {steps} .",
    "looking at the family tree , {x} turns out to be the {rel} of {z} , because {steps} .",
    "{x} is the {rel} of {z} -- {steps} , which settles it .",
    "the records show that {steps} ; that makes {x} the {rel} of {z} .",
]
VAL_FRAMES = [
    "the answer is {ans} . {steps} , so the arithmetic gives {ans} .",
    "{steps} . subtracting the years , we get {ans} .",
    "since {steps} , the difference comes to {ans} years .",
]
STEP_FRAMES = {
    "mother": "{h} is {t} 's mother", "father": "{h} is {t} 's father",
    "sister": "{h} is a sister of {t}", "brother": "{h} is a brother of {t}",
    "spouse": "{h} is married to {t}", "born": "{h} was born in {t}",
    "died": "{h} died in {t}", "parent": "{h} is a parent of {t}",
    "married": "{h} and {t} are married", "sibling": "{h} and {t} are siblings",
}


def _steps_text(lines, rng):
    parts = []
    for typ, head, body in lines:
        if typ != "check":
            continue
        pred, (h, t) = head
        parts.append(STEP_FRAMES.get(pred, "{h} %s {t}" % pred).format(h=h, t=t))
    if len(parts) > 1 and rng.random() < 0.5:
        return " , ".join(parts[:-1]) + " , and " + parts[-1]
    return " ; ".join(parts)


def gen_pairs(n, seed=0):
    """(trace_text, explanation) pairs from verified gold derivations."""
    from .kinship import FamilyWorld, name_pools, VALUE_PREDS
    rng = np.random.default_rng(seed)
    world = FamilyWorld(name_pools(300, 60, seed)[0], seed=seed)
    pairs = []
    for _ in range(n):
        k = [2, 3, 4, 5][int(rng.integers(4))]
        p, lines = world.sample(k, rng)
        trace = " ".join(" ".join([typ] + list(sum(([a[1][0], a[0], a[1][1]] for a in
                         ((head,) + tuple(body))), []))) + " ." for typ, head, body in lines)
        steps = _steps_text(lines, rng)
        x, z = p.goal[1]
        if p.goal[0] in VALUE_PREDS:
            frame = VAL_FRAMES[int(rng.integers(len(VAL_FRAMES)))]
            expl = frame.format(ans=p.answer, steps=steps)
        else:
            frame = REL_FRAMES[int(rng.integers(len(REL_FRAMES)))]
            expl = frame.format(x=x, z=z, rel=p.answer.replace("_", " "), steps=steps)
        pairs.append((f"trace {trace} explain", expl + " <end>"))
    return pairs


CORPUS_URLS = {
    # TinyStories V2 (GPT-4-written): THE tiny-model English corpus -- <10M-param models trained
    # on it produce fluent coherent English (Eldan & Li). Right register for a verbalizer.
    "tinystories": ("https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
                    "TinyStoriesV2-GPT4-train.txt", "txt", None),
    # Cosmopedia (HuggingFaceTB): Mixtral-written textbooks/stories/wikihow; richer register
    # than TinyStories while still synthetic-clean. 100k subset.
    "cosmopedia": ("https://huggingface.co/datasets/HuggingFaceTB/cosmopedia-100k/resolve/main/"
                   "data/train-00000-of-00002.parquet", "parquet", "text"),
}
# (FineWeb-Edu evaluated and rejected for this loader: smallest shard is 2.1GB and parquet
#  cannot be range-read -- add via the `datasets` library if that register is ever needed.)


def _fetch(source, cap_mb, root):
    url, fmt, col = CORPUS_URLS[source]
    cache = os.path.join(root, "data", f"{source}_{cap_mb}mb.txt")
    if os.path.exists(cache):
        return open(cache).read()
    import urllib.request
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "fer"})
    if fmt == "txt":
        with urllib.request.urlopen(req, timeout=180) as r:
            txt = r.read(cap_mb * 1024 * 1024).decode("utf-8", errors="ignore").lower()
    else:                                                  # parquet (pip install pandas pyarrow)
        import io
        import pandas as pd
        with urllib.request.urlopen(req, timeout=300) as r:
            df = pd.read_parquet(io.BytesIO(r.read()))
        txt = "\n\n".join(df[col].astype(str)).lower()[:cap_mb * 1024 * 1024]
    txt = txt[:txt.rfind(".") + 1]                         # end on a sentence boundary
    with open(cache, "w") as f:
        f.write(txt)
    return txt


def build_corpus(source="tinystories", cap_mb=24):
    """Broad English for phase A (linguistic competence from OUTSIDE the task data).
    Comma-separate to MIX registers (e.g. 'tinystories,cosmopedia'); cap is per source.
    Downloads + caches under data/; fails loudly rather than degrading silently."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parts = []
    for s in source.split(","):
        if s not in CORPUS_URLS:
            raise ValueError(f"unknown corpus {s!r}; available: {sorted(CORPUS_URLS)}")
        parts.append(_fetch(s, cap_mb, root))
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="verbalizer.pt")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--pre-steps", type=int, default=2000, dest="pre_steps")
    ap.add_argument("--pairs", type=int, default=3000)
    ap.add_argument("--corpus", default="tinystories")
    ap.add_argument("--corpus-mb", type=int, default=24, dest="corpus_mb")
    ap.add_argument("--sample", help="load a checkpoint and sample explanations")
    args = ap.parse_args()
    from model_io import BPEVocab

    if args.sample:
        ck = torch.load(args.sample, map_location=DEV, weights_only=False)
        m = ScratchpadLM(**ck["config"]).to(DEV)
        m.load_state_dict(ck["state_dict"])
        m.eval()
        V = BPEVocab(json_str=ck["tokenizer"])
        for trace, _ in gen_pairs(3, seed=99):
            ids = V.enc(trace)
            for _ in range(80):
                logits = m(torch.tensor([ids[-m.config["max_len"]:]], device=DEV))[0, -1]
                t = int(logits.argmax())
                ids.append(t)
                if "<end>" in V.decode(ids[-4:]):
                    break
            print(f">>> {trace[:90]}...\n    {V.decode(ids)[len(trace):]}\n")
        return

    corpus = build_corpus(args.corpus, args.corpus_mb)
    pairs = gen_pairs(args.pairs)
    print(f"corpus {len(corpus):,} chars | {len(pairs)} trace->explanation pairs")
    V = BPEVocab(texts=[corpus] + [a + " " + b for a, b in pairs], vocab_size=6000)
    m = ScratchpadLM(len(V), d=256, heads=8, pad=V.pad, max_len=512, loop=False,
                     pointer=True).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01)
    stream = torch.tensor(V.enc(corpus), dtype=torch.long)
    m.train()
    B, L = 16, 256
    print("phase A: broad-English pretraining (fluency from outside the task data)")
    for st in range(args.pre_steps):
        ix = torch.randint(0, max(1, stream.numel() - L - 1), (B,))
        x = torch.stack([stream[i:i + L + 1] for i in ix]).to(DEV)
        loss = F.cross_entropy(m(x[:, :-1]).reshape(-1, len(V)), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if st % max(1, args.pre_steps // 5) == 0:
            print(f"  pre {st}/{args.pre_steps} loss {loss.item():.3f}", flush=True)
    print("phase B: grounding finetune on (trace -> explanation)")
    enc_pairs = [V.enc(a + " " + b) for a, b in pairs]
    rng = np.random.default_rng(0)
    for st in range(args.steps):
        x = torch.full((B, 513), V.pad, dtype=torch.long)
        for r in range(B):
            s = enc_pairs[int(rng.integers(len(enc_pairs)))][:513]
            x[r, :len(s)] = torch.tensor(s)
        x = x.to(DEV)
        loss = F.cross_entropy(m(x[:, :-1]).reshape(-1, len(V)), x[:, 1:].reshape(-1),
                               ignore_index=V.pad)
        opt.zero_grad(); loss.backward(); opt.step()
        if st % max(1, args.steps // 10) == 0:
            print(f"  ft {st}/{args.steps} loss {loss.item():.3f}", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"state_dict": m.state_dict(), "config": m.config, "tokenizer": V.to_json()},
               args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
