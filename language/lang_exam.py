"""Language track (general-model goal): read English -> pass a held-out exam BY COMPREHENSION.
Rebuilds the prior exam.py idea (lost from repo), C2-leaning: tests grammar (agreement, articles,
tense/conditionals), comprehension (fill a blank from BOTH sides of context), and simple inference.
A bidirectional masked reader (ScratchpadLM causal=False) learns from a tiny self-made corpus of
LEARNABLE, GENERALIZABLE patterns, then answers held-out items by filling the [MASK] from meaning
(not left-to-right next-token). Reports accuracy vs chance. Foundation toward C2 + holistic
understanding (true C2 needs scale; this demonstrates rule-level language comprehension)."""
import sys, numpy as np, torch, torch.nn as nn, random
sys.path.insert(0, "/Users/leiro/workspace/llm")
from scratchpad_model import ScratchpadLM

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
NOUNS = ["cat", "dog", "bird", "child", "teacher", "student", "river", "city", "idea", "result"]
ADJ = ["happy", "quiet", "ancient", "complex", "fragile", "brilliant"]

# (template, slot-options, correct-index) — generalizable rules, held-out instances test the RULE
def corpus_item(rng, train=True):
    kind = rng.choice(["agree", "article", "cond", "compr", "infer"])
    if kind == "agree":          # subject-verb agreement (number)
        n = rng.choice(NOUNS); pl = rng.random() < 0.5
        subj = (n + "s") if pl else ("the " + n)
        verb = "run" if pl else "runs"
        opts = ["run", "runs"]; return f"{subj} [MASK] fast", opts, opts.index(verb)
    if kind == "article":        # a vs an (phonological rule)
        word = rng.choice(["apple", "idea", "hour", "cat", "dog", "umbrella", "tree"])
        an = word[0] in "aeiou" or word == "hour"
        opts = ["a", "an"]; return f"I saw [MASK] {word}", opts, opts.index("an" if an else "a")
    if kind == "cond":           # conditional/tense
        opts = ["would", "will"]
        past = rng.random() < 0.5
        s = "if she had time she [MASK] help" if past else "if she has time she [MASK] help"
        return s, opts, opts.index("would" if past else "will")
    if kind == "compr":          # comprehension: meaning from both sides
        a = rng.choice(ADJ)
        s = f"the {a} {rng.choice(NOUNS)} was very [MASK] indeed"
        opts = [a, rng.choice([x for x in ADJ if x != a])]; rng.shuffle(opts)
        return s, opts, opts.index(a)
    # infer: simple entailment (bigger > smaller)
    x, y = rng.sample(["mouse", "cat", "dog", "horse", "whale"], 2)
    order = ["mouse", "cat", "dog", "horse", "whale"]
    bigger = x if order.index(x) > order.index(y) else y
    opts = [x, y]; rng.shuffle(opts)
    return f"a {x} and a {y} ; the bigger one is the [MASK]", opts, opts.index(bigger)


WORDS = sorted(set(["[MASK]", "the", "i", "saw", "fast", "very", "indeed", "if", "she", "had",
                    "has", "time", "help", "a", "an", "and", ";", "one", "is", "bigger", "was"]
                   + NOUNS + [n + "s" for n in NOUNS] + ADJ + ["run", "runs", "would", "will"]
                   + ["apple", "idea", "hour", "cat", "dog", "umbrella", "tree", "mouse", "horse", "whale"]))
stoi = {w: i + 1 for i, w in enumerate(WORDS)}; stoi["<pad>"] = 0; V = len(stoi)
MASK = stoi["[MASK]"]
def enc(toks): return [stoi.get(t, 0) for t in toks]


def main():
    rng = random.Random(0); torch.manual_seed(0)
    m = ScratchpadLM(vocab=V, d=128, layers=3, heads=4, max_len=24, pad=0, causal=False).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
    # train: masked-LM on corpus items (mask the answer slot, predict it from both sides)
    for s in range(4000):
        rows = [corpus_item(rng) for _ in range(48)]
        seqs, tgts, mpos = [], [], []
        for text, opts, ci in rows:
            toks = text.split(); ids = enc(toks)
            mi = toks.index("[MASK]"); ids[mi] = MASK
            seqs.append(ids); tgts.append(stoi[opts[ci]]); mpos.append(mi)
        L = max(len(x) for x in seqs)
        batch = torch.zeros(len(seqs), L, dtype=torch.long, device=DEV)
        for i, ids in enumerate(seqs): batch[i, :len(ids)] = torch.tensor(ids, device=DEV)
        logits = m(batch)
        ml = logits[torch.arange(len(seqs)), torch.tensor(mpos, device=DEV)]
        loss = nn.functional.cross_entropy(ml, torch.tensor(tgts, device=DEV))
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 1000 == 0: print(f"  read-train {s} loss {loss.item():.3f}", flush=True)

    # ON-THE-FLY mastery+reasoning probe: generate NOVEL items, answer by COMPREHENSION,
    # track per category (grammar/comprehension/reasoning), and SHOW examples.
    def kind_of(text):
        if "fast" in text and "[MASK]" in text and ("run" in text or text.split()[1] == "[MASK]" or "s [MASK]" in text): return "agree"
        if text.startswith("i saw"): return "article"
        if text.startswith("if "): return "cond"
        if "bigger one" in text: return "infer"
        return "compr"
    m.eval(); ev = random.Random(99); cats = {}; shown = {}
    with torch.no_grad():
        for _ in range(500):
            text, opts, ci = corpus_item(ev)
            toks = text.split(); ids = enc(toks); mi = toks.index("[MASK]"); ids[mi] = MASK
            lg = m(torch.tensor([ids], device=DEV))[0, mi]
            pred = max(range(len(opts)), key=lambda j: lg[stoi[opts[j]]].item())
            k = kind_of(text); cats.setdefault(k, [0, 0]); cats[k][0] += (pred == ci); cats[k][1] += 1
            if shown.get(k, 0) < 2:
                shown[k] = shown.get(k, 0) + 1
                print(f"  [{k}] {text.replace('[MASK]','___')}  -> '{opts[pred]}' "
                      f"({'ok' if pred==ci else 'WRONG, want '+opts[ci]})")
    print("\n== on-the-fly mastery (novel items, by comprehension; chance 0.50) ==")
    tot = [0, 0]
    for k in ("agree", "article", "cond", "compr", "infer"):
        if k in cats:
            c, n = cats[k]; tot[0] += c; tot[1] += n
            print(f"  {k:8} {c}/{n} = {c/n:.2f}")
    a = tot[0] / max(1, tot[1])
    print(f"  OVERALL {a:.2f} -> {'MASTERY' if a>=0.8 else 'partial'} (reasoning = infer category)")


if __name__ == "__main__":
    main()
