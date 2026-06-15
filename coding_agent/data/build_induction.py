"""In-context RULE INDUCTION data (char-level): each block shows K examples of an UNNAMED
rule, then a query the model must answer by INFERRING the rule (not from a keyword). The same
input maps to different outputs across blocks (different rules), so memorizing input->output is
impossible — the model must read the demos and induce. Held-out rules test genuine generalization.

Block format:  in1>out1\nin2>out2\n...\ninK>outK\n   (all same rule; last pair is the query)
Each manifest record = ONE block (force --decode-objective causal). Output: /tmp/induct/manifest.jsonl
"""
import os, json, random

OUT = "/tmp/induct/manifest.jsonl"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
ALPHA = "abcdefghij"

# copy/permutation/selection rules (inducible by copy/induction heads)
RULES = {
    "reverse":  lambda s: s[::-1],
    "double":   lambda s: s + s,
    "dropfirst":lambda s: s[1:],
    "droplast": lambda s: s[:-1],
    "sort":     lambda s: "".join(sorted(s)),
    "rep2":     lambda s: "".join(c * 2 for c in s),
    "head":     lambda s: s[0],
    "tail":     lambda s: s[-1],
    # held-out (NOT trained) — tested by examples only:
    "swapfl":   lambda s: s[-1] + s[1:-1] + s[0] if len(s) > 1 else s,
    "firsttwo": lambda s: s[:2],
    "lasttwo":  lambda s: s[-2:],
    "revsort":  lambda s: "".join(sorted(s, reverse=True)),
}
TRAIN_RULES = ["reverse", "double", "dropfirst", "droplast", "sort", "rep2", "head", "tail"]
HELDOUT_RULES = ["swapfl", "firsttwo", "lasttwo", "revsort"]


def block(rng, rule, k=5, lo=3, hi=5):
    lines = []
    for _ in range(k):
        n = rng.randint(lo, hi)
        s = "".join(rng.choice(ALPHA) for _ in range(n))
        lines.append(f"{s}>{RULES[rule](s)}")
    return "\n".join(lines) + "\n"


def main():
    rng = random.Random(0)
    rows = []
    for i in range(90000):
        rule = rng.choice(TRAIN_RULES)
        b = block(rng, rule)
        chars = list(b)
        h = max(1, len(chars) // 2)
        rows.append({"split": "eval" if i % 50 == 0 else "train",
                     "text": chars[:h], "target": chars[h:],
                     "meta": {"source": "induct", "chunk_index": i}})
    rng.shuffle(rows)
    with open(OUT, "w") as o:
        for r in rows:
            o.write(json.dumps(r) + "\n")
    vocab = sorted(set("".join(c for r in rows for c in r["text"] + r["target"])))
    print(f"rows={len(rows)} vocab={len(vocab)} train_rules={len(TRAIN_RULES)} heldout={HELDOUT_RULES} MB={os.path.getsize(OUT)/1e6:.0f}")
    print("sample block:\n" + block(rng, "reverse"))


if __name__ == "__main__":
    main()
