"""Compact, EXECUTABLE, task<->code-coupled bash transcripts at CHARACTER level.

Each transcript: `Task: <op> <literal>` -> `$ python3 -c "<expr over the same literal>"` -> output.
The literal appears in BOTH task and command, so a small model learns conditioning by COPYING;
one-liners + char-level tokenization round-trip perfectly so the harness can actually run them.
Builds a char-level causal continuation manifest directly. Output: /tmp/chartool/manifest.jsonl
"""
import os, json, random

OUT = "/tmp/chartool/manifest.jsonl"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
W = 80                       # window in CHARS; causal seq = 2*W = 160 chars (fits a transcript)


def _str_op(rng):
    s = "".join(rng.choice("abcdefghij") for _ in range(rng.randint(3, 6)))
    op, (expr, out) = rng.choice([
        ("reverse",   (f"print('{s}'[::-1])", s[::-1])),
        ("uppercase", (f"print('{s}'.upper())", s.upper())),
        ("length",    (f"print(len('{s}'))", len(s))),
        ("first",     (f"print('{s}'[0])", s[0])),
        ("last",      (f"print('{s}'[-1])", s[-1])),
        ("sort",      (f"print(''.join(sorted('{s}')))", "".join(sorted(s)))),
        ("double",    (f"print('{s}'*2)", s * 2)),
    ])
    return f"Task: {op} '{s}'\n$ python3 -c \"{expr}\"\n{out}\n"


def _num_op(rng):
    import math
    n = rng.randint(2, 9)
    op, (expr, out) = rng.choice([
        ("factorial", (f"import math;print(math.factorial({n}))", math.factorial(n))),
        ("square",    (f"print({n}**2)", n ** 2)),
        ("double",    (f"print({n}*2)", n * 2)),
        ("successor", (f"print({n}+1)", n + 1)),
        ("iseven",    (f"print({n}%2==0)", n % 2 == 0)),
    ])
    return f"Task: {op} {n}\n$ python3 -c \"{expr}\"\n{out}\n"


def _list_op(rng):
    xs = [rng.randint(1, 30) for _ in range(rng.randint(2, 4))]
    op, (expr, out) = rng.choice([
        ("sum",    (f"print(sum({xs}))", sum(xs))),
        ("max",    (f"print(max({xs}))", max(xs))),
        ("min",    (f"print(min({xs}))", min(xs))),
        ("length", (f"print(len({xs}))", len(xs))),
        ("sorted", (f"print(sorted({xs}))", sorted(xs))),
    ])
    return f"Task: {op} {xs}\n$ python3 -c \"{expr}\"\n{out}\n"


def transcript(rng):
    op = rng.choice(["str", "str", "num", "num", "list", "list", "count", "fix", "fix"])
    if op == "str":
        return _str_op(rng)
    if op == "num":
        return _num_op(rng)
    if op == "list":
        return _list_op(rng)
    if op == "fix":
        # show a broken one-liner (missing closing ) ] or '), emit the repaired command
        s = "".join(rng.choice("abcdef") for _ in range(rng.randint(3, 5)))
        kind = rng.choice(["paren", "bracket", "quote"])
        if kind == "paren":
            broken = f"print(len('{s}')"            # missing )
            fixed = f"print(len('{s}'))"; out = len(s)
        elif kind == "bracket":
            broken = f"print('{s}'[::-1)"            # missing ]
            fixed = f"print('{s}'[::-1])"; out = s[::-1]
        else:
            broken = f"print('{s}.upper())"          # missing closing quote
            fixed = f"print('{s}'.upper())"; out = s.upper()
        return (f"Task: fix {broken}\n$ python3 -c \"{fixed}\"\n{out}\n")
    if op == "reverse":
        s = "".join(rng.choice("abcdefghij") for _ in range(rng.randint(3, 6)))
        return f"Task: reverse {s!r}\n$ python3 -c \"print({s!r}[::-1])\"\n{s[::-1]}\n"
    if op == "factorial":
        import math
        n = rng.randint(3, 9)
        return f"Task: factorial {n}\n$ python3 -c \"import math;print(math.factorial({n}))\"\n{math.factorial(n)}\n"
    if op == "sum":
        xs = [rng.randint(1, 20) for _ in range(rng.randint(2, 4))]
        return f"Task: sum {xs}\n$ python3 -c \"print(sum({xs}))\"\n{sum(xs)}\n"
    if op == "max":
        xs = [rng.randint(1, 50) for _ in range(rng.randint(2, 4))]
        return f"Task: max {xs}\n$ python3 -c \"print(max({xs}))\"\n{max(xs)}\n"
    if op == "count":
        s = "".join(rng.choice("aabbc") for _ in range(rng.randint(4, 7))); ch = rng.choice("abc")
        return f"Task: count {ch!r} in {s!r}\n$ python3 -c \"print({s!r}.count({ch!r}))\"\n{s.count(ch)}\n"
    if op == "length":
        s = "".join(rng.choice("abcdefg") for _ in range(rng.randint(2, 8)))
        return f"Task: length of {s!r}\n$ python3 -c \"print(len({s!r}))\"\n{len(s)}\n"
    s = "".join(rng.choice("abcdef") for _ in range(rng.randint(3, 6)))
    return f"Task: uppercase {s!r}\n$ python3 -c \"print({s!r}.upper())\"\n{s.upper()}\n"


def main():
    rng = random.Random(0)
    stream = "".join(transcript(rng) for _ in range(80000))
    chars = list(stream)
    rows = []
    ci = 0
    for i in range(0, len(chars) - W, W):
        tgt = chars[i + W:i + 2 * W]
        if not tgt:
            break
        rows.append({"split": "eval" if ci % 50 == 0 else "train",
                     "text": chars[i:i + W], "target": tgt,
                     "meta": {"source": "tool", "chunk_index": ci}})
        ci += 1
    rng.shuffle(rows)
    with open(OUT, "w") as o:
        for r in rows:
            o.write(json.dumps(r) + "\n")
    vocab = sorted(set(chars))
    print(f"rows={len(rows)} chars={len(chars)} W={W} (seq {2*W}) vocab={len(vocab)} manifestMB={os.path.getsize(OUT)/1e6:.0f}")
    print("sample transcript:\n" + stream[:120])


if __name__ == "__main__":
    main()
