"""Generate grounded bash-ReAct tool-use transcripts in the harness's exact protocol.

Teaches the model: instruction -> write code to a file with `$ cat > f.py`, run it with
`$ python3 f.py`, read output, and (for bug tasks) fix-then-rerun. Code is CORRECT and outputs
are really computed, so the protocol pairs with valid code. Output: /tmp/toolfmt/tool.txt
(plain text, consumed as a 'tool' source by build_codemix). Parametrically varied for breadth.
"""
import os, random, textwrap

OUT = "/tmp/toolfmt/tool.txt"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
SYS = ("You are a coding agent. To run a shell command write a line starting with '$ '. "
       "Write code to files, run it, read errors, and fix bugs.")

VARS = ["x", "n", "k", "val", "data", "num", "s", "arr"]
NAMES = ["compute", "solve", "run", "do_it", "main", "process", "calc"]


def transcript(task, code, fname, expected):
    body = code.rstrip("\n")
    return (f"{SYS}\nTask: {task}\n"
            f"$ cat > {fname} << 'EOF'\n{body}\nEOF\n"
            f"$ python3 {fname}\n{expected}\n")


def gen(rng):
    kind = rng.choice(["factorial", "reverse", "sumlist", "fizz", "isprime", "fixbug", "maxlist", "count"])
    v = rng.choice(VARS); fn = rng.choice(NAMES); f = f"{rng.choice(['sol','main','prog','run'])}.py"
    if kind == "factorial":
        n = rng.randint(3, 9); r = 1
        for i in range(2, n + 1): r *= i
        code = f"def {fn}({v}):\n    r = 1\n    for i in range(2, {v}+1):\n        r *= i\n    return r\nprint({fn}({n}))"
        return transcript(f"Write a function to compute the factorial of {n} and print it.", code, f, r)
    if kind == "reverse":
        s = "".join(rng.choice("abcdefgh") for _ in range(rng.randint(3, 6)))
        code = f"def {fn}({v}):\n    return {v}[::-1]\nprint({fn}({s!r}))"
        return transcript(f"Write a function that reverses the string {s!r} and print it.", code, f, s[::-1])
    if kind == "sumlist":
        xs = [rng.randint(1, 20) for _ in range(rng.randint(3, 5))]
        code = f"def {fn}({v}):\n    return sum({v})\nprint({fn}({xs}))"
        return transcript(f"Write a function that sums the list {xs} and print the result.", code, f, sum(xs))
    if kind == "maxlist":
        xs = [rng.randint(1, 50) for _ in range(rng.randint(3, 5))]
        code = f"def {fn}({v}):\n    return max({v})\nprint({fn}({xs}))"
        return transcript(f"Write a function returning the max of {xs} and print it.", code, f, max(xs))
    if kind == "count":
        s = "".join(rng.choice("aabbcc") for _ in range(rng.randint(4, 8))); ch = rng.choice("abc")
        code = f"print({s!r}.count({ch!r}))"
        return transcript(f"Count occurrences of {ch!r} in {s!r} and print it.", code, f, s.count(ch))
    if kind == "fizz":
        n = rng.randint(3, 6)
        lines = ["Fizz" if i % 3 == 0 else str(i) for i in range(1, n + 1)]
        code = (f"for i in range(1, {n}+1):\n    print('Fizz' if i%3==0 else i)")
        return transcript(f"Print 1..{n}, replacing multiples of 3 with Fizz.", code, f, "\n".join(lines))
    if kind == "isprime":
        n = rng.choice([7, 9, 11, 12, 13, 15])
        isp = n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))
        code = (f"def {fn}({v}):\n    return {v}>1 and all({v}%d for d in range(2,int({v}**0.5)+1))\n"
                f"print({fn}({n}))")
        return transcript(f"Write is-prime and print whether {n} is prime.", code, f, isp)
    # fixbug: model is shown a buggy file + error, then fixes it
    n = rng.randint(2, 6)
    buggy = f"def {fn}({v})\n    return {v}*2\nprint({fn}({n}))"
    fixed = f"def {fn}({v}):\n    return {v}*2\nprint({fn}({n}))"
    return (f"{SYS}\nTask: The file {f} has a syntax error (missing colon). Fix it and run.\n"
            f"$ python3 {f}\n  File \"{f}\", line 1\n    def {fn}({v})\n              ^\nSyntaxError: invalid syntax\n"
            f"$ cat > {f} << 'EOF'\n{fixed}\nEOF\n$ python3 {f}\n{n*2}\n")


def main():
    rng = random.Random(0)
    n = 120000
    with open(OUT, "w") as o:
        for _ in range(n):
            o.write(gen(rng) + "\n")
    mb = os.path.getsize(OUT) / 1e6
    print(f"wrote {n} tool-use transcripts -> {OUT} ({mb:.1f} MB)")
    print("--- sample ---")
    print(open(OUT).read(700))


if __name__ == "__main__":
    main()
