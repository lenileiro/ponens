"""Small bash-using coding-agent harness for the trained thinking model.

A ReAct-style loop: the model proposes shell commands (lines beginning with `$ ` or
fenced ```bash blocks); the harness executes them in a sandboxed temp workdir with a
timeout + denylist, feeds stdout/stderr back into the context, and loops until the task's
success check passes or max_steps is hit. Model-agnostic: works with any
{state_dict, vocab, model_config} multimodal/causal checkpoint.

This is the acceptance rig for: write code, run it, read errors, fix bugs, suggest fixes.
Usage: python /tmp/coding_agent.py <checkpoint.pt> [--task all|factorial|fixbug|...]
"""
import sys, os, re, json, subprocess, tempfile, shutil, argparse

sys.path.insert(0, "/Users/leiro/workspace/llm")
import torch
from thinking.multimodal import MultimodalLM, _batch_from_records, load_manifest
from thinking.trace import Vocab

TOK = re.compile(r"[a-z]+|[0-9]+|[^\sa-z0-9]")
DENY = re.compile(r"\brm\s+-rf\s+/|\bmkfs|\b:\(\)\s*\{|\bdd\s+if=|/dev/sd|\bshutdown|\breboot|\bcurl\b.*\|\s*sh|\bsudo\b")


# ---------------- model loading + generation ----------------
def load_agent_model(ckpt_path):
    import inspect
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg, itos = ck["model_config"], ck["vocab"]
    vocab = Vocab.__new__(Vocab)
    vocab.itos = itos
    vocab.stoi = {t: i for i, t in enumerate(itos)}
    vocab.pad, vocab.unk = 0, 1
    sig = inspect.signature(MultimodalLM.__init__)
    model = MultimodalLM(**{k: v for k, v in cfg.items() if k in sig.parameters})
    model.load_state_dict(ck["state_dict"], strict=False)
    model.eval()
    vd = cfg.get("view_dims", {}) or {}
    return model, vocab, vd, ("text_only" if not vd else "full")


def generate(model, vocab, vd, mode, prompt, n_new=64, temp=0.7, seed=0):
    # CAUSAL inference: the prompt lives in ids (the decoded sequence); the prefix txt is the
    # empty marker, matching how causal training builds decode_text. (Feeding the prompt into
    # txt instead silently breaks task-conditioning.) Char-aware decode.
    g = torch.Generator().manual_seed(seed)
    char_mode = sum(len(t) == 1 for t in vocab.itos) > 0.8 * len(vocab.itos)
    toks = list(prompt) if char_mode else (TOK.findall(prompt.lower()) or ["<empty_text>"])
    ids = [vocab.stoi.get(t, vocab.unk) for t in toks]
    txt = torch.tensor([[vocab.stoi.get("<empty_text>", vocab.unk)]], dtype=torch.long)
    n0 = len(ids)
    with torch.no_grad():
        for _ in range(n_new):
            logits = model([], txt, torch.tensor([ids]), mode=mode)[0, -1]
            if temp <= 0:
                nxt = int(logits.argmax())
            else:
                p = torch.softmax(logits / temp, -1)
                nxt = int(torch.multinomial(p, 1, generator=g))
            ids.append(nxt)
            if char_mode and "".join(vocab.itos[i] for i in ids[n0:]).count("\n") >= 3:
                break
    return ("" if char_mode else " ").join(vocab.itos[i] for i in ids[n0:])


# ---------------- bash execution sandbox ----------------
def run_bash(cmd, workdir, timeout=10):
    if DENY.search(cmd):
        return "", "BLOCKED: command matched safety denylist", 126
    try:
        p = subprocess.run(["bash", "-c", cmd], cwd=workdir, capture_output=True,
                           text=True, timeout=timeout,
                           env={**os.environ, "HOME": workdir})
        return p.stdout[-2000:], p.stderr[-2000:], p.returncode
    except subprocess.TimeoutExpired:
        return "", f"TIMEOUT after {timeout}s", 124


def detok(text):
    """Reconstruct code-ish text from space-joined word-level tokens:
    drop spaces around punctuation that binds (./_()[]{}:,;=<>) so 'sol . py'->'sol.py',
    'print ( x )'->'print(x)'. Imperfect (word-level tokenizer limitation) but lets simple
    commands execute."""
    t = text
    t = re.sub(r"\s+([.\,;:)\]}>])", r"\1", t)      # no space before these
    t = re.sub(r"([.([{<_/])\s+", r"\1", t)          # no space after these
    t = re.sub(r"\s*([_/=])\s*", r"\1", t)            # bind around _ / =
    return t


def extract_commands(text):
    text = detok(text)
    cmds = []
    for m in re.finditer(r"```(?:bash|sh)?\n(.*?)```", text, re.S):
        cmds += [l.strip() for l in m.group(1).splitlines() if l.strip()]
    for l in text.splitlines():
        s = l.strip()
        if s.startswith("$ "):
            cmds.append(s[2:].strip())
    return cmds


# ---------------- ReAct agent loop ----------------
SYSTEM = ("You are a coding agent. To run a shell command write a line starting with "
          "'$ '. Write code to files with cat, run it, read errors, and fix bugs.\n")


def agent_loop(model, vocab, vd, mode, task, max_steps=4, temp=0.7, verbose=True):
    workdir = tempfile.mkdtemp(prefix="agent_")
    if task.get("setup"):
        run_bash(task["setup"], workdir)
    ctx = SYSTEM + "Task: " + task["prompt"] + "\n"
    transcript = []
    try:
        for step in range(max_steps):
            gen = generate(model, vocab, vd, mode, ctx, n_new=task.get("n_new", 64),
                           temp=temp, seed=step)
            cmds = extract_commands(gen)
            transcript.append({"step": step, "model": gen, "cmds": cmds})
            if verbose:
                print(f"  [step {step}] model: {gen[:120]}")
                print(f"           parsed cmds: {cmds[:3]}")
            for c in cmds[:3]:
                out, err, rc = run_bash(c, workdir)
                ctx += f"$ {c}\n{out}{err}\n"
                transcript[-1].setdefault("runs", []).append({"cmd": c, "rc": rc, "out": out[:200], "err": err[:200]})
            if task.get("check"):
                ok, _, _ = run_bash(task["check"], workdir)
                if "PASS" in ok:
                    return {"task": task["name"], "solved": True, "steps": step + 1, "transcript": transcript}
        solved = False
        if task.get("check"):
            ok, _, _ = run_bash(task["check"], workdir)
            solved = "PASS" in ok
        return {"task": task["name"], "solved": solved, "steps": max_steps, "transcript": transcript}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# ---------------- coding / debug / fix tasks ----------------
TASKS = [
    {"name": "factorial",
     "prompt": "Create fact.py with a function factorial(n) and print factorial(5).",
     "check": "python3 fact.py 2>/dev/null | grep -q 120 && echo PASS"},
    {"name": "fixbug",
     "prompt": "The file buggy.py has a syntax error. Fix it so it runs and prints OK.",
     "setup": "printf 'def f(:\\n    print(\"OK\")\\nf()\\n' > buggy.py",
     "check": "python3 buggy.py 2>/dev/null | grep -q OK && echo PASS"},
    {"name": "reverse",
     "prompt": "Write rev.py: a function that reverses a string, print rev('abc').",
     "check": "python3 rev.py 2>/dev/null | grep -q cba && echo PASS"},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--task", default="all")
    ap.add_argument("--max-steps", type=int, default=4)
    ap.add_argument("--temp", type=float, default=0.7)
    args = ap.parse_args()
    model, vocab, vd, mode = load_agent_model(args.checkpoint)
    print(f"loaded {args.checkpoint} | vocab={len(vocab.itos)} mode={mode}")
    tasks = TASKS if args.task == "all" else [t for t in TASKS if t["name"] == args.task]
    results = []
    for t in tasks:
        print(f"\n=== TASK: {t['name']} ===")
        r = agent_loop(model, vocab, vd, mode, t, max_steps=args.max_steps, temp=args.temp)
        print(f"  -> solved={r['solved']} in {r['steps']} steps")
        results.append(r)
    n_solved = sum(r["solved"] for r in results)
    print(f"\n=== SCORE: {n_solved}/{len(results)} tasks solved ===")
    return results


if __name__ == "__main__":
    main()
