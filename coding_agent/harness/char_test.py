"""Char-level harness test: load the char model, generate `$ python3 -c "..."` from a
compact Task prompt, execute it in a sandbox, check the output. Tests task-conditioning +
executable round-trip on in-distribution NEW instances."""
import sys, re, subprocess, tempfile, shutil, inspect
sys.path.insert(0, "/tmp/charwt")
import torch
from thinking.multimodal import MultimodalLM
from thinking.trace import Vocab

CKPT = "/tmp/charmodel.pt"
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg, itos = ck["model_config"], ck["vocab"]
vocab = Vocab.__new__(Vocab); vocab.itos = itos
vocab.stoi = {t: i for i, t in enumerate(itos)}; vocab.pad, vocab.unk = 0, 1
char_mode = sum(len(t) == 1 for t in itos) > 0.8 * len(itos)
model = MultimodalLM(**{k: v for k, v in cfg.items() if k in inspect.signature(MultimodalLM.__init__).parameters})
model.load_state_dict(ck["state_dict"], strict=False); model.eval()
vd = cfg.get("view_dims", {}) or {}; mode = "text_only" if not vd else "full"
print(f"loaded char model | vocab={len(itos)} char_mode={char_mode}")


EMPTY = vocab.stoi.get("<empty_text>", vocab.unk)


def gen(prompt, n_new=70):
    # causal inference: prompt lives in ids (the decoded sequence); prefix txt is the
    # empty marker (matches how causal training builds decode_text). Extend ids greedily.
    toks = list(prompt) if char_mode else re.findall(r"[a-z]+|[0-9]+|[^\sa-z0-9]", prompt.lower())
    ids = [vocab.stoi.get(t, vocab.unk) for t in toks]
    txt = torch.tensor([[EMPTY]], dtype=torch.long)
    n0 = len(ids)
    with torch.no_grad():
        for _ in range(n_new):
            nxt = int(model([], txt, torch.tensor([ids]), mode=mode)[0, -1].argmax())
            ids.append(nxt)
            if char_mode and "".join(vocab.itos[i] for i in ids[n0:]).count("\n") >= 2:
                break
    return ("" if char_mode else " ").join(vocab.itos[i] for i in ids[n0:])


def run(cmd, timeout=8):
    wd = tempfile.mkdtemp(prefix="ct_")
    try:
        p = subprocess.run(["bash", "-c", cmd], cwd=wd, capture_output=True, text=True, timeout=timeout)
        return (p.stdout + p.stderr).strip()
    except Exception as e:
        return f"ERR {e}"
    finally:
        shutil.rmtree(wd, ignore_errors=True)


TASKS = [
    ("reverse 'gfed'", "defg"), ("uppercase 'abcd'", "ABCD"), ("length 'abcdefg'", "7"),
    ("first 'hij'", "h"), ("last 'hij'", "j"), ("sort 'dbca'", "abcd"), ("double 'ab'", "abab"),
    ("factorial 6", "720"), ("square 8", "64"), ("successor 9", "10"), ("iseven 6", "True"),
    ("sum [5, 10, 15]", "30"), ("max [3, 8, 1]", "8"), ("min [7, 2, 9]", "2"),
    ("sorted [3, 1, 2]", "[1, 2, 3]"), ("count 'a' in 'aabca'", "3"),
    ("fix print('abcd'[::-1)", "dcba"), ("fix print(len('abc')", "3"),
]
solved = 0
for task, expect in TASKS:
    g = gen(f"Task: {task}\n")
    cmd = next((l[2:].strip() for l in g.splitlines() if l.strip().startswith("$ ")), None)
    out = run(cmd) if cmd else ""
    ok = bool(cmd) and expect in out
    solved += ok
    print(f"\nTASK: {task}")
    print(f"  gen: {g[:80]!r}")
    print(f"  cmd: {cmd!r}")
    print(f"  out: {out[:60]!r} | expect {expect!r} -> {'PASS' if ok else 'fail'}")
print(f"\n=== SCORE: {solved}/{len(TASKS)} ===")
