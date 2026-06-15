"""RL-in-harness with VERIFIABLE rewards (the Composer-validated agentic technique, small/local).
Policy = char ScratchpadLM. For each task it samples K rollouts of a `$ python3 -c "..."` command;
we EXECUTE each in a sandbox; reward = (1 if output matches the expected result else 0). Dr.GRPO
update: advantage = reward - group_mean (NO std-normalize). Warmup (SFT on transcripts) gives a
non-chance base, then RL improves harness success. Reports success BEFORE vs AFTER RL."""
import sys, torch, torch.nn as nn, numpy as np, random, subprocess, tempfile, shutil
sys.path.insert(0, "/Users/leiro/workspace/llm")
from scratchpad_model import ScratchpadLM

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
ALPHA = "abcdefghij"


def task(rng):
    op = rng.choice(["reverse", "upper", "length", "first", "last"])
    s = "".join(rng.choice(ALPHA) for _ in range(rng.randint(3, 6)))
    expr, out = {
        "reverse": (f"print('{s}'[::-1])", s[::-1]),
        "upper":   (f"print('{s}'.upper())", s.upper()),
        "length":  (f"print(len('{s}'))", str(len(s))),
        "first":   (f"print('{s}'[0])", s[0]),
        "last":    (f"print('{s}'[-1])", s[-1]),
    }[op]
    prompt = f"Task: {op} '{s}'\n"
    full = prompt + f"$ python3 -c \"{expr}\"\n{out}\n"
    return prompt, full, out


CHARS = sorted(set("Task: '\n$ python3 -c\"()[]:.;=+*/abcdefghijklmnopqrstuvwxyz0123456789"))
stoi = {c: i + 1 for i, c in enumerate(CHARS)}; stoi["<p>"] = 0
itos = {i: c for c, i in stoi.items()}; V = len(stoi)
def enc(s): return [stoi.get(c, 0) for c in s]


def run(cmd):
    wd = tempfile.mkdtemp(prefix="rl_")
    try:
        p = subprocess.run(["bash", "-c", cmd], cwd=wd, capture_output=True, text=True, timeout=6)
        return (p.stdout + p.stderr).strip()
    except Exception:
        return ""
    finally:
        shutil.rmtree(wd, ignore_errors=True)


def extract(text):
    for l in text.splitlines():
        if l.strip().startswith("$ "):
            return l.strip()[2:].strip()
    return None


def model():
    m = ScratchpadLM(vocab=V, d=192, layers=4, heads=4, max_len=128, pad=0, loop=False, causal=True)
    return m.to(DEV)


def sample(m, prompt, n_new=60, temp=1.0, greedy=False):
    ids = enc(prompt); n0 = len(ids); gen_pos = []
    with torch.no_grad():
        for _ in range(n_new):
            logits = m(torch.tensor([ids], device=DEV))[0, -1]
            if greedy:
                nx = int(logits.argmax())
            else:
                nx = int(torch.multinomial(torch.softmax(logits / temp, -1), 1))
            ids.append(nx); gen_pos.append(len(ids) - 1)
            if itos.get(nx) == "\n" and "".join(itos.get(i, "") for i in ids[n0:]).count("\n") >= 2:
                break
    return ids, n0


def harness_success(m, rng, trials=40):
    ok = 0
    for _ in range(trials):
        prompt, _, exp = task(rng)
        ids, n0 = sample(m, prompt, greedy=True)
        cmd = extract("".join(itos.get(i, "") for i in ids[n0:]))
        if cmd and run(cmd) == exp:
            ok += 1
    return ok / trials


def main():
    rng = np.random.default_rng(0); pyrng = random.Random(0); torch.manual_seed(0)
    m = model(); opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
    # ---- warmup: SFT on full transcripts (non-chance base) ----
    for s in range(2500):
        full = [task(pyrng)[1] for _ in range(48)]
        L = max(len(f) for f in full)
        batch = torch.zeros(len(full), L, dtype=torch.long, device=DEV)
        for i, f in enumerate(full):
            e = enc(f); batch[i, :len(e)] = torch.tensor(e, device=DEV)
        logits = m(batch[:, :-1])
        loss = nn.functional.cross_entropy(logits.reshape(-1, V), batch[:, 1:].reshape(-1), ignore_index=0)
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 800 == 0: print(f"  warmup {s} loss {loss.item():.3f}", flush=True)
    before = harness_success(m, np.random.default_rng(7))
    print(f"== harness success AFTER warmup (pre-RL): {before:.2f}", flush=True)

    # ---- RL: Dr.GRPO with verifiable execution reward ----
    K = 8
    for step in range(400):
        prompt, _, exp = task(pyrng)
        rollouts, rewards = [], []
        for _ in range(K):
            ids, n0 = sample(m, prompt, temp=1.0)
            cmd = extract("".join(itos.get(i, "") for i in ids[n0:]))
            r = 1.0 if (cmd and run(cmd) == exp) else 0.0
            rollouts.append((ids, n0)); rewards.append(r)
        rewards = np.array(rewards)
        if rewards.std() < 1e-6:  # all same -> no signal
            continue
        adv = rewards - rewards.mean()                      # Dr.GRPO: no std-normalize
        # policy-gradient loss: recompute logprobs of generated tokens WITH grad
        loss = 0.0
        for (ids, n0), a in zip(rollouts, adv):
            if abs(a) < 1e-6: continue
            t = torch.tensor([ids], device=DEV)
            logp = torch.log_softmax(m(t)[0], -1)
            tok_lp = logp[torch.arange(n0 - 1, len(ids) - 1), t[0, n0:]]  # logprob of each generated token
            loss = loss - float(a) * tok_lp.sum()
        loss = loss / K
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0:
            print(f"  RL {step} mean_reward {rewards.mean():.2f}", flush=True)
    after = harness_success(m, np.random.default_rng(7))
    print(f"\n== harness success: pre-RL {before:.2f} -> post-RL {after:.2f} ==", flush=True)


if __name__ == "__main__":
    main()
