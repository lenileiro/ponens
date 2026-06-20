"""write_lm.py -- Phase 3 WRITING track: a small char-level causal LM that learns to
WRITE fluent English by reading a corpus, then (stretch) writes a short REASONED answer.

Self-contained. Reuses the repo engine `ScratchpadLM` as a plain causal decoder
(causal=True, pos_mode="rope", pointer=False). CPU only, short runs.

Corpus: reads ~5MB from /tmp/corpus1gb/*.txt if present, else a bundled tiny
public-domain-style English primer string.

Usage:
    python -m thinking.write_lm --selftest
    python -m thinking.write_lm --steps 3000 --out /tmp/writer.pt
    python -m thinking.write_lm --steps 3000 --reason   # also run reasoned-writing stretch
"""
import argparse
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

# coexist with another agent on the same box
torch.set_num_threads(int(os.environ.get("WRITE_LM_THREADS", "4")))

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scratchpad_model import ScratchpadLM  # noqa: E402


# ----------------------------------------------------------------------------- corpus
_BUNDLED = (
    "The sun rose over the quiet village and the people woke to begin their day. "
    "A small boy walked to the river to fetch water for his mother. "
    "On the way he met an old man who was sitting under a great oak tree. "
    "The old man smiled and said that the morning was the best part of the day. "
    "The boy nodded and went on his way, carrying the heavy bucket with both hands. "
    "When he came home his mother thanked him and gave him a warm piece of bread. "
    "After breakfast the children of the village ran out to play in the green fields. "
    "They chased each other through the tall grass and laughed until the sun was high. "
    "In the afternoon a gentle rain began to fall and the people went inside their homes. "
    "They lit small fires and told stories of the old days while the rain tapped the roof. "
    "An old woman spoke of a time when the river was wide and the forest was full of deer. "
    "The young ones listened with wide eyes and asked her many questions about the past. "
    "She told them that the land had always given the village what it needed to live. "
    "When the rain stopped the sky turned a soft gold and the birds began to sing again. "
    "The boy went back to the river and watched the water flow over the smooth stones. "
    "He thought about the words of the old man and decided that he too loved the morning. "
    "That night the village gathered around a large fire in the center of the square. "
    "They shared a simple meal of bread and fish and the warm soup that the women had made. "
    "The men played soft music on wooden pipes and the children danced until they were tired. "
    "One by one the families said good night and walked home under a wide field of stars. "
    "The boy lay in his bed and listened to the wind move through the leaves of the trees. "
    "He was glad to live in such a place and he closed his eyes and fell into a deep sleep. "
    "In his dream he walked along the river and the old man was there once more. "
    "The old man told him that every day was a gift and that he should use it well. "
    "When the boy woke the sun had risen again and a new day was waiting for him to begin. "
)


def load_corpus(max_bytes=5_000_000):
    d = "/tmp/corpus1gb"
    if os.path.isdir(d):
        # A.txt is a Middle-English dictionary (headwords/abbrevs) -> bad for fluent
        # prose; prefer the Project Gutenberg narrative files (B1/B2/C1/...).
        files = sorted(f for f in os.listdir(d)
                       if f.endswith(".txt") and not f.startswith("A"))
        if not files:
            files = sorted(f for f in os.listdir(d) if f.endswith(".txt"))
        if files:
            parts, total = [], 0
            for f in files:
                with open(os.path.join(d, f), "r", errors="ignore") as fh:
                    chunk = fh.read(max_bytes - total)
                parts.append(chunk)
                total += len(chunk)
                if total >= max_bytes:
                    break
            text = "".join(parts)
            if len(text) > 1000:
                return text, f"/tmp/corpus1gb ({total/1e6:.1f}MB)"
    # bundled fallback, repeated so there is enough to train on
    text = _BUNDLED * 40
    return text, f"bundled primer (x40, {len(text)/1e3:.0f}KB)"


class CharVocab:
    def __init__(self, text):
        # pad=0 reserved by the engine; build vocab starting at 1
        chars = sorted(set(text))
        self.itos = ["\x00"] + chars  # index 0 = pad
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.pad = 0

    @property
    def size(self):
        return len(self.itos)

    def encode(self, s):
        return [self.stoi.get(c, self.pad) for c in s]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids if 0 <= i < len(self.itos) and i != self.pad)


# ----------------------------------------------------------------------------- model
def build_model(vocab_size, d=256, layers=4, heads=4, max_len=256):
    return ScratchpadLM(
        vocab=vocab_size, d=d, layers=layers, heads=heads, max_len=max_len,
        pos_mode="rope", causal=True, pointer=False, loop=False,
    )


def get_batch(data, bs, seqlen, device):
    n = data.size(0)
    ix = torch.randint(0, n - seqlen - 1, (bs,))
    x = torch.stack([data[i:i + seqlen] for i in ix]).to(device)
    y = torch.stack([data[i + 1:i + seqlen + 1] for i in ix]).to(device)
    return x, y


@torch.no_grad()
def generate(model, vocab, prompt, n_new=160, temp=1.0, max_len=256, greedy=False):
    model.eval()
    dev = next(model.parameters()).device
    ids = vocab.encode(prompt)
    if not ids:
        ids = [vocab.stoi.get(" ", 1)]
    for _ in range(n_new):
        ctx = ids[-(max_len - 1):]
        x = torch.tensor([ctx], device=dev)
        logits = model(x)[0, -1]
        if greedy:
            nxt = int(torch.argmax(logits))
        else:
            probs = F.softmax(logits / max(temp, 1e-5), dim=-1)
            nxt = int(torch.multinomial(probs, 1))
        if nxt == vocab.pad:
            # avoid emitting pad; pick next-best non-pad
            logits[vocab.pad] = -1e9
            nxt = int(torch.argmax(logits))
        ids.append(nxt)
    return vocab.decode(ids)


def train(model, data, vocab, steps, device, bs=24, seqlen=128, lr=3e-3, log_every=200):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    model.train()
    curve = []
    t0 = time.time()
    seqlen = min(seqlen, model.config.get("max_len", 256) - 1)
    for step in range(1, steps + 1):
        x, y = get_batch(data, bs, seqlen, device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               y.reshape(-1), ignore_index=vocab.pad)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % log_every == 0 or step == 1:
            el = time.time() - t0
            ppl = math.exp(min(loss.item(), 20))
            print(f"  step {step:5d}/{steps}  loss {loss.item():.4f}  ppl {ppl:7.2f}  "
                  f"({el:.0f}s, {step/el:.1f} it/s)", flush=True)
            curve.append((step, loss.item()))
    return curve


# ----------------------------------------------------------------------------- reasoned writing (stretch)
# Generate (fact-set + question -> English answer sentence) examples, fine-tune the
# same LM on them, then prompt with held-out questions and read the written answer.
_NAMES = ["Anna", "Ben", "Cara", "Dan", "Eve", "Finn", "Gail", "Hugo"]
_CITIES = ["Paris", "Rome", "Cairo", "Tokyo", "Lima", "Oslo", "Delhi", "Quito"]
_JOBS = ["a doctor", "a baker", "a teacher", "a sailor", "a farmer", "a painter"]


def make_reason_example(rng):
    a, b = rng.sample(_NAMES, 2)
    city = rng.choice(_CITIES)
    job = rng.choice(_JOBS)
    facts = f"{a} lives in {city}. {a} is {job}. {b} is the friend of {a}. "
    qkind = rng.choice(["where", "job", "friend"])
    if qkind == "where":
        q = f"Where does {a} live? "
        ans = f"{a} lives in {city}."
    elif qkind == "job":
        q = f"What is the job of {a}? "
        ans = f"{a} is {job}."
    else:
        q = f"Who is the friend of {a}? "
        ans = f"{b} is the friend of {a}."
    return facts + q + "Answer: " + ans + "\n"


def reasoned_writing(model, vocab, device, steps=1500, log_every=200):
    import random
    rng = random.Random(0)
    print("\n[stretch] reasoned-writing fine-tune", flush=True)
    seqlen = min(160, model.config.get("max_len", 256) - 1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    model.train()
    for step in range(1, steps + 1):
        exs = [make_reason_example(rng) for _ in range(24)]
        enc = [vocab.encode(e) for e in exs]
        m = min(seqlen, max(len(e) for e in enc))
        xb, yb = [], []
        for e in enc:
            e = e[:m + 1]
            e = e + [vocab.pad] * (m + 1 - len(e))
            xb.append(e[:m]); yb.append(e[1:m + 1])
        x = torch.tensor(xb, device=device); y = torch.tensor(yb, device=device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               y.reshape(-1), ignore_index=vocab.pad)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % log_every == 0 or step == 1:
            print(f"  [reason] step {step:5d}/{steps}  loss {loss.item():.4f}", flush=True)
    # held-out eval: build fresh prompts, write the answer greedily up to newline
    import random as _r
    rng2 = _r.Random(999)
    print("\n[stretch] reasoned-writing held-out samples (greedy, stop at newline):", flush=True)
    correct = 0; total = 0
    model.eval()
    for _ in range(6):
        ex = make_reason_example(rng2)
        prompt, gold = ex.split("Answer: ")
        prompt = prompt + "Answer: "
        out = generate(model, vocab, prompt, n_new=60, greedy=True,
                       max_len=model.config.get("max_len", 256))
        written = out[len(prompt):].split("\n")[0].strip()
        gold = gold.strip()
        ok = written == gold
        correct += ok; total += 1
        print(f"  Q: {prompt.split('. ')[-1].replace('Answer: ','').strip()}", flush=True)
        print(f"     written: {written!r}", flush=True)
        print(f"     gold:    {gold!r}   {'OK' if ok else 'x'}", flush=True)
    print(f"[stretch] reasoned-writing exact-match: {correct}/{total}", flush=True)
    return correct, total


# ----------------------------------------------------------------------------- selftest / main
def selftest():
    torch.manual_seed(0)
    text = _BUNDLED * 2
    vocab = CharVocab(text)
    data = torch.tensor(vocab.encode(text), dtype=torch.long)
    model = build_model(vocab.size, d=64, layers=2, heads=4, max_len=64)
    dev = torch.device("cpu")
    model.to(dev)
    curve = train(model, data, vocab, steps=30, device=dev, bs=8, seqlen=48,
                  lr=3e-3, log_every=10)
    s = generate(model, vocab, "The ", n_new=20, greedy=True, max_len=64)
    assert isinstance(s, str) and len(s) >= 4
    assert curve[-1][1] < curve[0][1] + 1.0  # loss not exploding
    print("OK")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--seqlen", type=int, default=128)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--reason", action="store_true", help="run reasoned-writing stretch")
    ap.add_argument("--max-bytes", type=int, default=5_000_000)
    args = ap.parse_args(argv)

    if args.selftest:
        selftest()
        return

    torch.manual_seed(0)
    text, src = load_corpus(args.max_bytes)
    vocab = CharVocab(text)
    data = torch.tensor(vocab.encode(text), dtype=torch.long)
    dev = torch.device("cpu")
    print(f"corpus: {src}  chars={len(text):,}  vocab={vocab.size}", flush=True)
    model = build_model(vocab.size, d=args.d, layers=args.layers,
                        heads=args.heads, max_len=max(256, args.seqlen + 1))
    model.to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: ScratchpadLM d={args.d} layers={args.layers} heads={args.heads} "
          f"params={n_params/1e6:.2f}M  (causal, rope, no-pointer)", flush=True)

    print(f"\ntraining {args.steps} steps:", flush=True)
    curve = train(model, data, vocab, steps=args.steps, device=dev,
                  bs=args.bs, seqlen=args.seqlen, log_every=200)

    print("\nGENERATED SAMPLES:", flush=True)
    for prompt in ["The ", "Once upon", "She said"]:
        g = generate(model, vocab, prompt, n_new=200, greedy=True,
                     max_len=model.config.get("max_len", 256))
        print(f"\n[greedy] prompt={prompt!r}\n{g!r}", flush=True)
    for prompt, t in [("The ", 0.7), ("Once upon", 0.8)]:
        g = generate(model, vocab, prompt, n_new=200, temp=t,
                     max_len=model.config.get("max_len", 256))
        print(f"\n[temp={t}] prompt={prompt!r}\n{g!r}", flush=True)

    if args.reason:
        reasoned_writing(model, vocab, dev)

    if args.out:
        torch.save({"model": model.state_dict(), "config": model.config,
                    "itos": vocab.itos}, args.out)
        print(f"\nsaved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
