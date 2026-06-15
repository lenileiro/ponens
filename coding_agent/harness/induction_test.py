"""Test in-context rule induction: give K-1 demos of an unnamed rule + a query, the model
must INFER and apply. Categories: trained rules (new inputs), HELD-OUT rules (never trained,
examples only), and LENGTH-GEN (longer strings than training). Real test of rule-finding."""
import sys, torch, inspect, random
sys.path.insert(0, "/tmp/charwt")
from thinking.multimodal import MultimodalLM
from thinking.trace import Vocab
sys.path.insert(0, "/tmp")
from build_induction import RULES, TRAIN_RULES, HELDOUT_RULES, ALPHA

ck = torch.load("/tmp/inductmodel.pt", map_location="cpu", weights_only=False)
itos = ck["vocab"]; v = Vocab.__new__(Vocab); v.itos = itos
v.stoi = {t: i for i, t in enumerate(itos)}; v.pad, v.unk = 0, 1
m = MultimodalLM(**{k: val for k, val in ck["model_config"].items()
                    if k in inspect.signature(MultimodalLM.__init__).parameters})
m.load_state_dict(ck["state_dict"], strict=False); m.eval()
EMPTY = v.stoi.get("<empty_text>", v.unk)


def gen(prompt, n=12):
    ids = [v.stoi.get(c, v.unk) for c in prompt]
    txt = torch.tensor([[EMPTY]], dtype=torch.long); n0 = len(ids)
    with torch.no_grad():
        for _ in range(n):
            ids.append(int(m([], txt, torch.tensor([ids]), mode="text_only")[0, -1].argmax()))
            if v.itos[ids[-1]] == "\n":
                break
    return "".join(v.itos[i] for i in ids[n0:]).rstrip("\n")


def evaluate(rules, lo, hi, label, seed, trials=60):
    rng = random.Random(seed); ok = 0
    for _ in range(trials):
        rule = rng.choice(rules)
        demos = []
        for _ in range(4):
            s = "".join(rng.choice(ALPHA) for _ in range(rng.randint(lo, hi)))
            demos.append(f"{s}>{RULES[rule](s)}")
        q = "".join(rng.choice(ALPHA) for _ in range(rng.randint(lo, hi)))
        prompt = "\n".join(demos) + f"\n{q}>"
        pred = gen(prompt); want = RULES[rule](q)
        ok += (pred == want)
    print(f"  {label:38} {ok}/{trials} = {ok/trials:.2f}")
    return ok / trials


print("== in-context rule induction ==")
evaluate(TRAIN_RULES, 3, 5, "trained rules, new inputs", 1)
evaluate(HELDOUT_RULES, 3, 5, "HELD-OUT rules (novel, examples only)", 2)
evaluate(TRAIN_RULES, 6, 7, "trained rules, LONGER strings (len-gen)", 3)
