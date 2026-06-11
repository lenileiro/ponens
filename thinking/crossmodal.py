"""M-1: cross-modal FER probe — does a concept HEARD align with the same concept SEEN?

The defining FER question for a multimodal model. The word "red" arrives two ways: as pixels (a
red shape) and as speech (the spoken word "red", via the A-2 say-bank). A UNIFIED (factored)
representation places both near a single shared "red" concept, separable from the modality they
came in through; a FRACTURED one keeps two disjoint clusters (a vision-red and an audio-red that
never meet) -- the multimodal incarnation of the paper's split-concept pathology.

ONE encoder per modality feeds a SHARED concept space. We measure:
  - cross-modal alignment: cos(seen c, heard c) vs cos(seen c, heard c')  [same-concept margin]
  - modality leakage: are clusters organized by CONCEPT (UFR) or by MODALITY (FER)?
  - cross-modal retrieval: given a heard word, is the nearest seen concept the right one?

Arms: shared (both modalities project to one space, trained to align matching concepts) vs
separate (independent spaces, no alignment pressure -- the fractured baseline).

  python -m thinking.crossmodal --selftest
  python -m thinking.crossmodal --steps 600 --out runs/m1_crossmodal.json   (needs A-2 bank)
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from device import get_device

from .audio import spectrogram
from .vision import COLORS, SHAPES, render_object, sample_object
from .listen import WORDS, load_bank, _feats

DEV = get_device()
# concepts that exist in BOTH modalities: the color/shape words A-2 speaks AND vision renders
SHARED_CONCEPTS = [w for w in WORDS if w in COLORS or w in SHAPES]


class VisionTower(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(48, 64, 3, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d(1))
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(64, dim))

    def forward(self, x):
        return self.proj(self.conv(x))


class AudioTower(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 24, 3, padding=1), nn.GELU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(24, 48, 3, padding=1), nn.GELU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(48, 64, 3, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d((4, 4)))
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(64 * 16, dim))

    def forward(self, x):
        return self.proj(self.conv(x))


class CrossModal(nn.Module):
    def __init__(self, arm="shared", dim=64):
        super().__init__()
        self.arm = arm
        self.vis = VisionTower(dim)
        self.aud = AudioTower(dim)
        self.concept = nn.Linear(dim, len(SHARED_CONCEPTS))   # shared classifier head

    def embed_vision(self, x):
        return F.normalize(self.vis(x), dim=-1)

    def embed_audio(self, x):
        return F.normalize(self.aud(x), dim=-1)


def _vision_batch(concepts, rng, device):
    imgs, ys = [], []
    for c in concepts:
        base = sample_object(rng, slot="p0")
        from .vision import ObjectSpec
        if c in COLORS:
            spec = ObjectSpec("p0", c, base.shape, base.x, base.y, base.scale)
        else:
            spec = ObjectSpec("p0", base.color, c, base.x, base.y, base.scale)
        imgs.append(render_object(spec, size=32))
        ys.append(SHARED_CONCEPTS.index(c))
    return (torch.tensor(np.stack(imgs), dtype=torch.float32, device=device),
            torch.tensor(ys, device=device))


def _audio_batch(concepts, clips_by_word, rng, device):
    specs, ys = [], []
    for c in concepts:
        wi = WORDS.index(c)
        clip = clips_by_word[wi][int(rng.integers(len(clips_by_word[wi])))]
        specs.append({"wave": clip["wave"], "word": wi, "voice": clip["voice"]})
        ys.append(SHARED_CONCEPTS.index(c))
    x = _feats(specs, rng, device, noise=0.005)
    return x, torch.tensor(ys, device=device)


def train(arm, clips, steps, seed, dim=64, device=DEV):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    by_word = {}
    for c in clips:
        by_word.setdefault(c["word"], []).append(c)
    model = CrossModal(arm, dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(steps):
        model.train()
        cs = [SHARED_CONCEPTS[int(rng.integers(len(SHARED_CONCEPTS)))] for _ in range(32)]
        vx, vy = _vision_batch(cs, rng, device)
        ax, ay = _audio_batch(cs, by_word, rng, device)
        zv, za = model.embed_vision(vx), model.embed_audio(ax)
        # both modalities classify into the SHARED concept head (grounds the space)
        loss = F.cross_entropy(model.concept(zv), vy) + F.cross_entropy(model.concept(za), ay)
        if arm == "shared":
            # CLIP-style alignment: matching (seen c, heard c) pairs attract, mismatches repel
            logits = (zv @ za.t()) / 0.07
            tgt = torch.arange(len(cs), device=device)
            loss = loss + 0.5 * (F.cross_entropy(logits, tgt)
                                 + F.cross_entropy(logits.t(), tgt))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def probe(model, clips, device, seed=99):
    """Cross-modal alignment, modality leakage, and retrieval -- the FER metrics."""
    rng = np.random.default_rng(seed)
    by_word = {}
    for c in clips:
        by_word.setdefault(c["word"], []).append(c)
    model.eval()
    with torch.no_grad():
        vembs, aembs = {}, {}
        for c in SHARED_CONCEPTS:
            vx, _ = _vision_batch([c] * 8, rng, device)
            ax, _ = _audio_batch([c] * 8, by_word, rng, device)
            vembs[c] = model.embed_vision(vx).mean(0)
            aembs[c] = model.embed_audio(ax).mean(0)
    cos = lambda a, b: float(F.cosine_similarity(a[None], b[None]).item())
    same = [cos(vembs[c], aembs[c]) for c in SHARED_CONCEPTS]
    diff = [cos(vembs[c], aembs[d]) for c in SHARED_CONCEPTS for d in SHARED_CONCEPTS if c != d]
    # retrieval: nearest seen concept to each heard concept
    hits = 0
    for c in SHARED_CONCEPTS:
        sims = {d: cos(aembs[c], vembs[d]) for d in SHARED_CONCEPTS}
        hits += int(max(sims, key=sims.get) == c)
    # modality leakage: same-modality cohesion minus cross-modality cohesion (FER if >> 0)
    vv = [cos(vembs[c], vembs[d]) for c in SHARED_CONCEPTS for d in SHARED_CONCEPTS if c != d]
    leak = float(np.mean(vv) - np.mean(diff))
    return {"crossmodal_same_cos": float(np.mean(same)),
            "crossmodal_diff_cos": float(np.mean(diff)),
            "alignment_margin": float(np.mean(same) - np.mean(diff)),
            "retrieval_acc": hits / len(SHARED_CONCEPTS),
            "modality_leakage": leak}


def run(steps=600, seeds=(0, 1, 2), device=DEV):
    _manifest, clips = load_bank()
    report = {"experiment": "m1_crossmodal", "steps": steps,
              "concepts": SHARED_CONCEPTS, "arms": {}}
    for arm in ("separate", "shared"):
        rows = []
        for seed in seeds:
            model = train(arm, clips, steps, seed, device=device)
            rows.append(probe(model, clips, device))
        m = lambda k: float(np.mean([r[k] for r in rows]))
        report["arms"][arm] = {k: m(k) for k in rows[0]}
        a = report["arms"][arm]
        print(f"{arm:9} align-margin {a['alignment_margin']:+.2f}  retrieval "
              f"{a['retrieval_acc']:.2f}  modality-leak {a['modality_leakage']:+.2f}", flush=True)
    sh, sep = report["arms"]["shared"], report["arms"]["separate"]
    report["verdict"] = {
        "shared_aligns_modalities": sh["retrieval_acc"] > 0.6,
        "alignment_gain": sh["alignment_margin"] - sep["alignment_margin"],
        "retrieval_gain": sh["retrieval_acc"] - sep["retrieval_acc"],
        "unified_not_fractured": sh["retrieval_acc"] > 0.6 and sh["alignment_margin"] > 0.2,
    }
    print("verdict:", json.dumps(report["verdict"], indent=1), flush=True)
    return report


def selftest():
    _manifest, clips = load_bank()
    assert len(SHARED_CONCEPTS) >= 6, SHARED_CONCEPTS
    rng = np.random.default_rng(0)
    by_word = {}
    for c in clips:
        by_word.setdefault(c["word"], []).append(c)
    model = CrossModal("shared", dim=32)
    vx, vy = _vision_batch(SHARED_CONCEPTS[:4], rng, "cpu")
    ax, ay = _audio_batch(SHARED_CONCEPTS[:4], by_word, rng, "cpu")
    zv, za = model.embed_vision(vx), model.embed_audio(ax)
    assert zv.shape == za.shape == (4, 32)
    m = train("shared", clips, steps=2, seed=0, device="cpu")
    p = probe(m, clips, "cpu")
    assert -1 <= p["alignment_margin"] <= 2 and 0 <= p["retrieval_acc"] <= 1
    print("crossmodal selftest OK")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default="runs/m1_crossmodal.json")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
        return
    report = run(steps=args.steps, seeds=tuple(int(s) for s in args.seeds.split(",")))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
