"""A-2 LISTEN: real synthesized speech -> content words, speaker-invariant by construction.

The oracle is the OS speech synthesizer (`say` on macOS, `espeak-ng` on pods): every clip's
transcript is known because we generated it -- the speech analog of rendering scenes from a
sampled EDB. The factor structure is the classic one from the speech-LM literature (semantic vs
acoustic tokens): CONTENT (which word) x SPEAKER (which voice) x RATE (prosody jitter). The gate
that matters is HELD-OUT word x voice cells: transcribing a known word spoken by a voice never
heard saying it = speaker-invariant listening, the FER question in its audio-native form.

Arms (Image-2 recipe): factored (word+voice heads) vs swap (same word, different voice -> the
word readout must not move; supervised pairs from the renderer).

  python -m thinking.listen --build           # render the clip bank via `say` (cached)
  python -m thinking.listen --selftest
  python -m thinking.listen --steps 600 --out runs/a2_listen.json
"""
import argparse
import json
import os
import subprocess
import wave

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from device import get_device

from .audio import spectrogram

DEV = get_device()
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data",
                    "speech")
WORDS = ("red", "green", "blue", "yellow", "white", "purple", "orange", "cyan",
         "square", "circle", "triangle", "diamond")
VOICE_CANDIDATES = ("Samantha", "Daniel", "Albert", "Fred", "Karen", "Moira", "Rishi", "Tessa")
RATES = (160, 200, 240)
N_VOICES = 4
CLIP_SECONDS = 0.66                                        # pad/trim to fixed length
HOLDOUT_CELLS = ((0, 1), (3, 0), (5, 2), (7, 3), (8, 0), (10, 2), (11, 1), (2, 3))
#                 (word_idx, voice_idx) pairs never seen in training


def _available_voices():
    out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, timeout=30)
    names = {ln.split()[0] for ln in out.stdout.splitlines() if ln.strip()}
    return [v for v in VOICE_CANDIDATES if v in names][:N_VOICES]


def build_bank(root=ROOT):
    """Render WORDS x voices x RATES once; the transcript is the file name (oracle-sound)."""
    os.makedirs(root, exist_ok=True)
    voices = _available_voices()
    if len(voices) < N_VOICES:
        raise RuntimeError(f"need {N_VOICES} voices, found {voices}")
    manifest = {"voices": voices, "words": list(WORDS), "rates": list(RATES), "clips": []}
    for wi, word in enumerate(WORDS):
        for vi, voice in enumerate(voices):
            for ri, rate in enumerate(RATES):
                path = os.path.join(root, f"{wi}_{vi}_{ri}.wav")
                if not os.path.exists(path):
                    subprocess.run(["say", "-v", voice, "-r", str(rate), "-o", path,
                                    "--file-format=WAVE", "--data-format=LEI16@8000", word],
                                   check=True, timeout=60)
                manifest["clips"].append({"word": wi, "voice": vi, "rate": ri,
                                          "path": os.path.basename(path)})
    with open(os.path.join(root, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"speech bank: {len(manifest['clips'])} clips, voices {voices}")
    return manifest


def load_bank(root=ROOT):
    with open(os.path.join(root, "manifest.json")) as f:
        manifest = json.load(f)
    n = int(8000 * CLIP_SECONDS)
    clips = []
    for c in manifest["clips"]:
        w = wave.open(os.path.join(root, c["path"]))
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
        x = x / 32768.0
        nz = np.where(np.abs(x) > 0.01)[0]                 # trim leading/trailing silence
        if len(nz):
            x = x[max(0, nz[0] - 160):nz[-1] + 160]
        x = x[:n] if len(x) >= n else np.pad(x, (0, n - len(x)))
        clips.append({**c, "wave": x})
    return manifest, clips


def _split(clips):
    hold = set(HOLDOUT_CELLS)
    train = [c for c in clips if (c["word"], c["voice"]) not in hold]
    test = [c for c in clips if (c["word"], c["voice"]) in hold]
    return train, test


def _feats(batch, rng, device, noise=0.005):
    xs = []
    for c in batch:
        w = c["wave"]
        if noise:
            w = w + noise * rng.standard_normal(len(w)).astype(np.float32)
        xs.append(spectrogram(w))
    return torch.tensor(np.stack(xs), dtype=torch.float32, device=device)


class Listener(nn.Module):
    """Freq-preserving conv trunk; word + voice heads (factored or subspace-sliced)."""

    def __init__(self, arm="factored", dim=96, n_voices=N_VOICES):
        super().__init__()
        self.arm = arm
        self.conv = nn.Sequential(
            nn.Conv2d(1, 24, 3, padding=1), nn.GELU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(24, 48, 3, padding=1), nn.GELU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(48, 64, 3, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(64 * 16, dim), nn.LayerNorm(dim))
        if arm == "subspace":
            h = dim // 2
            self.word = nn.Linear(h, len(WORDS))
            self.voice = nn.Linear(dim - h, n_voices)
        else:
            self.word = nn.Linear(dim, len(WORDS))
            self.voice = nn.Linear(dim, n_voices)

    def parts(self, z):
        if self.arm == "subspace":
            h = z.shape[-1] // 2
            return z[:, :h], z[:, h:]
        return z, z

    def forward(self, x, return_embedding=False):
        z = self.proj(self.conv(x))
        zw, zv = self.parts(z)
        out = {"word": self.word(zw), "voice": self.voice(zv)}
        if return_embedding:
            out["embedding"] = z
        return out


def accuracy(model, clips, rng, device, n_pass=2):
    model.eval()
    got_w = got_v = total = 0
    with torch.no_grad():
        for _ in range(n_pass):
            for off in range(0, len(clips), 64):
                batch = clips[off:off + 64]
                out = model(_feats(batch, rng, device))
                yw = torch.tensor([c["word"] for c in batch], device=device)
                yv = torch.tensor([c["voice"] for c in batch], device=device)
                got_w += int(out["word"].argmax(-1).eq(yw).sum())
                got_v += int(out["voice"].argmax(-1).eq(yv).sum())
                total += len(batch)
    return got_w / total, got_v / total


def train_arm(arm, clips_train, steps, seed, inv_w=1.0, batch=32, lr=1e-3, device=DEV):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    batch_n = int(batch)
    by_word = {}
    for c in clips_train:
        by_word.setdefault(c["word"], []).append(c)
    model = Listener(arm).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(steps):
        model.train()
        batch_items = [clips_train[int(rng.integers(len(clips_train)))] for _ in range(batch_n)]
        x = _feats(batch_items, rng, device)
        yw = torch.tensor([c["word"] for c in batch_items], device=device)
        yv = torch.tensor([c["voice"] for c in batch_items], device=device)
        out = model(x, return_embedding=True)
        loss = F.cross_entropy(out["word"], yw) + F.cross_entropy(out["voice"], yv)
        if arm in ("swap", "subspace"):
            # SAME WORD, DIFFERENT VOICE: the word readout must not move (speaker invariance)
            pair = []
            for c in batch_items:
                alts = [a for a in by_word[c["word"]] if a["voice"] != c["voice"]]
                pair.append(alts[int(rng.integers(len(alts)))] if alts else c)
            o2 = model(_feats(pair, rng, device), return_embedding=True)
            if arm == "subspace":
                inv = F.mse_loss(model.parts(out["embedding"])[0],
                                 model.parts(o2["embedding"])[0])
            else:
                inv = F.mse_loss(out["word"], o2["word"])
            loss = loss + inv_w * inv
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def run(steps=600, seeds=(0, 1, 2), device=DEV):
    manifest, clips = load_bank()
    train_clips, test_clips = _split(clips)
    report = {"experiment": "a2_listen", "steps": steps,
              "bank": f"{len(WORDS)} words x {len(manifest['voices'])} voices x "
                      f"{len(RATES)} rates; {len(HOLDOUT_CELLS)} held-out cells",
              "arms": {}}
    for arm in ("factored", "swap", "subspace"):
        rows = []
        for seed in seeds:
            model = train_arm(arm, train_clips, steps, seed, device=device)
            rng = np.random.default_rng(7_000 + seed)
            seen_w, seen_v = accuracy(model, train_clips, rng, device)
            hold_w, hold_v = accuracy(model, test_clips, rng, device)
            rows.append({"seed": seed, "seen_word": seen_w, "holdout_word": hold_w,
                         "seen_voice": seen_v, "holdout_voice": hold_v})
        m = lambda k: float(np.mean([r[k] for r in rows]))
        report["arms"][arm] = {"seeds": rows, "seen_word": m("seen_word"),
                               "holdout_word": m("holdout_word"),
                               "holdout_voice": m("holdout_voice")}
        print(f"{arm:9} seen-word {m('seen_word'):.2f}  HOLDOUT-word {m('holdout_word'):.2f}  "
              f"holdout-voice {m('holdout_voice'):.2f}", flush=True)
    arms = report["arms"]
    best = max(arms, key=lambda a: arms[a]["holdout_word"])
    report["verdict"] = {
        "best_arm": best,
        "best_holdout_word": arms[best]["holdout_word"],
        "baseline_holdout_word": arms["factored"]["holdout_word"],
        "speaker_invariant_listening": arms[best]["holdout_word"] > 0.9,
    }
    print("verdict:", json.dumps(report["verdict"], indent=1), flush=True)
    return report


def selftest():
    if not os.path.exists(os.path.join(ROOT, "manifest.json")):
        build_bank()
    manifest, clips = load_bank()
    assert len(clips) == len(WORDS) * len(manifest["voices"]) * len(RATES)
    tr, te = _split(clips)
    assert te and all((c["word"], c["voice"]) in set(HOLDOUT_CELLS) for c in te)
    assert all((c["word"], c["voice"]) not in set(HOLDOUT_CELLS) for c in tr)
    n = int(8000 * CLIP_SECONDS)
    assert all(len(c["wave"]) == n for c in clips)
    rng = np.random.default_rng(0)
    x = _feats(clips[:2], rng, "cpu")
    for arm in ("factored", "swap", "subspace"):
        m = Listener(arm)
        out = m(x, return_embedding=True)
        assert out["word"].shape == (2, len(WORDS))
        model = train_arm(arm, tr[:40], steps=2, seed=0, device="cpu")
        assert model is not None
    print("listen selftest OK")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default="runs/a2_listen.json")
    args = ap.parse_args(argv)
    if args.build:
        build_bank()
        return
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
