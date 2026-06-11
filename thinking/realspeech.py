"""Real-data rung: Google Speech Commands (real human speech) -> does speaker-invariant listening
hold OFF the synthetic say-oracle? The honest transfer test the project keeps gating on.

GSC filenames encode the speaker (`<speaker>_nohash_<n>.wav`), so we can split by SPEAKER: train
on some voices, test on voices NEVER HEARD. This is the real-world version of A-2's held-out-voice
test. Arms transfer from A-2: factored (word head only) vs swap (same word, different speaker ->
the word readout must not move). 16kHz/1s clips match the vocoder pipeline.

  python -m thinking.realspeech --fetch --words bed,bird,cat,dog,down,eight,five,go,happy,house
  python -m thinking.realspeech --selftest
  python -m thinking.realspeech --steps 1500 --out runs/realspeech.json
"""
import argparse
import gzip
import json
import os
import tarfile
import urllib.request
import wave

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from device import get_device

DEV = get_device()
URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "gsc")
SR = 16000
N_FFT = 512
HOP = 128
N_BINS = N_FFT // 2 + 1
N_FRAMES = 128                                             # ~1s clip
DEFAULT_WORDS = ("bed", "bird", "cat", "dog", "down", "eight", "five", "go", "happy", "house")
_WIN = np.hanning(N_FFT).astype(np.float32)


def fetch(words, per_word=120, root=ROOT, byte_cap_gb=1.6):
    """Stream the tar.gz and extract up to per_word clips for each target word, recording the
    speaker id from the filename. Stops once every quota is met or the byte cap is hit."""
    os.makedirs(root, exist_ok=True)
    targets = set(words)
    got = {w: [] for w in words}
    cap = int(byte_cap_gb * 1e9)
    seen_bytes = [0]

    class _Counting:
        def __init__(self, f):
            self.f = f

        def read(self, n):
            b = self.f.read(n)
            seen_bytes[0] += len(b)
            return b

    print(f"streaming {URL} (cap {byte_cap_gb}GB)...")
    req = urllib.request.Request(URL, headers={"User-Agent": "ponens/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        gz = gzip.GzipFile(fileobj=_Counting(resp))
        tar = tarfile.open(fileobj=gz, mode="r|")
        for m in tar:
            if seen_bytes[0] > cap or all(len(got[w]) >= per_word for w in words):
                break
            if not m.name.endswith(".wav"):
                continue
            parts = m.name.split("/")
            if len(parts) < 2:
                continue
            word = parts[-2]
            if word not in targets or len(got[word]) >= per_word:
                continue
            speaker = os.path.basename(m.name).split("_nohash_")[0]
            data = tar.extractfile(m).read()
            out = os.path.join(root, f"{word}__{speaker}__{len(got[word])}.wav")
            with open(out, "wb") as f:
                f.write(data)
            got[word].append((speaker, os.path.basename(out)))
    manifest = {"words": list(words), "sr": SR,
                "clips": [{"word": w, "speaker": s, "path": p}
                          for w in words for (s, p) in got[w]]}
    with open(os.path.join(root, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    counts = {w: len(got[w]) for w in words}
    n_spk = len({c["speaker"] for c in manifest["clips"]})
    print(f"fetched {sum(counts.values())} clips, {n_spk} speakers, "
          f"{seen_bytes[0] / 1e9:.2f}GB streamed; per-word {counts}")
    return manifest


def _logmag(wave_f):
    n = SR
    wave_f = wave_f[:n] if len(wave_f) >= n else np.pad(wave_f, (0, n - len(wave_f)))
    cols = [np.abs(np.fft.rfft(wave_f[i:i + N_FFT] * _WIN))
            for i in range(0, len(wave_f) - N_FFT + 1, HOP)]
    s = np.log1p(np.stack(cols, 1).astype(np.float32))
    if s.shape[1] < N_FRAMES:
        s = np.pad(s, ((0, 0), (0, N_FRAMES - s.shape[1])))
    return s[:, :N_FRAMES]


def load(root=ROOT):
    with open(os.path.join(root, "manifest.json")) as f:
        manifest = json.load(f)
    words = manifest["words"]
    clips = []
    for c in manifest["clips"]:
        w = wave.open(os.path.join(root, c["path"]))
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        clips.append({"word": words.index(c["word"]), "speaker": c["speaker"],
                      "logmag": _logmag(x)})
    return manifest, clips


def speaker_split(clips, holdout_frac=0.25, seed=0):
    """Disjoint-by-SPEAKER split: test voices are never heard in training."""
    speakers = sorted({c["speaker"] for c in clips})
    rng = np.random.default_rng(seed)
    rng.shuffle(speakers)
    n_hold = max(1, int(len(speakers) * holdout_frac))
    hold = set(speakers[:n_hold])
    train = [c for c in clips if c["speaker"] not in hold]
    test = [c for c in clips if c["speaker"] in hold]
    return train, test, hold


class Listener(nn.Module):
    def __init__(self, arm, n_words, dim=128):
        super().__init__()
        self.arm = arm
        self.conv = nn.Sequential(
            nn.Conv2d(1, 24, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(48, 64, 3, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d((4, 4)))
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(64 * 16, dim), nn.LayerNorm(dim))
        self.word = nn.Linear(dim, n_words)

    def forward(self, x, return_embedding=False):
        z = self.proj(self.conv(x))
        out = {"word": self.word(z)}
        if return_embedding:
            out["embedding"] = z
        return out


def _batch(clips, idx, device, noise=0.01):
    X = torch.tensor(np.stack([clips[i]["logmag"] for i in idx]))[:, None].to(device)
    if noise:
        X = X + noise * torch.randn_like(X)
    y = torch.tensor([clips[i]["word"] for i in idx], device=device)
    return X, y


def accuracy(model, clips, device):
    model.eval()
    got = total = 0
    with torch.no_grad():
        for off in range(0, len(clips), 64):
            idx = list(range(off, min(off + 64, len(clips))))
            X, y = _batch(clips, idx, device, noise=0.0)
            got += int(model(X)["word"].argmax(-1).eq(y).sum())
            total += len(idx)
    return got / total


def train_arm(arm, train_clips, n_words, steps, seed, device=DEV, batch=32, inv_w=1.0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    by_word = {}
    for i, c in enumerate(train_clips):
        by_word.setdefault(c["word"], []).append(i)
    model = Listener(arm, n_words).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(steps):
        model.train()
        idx = list(rng.integers(0, len(train_clips), batch))
        X, y = _batch(train_clips, idx, device)
        out = model(X, return_embedding=True)
        loss = F.cross_entropy(out["word"], y)
        if arm == "swap":
            # SAME WORD, DIFFERENT SPEAKER -> word embedding must be invariant (real speakers)
            pair = []
            for i in idx:
                w = train_clips[i]["word"]
                alts = [j for j in by_word[w] if train_clips[j]["speaker"] != train_clips[i]["speaker"]]
                pair.append(alts[int(rng.integers(len(alts)))] if alts else i)
            Xp, _ = _batch(train_clips, pair, device)
            op = model(Xp, return_embedding=True)
            loss = loss + inv_w * F.mse_loss(out["embedding"], op["embedding"])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def run(steps=1500, seeds=(0, 1, 2), device=DEV):
    manifest, clips = load()
    n_words = len(manifest["words"])
    report = {"experiment": "realspeech_gsc", "steps": steps,
              "words": manifest["words"], "n_clips": len(clips),
              "n_speakers": len({c["speaker"] for c in clips}), "arms": {}}
    for arm in ("factored", "swap"):
        rows = []
        for seed in seeds:
            train_clips, test_clips, _hold = speaker_split(clips, seed=seed)
            model = train_arm(arm, train_clips, n_words, steps, seed, device=device)
            rows.append({"seed": seed, "seen": accuracy(model, train_clips, device),
                         "holdout_speaker": accuracy(model, test_clips, device)})
        m = lambda k: float(np.mean([r[k] for r in rows]))
        report["arms"][arm] = {"seeds": rows, "seen": m("seen"),
                               "holdout_speaker": m("holdout_speaker")}
        print(f"{arm:9} seen {m('seen'):.2f}  HELD-OUT-SPEAKER {m('holdout_speaker'):.2f}",
              flush=True)
    f, s = report["arms"]["factored"], report["arms"]["swap"]
    report["verdict"] = {
        "chance": 1 / n_words,
        "swap_gain": s["holdout_speaker"] - f["holdout_speaker"],
        "fer_law_holds_on_real_speech": s["holdout_speaker"] >= f["holdout_speaker"],
        "generalizes_to_unseen_speakers": max(f["holdout_speaker"], s["holdout_speaker"])
                                          > 3 / n_words,
    }
    print("verdict:", json.dumps(report["verdict"], indent=1), flush=True)
    return report


def selftest():
    if not os.path.exists(os.path.join(ROOT, "manifest.json")):
        print("no GSC bank yet; run --fetch first (skipping data-dependent checks)")
        return
    manifest, clips = load()
    assert clips and clips[0]["logmag"].shape == (N_BINS, N_FRAMES)
    tr, te, hold = speaker_split(clips)
    assert tr and te and not ({c["speaker"] for c in tr} & {c["speaker"] for c in te})
    n_words = len(manifest["words"])
    m = train_arm("swap", tr[:60], n_words, steps=3, seed=0, device="cpu")
    assert 0 <= accuracy(m, te[:40], "cpu") <= 1
    print("realspeech selftest OK")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--words", default=",".join(DEFAULT_WORDS))
    ap.add_argument("--per-word", type=int, default=120, dest="per_word")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default="runs/realspeech.json")
    args = ap.parse_args(argv)
    if args.fetch:
        fetch([w.strip() for w in args.words.split(",")], per_word=args.per_word)
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
