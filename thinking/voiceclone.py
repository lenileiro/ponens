"""Zero-shot voice cloning foundation: LISTEN to a voice, characterize it, GENERALIZE to unseen
voices -- the speaker encoder + its zero-shot verification metric.

The user's target: "listen to a human voice and mimic it WITHOUT extensive training." Modern
zero-shot voice cloning (VALL-E / GLM-TTS style: adapt from a ~3s acoustic prompt) hinges on a
SPEAKER ENCODER that maps a reference clip -> a voice embedding which CONDITIONS synthesis. The
hard, decisive requirement is that the encoder generalizes to speakers NEVER SEEN in training --
otherwise it memorizes a fixed set, not "listen and mimic anyone."

This trains the speaker encoder on real human voices (Google Speech Commands, 100s of speakers)
with a GE2E-style contrastive objective (same speaker -> close, different -> far), and measures
ZERO-SHOT speaker verification on HELD-OUT SPEAKERS: given two clips of a voice never trained on,
does the model tell same-vs-different correctly? High AUC/low EER on unseen speakers = the model
can characterize an arbitrary voice from listening, the prerequisite for cloning it.

  python -m thinking.voiceclone --fetch        # more GSC (many speakers, several clips each)
  python -m thinking.voiceclone --selftest
  python -m thinking.voiceclone --train --steps 6000 --out runs/voiceclone.json
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
GSC_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "gsc_spk")
SR = 16000
N_FFT = 512
HOP = 128
N_BINS = N_FFT // 2 + 1
CLIP = 16000                                               # 1s
_WIN = np.hanning(N_FFT).astype(np.float32)
# many words -> several clips per speaker (GE2E needs multiple utterances/speaker)
WORDS = ("bed bird cat dog down eight five four go happy house left nine no off on right "
         "seven six stop three tree two up yes zero").split()


def fetch(per_word=180, root=ROOT, byte_cap_gb=2.4):
    os.makedirs(root, exist_ok=True)
    targets = set(WORDS)
    got = {w: 0 for w in WORDS}
    cap = int(byte_cap_gb * 1e9)
    seen = [0]
    clips = []

    class _C:
        def __init__(s, f): s.f = f
        def read(s, n):
            b = s.f.read(n); seen[0] += len(b); return b

    print(f"streaming GSC (cap {byte_cap_gb}GB)...")
    req = urllib.request.Request(GSC_URL, headers={"User-Agent": "ponens/1.0"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        tar = tarfile.open(fileobj=gzip.GzipFile(fileobj=_C(resp)), mode="r|")
        for m in tar:
            if seen[0] > cap or all(got[w] >= per_word for w in WORDS):
                break
            if not m.name.endswith(".wav"):
                continue
            parts = m.name.split("/")
            if len(parts) < 2 or parts[-2] not in targets:
                continue
            w = parts[-2]
            if got[w] >= per_word:
                continue
            spk = os.path.basename(m.name).split("_nohash_")[0]
            data = tar.extractfile(m).read()
            name = f"{spk}__{w}__{got[w]}.wav"
            with open(os.path.join(root, name), "wb") as f:
                f.write(data)
            clips.append({"speaker": spk, "path": name})
            got[w] += 1
    json.dump({"clips": clips}, open(os.path.join(root, "manifest.json"), "w"))
    nspk = len({c["speaker"] for c in clips})
    print(f"fetched {len(clips)} clips, {nspk} speakers, {seen[0]/1e9:.2f}GB")
    return len(clips)


def _logmel(x, n_mels=40):
    n = CLIP
    x = x[:n] if len(x) >= n else np.pad(x, (0, n - len(x)))
    cols = [np.abs(np.fft.rfft(x[i:i + N_FFT] * _WIN)) for i in range(0, n - N_FFT + 1, HOP)]
    spec = np.stack(cols, 1).astype(np.float32)
    fb = _logmel.fb
    if fb is None:
        f = np.linspace(0, SR / 2, N_BINS)
        mmax = 2595 * np.log10(1 + (SR / 2) / 700)
        ctr = 700 * (10 ** (np.linspace(0, mmax, n_mels + 2) / 2595) - 1)
        fb = np.zeros((n_mels, N_BINS), np.float32)
        for mi in range(1, n_mels + 1):
            lo, ce, hi = ctr[mi - 1], ctr[mi], ctr[mi + 1]
            fb[mi - 1] = np.clip(np.minimum((f - lo) / (ce - lo + 1e-9), (hi - f) / (hi - ce + 1e-9)), 0, 1)
        _logmel.fb = fb
    return np.log1p(fb @ spec)                             # (n_mels, T)
_logmel.fb = None


def load(root=ROOT):
    man = json.load(open(os.path.join(root, "manifest.json")))
    by_spk = {}
    for c in man["clips"]:
        w = wave.open(os.path.join(root, c["path"]))
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        by_spk.setdefault(c["speaker"], []).append(_logmel(x))
    return {s: v for s, v in by_spk.items() if len(v) >= 2}   # need >=2 utts/speaker


def split_speakers(by_spk, holdout_frac=0.25, seed=0):
    spk = sorted(by_spk)
    rng = np.random.default_rng(seed)
    rng.shuffle(spk)
    k = max(2, int(len(spk) * holdout_frac))
    hold = set(spk[:k])
    return ({s: by_spk[s] for s in spk if s not in hold},
            {s: by_spk[s] for s in spk if s in hold})


class SpeakerEncoder(nn.Module):
    """log-mel -> L2-normalized voice embedding (the d-vector / acoustic-prompt encoder)."""

    def __init__(self, n_mels=40, d=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d((2, 2)))
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(128 * 4, d))

    def forward(self, x):                                  # (B, n_mels, T)
        return F.normalize(self.proj(self.conv(x.unsqueeze(1))), dim=-1)


def ge2e_loss(emb, n_spk, n_utt):
    """GE2E: each utterance's similarity to its own speaker-centroid (excl. itself) vs others."""
    e = emb.view(n_spk, n_utt, -1)
    centroids = e.mean(1)                                  # (n_spk, d)
    # leave-one-out centroid for the positive
    sums = e.sum(1, keepdim=True)
    loo = (sums - e) / (n_utt - 1)                         # (n_spk, n_utt, d)
    w, b = ge2e_loss.w, ge2e_loss.b
    flat = e.reshape(n_spk * n_utt, -1)
    sim = w * (flat @ centroids.t()) + b                   # (n_spk*n_utt, n_spk)
    pos = w * (F.normalize(loo, dim=-1).reshape(n_spk * n_utt, -1)
               * F.normalize(centroids, dim=-1).repeat_interleave(n_utt, 0)).sum(-1) + b
    idx = torch.arange(n_spk, device=emb.device).repeat_interleave(n_utt)
    sim = sim.clone()
    sim[torch.arange(n_spk * n_utt), idx] = pos
    return F.cross_entropy(sim, idx)
ge2e_loss.w = None
ge2e_loss.b = None


def _batch(by_spk, rng, n_spk=16, n_utt=4, device=DEV):
    spks = [s for s in by_spk if len(by_spk[s]) >= n_utt]
    chosen = [spks[i] for i in rng.permutation(len(spks))[:n_spk]]
    mats = []
    for s in chosen:
        utts = by_spk[s]
        pick = rng.permutation(len(utts))[:n_utt]
        for j in pick:
            mats.append(utts[j])
    return torch.tensor(np.stack(mats), dtype=torch.float32, device=device), len(chosen), n_utt


def verification_auc(model, by_spk, device=DEV, n_pairs=2000, seed=1):
    """Zero-shot: same-speaker vs different-speaker cosine on HELD-OUT speakers -> AUC + EER."""
    model.eval()
    rng = np.random.default_rng(seed)
    embs = {}
    with torch.no_grad():
        for s, utts in by_spk.items():
            x = torch.tensor(np.stack(utts), dtype=torch.float32, device=device)
            embs[s] = model(x).cpu().numpy()
    spk = [s for s in by_spk if len(by_spk[s]) >= 2]
    pos, neg = [], []
    for _ in range(n_pairs):
        s = spk[int(rng.integers(len(spk)))]
        i, j = rng.choice(len(embs[s]), 2, replace=False)
        pos.append(float(embs[s][i] @ embs[s][j]))
        s2 = spk[int(rng.integers(len(spk)))]
        while s2 == s:
            s2 = spk[int(rng.integers(len(spk)))]
        neg.append(float(embs[s][int(rng.integers(len(embs[s])))]
                         @ embs[s2][int(rng.integers(len(embs[s2])))]))
    pos, neg = np.array(pos), np.array(neg)
    # AUC via rank; EER via threshold sweep
    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
    order = scores.argsort()
    ranks = np.empty_like(order); ranks[order] = np.arange(len(scores))
    auc = (ranks[labels == 1].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg))
    eer = 1.0
    for th in np.linspace(scores.min(), scores.max(), 200):
        far = (neg >= th).mean(); frr = (pos < th).mean()
        if abs(far - frr) < abs(eer - frr) + 1e-9 and abs(far - frr) < 0.02:
            eer = (far + frr) / 2
    return {"auc": float(auc), "eer": float(eer),
            "same_mean": float(pos.mean()), "diff_mean": float(neg.mean())}


def train(steps=6000, seed=0, device=DEV, lr=1e-3, n_spk=16, n_utt=4):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    by_spk = load()
    train_spk, _ = split_speakers(by_spk, seed=seed)
    model = SpeakerEncoder().to(device)
    ge2e_loss.w = nn.Parameter(torch.tensor(10.0, device=device))
    ge2e_loss.b = nn.Parameter(torch.tensor(-5.0, device=device))
    opt = torch.optim.AdamW(list(model.parameters()) + [ge2e_loss.w, ge2e_loss.b], lr=lr)
    for st in range(1, steps + 1):
        model.train()
        x, ns, nu = _batch(train_spk, rng, n_spk=n_spk, n_utt=n_utt, device=device)
        loss = ge2e_loss(model(x), ns, nu)
        opt.zero_grad(); loss.backward(); opt.step()
        if st % max(1, steps // 8) == 0 or st == steps:
            print(f"  vc {st}/{steps} ge2e {loss.item():.4f}", flush=True)
    return model, by_spk


def run(steps=6000, seed=0, device=DEV):
    model, by_spk = train(steps=steps, seed=seed, device=device)
    train_spk, hold_spk = split_speakers(by_spk, seed=seed)
    seen = verification_auc(model, train_spk, device=device)
    unseen = verification_auc(model, hold_spk, device=device)
    report = {"experiment": "voiceclone_speaker_encoder", "steps": steps,
              "n_train_speakers": len(train_spk), "n_holdout_speakers": len(hold_spk),
              "seen_speaker_verification": seen, "ZEROSHOT_unseen_speaker_verification": unseen,
              "generalizes_to_unseen_voices": unseen["auc"] > 0.85}
    print(f"\nSEEN speakers  : AUC {seen['auc']:.3f} (same {seen['same_mean']:.2f} / diff {seen['diff_mean']:.2f})")
    print(f"UNSEEN (zero-shot): AUC {unseen['auc']:.3f} EER {unseen['eer']:.3f} "
          f"(same {unseen['same_mean']:.2f} / diff {unseen['diff_mean']:.2f})")
    print(f"generalizes to unseen voices (AUC>0.85): {report['generalizes_to_unseen_voices']}", flush=True)
    return report, model


def selftest():
    if not os.path.exists(os.path.join(ROOT, "manifest.json")):
        print("no GSC speaker bank; run --fetch first (skipping)")
        return
    by_spk = load()
    assert len(by_spk) >= 8
    tr, te = split_speakers(by_spk)
    assert not (set(tr) & set(te))
    m, _ = train(steps=3, seed=0, device="cpu")
    v = verification_auc(m, te, device="cpu", n_pairs=200)
    assert 0 <= v["auc"] <= 1
    print("voiceclone selftest OK")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--per-word", type=int, default=180, dest="per_word")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/voiceclone.json")
    ap.add_argument("--checkpoint", default="runs/voiceclone.pt")
    args = ap.parse_args(argv)
    if args.fetch:
        fetch(per_word=args.per_word); return
    if args.selftest:
        selftest(); return
    if args.train:
        report, model = run(steps=args.steps, seed=args.seed)
        os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
        torch.save({"state_dict": model.state_dict()}, args.checkpoint)
        json.dump(report, open(args.out, "w"), indent=1)
        print(f"saved -> {args.out}")
        return
    ap.error("choose --fetch / --selftest / --train")


if __name__ == "__main__":
    main()
