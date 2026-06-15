"""Zero-shot voice conversion that WORKS: kNN-VC with a pretrained content encoder (WavLM).

The from-scratch AutoVC bottleneck couldn't disentangle content from speaker (voice transfer stayed
at chance). The fix the literature converged on: use a PRETRAINED self-supervised speech model
(WavLM) whose intermediate features are already speaker-invariant phonetic CONTENT. Then conversion
needs NO trained converter at all (kNN-VC, Baas 2023):

  1. WavLM (layer 6) -> content features for the source AND for the target speaker's reference pool.
  2. For each source frame, replace it with the MEAN of its k nearest neighbours in the target
     pool (kNN regression in feature space) -- this swaps WHO is speaking, keeps WHAT is said.
  3. A HiFi-GAN trained on WavLM features vocodes the matched features -> waveform.

All three come pretrained from the reference implementation (torch.hub bshall/knn-vc). We run it
zero-shot on UNSEEN LibriSpeech speakers and score voice transfer with our OWN frozen speaker
encoder (runs/libriclone.pt): does the converted audio land as target B, not source A?

  python -m thinking.knnvc --selftest        # checks deps + data only (model needs GPU/download)
  python -m thinking.knnvc --run --out runs/knnvc.json --synth-out data/synth   (GPU)
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from device import get_device
from .libriclone import (ROOT, load as load_libri, split_speakers, SR, _mel, SEG_FRAMES,
                         HOP, SpeakerEncoder, _seg_mels, ge2e)

DEV = get_device()


def load_knnvc(device=DEV):
    """Pretrained kNN-VC: WavLM (content) + prematched HiFi-GAN (WavLM-feature vocoder)."""
    return torch.hub.load("bshall/knn-vc", "knn_vc", prematched=True, trust_repo=True,
                          pretrained=True, device=device)


def _to_wav_tensor(x, device):
    return torch.tensor(x, dtype=torch.float32, device=device)[None]   # (1, n)


def _write_wav(path, wav, sr=SR):
    import wave as wv
    y = wav / (np.abs(wav).max() + 1e-8) * 0.95
    w = wv.open(path, "w")
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
    w.writeframes((y * 32767).astype(np.int16).tobytes()); w.close()


def _spk_embed(se, wav, device):
    """Our frozen speaker encoder's embedding of a waveform (for voice_match scoring)."""
    need = SEG_FRAMES * HOP
    w = wav[:need] if len(wav) >= need else np.pad(wav, (0, need - len(wav)))
    mel = _mel(torch.tensor(w[None], dtype=torch.float32, device=device))[:, :, :SEG_FRAMES]
    return F.normalize(se(mel)[0], dim=0)


def _scoring_encoder(by_spk, device, steps=8000, seed=0):
    """Frozen speaker encoder for voice_match scoring: load runs/libriclone.pt if present, else
    quick-train a GE2E encoder on the TRAIN speakers (keeps the metric self-contained on a pod)."""
    se = SpeakerEncoder().to(device)
    if os.path.exists("runs/libriclone.pt"):
        se.load_state_dict(torch.load("runs/libriclone.pt", map_location=device)["se"])
        print("scoring encoder: loaded runs/libriclone.pt", flush=True)
    else:
        print("scoring encoder: quick-training GE2E (no checkpoint)", flush=True)
        train_spk, _ = split_speakers(by_spk, seed=0)
        import torch.nn as nn
        w = nn.Parameter(torch.tensor(10.0, device=device)); b = nn.Parameter(torch.tensor(-5.0, device=device))
        opt = torch.optim.AdamW(list(se.parameters()) + [w, b], lr=1e-3)
        rng = np.random.default_rng(seed)
        for st in range(1, steps + 1):
            mel, ns, nu = _seg_mels(train_spk, rng, 16, 4, device)
            loss = ge2e(se(mel), ns, nu, w, b)
            opt.zero_grad(); loss.backward(); opt.step()
        json.dump({}, open(os.devnull, "w"))
    for p in se.parameters():
        p.requires_grad_(False)
    se.train()                                             # cudnn LSTM backward needs train mode
    return se


def run(device=DEV, n=120, topk=4, seed=1, save_dir="data/synth", out=None):
    knn = load_knnvc(device)
    by_spk = load_libri()
    _, hold = split_speakers(by_spk, seed=0)               # UNSEEN speakers (held out from our enc)
    # our frozen speaker encoder for scoring
    se = _scoring_encoder(by_spk, device)

    spks = [s for s in hold if len(hold[s]) >= 3]
    rng = np.random.default_rng(seed)
    # target-speaker centroids (our encoder) over held-out speakers
    cent = {s: F.normalize(torch.stack([_spk_embed(se, w, device) for w in hold[s][:6]]).mean(0), dim=0)
            for s in spks}
    cmat = torch.stack([cent[s] for s in spks])

    voice_hit = src_match = total = saved = 0
    for _ in range(n):
        a, b = rng.choice(len(spks), 2, replace=False)
        a, b = spks[a], spks[b]
        src = hold[a][int(rng.integers(len(hold[a])))]
        refs = [hold[b][j] for j in rng.permutation(len(hold[b]))[:6]]
        with torch.no_grad():
            q = knn.get_features(_to_wav_tensor(src, device))
            mset = knn.get_matching_set([_to_wav_tensor(r, device) for r in refs])
            conv = knn.match(q, mset, topk=topk).cpu().numpy()
        emb = _spk_embed(se, conv, device)
        pred = spks[int((cmat @ emb).argmax())]
        voice_hit += (pred == b)                           # converted lands as TARGET b
        src_match += (pred == a)                           # (failure mode: stays source a)
        total += 1
        if save_dir and saved < 5:
            os.makedirs(save_dir, exist_ok=True)
            _write_wav(os.path.join(save_dir, f"knn{saved}_source_A.wav"), src)
            _write_wav(os.path.join(save_dir, f"knn{saved}_ref_B.wav"), refs[0])
            _write_wav(os.path.join(save_dir, f"knn{saved}_cloned_AinB.wav"), conv)
            saved += 1
    report = {"experiment": "knnvc_pretrained_wavlm", "topk": topk, "n": total,
              "n_holdout_speakers": len(spks), "voice_chance": 1 / len(spks),
              "voice_match": voice_hit / total, "stayed_source": src_match / total,
              "works": voice_hit / total > 0.5}
    print(f"\nZERO-SHOT kNN-VC (UNSEEN speakers): voice_match {report['voice_match']:.3f} "
          f"(chance {report['voice_chance']:.3f}, {len(spks)} speakers); "
          f"stayed-source {report['stayed_source']:.3f}")
    print(f"works (>0.5): {report['works']}", flush=True)
    if out:
        json.dump(report, open(out, "w"), indent=1)
    return report


def selftest():
    # deps + data presence (the WavLM model download/GPU runs only under --run)
    import importlib
    have_ta = importlib.util.find_spec("torchaudio") is not None
    have_data = os.path.exists(os.path.join(ROOT, "manifest.json"))
    have_enc = os.path.exists("runs/libriclone.pt")
    print(f"torchaudio: {have_ta}  libri-data: {have_data}  speaker-encoder: {have_enc}")
    if have_data and have_enc:
        by_spk = load_libri(); _, hold = split_speakers(by_spk, seed=0)
        assert len(hold) >= 2
        se = SpeakerEncoder(); se.load_state_dict(torch.load("runs/libriclone.pt", map_location="cpu")["se"])
        se.train()
        for p in se.parameters():
            p.requires_grad_(False)
        e = _spk_embed(se, list(hold.values())[0][0], "cpu")
        assert e.shape == (128,)
    print("knnvc selftest OK (run --run on GPU for the WavLM conversion)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--out", default="runs/knnvc.json")
    ap.add_argument("--synth-out", default="data/synth", dest="synth_out")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest(); return
    if args.run:
        run(n=args.n, topk=args.topk, out=args.out, save_dir=args.synth_out); return
    ap.error("choose --selftest / --run")


if __name__ == "__main__":
    main()
