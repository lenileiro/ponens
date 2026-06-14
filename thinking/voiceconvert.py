"""Zero-shot voice cloning, the SPEAKING half: render content in an UNSEEN voice from a reference.

The encoder (voiceclone.py) listens and characterizes any voice (0.84 AUC on unseen speakers).
This is the synthesis side -- AutoVC-style any-to-any voice conversion: a CONTENT encoder with a
tight bottleneck (so speaker identity is squeezed OUT of the content code) + the frozen speaker
embedding from a REFERENCE clip -> decoder -> mel. Because content is bottlenecked, the voice must
come from the reference embedding, so swapping in an unseen speaker's reference makes the output
speak in that voice -- with no training on that speaker.

Zero-shot test (held-out speakers): content from source A + reference from UNSEEN speaker B ->
generated mel. Measured two ways, both with frozen probes:
  voice_match : re-embed the generated mel; is it nearest target B (vs other speakers)? = mimicry
  content_acc : a frozen word classifier reads the source word off the generated mel = content kept

  python -m thinking.voiceconvert --selftest
  python -m thinking.voiceconvert --train --steps 8000 --out runs/voiceconvert.json
  (requires data/gsc_spk + runs/voiceclone.pt from thinking.voiceclone)
"""
import argparse
import json
import os
import wave

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from device import get_device
from .voiceclone import (SpeakerEncoder, _logmel, ROOT, WORDS, SR, split_speakers, load as _load_spk)

DEV = get_device()
N_MELS = 40


def load_with_words(root=ROOT):
    """by_spk -> {speaker: [(word_idx, mel), ...]} (need word labels for the content metric)."""
    man = json.load(open(os.path.join(root, "manifest.json")))
    by_spk = {}
    for c in man["clips"]:
        w = wave.open(os.path.join(root, c["path"]))
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        word = c["path"].split("__")[1]
        by_spk.setdefault(c["speaker"], []).append((WORDS.index(word), _logmel(x)))
    return {s: v for s, v in by_spk.items() if len(v) >= 2}


class ContentEncoder(nn.Module):
    """mel -> bottlenecked content code (small dim + temporal downsample = AutoVC disentangler)."""

    def __init__(self, n_mels=N_MELS, content_dim=4, down=4):
        super().__init__()
        self.down = down
        self.net = nn.Sequential(
            nn.Conv1d(n_mels, 128, 5, padding=2), nn.GELU(),
            nn.Conv1d(128, 128, 5, padding=2), nn.GELU(),
            nn.Conv1d(128, content_dim, 1))                # TIGHT bottleneck (dim 4, down 4):
        #                                                    squeeze speaker identity OUT of content

    def forward(self, mel):                                # (B, n_mels, T)
        c = self.net(mel)
        return F.avg_pool1d(c, self.down)                  # temporal bottleneck -> (B, content_dim, T/down)


class Decoder(nn.Module):
    """content code + speaker embedding -> reconstructed mel."""

    def __init__(self, n_mels=N_MELS, content_dim=4, spk_dim=128, down=4):
        super().__init__()
        self.down = down
        self.spk = nn.Linear(spk_dim, 64)
        self.net = nn.Sequential(
            nn.Conv1d(content_dim + 64, 256, 5, padding=2), nn.GELU(),
            nn.Conv1d(256, 256, 5, padding=2), nn.GELU(),
            nn.Conv1d(256, n_mels, 1))

    def forward(self, content, spk_emb, out_T):
        c = F.interpolate(content, size=out_T, mode="nearest")     # upsample content to mel length
        s = self.spk(spk_emb)[:, :, None].expand(-1, -1, out_T)
        return self.net(torch.cat([c, s], 1))


class VoiceConverter(nn.Module):
    def __init__(self, speaker_encoder):
        super().__init__()
        self.content = ContentEncoder()
        self.decoder = Decoder()
        self.spk_enc = speaker_encoder                     # frozen

    def convert(self, source_mel, ref_mel):
        with torch.no_grad():
            spk = self.spk_enc(ref_mel)
        c = self.content(source_mel)
        return self.decoder(c, spk, source_mel.shape[-1])


def _word_classifier(by_spk, device, steps=600, seed=0):
    """Frozen content probe: mel -> word (verifies the converted output keeps the source word)."""
    X, y = [], []
    for utts in by_spk.values():
        for wi, mel in utts:
            X.append(mel); y.append(wi)
    X = torch.tensor(np.stack(X), dtype=torch.float32, device=device).unsqueeze(1)
    y = torch.tensor(y, device=device)
    net = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
                        nn.Conv2d(32, 64, 3, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d((4, 4)),
                        nn.Flatten(), nn.Linear(64 * 16, 128), nn.GELU(),
                        nn.Linear(128, len(WORDS))).to(device)
    torch.manual_seed(seed); opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        i = torch.tensor(rng.integers(0, len(X), 64))
        loss = F.cross_entropy(net(X[i] + 0.01 * torch.randn_like(X[i])), y[i])
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net


def _pad_to(mels, T):
    out = []
    for m in mels:
        m = m[:, :T] if m.shape[1] >= T else np.pad(m, ((0, 0), (0, T - m.shape[1])))
        out.append(m)
    return np.stack(out)


def train(steps=8000, seed=0, device=DEV, batch=32, lr=1e-3, T=98):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    by_spk = load_with_words()
    train_spk, _ = split_speakers(by_spk, seed=seed)
    spk_enc = SpeakerEncoder().to(device)
    spk_enc.load_state_dict(torch.load("runs/voiceclone.pt", map_location=device)["state_dict"])
    spk_enc.eval()
    for p in spk_enc.parameters():
        p.requires_grad_(False)
    model = VoiceConverter(spk_enc).to(device)
    opt = torch.optim.AdamW(list(model.content.parameters()) + list(model.decoder.parameters()), lr=lr)
    spks = [s for s in train_spk if len(train_spk[s]) >= 2]
    for st in range(1, steps + 1):
        model.train()
        src, ref, xref = [], [], []
        for _ in range(batch):
            s = spks[int(rng.integers(len(spks)))]
            utts = train_spk[s]
            i, j = rng.choice(len(utts), 2, replace=False)   # source + same-spk ref (recon)
            src.append(utts[i][1]); ref.append(utts[j][1])
            o = spks[int(rng.integers(len(spks)))]           # a DIFFERENT speaker (conversion)
            while o == s:
                o = spks[int(rng.integers(len(spks)))]
            xref.append(train_spk[o][int(rng.integers(len(train_spk[o])))][1])
        src = torch.tensor(_pad_to(src, T), dtype=torch.float32, device=device)
        ref = torch.tensor(_pad_to(ref, T), dtype=torch.float32, device=device)
        xref = torch.tensor(_pad_to(xref, T), dtype=torch.float32, device=device)
        # 1) RECON: content(X) + same-speaker ref -> reconstruct X
        recon = F.l1_loss(model.convert(src, ref), src)
        # 2) SPEAKER-CONSISTENCY: content(X) + DIFFERENT-speaker ref -> output must take that voice
        conv = model.convert(src, xref)
        with torch.no_grad():
            tgt_emb = F.normalize(model.spk_enc(xref), dim=-1)
        gen_emb = F.normalize(model.spk_enc(conv), dim=-1)
        spk_consist = (1 - (gen_emb * tgt_emb).sum(-1)).mean()   # cosine: gen voice == target voice
        loss = recon + 3.0 * spk_consist
        opt.zero_grad(); loss.backward(); opt.step()
        if st % max(1, steps // 8) == 0 or st == steps:
            print(f"  vcv {st}/{steps} recon {recon.item():.4f} spk-consist {spk_consist.item():.4f}",
                  flush=True)
    return model, by_spk


def evaluate(model, by_spk, device=DEV, n=400, seed=1, T=98):
    """Zero-shot conversion on HELD-OUT speakers: content from A in UNSEEN voice B."""
    train_spk, hold = split_speakers(by_spk, seed=0)
    wclf = _word_classifier(by_spk, device)
    model.eval()
    rng = np.random.default_rng(seed)
    # target-speaker embedding centroids (held-out)
    cent = {}
    with torch.no_grad():
        for s, utts in hold.items():
            mels = torch.tensor(_pad_to([m for _, m in utts], T), dtype=torch.float32, device=device)
            cent[s] = F.normalize(model.spk_enc(mels).mean(0), dim=0)
    cent_mat = torch.stack(list(cent.values())); cent_spk = list(cent)
    spks = [s for s in hold if len(hold[s]) >= 1]
    voice_hit = content_hit = total = 0
    with torch.no_grad():
        for _ in range(n):
            a = spks[int(rng.integers(len(spks)))]         # source content speaker (unseen)
            b = spks[int(rng.integers(len(spks)))]         # target VOICE speaker (unseen, != a)
            while b == a:
                b = spks[int(rng.integers(len(spks)))]
            wi, src_mel = hold[a][int(rng.integers(len(hold[a])))]
            ref_mel = hold[b][int(rng.integers(len(hold[b])))][1]
            src = torch.tensor(_pad_to([src_mel], T), dtype=torch.float32, device=device)
            ref = torch.tensor(_pad_to([ref_mel], T), dtype=torch.float32, device=device)
            gen = model.convert(src, ref)                  # A's word in B's voice
            # voice: re-embed generated; nearest held-out centroid should be B
            emb = F.normalize(model.spk_enc(gen)[0], dim=0)
            pred_b = cent_spk[int((cent_mat @ emb).argmax())]
            voice_hit += (pred_b == b)
            # content: word classifier reads source word off the generated mel
            content_hit += int(wclf(gen.unsqueeze(1)).argmax(-1).item() == wi)
            total += 1
    return {"voice_match": voice_hit / total, "content_acc": content_hit / total,
            "n_holdout_speakers": len(hold), "voice_chance": 1 / len(hold), "n": total}


def run(steps=8000, seed=0, device=DEV):
    model, by_spk = train(steps=steps, seed=seed, device=device)
    ev = evaluate(model, by_spk, device=device)
    report = {"experiment": "voiceconvert_zeroshot", "steps": steps, **ev,
              "zero_shot_clone_works": ev["voice_match"] > 5 * ev["voice_chance"] and ev["content_acc"] > 0.4}
    print(f"\nZERO-SHOT voice conversion (UNSEEN source & target speakers):")
    print(f"  voice_match {ev['voice_match']:.3f} (chance {ev['voice_chance']:.3f}, {ev['n_holdout_speakers']} speakers)")
    print(f"  content_acc {ev['content_acc']:.3f}")
    print(f"  zero-shot clone works: {report['zero_shot_clone_works']}", flush=True)
    return report, model


def selftest():
    if not os.path.exists(os.path.join(ROOT, "manifest.json")) or not os.path.exists("runs/voiceclone.pt"):
        print("need data/gsc_spk + runs/voiceclone.pt (run thinking.voiceclone --fetch --train); skipping")
        return
    by_spk = load_with_words(); assert len(by_spk) >= 8
    se = SpeakerEncoder(); se.load_state_dict(torch.load("runs/voiceclone.pt", map_location="cpu")["state_dict"])
    m = VoiceConverter(se)
    src = torch.randn(2, N_MELS, 98); ref = torch.randn(2, N_MELS, 98)
    out = m.convert(src, ref); assert out.shape == (2, N_MELS, 98), out.shape
    model, bs = train(steps=3, seed=0, device="cpu")
    ev = evaluate(model, bs, device="cpu", n=20)
    assert "voice_match" in ev and "content_acc" in ev
    print("voiceconvert selftest OK")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/voiceconvert.json")
    ap.add_argument("--checkpoint", default="runs/voiceconvert.pt")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest(); return
    if args.train:
        report, model = run(steps=args.steps, seed=args.seed)
        os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
        torch.save({"state_dict": model.state_dict()}, args.checkpoint)
        json.dump(report, open(args.out, "w"), indent=1)
        print(f"saved -> {args.out}")
        return
    ap.error("choose --selftest / --train")


if __name__ == "__main__":
    main()
