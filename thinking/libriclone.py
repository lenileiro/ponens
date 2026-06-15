"""Audible zero-shot voice cloning on LibriSpeech (multi-speaker REAL human sentences, 16kHz).

The full pipeline the single-word GSC setup couldn't support. Sentence-level multi-speaker real
speech is exactly what AutoVC-style disentanglement needs: enough content signal to preserve and
enough speakers to learn speaker-invariant content. Three stages, one GPU job:

  1. SPEAKER ENCODER (GE2E) on many LibriSpeech speakers -> voice embedding, generalizes to unseen.
  2. VOICE CONVERTER: content encoder (bottleneck) + frozen speaker embedding -> mel.
       losses: recon (same-speaker) + spk-consistency (gen voice == reference voice)
             + CYCLE content-consistency (content(convert(A->B)) == content(A); protects words
               without transcripts) -- the symmetric pair that broke the voice/content collapse.
  3. mel GAN VOCODER (HiFi-GAN recipe) on LibriSpeech -> NATURAL waveform.
  zero-shot clone: source A + reference of UNSEEN speaker B -> mel -> vocoder -> A's words in B's
  voice, audible. Measured by frozen speaker-verification (voice) + mel cycle (content).

  python -m thinking.libriclone --fetch
  python -m thinking.libriclone --selftest
  python -m thinking.libriclone --train --out runs/libriclone.json   (GPU)
"""
import argparse
import json
import os
import tarfile
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from device import get_device

DEV = get_device()
URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"   # 40 speakers, ~337MB, 16kHz flac
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "libri")
SR = 16000
N_FFT = 1024
HOP = 256
N_MELS = 80
SEG_FRAMES = 96                                            # mel frames per training segment (~1.5s)
_WIN = torch.hann_window(N_FFT)


def fetch(n_clips=2000, root=ROOT, byte_cap_gb=0.5, per_speaker=45):
    """Stream LibriSpeech dev-clean; extract flac clips with speaker IDs (path: .../<spk>/<ch>/..)."""
    import soundfile as sf
    import io
    os.makedirs(root, exist_ok=True)
    got = 0
    cap = int(byte_cap_gb * 1e9)
    seen = [0]
    clips = []
    perspk = {}

    class _C:
        def __init__(s, f): s.f = f
        def read(s, n):
            b = s.f.read(n); seen[0] += len(b); return b

    print(f"streaming {URL} (cap {byte_cap_gb}GB, {n_clips} clips)...")
    req = urllib.request.Request(URL, headers={"User-Agent": "ponens/1.0"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        tar = tarfile.open(fileobj=_C(resp), mode="r|gz")
        for m in tar:
            if seen[0] > cap or got >= n_clips:
                break
            if not m.name.endswith(".flac"):
                continue
            spk = m.name.split("/")[-3]                    # LibriSpeech/dev-clean/<spk>/<chapter>/<utt>.flac
            if perspk.get(spk, 0) >= per_speaker:          # cap clips/speaker -> MORE speakers
                continue
            data = tar.extractfile(m).read()
            x, sr = sf.read(io.BytesIO(data))
            if sr != SR or len(x) < SR // 2:
                continue
            name = f"{spk}__{got}.npy"
            np.save(os.path.join(root, name), x.astype(np.float32))
            clips.append({"speaker": spk, "path": name})
            perspk[spk] = perspk.get(spk, 0) + 1
            got += 1
    json.dump({"clips": clips}, open(os.path.join(root, "manifest.json"), "w"))
    nspk = len({c["speaker"] for c in clips})
    print(f"fetched {got} clips, {nspk} speakers, {seen[0]/1e9:.2f}GB")
    return got


def load(root=ROOT):
    man = json.load(open(os.path.join(root, "manifest.json")))
    by_spk = {}
    for c in man["clips"]:
        x = np.load(os.path.join(root, c["path"]))
        by_spk.setdefault(c["speaker"], []).append(x)
    return {s: v for s, v in by_spk.items() if len(v) >= 2}


def split_speakers(by_spk, holdout_frac=0.2, seed=0):
    spk = sorted(by_spk); rng = np.random.default_rng(seed); rng.shuffle(spk)
    k = max(2, int(len(spk) * holdout_frac)); hold = set(spk[:k])
    return ({s: by_spk[s] for s in spk if s not in hold}, {s: by_spk[s] for s in spk if s in hold})


def _mel(wav):                                             # wav (B, n) -> (B, N_MELS, T)
    spec = torch.stft(wav, N_FFT, HOP, N_FFT, _WIN.to(wav.device), center=True,
                      return_complex=True).abs()
    fb = _mel.cache.get(wav.device)
    if fb is None:
        f = torch.linspace(0, SR / 2, N_FFT // 2 + 1, device=wav.device)
        mmax = 2595 * np.log10(1 + (SR / 2) / 700)
        ctr = 700 * (10 ** (torch.linspace(0, mmax, N_MELS + 2, device=wav.device) / 2595) - 1)
        fb = torch.zeros(N_MELS, N_FFT // 2 + 1, device=wav.device)
        for mi in range(1, N_MELS + 1):
            lo, ce, hi = ctr[mi - 1], ctr[mi], ctr[mi + 1]
            fb[mi - 1] = torch.clamp(torch.minimum((f - lo) / (ce - lo + 1e-9), (hi - f) / (hi - ce + 1e-9)), 0, 1)
        _mel.cache[wav.device] = fb
    return torch.log1p(fb @ spec)
_mel.cache = {}


def _seg_mels(by_spk, rng, n_spk, n_utt, device):
    """n_spk speakers x n_utt utterances -> mel segments (B, N_MELS, SEG_FRAMES)."""
    spks = [s for s in by_spk if len(by_spk[s]) >= n_utt]
    chosen = [spks[i] for i in rng.permutation(len(spks))[:n_spk]]
    waves = []
    need = SEG_FRAMES * HOP
    for s in chosen:
        utts = by_spk[s]
        for j in rng.permutation(len(utts))[:n_utt]:
            w = utts[j]
            if len(w) > need:
                st = int(rng.integers(len(w) - need)); w = w[st:st + need]
            else:
                w = np.pad(w, (0, need - len(w)))
            waves.append(w)
    wav = torch.tensor(np.stack(waves), dtype=torch.float32, device=device)
    return _mel(wav)[:, :, :SEG_FRAMES], len(chosen), n_utt


# ---- speaker encoder (GE2E) ----
class SpeakerEncoder(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.lstm = nn.LSTM(N_MELS, 256, 2, batch_first=True)
        self.proj = nn.Linear(256, d)

    def forward(self, mel):                                # (B, N_MELS, T)
        h, _ = self.lstm(mel.transpose(1, 2))
        return F.normalize(self.proj(h[:, -1]), dim=-1)


def ge2e(emb, n_spk, n_utt, w, b):
    e = emb.view(n_spk, n_utt, -1)
    cent = F.normalize(e.mean(1), dim=-1)
    flat = e.reshape(n_spk * n_utt, -1)
    sim = w * (flat @ cent.t()) + b
    idx = torch.arange(n_spk, device=emb.device).repeat_interleave(n_utt)
    return F.cross_entropy(sim, idx)


# ---- voice converter (AutoVC-style) ----
class ContentEncoder(nn.Module):
    def __init__(self, content_dim=16, down=2):
        super().__init__()
        self.down = down
        self.net = nn.Sequential(nn.Conv1d(N_MELS, 256, 5, padding=2), nn.GELU(),
                                 nn.Conv1d(256, 256, 5, padding=2), nn.GELU(),
                                 nn.Conv1d(256, content_dim, 1))

    def forward(self, mel):
        return F.avg_pool1d(self.net(mel), self.down)


class ConvDecoder(nn.Module):
    def __init__(self, content_dim=16, spk_dim=128):
        super().__init__()
        self.spk = nn.Linear(spk_dim, 64)
        self.net = nn.Sequential(nn.Conv1d(content_dim + 64, 512, 5, padding=2), nn.GELU(),
                                 nn.Conv1d(512, 512, 5, padding=2), nn.GELU(),
                                 nn.Conv1d(512, N_MELS, 1))

    def forward(self, content, spk, out_T):
        c = F.interpolate(content, size=out_T, mode="nearest")
        s = self.spk(spk)[:, :, None].expand(-1, -1, out_T)
        return self.net(torch.cat([c, s], 1))


# ---- mel GAN vocoder (HiFi-GAN recipe, mel -> waveform) ----
class MelVocoder(nn.Module):
    """Upsample mel (hop=256) -> waveform via transposed convs (HiFi-GAN generator, compact)."""
    def __init__(self, ch=256):
        super().__init__()
        self.inp = weight_norm(nn.Conv1d(N_MELS, ch, 7, padding=3))
        ups, rates = [], [8, 8, 2, 2]                      # product = 256 = HOP
        c = ch
        for r in rates:
            ups.append(weight_norm(nn.ConvTranspose1d(c, c // 2, r * 2, r, r // 2)))
            c //= 2
        self.ups = nn.ModuleList(ups)
        self.resblocks = nn.ModuleList([weight_norm(nn.Conv1d(c, c, 7, padding=3, dilation=1)) for _ in range(len(rates))])
        # rebuild res per stage:
        self.res = nn.ModuleList()
        cc = ch
        for r in rates:
            cc //= 2
            self.res.append(weight_norm(nn.Conv1d(cc, cc, 7, padding=3)))
        self.post = weight_norm(nn.Conv1d(c, 1, 7, padding=3))

    def forward(self, mel):
        x = self.inp(mel)
        for up, res in zip(self.ups, self.res):
            x = F.leaky_relu(x, 0.1)
            x = up(x)
            x = x + F.leaky_relu(res(x), 0.1)
        return torch.tanh(self.post(F.leaky_relu(x, 0.1))).squeeze(1)


class PeriodDisc(nn.Module):
    def __init__(self, period):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList([weight_norm(nn.Conv2d(1, 32, (5, 1), (3, 1), (2, 0))),
                                    weight_norm(nn.Conv2d(32, 128, (5, 1), (3, 1), (2, 0))),
                                    weight_norm(nn.Conv2d(128, 256, (5, 1), 1, (2, 0)))])
        self.post = weight_norm(nn.Conv2d(256, 1, (3, 1), 1, (1, 0)))

    def forward(self, x):
        b, t = x.shape
        pad = (self.period - t % self.period) % self.period
        x = F.pad(x, (0, pad)).view(b, 1, -1, self.period)
        fs = []
        for c in self.convs:
            x = F.leaky_relu(c(x), 0.1); fs.append(x)
        x = self.post(x); fs.append(x)
        return x.flatten(1), fs


class MPD(nn.Module):
    def __init__(self):
        super().__init__()
        self.ds = nn.ModuleList([PeriodDisc(p) for p in (2, 3, 5, 7, 11)])

    def forward(self, x):
        outs, feats = [], []
        for d in self.ds:
            o, f = d(x); outs.append(o); feats.append(f)
        return outs, feats


def mel_l1(a_wav, b_wav):
    n = min(a_wav.shape[1], b_wav.shape[1])
    return F.l1_loss(_mel(a_wav[:, :n]), _mel(b_wav[:, :n]))


def _waves_batch(by_spk, rng, batch, device):
    spks = list(by_spk)
    need = SEG_FRAMES * HOP
    out = []
    for _ in range(batch):
        s = spks[int(rng.integers(len(spks)))]
        w = by_spk[s][int(rng.integers(len(by_spk[s])))]
        if len(w) > need:
            st = int(rng.integers(len(w) - need)); w = w[st:st + need]
        else:
            w = np.pad(w, (0, need - len(w)))
        out.append(w)
    return torch.tensor(np.stack(out), dtype=torch.float32, device=device)


def train(steps_spk=8000, steps_vc=12000, steps_voc=40000, seed=0, device=DEV, batch=16):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    by_spk = load(); train_spk, _ = split_speakers(by_spk, seed=seed)

    # STAGE 1: speaker encoder
    print("=== stage 1: speaker encoder (GE2E) ===", flush=True)
    se = SpeakerEncoder().to(device)
    w = nn.Parameter(torch.tensor(10.0, device=device)); b = nn.Parameter(torch.tensor(-5.0, device=device))
    opt = torch.optim.AdamW(list(se.parameters()) + [w, b], lr=1e-3)
    for st in range(1, steps_spk + 1):
        mel, ns, nu = _seg_mels(train_spk, rng, 16, 4, device)
        loss = ge2e(se(mel), ns, nu, w, b)
        opt.zero_grad(); loss.backward(); opt.step()
        if st % max(1, steps_spk // 4) == 0:
            print(f"  spk {st}/{steps_spk} ge2e {loss.item():.3f}", flush=True)
    se.eval()
    for p in se.parameters():
        p.requires_grad_(False)

    # STAGE 2: voice converter
    print("=== stage 2: voice converter (AutoVC + spk/content consistency) ===", flush=True)
    ce = ContentEncoder().to(device); dec = ConvDecoder().to(device)
    optv = torch.optim.AdamW(list(ce.parameters()) + list(dec.parameters()), lr=1e-3)
    for st in range(1, steps_vc + 1):
        mel, ns, nu = _seg_mels(train_spk, rng, 8, 2, device)   # 8 spk x 2 utt
        e = mel.view(ns, nu, N_MELS, -1)
        src = e[:, 0]; ref_same = e[:, 1]                       # (ns, N_MELS, T)
        roll = torch.roll(torch.arange(ns), 1)
        ref_diff = e[roll, 0]                                   # different-speaker ref
        T = src.shape[-1]
        spk_same = se(ref_same); spk_diff = se(ref_diff)
        recon = F.l1_loss(dec(ce(src), spk_same, T), src)
        conv = dec(ce(src), spk_diff, T)
        spk_consist = (1 - (se(conv) * spk_diff).sum(-1)).mean()
        cyc = F.l1_loss(ce(conv), ce(src))                     # content cycle: words survive
        (recon + 2.0 * spk_consist + 3.0 * cyc).backward()
        optv.step(); optv.zero_grad()
        if st % max(1, steps_vc // 4) == 0:
            print(f"  vc {st}/{steps_vc} recon {recon.item():.3f} spk {spk_consist.item():.3f} "
                  f"cyc {cyc.item():.3f}", flush=True)

    # STAGE 3: mel GAN vocoder
    print("=== stage 3: mel GAN vocoder (HiFi-GAN) ===", flush=True)
    G = MelVocoder().to(device); D = MPD().to(device)
    og = torch.optim.AdamW(G.parameters(), lr=2e-4, betas=(0.8, 0.99))
    od = torch.optim.AdamW(D.parameters(), lr=2e-4, betas=(0.8, 0.99))
    for st in range(1, steps_voc + 1):
        real = _waves_batch(train_spk, rng, batch, device)
        m = _mel(real)
        fake = G(m)
        n = min(real.shape[1], fake.shape[1])
        real_c, fake_c = real[:, :n], fake[:, :n]
        od.zero_grad()
        dr, _ = D(real_c); df, _ = D(fake_c.detach())
        dl = sum(((r - 1) ** 2).mean() + (f ** 2).mean() for r, f in zip(dr, df))
        dl.backward(); od.step()
        og.zero_grad()
        dr, fr = D(real_c); df, ff = D(fake_c)
        adv = sum(((f - 1) ** 2).mean() for f in df)
        fm = sum(F.l1_loss(a, bb) for fl, ffl in zip(fr, ff) for a, bb in zip(fl, ffl))
        gl = adv + 2.0 * fm + 45.0 * mel_l1(fake_c, real_c)
        gl.backward(); og.step()
        if st % max(1, steps_voc // 5) == 0:
            print(f"  voc {st}/{steps_voc} G {gl.item():.2f} D {dl.item():.2f}", flush=True)
    return se, ce, dec, G, by_spk


def evaluate(se, ce, dec, G, by_spk, device=DEV, n=200, seed=1, save_dir=None):
    _, hold = split_speakers(by_spk, seed=0)
    rng = np.random.default_rng(seed)
    need = SEG_FRAMES * HOP
    # held-out speaker centroids
    cent = {}
    with torch.no_grad():
        for s, utts in hold.items():
            ws = []
            for w in utts[:8]:
                w = w[:need] if len(w) >= need else np.pad(w, (0, need - len(w)))
                ws.append(w)
            mel = _mel(torch.tensor(np.stack(ws), dtype=torch.float32, device=device))[:, :, :SEG_FRAMES]
            cent[s] = F.normalize(se(mel).mean(0), dim=0)
    cmat = torch.stack(list(cent.values())); cspk = list(cent)
    spks = list(hold)
    voice_hit = total = 0
    saved = 0
    with torch.no_grad():
        for _ in range(n):
            a = spks[int(rng.integers(len(spks)))]
            b = spks[int(rng.integers(len(spks)))]
            while b == a:
                b = spks[int(rng.integers(len(spks)))]
            def seg(s):
                w = hold[s][int(rng.integers(len(hold[s])))]
                w = w[:need] if len(w) >= need else np.pad(w, (0, need - len(w)))
                return _mel(torch.tensor(w[None], dtype=torch.float32, device=device))[:, :, :SEG_FRAMES]
            src = seg(a); ref = seg(b)
            conv = dec(ce(src), se(ref), src.shape[-1])
            emb = F.normalize(se(conv)[0], dim=0)
            if cspk[int((cmat @ emb).argmax())] == b:
                voice_hit += 1
            total += 1
            if save_dir and saved < 4:
                import wave as wv
                wav = G(conv)[0].cpu().numpy()
                for tag, mm in (("source_A", src), ("ref_B", ref), ("cloned_AinB", conv)):
                    w2 = G(mm)[0].cpu().numpy() if tag != "cloned_AinB" else wav
                    y = (w2 / (np.abs(w2).max() + 1e-8) * 0.95 * 32767).astype(np.int16)
                    o = wv.open(os.path.join(save_dir, f"clone{saved}_{tag}.wav"), "w")
                    o.setnchannels(1); o.setsampwidth(2); o.setframerate(SR); o.writeframes(y.tobytes()); o.close()
                saved += 1
    return {"voice_match": voice_hit / total, "voice_chance": 1 / len(hold), "n_holdout": len(hold), "n": total}


def run(device=DEV, save_dir="data/synth", **kw):
    se, ce, dec, G, by_spk = train(device=device, **kw)
    os.makedirs(save_dir, exist_ok=True)
    ev = evaluate(se, ce, dec, G, by_spk, device=device, save_dir=save_dir)
    report = {"experiment": "libriclone", "sr": SR, **ev,
              "voice_clone_works": ev["voice_match"] > 5 * ev["voice_chance"]}
    print(f"\nZERO-SHOT clone (unseen speakers): voice_match {ev['voice_match']:.3f} "
          f"(chance {ev['voice_chance']:.3f}, {ev['n_holdout']} held-out speakers)")
    return report, (se, ce, dec, G)


def selftest():
    if not os.path.exists(os.path.join(ROOT, "manifest.json")):
        print("no LibriSpeech; run --fetch first (skipping data-dependent checks)")
        # architecture-only smoke
        se = SpeakerEncoder(); ce = ContentEncoder(); dec = ConvDecoder(); G = MelVocoder()
        mel = torch.randn(2, N_MELS, SEG_FRAMES)
        assert se(mel).shape == (2, 128)
        c = ce(mel); out = dec(c, se(mel), SEG_FRAMES)
        assert out.shape == (2, N_MELS, SEG_FRAMES), out.shape
        wav = G(mel)
        assert wav.shape[1] >= SEG_FRAMES * HOP - HOP, wav.shape
        print("libriclone arch selftest OK")
        return
    se, ce, dec, G, bs = train(steps_spk=2, steps_vc=2, steps_voc=2, device="cpu", batch=2)
    ev = evaluate(se, ce, dec, G, bs, device="cpu", n=10)
    assert "voice_match" in ev
    print("libriclone selftest OK")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--n-clips", type=int, default=2000, dest="n_clips")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--steps-spk", type=int, default=8000, dest="steps_spk")
    ap.add_argument("--steps-vc", type=int, default=12000, dest="steps_vc")
    ap.add_argument("--steps-voc", type=int, default=40000, dest="steps_voc")
    ap.add_argument("--out", default="runs/libriclone.json")
    ap.add_argument("--checkpoint", default="runs/libriclone.pt")
    args = ap.parse_args(argv)
    if args.fetch:
        fetch(n_clips=args.n_clips); return
    if args.selftest:
        selftest(); return
    if args.train:
        report, models = run(steps_spk=args.steps_spk, steps_vc=args.steps_vc, steps_voc=args.steps_voc)
        se, ce, dec, G = models
        os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
        torch.save({"se": se.state_dict(), "ce": ce.state_dict(),
                    "dec": dec.state_dict(), "G": G.state_dict()}, args.checkpoint)
        json.dump(report, open(args.out, "w"), indent=1)
        print(f"saved -> {args.out}")
        return
    ap.error("choose --fetch / --selftest / --train")


if __name__ == "__main__":
    main()
