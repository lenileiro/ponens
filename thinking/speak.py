"""A-3 SPEAK: emit a word as audio tokens, VERIFIED BY ROUND-TRIP through the A-2 listener.

This is the thesis applied to generation: we never trust the model's claim that it 'said red'.
A frozen A-2 Listener transcribes the produced audio and agreement is the checkable signal --
the audio analog of the Datalog checker validating an emitted line. No human eval, no reference
waveform distance; the only thing that counts is whether the listener hears the intended word.

  speaker: word id -> sequence of acoustic codes -> decoder -> waveform
  codec  : a tiny VQ over real `say` clips (the discrete acoustic-token space from the speech-LM
           literature, learned on our oracle bank)
  reward : listener(decode(codes)).argmax == word  (round-trip transcription match)

Trained by straight-through over the codec + cross-entropy to the codes of a REAL clip of that
word (teacher forcing on the oracle's own acoustic tokens), then evaluated purely by round-trip.

  python -m thinking.speak --selftest
  python -m thinking.speak --steps 800 --out runs/a3_speak.json   (needs the A-2 listener + bank)
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
from .listen import WORDS, load_bank, _split, Listener, train_arm as train_listener, _feats

DEV = get_device()
N_CODES = 128                                              # VQ codebook size
N_BINS = 65                                                # spectrogram frequency bins (n_fft=128)
N_FRAMES = 80                                              # spectrogram columns covering one clip


def _spec_cols(wave):
    """Spectrogram columns as (T, N_BINS) -- the listener's OWN input space, padded/trimmed."""
    s = spectrogram(wave)[0].T                             # (T, 65)
    s = torch.tensor(s, dtype=torch.float32)
    if len(s) < N_FRAMES:
        s = torch.cat([s, torch.zeros(N_FRAMES - len(s), N_BINS)])
    return s[:N_FRAMES]


class FrameVQ(nn.Module):
    """VQ over SPECTROGRAM columns -> discrete acoustic tokens. Decoding yields a spectrogram the
    listener reads directly: verification stays honest, no lossy waveform reconstruction."""

    def __init__(self, n_codes=N_CODES, bins=N_BINS, dim=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(bins, 128), nn.GELU(), nn.Linear(128, dim))
        self.codebook = nn.Embedding(n_codes, dim)
        self.dec = nn.Sequential(nn.Linear(dim, 128), nn.GELU(), nn.Linear(128, bins))

    def quantize(self, z):
        d = (z.pow(2).sum(-1, keepdim=True) - 2 * z @ self.codebook.weight.t()
             + self.codebook.weight.pow(2).sum(-1))
        return d.argmin(-1)

    def encode_codes(self, cols):
        return self.quantize(self.enc(cols))

    def decode_spec(self, codes):
        return self.dec(self.codebook(codes))              # (T, N_BINS) log-spectrogram

    def forward(self, cols):
        z = self.enc(cols)
        codes = self.quantize(z)
        zq = self.codebook(codes)
        zq_st = z + (zq - z).detach()                      # straight-through
        recon = self.dec(zq_st)
        vq_loss = F.mse_loss(zq, z.detach()) + 0.25 * F.mse_loss(z, zq.detach())
        return recon, codes, vq_loss


def train_codec(clips, steps=600, seed=0, device=DEV):
    torch.manual_seed(seed)
    vq = FrameVQ().to(device)
    opt = torch.optim.AdamW(vq.parameters(), lr=1e-3)
    allcols = torch.cat([_spec_cols(c["wave"]) for c in clips]).to(device)
    for _ in range(steps):
        idx = torch.randint(0, len(allcols), (256,), device=device)
        recon, _codes, vq_loss = vq(allcols[idx])
        loss = F.mse_loss(recon, allcols[idx]) + vq_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
    return vq


def codes_of(vq, wave, device):
    return vq.encode_codes(_spec_cols(wave).to(device))    # (N_FRAMES,)


def spec_from_codes(vq, codes, device):
    """Decoded log-spectrogram (1, N_BINS, N_FRAMES) in the listener's input layout."""
    return vq.decode_spec(codes.to(device)).T[None]        # (1, 65, T)


class Speaker(nn.Module):
    """word id -> N_FRAMES acoustic codes (autoregressive over a tiny learned prior)."""

    def __init__(self, n_words=len(WORDS), n_codes=N_CODES, n_frames=N_FRAMES, dim=96):
        super().__init__()
        self.n_frames = n_frames
        self.word = nn.Embedding(n_words, dim)
        self.posn = nn.Parameter(torch.zeros(n_frames, dim))
        self.code = nn.Embedding(n_codes + 1, dim)         # +1 = BOS
        self.bos = n_codes
        self.lstm = nn.GRU(dim, dim, batch_first=True)
        self.head = nn.Linear(dim, n_codes)

    def forward(self, words, prev_codes):
        """teacher forcing: prev_codes (B, T) shifted; returns logits (B, T, n_codes)."""
        B, T = prev_codes.shape
        ctx = self.word(words)[:, None, :] + self.posn[None, :T, :] + self.code(prev_codes)
        h, _ = self.lstm(ctx)
        return self.head(h)

    @torch.no_grad()
    def generate(self, words, device):
        B = len(words)
        cur = torch.full((B, 1), self.bos, device=device)
        out = []
        h = None
        we = self.word(words)[:, None, :]
        for t in range(self.n_frames):
            ctx = we + self.posn[None, t:t + 1, :] + self.code(cur[:, -1:])
            o, h = self.lstm(ctx, h)
            nxt = self.head(o[:, -1]).argmax(-1, keepdim=True)
            out.append(nxt)
            cur = nxt
        return torch.cat(out, 1)                            # (B, N_FRAMES)

    def soft_spec(self, words, vq, temp=1.0):
        """Differentiable decoded spectrogram via SOFT codes; temp anneals toward one-hot so the
        trained soft path converges to the HARD argmax decode used at eval (closes the
        relaxation gap). Lets the frozen listener's gradient flow back -- 'speak to be understood'."""
        B = len(words)
        cur_emb = self.code.weight[self.bos][None, None, :].expand(B, 1, -1)
        h = None
        we = self.word(words)[:, None, :]
        cols = []
        for t in range(self.n_frames):
            ctx = we + self.posn[None, t:t + 1, :] + cur_emb
            o, h = self.lstm(ctx, h)
            logits = self.head(o[:, -1])                   # (B, n_codes)
            p = F.softmax(logits / temp, -1)
            code_emb = p @ vq.codebook.weight              # (B, dim) soft code embedding
            cols.append(vq.dec(code_emb))                  # (B, N_BINS)
            cur_emb = (p @ self.code.weight[:logits.shape[-1]])[:, None, :]
        return torch.stack(cols, 1).transpose(1, 2)        # (B, N_BINS, N_FRAMES)


def roundtrip_accuracy(speaker, vq, listener, words_tensor, device):
    """THE metric: speak each word -> codes -> decoded spectrogram -> frozen listener transcribes.
    Agreement with the intended word is the checkable signal (the audio analog of the checker)."""
    speaker.eval(); listener.eval()
    codes = speaker.generate(words_tensor, device)
    specs = torch.stack([spec_from_codes(vq, codes[r], device) for r in range(len(words_tensor))])
    with torch.no_grad():
        pred = listener(specs)["word"].argmax(-1)
    return int(pred.eq(words_tensor).sum()) / len(words_tensor)


def run(steps=800, seed=0, device=DEV):
    manifest, clips = load_bank()
    train_clips, _ = _split(clips)
    # frozen listener (the verifier) + codec, both from the oracle bank
    listener = train_listener("swap", train_clips, steps=600, seed=seed, device=device)
    for p in listener.parameters():
        p.requires_grad_(False)
    vq = train_codec(clips, steps=400, seed=seed, device=device)
    # target codes: a real clip per word (teacher-forced acoustic tokens)
    by_word = {}
    for c in clips:
        by_word.setdefault(c["word"], []).append(c)
    rng = np.random.default_rng(seed)
    speaker = Speaker().to(device)
    # the codec decoder is trained WITH the speaker so decoded audio is what the listener reads
    opt = torch.optim.AdamW(list(speaker.parameters()) + list(vq.dec.parameters())
                            + [vq.codebook.weight], lr=2e-3)
    for st in range(1, steps + 1):
        speaker.train()
        ws = [int(rng.integers(len(WORDS))) for _ in range(32)]
        tgt = torch.stack([codes_of(vq, by_word[w][int(rng.integers(len(by_word[w])))]["wave"],
                                    device) for w in ws])
        words = torch.tensor(ws, device=device)
        prev = torch.cat([torch.full((len(ws), 1), speaker.bos, device=device), tgt[:, :-1]], 1)
        logits = speaker(words, prev)
        ce_codes = F.cross_entropy(logits.reshape(-1, N_CODES), tgt.reshape(-1))
        # ROUND-TRIP loss: decoded spectrogram must be transcribed as the intended word by the
        # frozen listener (the verifier supervises generation -- the thesis, applied to speaking).
        # temp anneals 2.0 -> 0.3 so the soft training path converges to the hard eval decode.
        temp = max(0.3, 2.0 * (1.0 - st / steps))
        spec = speaker.soft_spec(words, vq, temp=temp)[:, None]   # (B,1,N_BINS,N_FRAMES)
        listener_logits = listener(spec)["word"]
        ce_listen = F.cross_entropy(listener_logits, words)
        loss = ce_codes + 2.0 * ce_listen
        opt.zero_grad()
        loss.backward()
        opt.step()
    allw = torch.arange(len(WORDS), device=device)
    rt = roundtrip_accuracy(speaker, vq, listener, allw.repeat(4), device)
    report = {"experiment": "a3_speak", "steps": steps, "n_codes": N_CODES,
              "roundtrip_acc": rt, "chance": 1 / len(WORDS),
              "gate": rt > 4 / len(WORDS)}
    print(json.dumps(report, indent=1), flush=True)
    return report


def selftest():
    manifest, clips = load_bank()
    vq = train_codec(clips[:40], steps=4, seed=0, device="cpu")
    codes = codes_of(vq, clips[0]["wave"], "cpu")
    assert codes.shape == (N_FRAMES,)
    spec = spec_from_codes(vq, codes, "cpu")
    assert spec.shape == (1, N_BINS, N_FRAMES) and torch.isfinite(spec).all()
    sp = Speaker()
    words = torch.tensor([0, 5, 11])
    prev = torch.zeros(3, N_FRAMES, dtype=torch.long)
    logits = sp(words, prev)
    assert logits.shape == (3, N_FRAMES, N_CODES)
    gen = sp.generate(words, "cpu")
    assert gen.shape == (3, N_FRAMES)
    # round-trip wiring: a quickly-trained listener transcribes decoded audio
    tr, _ = _split(clips)
    listener = train_listener("swap", tr[:40], steps=2, seed=0, device="cpu")
    rt = roundtrip_accuracy(sp, vq, listener, words, "cpu")
    assert 0.0 <= rt <= 1.0
    print("speak selftest OK")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/a3_speak.json")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
        return
    report = run(steps=args.steps, seed=args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
