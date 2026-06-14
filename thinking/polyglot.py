"""A-4b POLYGLOT: learn a language's pronunciation rules from a few listening examples, then
pronounce ANY word in that language. The user's target: "know how to pronounce anything given the
speech rules for any language, and learn the rules from a few listening examples."

This is the pronunciation-rule learning thesis applied to speech. Phonemes are universal (fixed
audio units); a LANGUAGE is a grapheme->phoneme rule system (which letter sounds like which
phoneme), random per language. The model never sees a language's rules stated -- it must infer
them by listening to K (word, audio) support pairs, then apply them to a held-out query word.

Meta-learning setup, evaluated on HELD-OUT LANGUAGES (rule systems never seen in training):
  support: K words + their audio (phoneme spectrograms) in language L
  query  : a new word in L -> predict its phoneme sequence (then render to audio)
  metric : phoneme accuracy on query words, held-out languages (chance = 1/n_phonemes)

If accuracy >> chance on unseen languages, the model learned to LEARN pronunciation rules from
listening -- not to memorize one language. Includes a DIGRAPH variant (some letter PAIRS map to a
phoneme) to test multi-character rule inference.

  python -m thinking.polyglot --selftest
  python -m thinking.polyglot --steps 8000 --out runs/a4b_polyglot.json
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from device import get_device
from .audio import render_tone, spectrogram, PITCHES, TIMBRES

DEV = get_device()
N_PHONEMES = 16
N_LETTERS = 16                                             # alphabet a..p
PHON_FRAMES = 8                                            # spectrogram cols per phoneme
N_BINS = 65
MIN_LEN, MAX_LEN = 3, 6
SUPPORT_K = 24                                             # listening examples per language
PITCH_LIST = list(PITCHES)
HOLDOUT_LANG_SEED0 = 900_000                               # held-out languages drawn from here up


def _phoneme_audio_bank():
    """16 fixed, distinct phoneme sounds (universal across languages). Each = a short tone with a
    distinct pitch x timbre, rendered to a PHON_FRAMES spectrogram block."""
    bank = []
    rng = np.random.default_rng(12345)
    for p in range(N_PHONEMES):
        pitch = PITCH_LIST[p % len(PITCH_LIST)]
        timbre = TIMBRES[(p // len(PITCH_LIST)) % len(TIMBRES)]
        env = ("flat", "decay", "attack")[p % 3]
        wave = render_tone(pitch, timbre, env, detune=0.0, amp=1.0, phase=0.0, noise=0.0,
                           rng=rng)
        s = spectrogram(wave)[0]                           # (65, T)
        col = s[:, :PHON_FRAMES] if s.shape[1] >= PHON_FRAMES else np.pad(
            s, ((0, 0), (0, PHON_FRAMES - s.shape[1])))
        bank.append(col.astype(np.float32))
    return np.stack(bank)                                  # (N_PHONEMES, 65, PHON_FRAMES)


PHON_BANK = _phoneme_audio_bank()


def make_language(seed, digraph=False):
    """A grapheme->phoneme rule system: each letter -> a phoneme (random, many-to-one allowed).
    digraph: a few letter PAIRS override to a different phoneme (multi-char rules)."""
    rng = np.random.default_rng(seed)
    letter_map = rng.integers(0, N_PHONEMES, size=N_LETTERS)
    digraphs = {}
    if digraph:
        for _ in range(3):
            a, b = int(rng.integers(N_LETTERS)), int(rng.integers(N_LETTERS))
            digraphs[(a, b)] = int(rng.integers(N_PHONEMES))
    return {"letter_map": letter_map, "digraphs": digraphs}


def pronounce_word(letters, lang):
    """Apply the language's rules: letter sequence -> phoneme sequence (digraphs consume 2)."""
    phons, i = [], 0
    while i < len(letters):
        if i + 1 < len(letters) and (letters[i], letters[i + 1]) in lang["digraphs"]:
            phons.append(lang["digraphs"][(letters[i], letters[i + 1])])
            i += 2
        else:
            phons.append(int(lang["letter_map"][letters[i]]))
            i += 1
    return phons


def sample_word(rng):
    n = int(rng.integers(MIN_LEN, MAX_LEN + 1))
    return [int(rng.integers(N_LETTERS)) for _ in range(n)]


def phoneme_audio(phons):
    """Concatenate phoneme spectrogram blocks -> (65, len*PHON_FRAMES)."""
    return np.concatenate([PHON_BANK[p] for p in phons], axis=1)


class PhonemeEar(nn.Module):
    """Reads one phoneme audio block -> embedding (identifies the phoneme by listening)."""

    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(N_BINS * PHON_FRAMES, d), nn.GELU(),
                                 nn.Linear(d, d), nn.LayerNorm(d))

    def forward(self, blocks):                             # (..., 65, PHON_FRAMES)
        flat = blocks.reshape(*blocks.shape[:-2], N_BINS * PHON_FRAMES)
        return self.net(flat)


class Polyglot(nn.Module):
    """In-context pronunciation: read K (letter, heard-phoneme) support pairs, infer the
    letter->phoneme rule, apply to query letters. A transformer over the support set + query lets
    it handle context-dependent rules (digraphs) beyond a pure per-letter lookup."""

    def __init__(self, d=192, heads=6, layers=4):
        super().__init__()
        self.letter = nn.Embedding(N_LETTERS, d)
        self.ear = PhonemeEar(d)
        self.role = nn.Embedding(2, d)                     # 0 = support, 1 = query
        self.posn = nn.Embedding(MAX_LEN + 2, d)           # within-word position
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, activation="gelu",
                                           batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d, N_PHONEMES)

    def forward(self, support_letters, support_audio, support_pos, support_mask,
                query_letters, query_pos, query_mask):
        # support tokens: letter emb + heard-phoneme emb (the (grapheme, sound) observation)
        sup = (self.letter(support_letters) + self.ear(support_audio)
               + self.role(torch.zeros_like(support_letters)) + self.posn(support_pos))
        qry = (self.letter(query_letters)
               + self.role(torch.ones_like(query_letters)) + self.posn(query_pos))
        seq = torch.cat([sup, qry], dim=1)
        mask = torch.cat([support_mask, query_mask], dim=1)
        h = self.enc(seq, src_key_padding_mask=mask)
        q = h[:, sup.shape[1]:]                            # query positions
        return self.head(q)                                # (B, query_len, N_PHONEMES)


def _build_episode(seed, rng, digraph, device):
    """One language: K support words (cover the alphabet) + query words; tensors padded."""
    lang = make_language(seed, digraph=digraph)
    # support words, ensuring every letter appears at least once
    sup_words = [sample_word(rng) for _ in range(SUPPORT_K)]
    covered = set(l for w in sup_words for l in w)
    for missing in set(range(N_LETTERS)) - covered:
        sup_words[int(rng.integers(SUPPORT_K))] = [missing] + sample_word(rng)[:MAX_LEN - 1]
    qry_words = [sample_word(rng) for _ in range(8)]
    return lang, sup_words, qry_words


def _pack_support(sup_words, lang, device):
    letters, audios, poss = [], [], []
    for w in sup_words:
        phons = pronounce_word(w, lang)
        for pos, (lt, ph) in enumerate(zip(w, phons)):
            letters.append(lt)
            audios.append(PHON_BANK[ph])
            poss.append(min(pos, MAX_LEN + 1))
    L = len(letters)
    return (torch.tensor(letters, device=device),
            torch.tensor(np.stack(audios), dtype=torch.float32, device=device),
            torch.tensor(poss, device=device), L)


def _pack_batch(seeds, rng, digraph, device):
    episodes = [_build_episode(s, rng, digraph, device) for s in seeds]
    sup_packed = [_pack_support(sw, lang, device) for lang, sw, _ in episodes]
    max_sup = max(p[3] for p in sup_packed)
    B = len(seeds)
    sL = torch.zeros(B, max_sup, dtype=torch.long, device=device)
    sA = torch.zeros(B, max_sup, N_BINS, PHON_FRAMES, device=device)
    sP = torch.zeros(B, max_sup, dtype=torch.long, device=device)
    sM = torch.ones(B, max_sup, dtype=torch.bool, device=device)
    for b, (lt, au, po, L) in enumerate(sup_packed):
        sL[b, :L], sA[b, :L], sP[b, :L], sM[b, :L] = lt, au, po, False
    # queries: flatten all query words per episode into one padded query sequence
    qmax = MAX_LEN * 8
    qL = torch.zeros(B, qmax, dtype=torch.long, device=device)
    qP = torch.zeros(B, qmax, dtype=torch.long, device=device)
    qM = torch.ones(B, qmax, dtype=torch.bool, device=device)
    qY = torch.full((B, qmax), -100, dtype=torch.long, device=device)
    for b, (lang, _sw, qw) in enumerate(episodes):
        j = 0
        for w in qw:
            phons = pronounce_word(w, lang)
            for lt, ph in zip(w, phons):
                if j >= qmax:
                    break
                qL[b, j], qP[b, j], qM[b, j], qY[b, j] = lt, min(j % MAX_LEN, MAX_LEN + 1), False, ph
                j += 1
    return sL, sA, sP, sM, qL, qP, qM, qY


def train(steps=8000, seed=0, device=DEV, batch=16, lr=3e-4, digraph=False, d=192):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Polyglot(d=d).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for st in range(1, steps + 1):
        model.train()
        seeds = [int(rng.integers(0, HOLDOUT_LANG_SEED0)) for _ in range(batch)]  # TRAIN langs
        sL, sA, sP, sM, qL, qP, qM, qY = _pack_batch(seeds, rng, digraph, device)
        logits = model(sL, sA, sP, sM, qL, qP, qM)
        loss = F.cross_entropy(logits.reshape(-1, N_PHONEMES), qY.reshape(-1), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if st % max(1, steps // 10) == 0 or st == steps:
            print(f"  a4b {st}/{steps} loss {loss.item():.4f}", flush=True)
    return model


def evaluate(model, n_langs=60, seed=HOLDOUT_LANG_SEED0, device=DEV, digraph=False):
    """Phoneme accuracy on HELD-OUT languages (rule systems never trained on)."""
    model.eval()
    rng = np.random.default_rng(7)
    correct = total = 0
    with torch.no_grad():
        for i in range(0, n_langs, 16):
            seeds = [seed + i + j for j in range(min(16, n_langs - i))]
            sL, sA, sP, sM, qL, qP, qM, qY = _pack_batch(seeds, rng, digraph, device)
            pred = model(sL, sA, sP, sM, qL, qP, qM).argmax(-1)
            m = qY != -100
            correct += int((pred.eq(qY) & m).sum())
            total += int(m.sum())
    return correct / max(1, total)


def run(steps=8000, seed=0, device=DEV):
    report = {"experiment": "a4b_polyglot", "steps": steps, "n_phonemes": N_PHONEMES,
              "n_letters": N_LETTERS, "support_k": SUPPORT_K, "chance": 1 / N_PHONEMES}
    print("=== per-letter rules ===")
    m = train(steps=steps, seed=seed, device=device, digraph=False)
    report["holdout_lang_phoneme_acc"] = evaluate(m, device=device, digraph=False)
    print("=== + digraph (multi-char) rules ===")
    md = train(steps=steps, seed=seed, device=device, digraph=True)
    report["holdout_lang_digraph_acc"] = evaluate(md, device=device, digraph=True)
    report["learns_rules_from_listening"] = report["holdout_lang_phoneme_acc"] > 0.8
    print(f"\nHELD-OUT LANGUAGE phoneme acc: {report['holdout_lang_phoneme_acc']:.3f} "
          f"(chance {1/N_PHONEMES:.3f})")
    print(f"  with digraph rules: {report['holdout_lang_digraph_acc']:.3f}")
    print(f"learns rules from listening (>0.8): {report['learns_rules_from_listening']}", flush=True)
    return report, m


def selftest():
    assert PHON_BANK.shape == (N_PHONEMES, N_BINS, PHON_FRAMES)
    lang = make_language(123, digraph=True)
    w = [0, 1, 2, 3]
    phons = pronounce_word(w, lang)
    assert len(phons) <= len(w) and all(0 <= p < N_PHONEMES for p in phons)
    # different languages pronounce the same word differently (rules matter)
    l1, l2 = make_language(1), make_language(2)
    assert pronounce_word([0, 1, 2, 3, 4], l1) != pronounce_word([0, 1, 2, 3, 4], l2)
    aud = phoneme_audio(phons)
    assert aud.shape[0] == N_BINS
    rng = np.random.default_rng(0)
    sL, sA, sP, sM, qL, qP, qM, qY = _pack_batch([1, 2], rng, False, "cpu")
    m = Polyglot(d=64, heads=4, layers=2)
    out = m(sL, sA, sP, sM, qL, qP, qM)
    assert out.shape[-1] == N_PHONEMES
    model = train(steps=3, seed=0, device="cpu")
    acc = evaluate(model, n_langs=8, device="cpu")
    assert 0 <= acc <= 1
    print("polyglot selftest OK")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/a4b_polyglot.json")
    ap.add_argument("--checkpoint", default="runs/a4b_polyglot.pt")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
        return
    report, model = run(steps=args.steps, seed=args.seed)
    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, args.checkpoint)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
