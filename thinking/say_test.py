"""TEST: can the trained model pronounce ARBITRARY text given to it post-training?

Reading the training set back is not proof of speech -- a model can overfit. The real test the user
asked for: hand it NOVEL sentences it never trained on (proper names, numbers, tongue-twisters,
unusual words), synthesize them, then run an INDEPENDENT speech recognizer (Whisper) over the audio.
If the ASR transcribes the words back correctly, the model truly PRONOUNCED arbitrary text -- an
objective, self-checking loop (no human listening required), the same closed-loop idea as the rest
of the stack.

Metric: word error rate (WER) of Whisper's transcript vs. the input text, averaged over the prompts.
PASS when mean WER is low enough that the words are clearly intelligible.

  python -m thinking.say_test --selftest                 # WER math + deps, no model
  python -m thinking.say_test --run --out runs/say_test.json --synth-out data/synth   (GPU; Whisper)
"""
import argparse
import json
import os
import re

import numpy as np

# Arbitrary, HELD-OUT prompts (not LJSpeech sentences): names, numbers, twisters, rare words.
PROMPTS = [
    "the quick brown fox jumps over the lazy dog",
    "she sells sea shells by the sea shore",
    "peter piper picked a peck of pickled peppers",
    "doctor olivia martinez lives on maple street",
    "the train departs at quarter past seven tomorrow",
    "how much wood would a woodchuck chuck if it could",
    "the museum opened in nineteen ninety three",
    "please remember to bring your umbrella and notebook",
    "a journey of a thousand miles begins with a single step",
    "the curious children explored the ancient stone castle",
]


def normalize(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).split()


def wer(ref, hyp):
    """Word error rate via Levenshtein over word tokens."""
    r, h = normalize(ref), normalize(hyp)
    if not r:
        return 0.0 if not h else 1.0
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=int)
    d[:, 0] = np.arange(len(r) + 1); d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
    return d[len(r), len(h)] / len(r)


def _asr_pipe(device):
    from transformers import pipeline
    dev = 0 if str(device).startswith("cuda") else -1
    return pipeline("automatic-speech-recognition", model="openai/whisper-base.en", device=dev)


def run(ckpt="runs/tts_fast.pt", out=None, synth_out="data/synth", device=None):
    from device import get_device
    from .prosody import speak, _write, SR
    device = device or get_device()
    asr = _asr_pipe(device)
    rows, wers = [], []
    os.makedirs(synth_out, exist_ok=True)
    for i, text in enumerate(PROMPTS):
        wav = speak(text, ckpt)
        path = os.path.join(synth_out, f"say_test_{i}.wav")
        _write(path, wav, SR)
        hyp = asr(path)["text"].strip()
        e = wer(text, hyp); wers.append(e); rows.append({"ref": text, "asr": hyp, "wer": round(e, 3)})
        print(f"  [{i}] WER {e:.2f}  ref: {text!r}\n        asr: {hyp!r}", flush=True)
    mean = float(np.mean(wers))
    report = {"experiment": "say_test_arbitrary_text_asr_roundtrip", "n": len(PROMPTS),
              "asr_model": "whisper-base.en", "mean_wer": mean,
              "intelligible_frac": float(np.mean([w < 0.5 for w in wers])),
              "pass": mean < 0.35, "rows": rows}
    print(f"\nARBITRARY-TEXT PRONUNCIATION: mean WER {mean:.3f} over {len(PROMPTS)} novel prompts; "
          f"pass(<0.35): {report['pass']}", flush=True)
    if out:
        json.dump(report, open(out, "w"), indent=1)
    return report


def selftest():
    assert wer("hello world", "hello world") == 0.0
    assert wer("the cat sat", "the cat sat") == 0.0
    assert abs(wer("the cat sat", "the dog sat") - 1 / 3) < 1e-9
    assert wer("a b c d", "") == 1.0
    import importlib.util as u
    print("transformers:", "yes" if u.find_spec("transformers") else "NO (pip on pod)")
    print("say_test selftest OK")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--ckpt", default="runs/tts_fast.pt")
    ap.add_argument("--out", default="runs/say_test.json")
    ap.add_argument("--synth-out", dest="synth_out", default="data/synth")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest(); return
    if args.run:
        run(ckpt=args.ckpt, out=args.out, synth_out=args.synth_out); return
    ap.error("choose --selftest / --run")


if __name__ == "__main__":
    main()
