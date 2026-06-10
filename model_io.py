"""Load a trained A1 model checkpoint (saved by learn_run with FER_SAVE) for chat + testing.

Checkpoint = {state_dict, config, itos, token}. Rebuilds the ScratchpadLM and a vocab whose
enc/decode match how it was trained (char or word level), so chat.py and a1_test.py can talk to
the exact trained weights.
"""
import re
import torch
from scratchpad_model import ScratchpadLM
from device import get_device

DEV = get_device()
_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[.?!,;]")


class LoadedVocab:
    def __init__(self, itos, token):
        self.itos = itos
        self.stoi = {t: i for i, t in enumerate(itos)}
        self.token = token
        self.pad = 0
        self.unk = self.stoi.get("<unk>", 0)

    def __len__(self):
        return len(self.itos)

    def toks(self, s):
        return _WORD.findall(s) if self.token == "word" else list(s)

    def enc(self, s):
        return [self.stoi.get(t, self.unk) for t in self.toks(s)]

    def decode(self, ids):
        ts = [self.itos[i] for i in ids if 0 <= i < len(self.itos)]
        if self.token != "word":
            return "".join(ts)
        out = ""
        for t in ts:
            out += (t if t in ".?!,;" else (" " + t if out else t))
        return out


class BPEVocab:
    """Byte-level BPE (HuggingFace tokenizers) — no <unk> (every byte representable), shared by
    training (build from texts) and loading (rebuild from saved JSON). Same interface as the others."""
    token = "bpe"

    def __init__(self, texts=None, json_str=None, vocab_size=4000):
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
        if json_str is not None:
            self.tk = Tokenizer.from_str(json_str)
        else:
            self.tk = Tokenizer(models.BPE(unk_token=None))
            self.tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
            self.tk.decoder = decoders.ByteLevel()
            self.tk.train_from_iterator(list(texts),
                                        trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<pad>"]))
        self.pad = self.tk.token_to_id("<pad>") or 0
        self.unk = -1                                    # byte-level BPE has no OOV
        self.stoi = self.tk.get_vocab()

    def __len__(self):
        return self.tk.get_vocab_size()

    def enc(self, s):
        return self.tk.encode(s).ids

    def toks(self, s):
        return self.tk.encode(s).tokens

    def decode(self, ids):
        return self.tk.decode([int(i) for i in ids if int(i) != self.pad])

    def to_json(self):
        return self.tk.to_str()


def load(path):
    c = torch.load(path, map_location=DEV)
    m = ScratchpadLM(**c["config"]).to(DEV)
    m.load_state_dict(c["state_dict"]); m.eval()
    V = BPEVocab(json_str=c["tokenizer"]) if c["token"] == "bpe" else LoadedVocab(c["itos"], c["token"])
    return m, V, c["config"]["max_len"]
