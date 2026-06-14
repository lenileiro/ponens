"""Small token vocabulary helper for manifest-driven decoders.

Current text and multimodal training data supplies its own inputs and targets. This module only
owns token/id conversion and the minimal extraction punctuation used by the text rung; it does
not render synthetic questions, proof traces, or templates.
"""

PAD, UNK = "<pad>", "<unk>"
KEYWORDS = ("extract", "fact", "done", ".")


class Vocab:
    def __init__(self, tokens):
        self.itos = [PAD, UNK] + sorted(set(tokens) | set(KEYWORDS))
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.pad, self.unk = 0, 1

    def __len__(self):
        return len(self.itos)

    def enc(self, tokens):
        return [self.stoi.get(t, self.unk) for t in tokens]

    def dec(self, ids):
        return [self.itos[i] for i in ids]
