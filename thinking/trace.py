"""Small token vocabulary helper for manifest-driven decoders.

Current text and multimodal training data supplies its own inputs and targets. This module only
owns token/id conversion and the minimal extraction punctuation used by the text rung; it does
not render synthetic questions, proof traces, or templates.
"""

PAD, UNK = "<pad>", "<unk>"
KEYWORDS = ("extract", "fact", "done", ".")


class Vocab:
    def __init__(self, tokens, max_size=None):
        keywords = set(KEYWORDS)
        if max_size and int(max_size) > 0:
            # Keep the most frequent tokens (control keywords always retained);
            # the long tail falls back to <unk>. Caps embedding-table blow-up on
            # large corpora where most tokens appear once and never train.
            from collections import Counter
            budget = max(0, int(max_size) - 2 - len(keywords))  # 2 = PAD, UNK
            counts = Counter(t for t in tokens if t not in keywords)
            kept = {t for t, _ in counts.most_common(budget)}
            vocab_tokens = keywords | kept
        else:
            vocab_tokens = set(tokens) | keywords
        self.itos = [PAD, UNK] + sorted(vocab_tokens)
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.pad, self.unk = 0, 1

    def __len__(self):
        return len(self.itos)

    def enc(self, tokens):
        return [self.stoi.get(t, self.unk) for t in tokens]

    def dec(self, ids):
        return [self.itos[i] for i in ids]
