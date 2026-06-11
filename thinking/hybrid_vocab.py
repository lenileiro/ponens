"""Hybrid byte-BPE vocabulary for training LANGUAGE and REASONING in one model.

The contract (the Lang-0 rung): free English is byte-BPE so the model learns real fluency, but
every token the CHECKER touches is reserved ATOMIC -- trace keywords, kinship/value/answer
predicates, the slot tokens p0..p159, digits-as-years, and the punctuation the grammar parses.
A canonical trace therefore survives tokenization token-for-token (parse/check round-trips), while
surface sentences fragment normally. This is what lets one model read NL, reason canonically, and
write NL without the reasoning layer inheriting BPE's name-fragmentation (rung-0 finding: unseen
fragmented entities derail structural circuits -- here entities are always slots).
"""
from .world import N_SLOTS
from .trace import KEYWORDS
from . import kinship as K


def canonical_specials():
    """Every token the GoalChecker / trace parser must see as one indivisible id."""
    toks = list(KEYWORDS)                                   # think/so/and/answer/check/... + . ?
    toks += ["extract", "fact", "done", "write", "compute", "minus", "define", ":"]
    toks += list(K.BASE_PREDS)                              # mother father sister ... born died
    for hp, _b in K.RULES:                                  # parent grandparent ancestor ...
        toks.append(hp[0])
    toks += list(K.VALUE_PREDS) + list(K.ANSWER_PREDS)      # age_*/older_by/who_* + kinship answers
    toks += [f"p{i}" for i in range(N_SLOTS)]               # anonymized entity slots
    seen, out = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out
