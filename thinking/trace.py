"""Trace language: token-level rendering, parsing, vocab, and batch packing.

The vocabulary operates on TOKEN LISTS end to end (never free text), so the surface language can
contain any token without tokenizer-regex loss. Format (premises-first — the conclusion's new
entity is emitted only AFTER its supporting fact, making every content prediction a verbatim
induction lookup):

  facts   'a r b .'  per EDB fact
  query   'query far a ?'
  step    'think a far b and b r c so a far c .'
  answer  'answer c .'
"""
from dataclasses import dataclass, field
import numpy as np
import torch

PAD, UNK = "<pad>", "<unk>"
KEYWORDS = ("think", "so", "and", "because", "answer", "query", "path", "end", ".", "?",
            "needs", "check", "extract", "fact", "done")


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


def atom_tokens(atom):
    """Infix surface for a binary atom: ('far', ('a', 'b')) -> ['a', 'far', 'b']."""
    return [atom[1][0], atom[0], atom[1][1]]


def parse_atom(toks):
    return (toks[1], (toks[0], toks[2])) if len(toks) == 3 else None


@dataclass
class Example:
    tokens: list                      # full supervised sequence
    aux: list                         # [(predictor_pos, gold_ctx_pos)] lookup supervision targets
    meta: dict = field(default_factory=dict)


def atom_text(atom):
    return f"{atom[0]}({atom[1][0]},{atom[1][1]})"


def line_rule_id(typ, head, body):
    """Stable semantic label for a proof line, used by FER/UFR probes."""
    if typ == "check":
        return f"check:{head[0]}"
    if typ != "think":
        return typ
    bpreds = tuple(b[0] for b in body)
    if head[0] == "ancestor":
        if bpreds == ("parent",):
            return "ancestor_base"
        if bpreds == ("ancestor", "parent"):
            return "ancestor_forward"
        if bpreds == ("parent", "ancestor"):
            return "ancestor_backward"
    return f"{head[0]}<-{'+'.join(bpreds)}"


def line_meta(typ, head, body, start, end, depth_index=None):
    return {
        "kind": typ,
        "rule_id": line_rule_id(typ, head, body),
        "pred": head[0],
        "head": atom_text(head),
        "body": [atom_text(b) for b in body],
        "start": start,
        "end": end,
        "depth_index": depth_index,
    }


def render_goal_line(typ, head, body=()):
    """Render one canonical proof line, without the final period.

    Keeping this in the trace module avoids scattering grammar assumptions through decoders.
    """
    if typ == "check":
        return ["check"] + atom_tokens(head)
    if typ != "think":
        raise ValueError(f"unknown goal line type {typ!r}")
    toks = ["think"]
    for i, b in enumerate(body):
        if i:
            toks.append("and")
        toks += atom_tokens(b)
    return toks + ["so"] + atom_tokens(head)


def render_example(problem, steps, sup="steps"):
    """Render a Problem (+ gold derivation steps) into tokens with aux lookup positions.
    aux pairs the position whose NEXT token is an in-context lookup with the position of the
    answer inside the facts — the attention-supervision (and pointer-head) training signal."""
    toks, tailpos, aux = [], {}, []
    for pred, (h, t) in problem.edb:
        tailpos[(pred, h, t)] = len(toks) + 2
        toks += [h, pred, t, "."]
    if sup == "steps":
        toks += ["query", problem.goal[0], problem.head, "?"]
        for head, body in steps:
            base = len(toks)
            toks.append("think")
            for j, b in enumerate(body):
                if j:
                    toks.append("and")
                if b[0] == "r":       # the lookup: tail of an EDB fact (predictor = the 'r' token)
                    aux.append((len(toks) + 1, tailpos[(b[0],) + b[1]]))
                toks += atom_tokens(b)
            toks += ["so"] + atom_tokens(head) + ["."]
        toks += ["answer", problem.answer, "."]
    elif sup == "path":               # datalog_reason baseline: emit the entity path
        toks += ["path", problem.head]
        node = problem.head
        base = len(toks)
        chain = [problem.head]
        nxt = {h: t for pred, (h, t) in problem.edb if pred == "r"}
        while node in nxt:
            node = nxt[node]
            chain.append(node)
        for i in range(len(chain) - 1):                    # predictor of e[i+1] is the e[i] token
            aux.append((base - 1 + i, tailpos[("r", chain[i], chain[i + 1])]))
        toks += chain[1:] + ["end", "."]
    else:
        raise ValueError(f"unknown sup {sup!r}")
    return Example(tokens=toks, aux=aux)


def _pick(variants, rng):
    if isinstance(variants[0], (list, tuple)) and not isinstance(variants[0], str):
        return variants[int(rng.integers(len(variants))) if rng is not None else 0]
    return variants                                        # single template (back-compat)


def render_prompt(problem, templates, question, rng=None):
    """NL surface for facts + question, with SYNONYM/INVERSE template variants chosen per fact
    (rng=None -> first variant, deterministic). templates: {pred: [variant, ...]} where a variant
    is a token tuple with {h}/{t} slots (inverse variants put {t} first). question: list of
    (variant, pred_or_None) -- pred-specific phrasings (e.g. the ancestor-or-descendant antonym
    contrast) are only used when the gold answer matches. Returns (tokens, slot, first_pos)."""
    toks, slot, first_pos = [], {}, {}
    for pred, (h, t) in problem.edb:
        tpl = _pick(templates[pred], rng)
        base = len(toks)
        for w in tpl:
            toks.append(h if w == "{h}" else t if w == "{t}" else w)
        slot[(pred, h, t)] = {h: base + tpl.index("{h}"), t: base + tpl.index("{t}")}
        first_pos.setdefault(h, base + tpl.index("{h}"))
        first_pos.setdefault(t, base + tpl.index("{t}"))
    x, y = problem.goal[1]
    gp = problem.goal[0]
    qs = [v for v, p in question
          if p is None or p == gp or (isinstance(p, tuple) and gp in p)]
    q = qs[int(rng.integers(len(qs))) if rng is not None else 0]
    toks += [x if w == "{h}" else y if w == "{t}" else w for w in q]
    return toks, slot, first_pos


def render_goal_example(problem, lines, templates, question, rng=None):
    """GOAL-DIRECTED rendering: NL facts (synonym-varied templated surface) + NL question +
    preorder proof-tree decomposition. 'think H needs B1 and B2 .' opens subgoals; 'check F .'
    discharges an EDB leaf. Heads contain no new entities (the query pair / an open subgoal), so
    every NEW entity first appears inside a body atom -- a context lookup, which is where aux
    supervision points."""
    toks, slot, first_pos = render_prompt(problem, templates, question, rng)
    x, y = problem.goal[1]
    aux = []
    metas = []
    last_pos = dict(first_pos)                             # nearest antecedent of every entity
    for i, tk in enumerate(toks):                          # question mentions update antecedents
        if tk in last_pos:
            last_pos[tk] = i

    def emit_atom(atom):
        key = (atom[0],) + atom[1]
        for tk in atom_tokens(atom):
            if tk in last_pos:                             # DENSE binding supervision: EVERY
                gold = (slot[key][tk] if key in slot and tk in slot[key]   # occurrence, exact
                        else last_pos[tk])                 # fact slot when known, else antecedent
                aux.append((len(toks) - 1, gold))
                last_pos[tk] = len(toks)
            toks.append(tk)

    rule_counts = {}
    for typ, head, body in lines:
        line_start = len(toks)
        rid = line_rule_id(typ, head, body)
        rule_counts[rid] = rule_counts.get(rid, 0) + 1
        if typ == "think":                                 # PREMISES-FIRST WITHIN THE LINE (the
            toks.append("think")                           # chain world's decisive format): body
            for j, b in enumerate(body):                   # atoms echo known facts, THEN 'so',
                if j:                                      # THEN the conclusion -- every token a
                    toks.append("and")                     # local prediction
                emit_atom(b)
            toks.append("so")
            emit_atom(head)
            toks.append(".")
        else:                                              # check
            toks.append("check")
            emit_atom(head)
            toks.append(".")
        metas.append(line_meta(typ, head, body, line_start, len(toks), rule_counts[rid]))
    answer_start = len(toks)
    toks += ["answer", problem.answer, "."]
    metas.append({
        "kind": "answer",
        "rule_id": "answer",
        "pred": problem.goal[0],
        "head": str(problem.answer),
        "body": [],
        "start": answer_start,
        "end": len(toks),
        "depth_index": None,
    })
    return Example(tokens=toks, aux=aux, meta={"lines": metas})


def render_extraction_example(problem, templates, question, rng=None):
    """PHASE-1 READING supervision: NL surface -> exhaustive canonical fact list.
      <NL facts> extract fact a mother b . fact a born 1726 . ... done .
    Fact lines follow the surface sentence order (alignment eases learning); every emitted
    entity/year token gets aux supervision pointing at its slot in the source sentence."""
    toks, slot, first_pos = render_prompt(problem, templates, question, rng)
    while toks and toks[-1] != ".":                        # strip the question: reading-only task
        toks.pop()
    aux = []
    toks.append("extract")
    for pred, (h, t) in problem.edb:
        key = (pred, h, t)
        toks.append("fact")
        for tk in (h, pred, t):
            if key in slot and tk in slot[key]:
                aux.append((len(toks) - 1, slot[key][tk]))
            toks.append(tk)
        toks.append(".")
    toks += ["done", "."]
    return Example(tokens=toks, aux=aux)


def render_math_example(y1, y2):
    """MATH drill: the computation skill the age questions rely on, taught explicitly.
      compute 1967 minus 1900 : 67 .
    Tested on UNSEEN pairs (--mode math): exact-match on novel operands separates KNOWING
    subtraction from memorizing a difference table."""
    return Example(tokens=["compute", str(y2), "minus", str(y1), ":", str(y2 - y1), "."],
                   aux=[])


def render_def_example(word, sentence, level):
    """VOCABULARY lesson: define a word at an education level. For relation words this is the
    RULE stated in English -- declarative rule knowledge, testable.
      define elementary grandmother : a grandmother is your mom 's mom or your dad 's mom ."""
    return Example(tokens=["define", level, word, ":"] + list(sentence), aux=[])


def render_write_example(fact, sentence, level):
    """WRITING exercise: canonical fact + education-level cue -> that level's English sentence.
      write midschool anna mother bora : anna , who is bora 's mother , ... .
    The inverse of extraction; graded by matching a valid pattern of the level (slots correct)."""
    pred, (h, t) = fact
    toks = ["write", level, h, pred, t, ":"]
    toks += [h if w == "{h}" else t if w == "{t}" else w for w in sentence]
    return Example(tokens=toks, aux=[])


def parse_fact_line(toks):
    """['fact','a','mother','b'] -> ('mother', ('a', 'b')) | None."""
    if len(toks) == 4 and toks[0] == "fact":
        return (toks[2], (toks[1], toks[3]))
    return None


def parse_goal_line(toks):
    """One emitted line -> ('think', head, body) | ('check', fact, ()) | None.
    think lines are premises-first: think <B1> and <B2> so <H>"""
    if len(toks) == 4 and toks[0] == "check":
        f = parse_atom(toks[1:4])
        return ("check", f, ()) if f else None
    if len(toks) >= 8 and toks[0] == "think" and "so" in toks:
        iso = toks.index("so")
        head = parse_atom(toks[iso + 1:iso + 4])
        if len(toks) != iso + 4:
            return None
        body, cur = [], []
        for t in toks[1:iso]:
            if t == "and":
                body.append(parse_atom(cur))
                cur = []
            else:
                cur.append(t)
        body.append(parse_atom(cur))
        if head is None or any(b is None for b in body):
            return None
        return ("think", head, tuple(body))
    return None


def parse_line(toks):
    """Parse one emitted think-line -> (head_atom, (body_atoms...)) or None if malformed.
    ['think','a','far','b','and','b','r','c','so','a','far','c'] is the canonical shape."""
    if len(toks) < 8 or toks[0] != "think" or "so" not in toks:
        return None
    iso = toks.index("so")
    head = parse_atom(toks[iso + 1:iso + 4])
    body, cur = [], []
    for t in toks[1:iso]:
        if t == "and":
            body.append(parse_atom(cur))
            cur = []
        else:
            cur.append(t)
    body.append(parse_atom(cur))
    if head is None or any(b is None for b in body) or len(toks) != iso + 4:
        return None
    return head, tuple(body)


def build_vocab(examples, extra_tokens=()):
    toks = [t for ex in examples for t in ex.tokens]
    return Vocab(list(toks) + list(extra_tokens))


def pack_batch(seqs, block, batch, pad, rng):
    """Pack complete encoded examples into (batch, block+1) rows — never truncated mid-example
    (random windows that cut facts off TRAIN hallucination), remainder padded. Returns the row
    tensor and the offset-shifted aux supervision triples (row, predictor_pos, gold_pos)."""
    x = torch.full((batch, block + 1), pad, dtype=torch.long)
    sup = []
    for r in range(batch):
        o = 0
        while True:
            ids, aux = seqs[int(rng.integers(0, len(seqs)))]
            if o + len(ids) > block + 1:
                break
            x[r, o:o + len(ids)] = torch.tensor(ids)
            sup += [(r, o + p, o + c) for p, c in aux if o + p < block]
            o += len(ids)
    return x, sup


def build_rule_vocab(examples, kinds=("think", "check")):
    """Stable rule/action label map from rendered trace metadata."""
    labels = set()
    for ex in examples:
        for m in ex.meta.get("lines", []):
            if m.get("kind") in kinds:
                labels.add(m["rule_id"])
    return {r: i for i, r in enumerate(sorted(labels))}


def pack_batch_with_meta(seqs, block, batch, pad, rng, rule_stoi=None,
                         kinds=("think", "check")):
    """Pack examples plus optional rule labels and line spans.

    seqs entries are (ids, aux, meta). rule_targets is aligned to model input positions
    x[:, :-1], therefore its second dimension is `block`.
    """
    x = torch.full((batch, block + 1), pad, dtype=torch.long)
    sup, spans = [], []
    rule_targets = torch.full((batch, block), -100, dtype=torch.long) if rule_stoi else None
    kinds = set(kinds)
    for r in range(batch):
        o = 0
        while True:
            ids, aux, meta = seqs[int(rng.integers(0, len(seqs)))]
            if o + len(ids) > block + 1:
                break
            x[r, o:o + len(ids)] = torch.tensor(ids)
            sup += [(r, o + p, o + c) for p, c in aux if o + p < block]
            if rule_stoi:
                for m in meta.get("lines", []):
                    rid = m.get("rule_id")
                    if m.get("kind") not in kinds or rid not in rule_stoi:
                        continue
                    s, e = o + int(m["start"]), min(o + int(m["end"]), block)
                    if e <= s or s >= block:
                        continue
                    s = max(0, s)
                    label = rule_stoi[rid]
                    rule_targets[r, s:e] = label
                    spans.append((r, s, e, label))
            o += len(ids)
    return x, sup, rule_targets, spans
