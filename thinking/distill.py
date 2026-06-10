"""DATA DISTILLATION: a frontier teacher writes the surface language; the agent must read it.

`claude -p` (the gen_corpus pattern) generates a large paraphrase BANK per fact predicate and per
question type. Every variant is MECHANICALLY validated (slot discipline, charset, length) -- the
teacher provides diversity, never ground truth (facts/rules stay oracle-generated). The bank is
split TRAIN/EVAL so that held-out phrasings measure LANGUAGE UNDERSTANDING: at eval the agent
reads sentences whose surface it has never seen, expressing facts whose semantics it has.

The bank is LEVELED for a language curriculum (the agent grows up):
  kindergarten  very short sentences, words a 5-year-old knows
  midschool     subordinate clauses, richer vocabulary (a 12-year-old's English)

  python -m thinking.distill --go            # generate via claude -p -> surfaces.json
  python -m thinking.distill --show          # inspect the bank
"""
import argparse
import json
import os
import re
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BANK = os.path.join(ROOT, "surfaces.json")

FACT_SPECS = {
    "mother": "X is the mother of Y (X = female parent of Y). The sentence must unambiguously "
              "mean female parent -- 'parent' alone or gender-ambiguous wording is WRONG.",
    "father": "X is the father of Y (X = male parent of Y). Must unambiguously mean male parent.",
    "sister": "X is a sister of Y (X = female sibling of Y). Must unambiguously mean female sibling.",
    "brother": "X is a brother of Y (X = male sibling of Y). Must unambiguously mean male sibling.",
    "spouse": "X is married to Y. Gender-neutral.",
    "born": "X was born in the year Y (Y is a 4-digit year).",
    "died": "X died in the year Y (Y is a 4-digit year).",
}
QUESTION_SPECS = {
    "rel": ("ask what family relation X is to Y (open question, no relation named)",
            "ANSWER_PREDS"),
    "age_at_death": ("ask how old X was when X died", None),
    "age_when": ("ask how old X was when Y was born", None),
    "age_when_died": ("ask how old X was when Y died", None),
    "age_in_year": ("ask how old X would be in the year Y (hypothetical, may be future)", None),
    "older_by": ("ask by how many years X is older than Y (equivalently: how many years younger "
                 "Y is than X)", None),
    "who_older": ("ask which of X and Y is older / was born first", None),
    "who_younger": ("ask which of X and Y is younger / was born last", None),
}

# word definitions: vocabulary lessons -- and for relation words these are THE RULES IN ENGLISH
DEF_SPECS = {
    "mother": "a female parent", "father": "a male parent",
    "sister": "a female sibling", "brother": "a male sibling",
    "spouse": "a person someone is married to",
    "born": "to come into the world; the year of birth",
    "died": "to pass away; the year of death",
    "grandmother": "the mother of one of your parents",
    "grandfather": "the father of one of your parents",
    "great_grandmother": "the mother of one of your grandparents",
    "great_grandfather": "the father of one of your grandparents",
    "aunt": "a sister of one of your parents", "uncle": "a brother of one of your parents",
    "cousin": "a child of your aunt or uncle",
    "nephew": "a son of your sibling", "niece": "a daughter of your sibling",
    "mother_in_law": "the mother of your spouse", "father_in_law": "the father of your spouse",
    "ancestor": "a person you descend from: a parent, a parent of a parent, and so on",
}

LEVELS = {                                                 # the education ladder, in order
    "preschool": "STYLE: preschool English. Tiny sentences (3-6 words), only the most basic "
                 "words a 3-year-old knows.",
    "kindergarten": "STYLE: kindergarten English. Very short (4-8 words), simple everyday words "
                    "a 5-year-old knows. No subordinate clauses.",
    "elementary": "STYLE: elementary-school English. 6-12 words, simple connectives (and, so, "
                  "because), vocabulary of an 8-year-old.",
    "midschool": "STYLE: middle-school English. 8-16 words, subordinate clauses, appositives and "
                 "relative clauses welcome; vocabulary of a 12-year-old.",
    "highschool": "STYLE: high-school English. 10-20 words, formal register, passive voice and "
                  "nominalizations welcome; vocabulary of a 16-year-old.",
    "undergraduate": "STYLE: university-essay English. 12-24 words, precise hedged academic "
                     "register, complex clause structure.",
    "graduate": "STYLE: graduate-academic English. 14-26 words, dense nominal style as in a "
                "journal article; technical kinship terminology welcome.",
    "scholar": "STYLE: scholarly/genealogical register. 12-26 words, precise formal prose as in "
               "a biography, archive record, or genealogy; literary constructions welcome.",
}

PROMPT = """Write {n} different natural English sentence patterns that each express EXACTLY this:
{spec}

{style}

Rules:
- use the literal placeholders {{h}} for X and {{t}} for Y (each exactly once{t_opt})
- lowercase words, only letters, commas and the placeholders; end with '{end}'
- one pattern per line, no numbering, no quotes, no extra text
- vary structure genuinely (clefts, appositives, inversions, relative clauses), not just synonyms
- every pattern must preserve the EXACT meaning including gender/direction; when in doubt, be explicit"""


def _teacher(spec, n, need_t, is_q, level):
    p = PROMPT.format(n=n, spec=spec, style=LEVELS[level], end="?" if is_q else ".",
                      t_opt="" if need_t else "; {t} may be omitted")
    out = subprocess.run(["claude", "-p", p], capture_output=True, text=True, timeout=300)
    return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]


def _validate(line, need_t, is_q, no_slots=False):
    """Mechanical guard: slot discipline + closed charset + sane length. Returns tokens|None."""
    if no_slots:
        if "{h}" in line or "{t}" in line:
            return None
    elif line.count("{h}") != 1 or (need_t and line.count("{t}") != 1) or line.count("{t}") > 1:
        return None
    end = "?" if is_q else "."
    if not line.endswith(end):
        line = line.rstrip(".?!") + " " + end
    toks = line.replace(",", " , ").replace(end, f" {end}").split()
    if not 3 <= len(toks) <= 32:                           # preschool short .. graduate long
        return None
    for t in toks:
        if t in ("{h}", "{t}", ",", ".", "?"):
            continue
        if not re.fullmatch(r"[a-z]+(?:'s)?", t):
            return None
    return tuple(toks)


def generate(n_per=18):
    bank = {"templates": {}, "questions": {}}
    for level in LEVELS:
        bank["templates"][level] = {}
        bank["questions"][level] = {}
        for pred, spec in FACT_SPECS.items():
            got, seen = [], set()
            for ln in _teacher(spec, n_per, need_t=True, is_q=False, level=level):
                tk = _validate(ln, need_t=True, is_q=False)
                if tk and tk not in seen:
                    seen.add(tk)
                    got.append(list(tk))
            bank["templates"][level][pred] = got
            print(f"  {level}/{pred}: {len(got)} valid variants", flush=True)
        for qkey, (spec, _) in QUESTION_SPECS.items():
            need_t = qkey not in ("age_at_death",)
            got, seen = [], set()
            for ln in _teacher("a QUESTION: " + spec, n_per, need_t=need_t, is_q=True,
                               level=level):
                tk = _validate(ln, need_t=need_t, is_q=True)
                if tk and tk not in seen:
                    seen.add(tk)
                    got.append(list(tk))
            bank["questions"][level][qkey] = got
            print(f"  {level}/Q/{qkey}: {len(got)} valid variants", flush=True)
    return bank


def split_bank(bank, eval_frac=0.25):
    """Deterministic train/eval phrasing split (held-out surfaces = the language test)."""
    out = {"templates": {}, "questions": {}, "eval_templates": {}, "eval_questions": {}}
    for sect, esect in (("templates", "eval_templates"), ("questions", "eval_questions")):
        for level, d in bank[sect].items():
            out[sect][level] = {}
            out[esect][level] = {}
            for k, vs in d.items():
                cut = max(1, int(len(vs) * eval_frac))
                out[esect][level][k] = vs[:cut]            # teacher order is arbitrary -> fine
                out[sect][level][k] = vs[cut:]
    return out


def generate_definitions(n_per=10):
    """Leveled DEFINITION patterns per word: 'a grandmother is the mother of a parent .' etc."""
    defs = {}
    for level in LEVELS:
        defs[level] = {}
        for word, meaning in DEF_SPECS.items():
            spec = (f"a DEFINITION of the word '{word.replace('_', ' ')}' meaning exactly: "
                    f"{meaning}. The sentence must define the word itself (start naturally, "
                    f"e.g. 'a {word.replace('_', ' ')} is ...'), mention no specific people.")
            got, seen = [], set()
            for ln in _teacher(spec, n_per, need_t=False, is_q=False, level=level):
                tk = _validate(ln, need_t=False, is_q=False, no_slots=True)
                if tk and tk not in seen:
                    seen.add(tk)
                    got.append(list(tk))
            defs[level][word] = got
            print(f"  {level}/def/{word}: {len(got)} valid", flush=True)
    return defs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true", help="generate via claude -p (frontier calls)")
    ap.add_argument("--definitions", action="store_true",
                    help="generate word definitions and merge into the existing bank")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    if args.definitions:
        b = json.load(open(BANK)) if os.path.exists(BANK) else \
            {"templates": {}, "questions": {}, "eval_templates": {}, "eval_questions": {}}
        d = generate_definitions(args.n if args.n != 24 else 10)
        b["definitions"], b["eval_definitions"] = {}, {}
        for lv, words in d.items():
            b["definitions"][lv], b["eval_definitions"][lv] = {}, {}
            for w, vs in words.items():
                cut = max(1, len(vs) // 4)
                b["eval_definitions"][lv][w] = vs[:cut]
                b["definitions"][lv][w] = vs[cut:]
        json.dump(b, open(BANK, "w"), indent=1)
        print(f"definitions merged -> {BANK}")
        return
    if args.show:
        b = json.load(open(BANK))
        for level, d in b["templates"].items():
            for k, vs in d.items():
                print(f"{level}/{k} ({len(vs)} train / {len(b['eval_templates'][level][k])} eval):")
                for v in vs[:2]:
                    print("   ", " ".join(v))
        return
    if not args.go:
        print(__doc__)
        return
    bank = split_bank(generate(args.n))
    json.dump(bank, open(BANK, "w"), indent=1)
    print(f"saved -> {BANK}")


if __name__ == "__main__":
    main()
