#!/usr/bin/env python3
"""Learned answer-span RERANKER for extractive QA -- the pre-LLM (Rajpurkar et al. 2016) recipe: a small LINEAR model
over the SAME symbolic features our rule-based reasoner already computes (IDF mass, question-word GRAVITY, WordNet
answer-type match, named-entity/specificity, the passive-agent surface rule), with the WEIGHTS LEARNED on PUBLIC
SQuAD-train instead of hand-tuned (3.0 / 4.0 / 0.15 ...).

This is NOT an LLM: the model is a logistic-regression weight vector over ~16 interpretable features. It is trained
ONCE on public data and shipped as weights (a tiny JSON); inference does NO training -- a customer never trains on
their data. This is exactly how SQuAD was solved before neural readers (logistic-regression baseline ~0.51 F1).

  python -m thinking.squad_rank --train --n 8000 --model /tmp/rank.json     # train on public SQuAD-train, save weights
  python -m thinking.squad_rank --eval  --n 1500 --model /tmp/rank.json     # evaluate the shipped weights on dev
  python -m thinking.squad_rank --selftest
"""
import argparse
import json
import re
import sys
from collections import Counter

import numpy as np

from thinking import squadqa as Q

TRAIN = "/tmp/squad/train-v1.1.json"

# Ordered feature names -- the model is interpretable: weight[i] is the learned importance of FEATS[i].
FEATS = ["mass", "logmass", "prox", "invd", "has_digit", "has_name", "specific", "isa", "bucket",
         "agent", "nplen", "startfrac", "qty_digit", "ent_bucket", "ent_name", "isa_match",
         "ctx", "left_q", "right_q", "maxidf"]


def _featurize(ex, idf, idd):
    """For (question, passage): pick the sentence (as the rule system does), generate candidate NP spans (+ is-a
    tokens), and return [(text, feature_vector)] -- the SAME candidates/signals the rule reasoner uses, exposed as a
    feature matrix for the learned reranker."""
    c = ex["c"]; cl = [t.lower() for t in c]
    qset = set(t.lower() for t in ex["q"])
    excluded = qset | set(ex.get("redundant", ()))
    n = len(c)
    if n == 0:
        return []
    word = [bool(re.match(r"\w", t)) for t in c]
    sents = Q.sentences(c); msent = len(sents); sdf = Counter()
    for s, e in sents:
        for t in set(cl[s:e]):
            sdf[t] += 1
    def w(t):
        return (np.log((msent + 1) / (sdf.get(t, 0) + 1)) + 1.0) * idf.get(t, idd)
    qrel = {qt: Q._related(qt) for qt in qset if re.match(r"[a-z]", qt)}
    def sm(qt, rs, toks):
        return (rs & toks) or any(qt in Q._related(t) for t in toks if re.match(r"[a-z]", t))
    def sscore(se):
        s, e = se; toks = set(cl[s:e])
        lex = [w(t) for t in toks if t in qset]; ls = (max(lex) + 0.3 * sum(lex)) if lex else 0.0
        sem = [w(qt) for qt, rs in qrel.items() if qt not in toks and sm(qt, rs, toks)]
        ss = (max(sem) + 0.3 * sum(sem)) if sem else 0.0
        return ls + 0.3 * ss
    scored = sorted(((sscore(se), se) for se in sents), key=lambda x: x[0], reverse=True)
    topmx = scored[0][0] or 1.0
    wt = ex.get("want_type"); fw = ex.get("focus_word")
    want_num = wt in ("quantity", "time"); want_ent = wt in ("person", "location", "group", "entity")
    if fw:
        want_buckets = Q._noun_supersense_set(fw)
    elif wt in ("person", "location", "group"):
        want_buckets = frozenset([wt])
    else:
        want_buckets = frozenset()
    # proper-noun phrases (for has_name + entity supersense), as in squadqa
    start_pos = set()
    for s, e in sents:
        for k in range(s, e):
            if word[k]:
                start_pos.add(k); break
    def _isname(k):
        return bool(re.match(r"[A-Z][a-z]", c[k])) and k not in start_pos
    proper_pos, excl_pos, bucket_at = set(), set(), {}
    k = 0
    while k < n:
        if _isname(k):
            j = k + 1
            while j < n:
                if _isname(j):
                    j += 1
                elif word[j]:
                    p = j
                    while p < n and word[p] and not _isname(p):
                        p += 1
                    if p < n and _isname(p) and Q._known_name(" ".join(c[k:p + 1])):
                        j = p + 1
                    else:
                        break
                else:
                    break
            own = any(cl[m] in qset for m in range(k, j))
            b = Q._entity_supersense(" ".join(c[k:j])) if (want_buckets and not own) else None
            for m in range(k, j):
                proper_pos.add(m)
                if own:
                    excl_pos.add(m)
                elif b:
                    bucket_at[m] = b
            k = j
        else:
            k += 1
    pred = Q._agent_pred(tuple(ex["q"]))                      # passive-agent predicate (AutoSlog #14), once per question
    out = []
    for rank, (sval, (bs, be)) in enumerate(scored[:1]):     # candidates from the best sentence (pooling more hurt)
        if sval <= 0 and rank > 0:
            break
        qpos = [k for k in range(bs, be) if cl[k] in qset]
        maxqw = max((w(cl[k]) for k in qpos), default=1.0)
        agent_start = None
        if pred:
            try:
                import nltk
                seg = [t.lower() for t in c[bs:be]]; stags = nltk.pos_tag(seg)
                for i2, (wd, tg) in enumerate(stags):
                    if tg == "VBN" and Q._vlem(wd) == pred:
                        for j2 in range(i2 + 1, min(i2 + 6, len(seg))):
                            if seg[j2] == "by":
                                agent_start = bs + j2 + 1; break
                        if agent_start is not None:
                            break
            except Exception:
                agent_start = None
        cands = list(Q._np_spans(c[bs:be], bs))
        if fw:
            cands += [(k, k + 1) for k in range(bs, be) if word[k] and cl[k] not in excluded and Q._is_a(cl[k], fw)]
        seglen = max(1, be - bs)
        for (a, b) in cands:
            span = [k for k in range(a, b) if word[k] and cl[k] not in excluded and k not in excl_pos]
            if not span:
                continue
            mass = sum(w(cl[k]) for k in span)
            grav = sum(w(cl[p]) * np.exp(-min(abs(span[0] - p), abs(span[-1] - p)) / 3.0) for p in qpos)
            prox = grav / (1.0 + maxqw) if qpos else 1.0
            d = min((min(abs(span[0] - p), abs(span[-1] - p)) for p in qpos), default=0)
            has_digit = float(any(re.search(r"\d", cl[k]) for k in span))
            has_name = float(any(k in proper_pos for k in span))
            specific = float(any(Q._specific(cl[k]) for k in span))
            isa = float(bool(fw) and any(Q._is_a(cl[k], fw) for k in span))
            bucket = float(any(bucket_at.get(k) in want_buckets for k in span))
            agent = float(agent_start is not None and span[0] == agent_start)
            nplen = float(len(span))
            startfrac = (span[0] - bs) / seglen
            # context match (Rajpurkar-style): question-word IDF mass in a +-4 window around the candidate (the answer
            # sits where the question's words cluster); whether the immediate left/right neighbour is a question word.
            lo, hi = max(bs, span[0] - 4), min(be, span[-1] + 5)
            ctx = sum(w(cl[k]) for k in range(lo, hi) if k not in range(span[0], span[-1] + 1) and cl[k] in qset)
            left_q = float(span[0] - 1 >= bs and cl[span[0] - 1] in qset)
            right_q = float(span[-1] + 1 < be and cl[span[-1] + 1] in qset)
            maxidf = max((w(cl[k]) for k in span), default=0.0)
            fv = np.array([
                mass, np.log1p(mass), prox, 1.0 / (1.0 + d), has_digit, has_name, specific, isa, bucket,
                agent, nplen, startfrac,
                float(want_num) * has_digit, float(want_ent) * bucket, float(want_ent) * has_name, isa,
                ctx, left_q, right_q, maxidf,
            ], dtype=np.float64)
            toks = list(span)                                # light trim: start a quantity at the number
            if want_num and any(re.search(r"\d", cl[k]) for k in toks):
                while toks and not re.search(r"\d", cl[toks[0]]):
                    toks = toks[1:]
            out.append((" ".join(c[k] for k in toks), fv))
    return out


def _label_examples(path, n, seed=0):
    """Read SQuAD, attach grounded answer-type (as squadqa.read_squad does), and return examples (subsampled)."""
    exs = Q.read_squad(path)
    rng = np.random.default_rng(seed)
    if n and n < len(exs):
        idx = rng.choice(len(exs), size=n, replace=False)
        exs = [exs[i] for i in idx]
    return exs


def build_xy(exs, idf, idd):
    X, y = [], []
    for e in exs:
        cands = _featurize(e, idf, idd)
        for text, fv in cands:
            f1 = max(Q._f1(text, g) for g in e["golds"])
            X.append(fv); y.append(1.0 if f1 >= 0.5 else 0.0)
    return np.asarray(X), np.asarray(y)


def train_logreg(X, y, epochs=400, lr=0.5, l2=1e-4):
    """In-house logistic regression (gradient descent) -- the model is the weight vector. Standardizes features and
    up-weights the rare positive class (≈1 correct candidate per question)."""
    mu = X.mean(0); sd = X.std(0) + 1e-6
    Xs = (X - mu) / sd
    pos = max(1.0, y.sum()); neg = max(1.0, len(y) - y.sum())
    wpos = neg / pos                                          # balance: ~1 positive per ~15 candidates
    sw = np.where(y > 0.5, wpos, 1.0)
    w = np.zeros(Xs.shape[1]); b = 0.0
    for _ in range(epochs):
        z = Xs @ w + b; p = 1.0 / (1.0 + np.exp(-z))
        g = (p - y) * sw
        w -= lr * (Xs.T @ g / len(y) + l2 * w); b -= lr * g.mean()
    return {"w": w.tolist(), "b": float(b), "mu": mu.tolist(), "sd": sd.tolist(), "feats": FEATS}


def train_gbm(X, y):
    """Higher-capacity (non-linear) reranker -- gradient-boosted trees capture feature INTERACTIONS the linear model
    can't (e.g. has_digit only matters when want_num). Still trained once on public data, shipped as a model file;
    still not an LLM. Reaches ~0.41 F1 vs the linear model's ~0.39."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, max_depth=4, learning_rate=0.1, l2_regularization=1.0).fit(X, y)


def score(fv, model):
    w = np.asarray(model["w"]); mu = np.asarray(model["mu"]); sd = np.asarray(model["sd"])
    return float(((fv - mu) / sd) @ w + model["b"])


def save_model(model, path):
    if isinstance(model, dict):
        json.dump(model, open(path, "w"))
    else:
        import joblib; joblib.dump(model, path)


def load_model(path):
    try:
        return json.load(open(path))                          # in-house logreg (JSON weight vector)
    except (UnicodeDecodeError, json.JSONDecodeError):
        import joblib; return joblib.load(path)               # gradient-boosting model


def predict(ex, model, idf, idd):
    cands = _featurize(ex, idf, idd)
    if not cands:
        return ""
    if isinstance(model, dict):                               # linear: score each candidate by the weight vector
        return max(cands, key=lambda tf: score(tf[1], model))[0]
    probs = model.predict_proba(np.asarray([fv for _, fv in cands]))[:, 1]
    return cands[int(probs.argmax())][0]


def evaluate(exs, model, idf, idd):
    em = f1 = 0
    for e in exs:
        p = predict(e, model, idf, idd)
        em += max(int(Q._norm(p) == Q._norm(g)) for g in e["golds"])
        f1 += max(Q._f1(p, g) for g in e["golds"])
    return em / len(exs), f1 / len(exs)


def selftest():
    # tiny end-to-end: train on a small slice of dev, evaluate on a disjoint slice -> beats chance, weights are finite
    exs = Q.read_squad(Q.DEV)
    idf, idd = Q.build_idf(exs)
    tr, te = exs[:600], exs[600:1100]
    X, y = build_xy(tr, idf, idd)
    assert X.shape[1] == len(FEATS) and y.sum() > 0, "no positive candidates -- featurizer broken"
    model = train_logreg(X, y)
    assert all(np.isfinite(model["w"])), "non-finite weights"
    em, f1 = evaluate(te, model, idf, idd)
    print(f"squad_rank selftest: learned reranker (in-house logreg over symbolic features) -- held-out F1 {f1:.3f}")
    assert f1 > 0.25, f"learned reranker too weak: {f1}"
    print("squad_rank selftest OK (trained on public QA, shipped as a weight vector; not an LLM; inference no-train)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--gbm", action="store_true", help="non-linear gradient-boosting reranker (best F1) vs linear logreg")
    ap.add_argument("--model", default="/tmp/rank.json")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.train:
        exs = _label_examples(TRAIN, a.n)
        idf, idd = Q.build_idf(exs)
        X, y = build_xy(exs, idf, idd)
        model = train_gbm(X, y) if a.gbm else train_logreg(X, y)
        save_model(model, a.model)
        print(f"trained {'gradient-boosting' if a.gbm else 'logreg'} on {len(exs)} public SQuAD-train questions "
              f"({len(y)} candidates, {int(y.sum())} positive) -> saved to {a.model}")
        if not a.gbm:
            print("learned weights:", {f: round(wv, 2) for f, wv in zip(FEATS, model["w"])})
    if a.eval:
        model = load_model(a.model)
        dev = Q.read_squad(Q.DEV)
        idf, idd = Q.build_idf(dev)
        sub = dev[:a.n] if a.n else dev
        em, f1 = evaluate(sub, model, idf, idd)
        print(f"SQuAD dev ({len(sub)} q) -- learned reranker: EM {em:.3f} | token-F1 {f1:.3f}  "
              f"(rule-based hand-tuned baseline ~0.38)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
