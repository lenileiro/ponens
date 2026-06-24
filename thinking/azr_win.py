#!/usr/bin/env python3
"""azr_win -- THE COMBINED SOLVER ("find a winner"). One model that THINKS and ITERATES, fusing every
technique we built into a single multi-part problem solver that returns MULTIPLE verified answers:

  * BACKBONE -- iterate-execute-residual (azr_iter): build the solution one executor-verified step at a time
    (a depth-d task -> d single-step inductions). This is the multi-part solving.
  * POLICY by RL -- GRPO/Dr.GRPO (grpo_cot): the step-proposer is trained not only by imitation of verified
    chains (ReST-EM SFT, positive-only) but by GROUP-RELATIVE policy gradient over G sampled rollouts, reward
    = closeness-to-target (+1 if solved). Steps that move TOWARD the target are up-weighted, steps that veer
    away down-weighted -- a signal SFT-on-successes throws away. This lifts base solve@1.
  * MULTIPLE ANSWERS -- best-of-N with a SOUND verifier returns every distinct verified program (the solution
    is allowed to have many forms; we keep them all). Test-time search MULTIPLIES the lifted base rate.
  * COMPOUNDING -- library discovery abstracts recurring concepts into macros, raising the depth ceiling.

HYBRID training = ReST-EM SFT (strong positive signal) + GRPO (relative success/failure signal). The win
condition: combined > SFT-only at depth 5 (higher solve@1, so best-of-N multiplies to a higher ceiling).

    python -m thinking.azr_win --selftest
    python -m thinking.azr_win --maxdepth 5 --concepts 6 --rounds 3000 --grpo
"""
import argparse
import math
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from thinking.azr import NBASE, NOP, BASE_OPS, Library, execute, apply_ops_np
from thinking.azr_struct import (make_concepts, gen_task_struct, concepts_recovered, abstract_struct,
                                 refactor)
from thinking.azr_iter import (StepProposer, solve_iter, solve_iter_batch, solve_iter_gpu, closeness,
                               chain_train_pairs, collect_solve, evaluate, enable_tf32, amp)


# ===================================================================================================
# GRPO over the iterate-execute-residual policy: each rollout is a chain of single-step decisions; the
# reward is closeness-to-target (+1 if it solves) -- DENSE, so even unsolved tasks give gradient.
# ===================================================================================================
def grpo_rollout(prop, task, lib, A, L, max_steps, device, temp=1.0):
    """ONE stochastic rollout of the step policy. Keeps grad on the chosen-op log-probs (no torch.no_grad).
    Returns (sum_logp, reward, solved)."""
    demos = task[1]
    states = [list(x) for (x, _y) in demos]
    Y = [list(y) for (_x, y) in demos]
    active = [op for op, a in enumerate(lib.active()) if a and op != NOP]
    amask = torch.full((prop.vocab,), float("-inf"), device=device)
    amask[active] = 0.0
    sum_logp = torch.zeros((), device=device)
    tg = torch.tensor([Y], device=device)
    for _ in range(max_steps):
        if all(states[i] == Y[i] for i in range(len(Y))):
            break
        st = torch.tensor([states], device=device)
        lp = F.log_softmax(prop(st, tg)[0] / temp + amask, -1)        # over ACTIVE ops only
        op = int(torch.distributions.Categorical(logits=lp).sample())
        sum_logp = sum_logp + lp[op]
        states = [execute([op], s, A, lib) for s in states]
    solved = all(states[i] == Y[i] for i in range(len(Y)))
    reward = closeness(states, Y, L) + (1.0 if solved else 0.0)        # dense + solve bonus
    return sum_logp, reward, solved


def grpo_step(prop, tasks, lib, A, L, max_steps, device, G=6, temp=1.0):
    """Dr.GRPO: for each task sample G rollouts, advantage = reward - group_mean (no std norm). Accumulate
    -advantage*logp across tasks. Returns (loss_tensor_or_None, mean_solve_rate)."""
    terms, solves = [], []
    for t in tasks:
        rolls = [grpo_rollout(prop, t, lib, A, L, max_steps, device, temp) for _ in range(G)]
        rewards = np.array([r for (_lp, r, _s) in rolls])
        solves.append(np.mean([float(s) for (_lp, _r, s) in rolls]))
        if rewards.std() < 1e-6:                                       # no relative signal
            continue
        adv = rewards - rewards.mean()
        for (lp, _r, _s), a in zip(rolls, adv):
            if abs(a) > 1e-6:
                terms.append(-float(a) * lp)
    if not terms:
        return None, float(np.mean(solves)) if solves else 0.0
    return torch.stack(terms).mean(), float(np.mean(solves))


def grpo_step_process(prop, tasks, lib, A, L, max_steps, device, G=6, temp=1.0):
    """Dr.GRPO with the PER-STEP PROCESS REWARD -- BATCHED. All B = len(tasks)*G rollouts advance together in
    ONE proposer forward per step (vs B=1 per step per rollout); states/ops are applied with vectorized numpy.
    Per-step reward = executor-grounded Δcloseness (the verifier's progress signal) + terminal solve bonus;
    per-step credit via RETURN-TO-GO with a PER-STEP, PER-TASK group baseline (mean baseline, no std/length
    norm). A 'detour' op still gets credit from the later solve. ~50x fewer forward passes than the naive loop.
    """
    Xs, Ys, owner = [], [], []
    for ti, t in enumerate(tasks):
        demos = t[1]
        X = [list(x) for (x, _y) in demos]; Y = [list(y) for (_x, y) in demos]
        for _ in range(G):
            Xs.append(X); Ys.append(Y); owner.append(ti)
    B = len(Xs)
    states = np.array(Xs); targets = np.array(Ys)                      # (B,K,L)
    flatY = targets.reshape(B, -1)
    tg = torch.tensor(targets, device=device)
    active = [op for op, a in enumerate(lib.active()) if a and op != NOP]
    amask = torch.full((prop.vocab,), float("-inf"), device=device); amask[active] = 0.0
    arangeB = torch.arange(B, device=device)
    done = (states.reshape(B, -1) == flatY).all(1)
    prev = (states.reshape(B, -1) == flatY).mean(1)                    # closeness per row
    logps = [[] for _ in range(B)]; rewards = [[] for _ in range(B)]
    for _step in range(max_steps):
        if done.all():
            break
        st = torch.tensor(states, device=device)
        lp = F.log_softmax(prop(st, tg) / temp + amask, -1)            # (B, vocab) -- ONE forward, all rollouts
        ops_t = torch.distributions.Categorical(logits=lp).sample()    # (B,)
        chosen = lp[arangeB, ops_t]                                    # (B,) keeps grad
        ops = ops_t.detach().cpu().numpy()
        states = apply_ops_np(states, ops, lib, A)
        now = (states.reshape(B, -1) == flatY).mean(1)
        for b in range(B):
            if not done[b]:
                logps[b].append(chosen[b]); rewards[b].append(float(now[b] - prev[b]))
        prev = now
        done = (states.reshape(B, -1) == flatY).all(1)
    solved = (states.reshape(B, -1) == flatY).all(1)
    for b in range(B):
        if solved[b] and rewards[b]:
            rewards[b][-1] += 1.0                                      # terminal solve bonus
    terms, solves = [], []
    for ti in range(len(tasks)):
        idxs = [b for b in range(B) if owner[b] == ti]
        solves.append(float(np.mean([float(solved[b]) for b in idxs])))
        rtg = {}
        for b in idxs:
            g, acc = [0.0] * len(rewards[b]), 0.0
            for k in reversed(range(len(rewards[b]))):
                acc += rewards[b][k]; g[k] = acc
            rtg[b] = g
        maxT = max((len(rtg[b]) for b in idxs), default=0)
        baseline = [float(np.mean([rtg[b][k] for b in idxs if k < len(rtg[b])])) for k in range(maxT)]
        for b in idxs:
            for k in range(len(logps[b])):
                adv = rtg[b][k] - baseline[k]
                if abs(adv) > 1e-8:
                    terms.append(-adv * logps[b][k])
    if not terms:
        return None, float(np.mean(solves)) if solves else 0.0
    return torch.stack(terms).mean(), float(np.mean(solves))


def collect_answers(prop, task, lib, A, L, max_steps, beam, device, samples=16, temp=0.9):
    """MULTIPLE ANSWERS: return every DISTINCT executor-verified program found over N rollouts (sound -- each
    truly reproduces all demos). Greedy first, then sampled for diversity."""
    found = {}
    for s in range(samples):
        chain = solve_iter(prop, task, lib, A, L, max_steps, beam, device,
                           temp=(0.0 if s == 0 else temp))
        if chain is not None:
            found[tuple(o for o in chain if o != NOP)] = chain
    return list(found.values())


# ===================================================================================================
# Persist / load the trained solver (proposer weights + discovered library + concepts) and ANSWER a prompt.
# ===================================================================================================
def save_model(path, prop, lib, concepts, A, L, K, d, max_macros, layers=3, heads=4):
    torch.save({"state_dict": prop.state_dict(), "A": A, "L": L, "K": K, "d": d, "layers": layers,
                "heads": heads, "cont": getattr(prop, "cont", False), "max_macros": max_macros,
                "macros": {int(k): list(v) for k, v in lib.macros.items()},
                "concepts": [list(c) for c in concepts]}, path)


def load_model(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    lib = Library(max_macros=ck["max_macros"])
    lib.macros = {int(k): list(v) for k, v in ck["macros"].items()}
    prop = StepProposer(ck["A"], ck["L"], lib.vocab, d=ck["d"], h=ck.get("heads", 4),
                        layers=ck.get("layers", 3), cont=ck.get("cont", False)).to(device)
    prop.load_state_dict(ck["state_dict"]); prop.eval()
    return prop, lib, [list(c) for c in ck["concepts"]], ck["A"], ck["L"], ck["K"]


def render(prog, lib):
    """op-id program -> readable base-op names (macros expanded)."""
    base = [b for op in prog if op != NOP for b in lib.expand(op)]
    return " ".join(BASE_OPS[b] for b in base) or "(identity)"


def solve_prompt(prop, lib, concepts, A, L, K, depth, device, beam=8, samples=16, seed=0):
    """ANSWER A PROMPT: a prompt = K input->output examples of a hidden transformation. The model INDUCES the
    program (iterate-execute-residual + best-of-N, sound-verified), returns every distinct verified program
    (multiple answers), and APPLIES each to a fresh query input -> predicted output."""
    rng = np.random.default_rng(seed)
    prog, demos, xq, yq = gen_task_struct(rng, concepts, A, L, K, depth)
    task = (prog, demos, xq, yq)
    answers = collect_answers(prop, task, lib, A, L, depth * 2 + 2, beam, device, samples=samples)
    print(f"\n== PROMPT (depth-{depth} hidden transformation; {K} examples) ==", flush=True)
    for (x, y) in demos:
        print(f"   {x}  ->  {y}", flush=True)
    print(f"   QUERY: {xq}  ->  ?", flush=True)
    if not answers:
        print("   (the solver could not find a verified program -- abstaining)", flush=True)
        return
    print(f"\n   model found {len(answers)} VERIFIED program(s) (each reproduces all {K} examples):", flush=True)
    preds = set()
    for i, chain in enumerate(answers):
        pred = execute([o for o in chain if o != NOP], list(xq), A, lib)
        preds.add(tuple(pred))
        tag = "  <- uses discovered macro(s)" if any(o >= NBASE for o in chain) else ""
        print(f"     [{i+1}] {render(chain, lib)}   =>  query {xq} -> {pred}{tag}", flush=True)
    answer = list(list(preds)[0]) if len(preds) == 1 else "(programs disagree on query)"
    print(f"\n   ANSWER: {xq} -> {answer}   [{'CORRECT' if answer == yq else 'vs truth ' + str(yq)}]", flush=True)
    print(f"   (sound: every program above truly reproduces the examples; true hidden prog = {render(prog, lib)})",
          flush=True)


def selfplay(A=6, L=4, K=4, n_concepts=6, maxdepth=5, d=160, layers=3, heads=4, rounds=3000, bs=32, beam=8,
             lr=1.5e-3, max_macros=24, abstract_every=100, collect_n=6, collect_temp=0.9,
             use_grpo=True, grpo_mode="process", G=6, grpo_bs=12, grpo_weight=1.0, save_path=None,
             ckpt_every=0, frontier_thresh=0.7, cont=False, device=None, seeds=1, verbose=True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    enable_tf32(device)                                          # H100: free TF32 matmul speedup
    res = []
    for seed in range(seeds):
        torch.manual_seed(seed); rng = np.random.default_rng(seed)
        concepts = make_concepts(rng, n_concepts, A, L)
        lib = Library(max_macros=max_macros)
        prop = StepProposer(A, L, lib.vocab, d=d, h=heads, layers=layers, cont=cont).to(device)
        opt = torch.optim.AdamW(prop.parameters(), lr=lr, weight_decay=0.01)
        frontier_depth, recent, vbuf = 1, [], []
        for rnd in range(rounds):
            max_steps = frontier_depth * 2 + 2
            tasks = [gen_task_struct(rng, concepts, A, L, K, int(rng.integers(1, frontier_depth + 1)))
                     for _ in range(bs)]
            # --- ReST-EM SFT on verified chains (positive signal); batched greedy + sampled fallback ---
            on_cuda = "cuda" in str(device)
            gsolve = solve_iter_gpu if on_cuda else solve_iter_batch
            greedy = gsolve(prop, tasks, lib, A, L, max_steps, beam, device)
            if on_cuda and collect_n > 1:                               # BATCHED fallback (all unsolved at once)
                un = [ti for ti in range(len(tasks)) if greedy[ti] is None]
                for _s in range(collect_n - 1):
                    if not un:
                        break
                    res = solve_iter_gpu(prop, [tasks[ti] for ti in un], lib, A, L, max_steps, beam,
                                         device, temp=collect_temp)
                    nxt = []
                    for j, ti in enumerate(un):
                        if res[j] is not None:
                            greedy[ti] = res[j]
                        else:
                            nxt.append(ti)
                    un = nxt
            pairs, nsolved = [], 0
            for ti, t in enumerate(tasks):
                chain = greedy[ti]
                if chain is None and not on_cuda:                       # CPU: per-task sampled fallback
                    for _s in range(max(0, collect_n - 1)):
                        chain = solve_iter(prop, t, lib, A, L, max_steps, beam, device, temp=collect_temp)
                        if chain is not None:
                            break
                if chain is not None:
                    nsolved += 1; vbuf.append(chain)
                    X = [list(x) for (x, _y) in t[1]]; Y = [list(y) for (_x, y) in t[1]]
                    rc = refactor(chain, lib, len(chain) + len(lib.macros) + 4)
                    rc = [o for o in rc if o != NOP] or chain
                    pairs += chain_train_pairs(X, Y, rc, lib, A)
            recent.append(nsolved / bs)
            if pairs:
                prop.train(); rng.shuffle(pairs)
                st = torch.tensor([p[0] for p in pairs], device=device)
                tg = torch.tensor([p[1] for p in pairs], device=device)
                op = torch.tensor([p[2] for p in pairs], device=device)
                with amp(device):                               # bf16 forward on cuda
                    loss = F.cross_entropy(prop(st, tg), op)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(prop.parameters(), 1.0); opt.step()
            # --- GRPO relative update (success+failure signal) ---
            if use_grpo:
                gtasks = [gen_task_struct(rng, concepts, A, L, K, int(rng.integers(1, frontier_depth + 1)))
                          for _ in range(grpo_bs)]
                prop.train()
                gfn = grpo_step_process if grpo_mode == "process" else grpo_step
                gloss, _gsolve = gfn(prop, gtasks, lib, A, L, max_steps, device, G=G)
                if gloss is not None:
                    opt.zero_grad(); (grpo_weight * gloss).backward()
                    torch.nn.utils.clip_grad_norm_(prop.parameters(), 1.0); opt.step()
            if rnd > 0 and rnd % abstract_every == 0 and vbuf:
                abstract_struct(lib, vbuf[-3000:]); vbuf = vbuf[-3000:]
            if len(recent) >= 40 and np.mean(recent[-40:]) > frontier_thresh and frontier_depth < maxdepth:
                frontier_depth += 1; recent = []
                if verbose:
                    print(f"  s{seed} r{rnd}: frontier -> {frontier_depth}", flush=True)
            if save_path and seed == 0 and ckpt_every > 0 and rnd > 0 and rnd % ckpt_every == 0:
                save_model(save_path, prop, lib, concepts, A, L, K, d, max_macros, layers=layers, heads=heads)
                if verbose:
                    print(f"  s{seed} r{rnd}: checkpoint -> {save_path} "
                          f"(fetchable now; survives timeout)", flush=True)
            if verbose and rnd % max(1, rounds // 10) == 0:
                acc = evaluate(prop, rng, concepts, A, L, K, range(1, maxdepth + 1),
                               lib, maxdepth * 2 + 2, beam, device, n=80)
                print(f"  s{seed} r{rnd:5d} (frontier {frontier_depth}, macros {len(lib.macros)}, "
                      f"concepts {concepts_recovered(lib, concepts, A, L)}/{len(concepts)}): "
                      + " ".join(f"d{k}:{v:.2f}" for k, v in acc.items()), flush=True)
        final = evaluate(prop, rng, concepts, A, L, K, range(1, maxdepth + 1), lib,
                         maxdepth * 2 + 2, beam, device, n=200)
        bestN = evaluate(prop, rng, concepts, A, L, K, range(1, maxdepth + 1), lib,
                         maxdepth * 2 + 2, beam, device, n=200, samples=16, temp=0.9)
        rec = concepts_recovered(lib, concepts, A, L)
        res.append((final, rec, len(lib.macros), bestN))
        if save_path and seed == 0:
            save_model(save_path, prop, lib, concepts, A, L, K, d, max_macros, layers=layers, heads=heads)
            if verbose:
                print(f"  saved trained solver -> {save_path}", flush=True)
        if verbose:
            tag = "COMBINED (SFT+GRPO)" if use_grpo else "SFT-only"
            print(f"  s{seed} FINAL [{tag}]: concepts {rec}/{len(concepts)}, macros {len(lib.macros)}", flush=True)
            print("    solve@1   : " + " ".join(f"d{k}:{v:.3f}" for k, v in final.items()), flush=True)
            print("    best-of-16: " + " ".join(f"d{k}:{v:.3f}" for k, v in bestN.items()), flush=True)
    return res


def selftest():
    """CPU, tiny: the COMBINED solver (SFT+GRPO) must solve depth-2 above chance, and best-of-N >= solve@1."""
    res = selfplay(A=5, L=4, K=4, n_concepts=3, maxdepth=2, d=64, rounds=250, bs=24, beam=5,
                   max_macros=6, abstract_every=120, collect_n=4, use_grpo=True, G=5, grpo_bs=8,
                   seeds=1, device="cpu", verbose=False)
    final, rec, nmac, bestN = res[0]
    print(f"  selftest: concepts {rec}/3, macros {nmac} | combined solve@1 d2:{final[2]:.2f} | "
          f"best-of-16 d2:{bestN[2]:.2f}")
    assert final[2] > 0.4, f"combined solver failed to compose depth-2 ({final[2]:.2f})"
    assert bestN[2] >= final[2], "best-of-N (multiple answers) should not hurt"
    print("azr_win selftest OK")
    return 0


# ===================================================================================================
# REASONING over real data: azr_win's verified search synthesizes an explainable RULE from a prompt. Same
# paradigm (propose -> verify -> keep -> iterate), now with a HELD-OUT verifier (a condition is kept only if
# it improves accuracy on held-out prompt rows -> "verified" = GENERALIZES, not memorizes) and a self-play
# LEARNED-TO-REASON proposer (RuleProposer scores which condition to propose next). Runtime = pure reasoning,
# zero training on the problem; the proposer is trained once on SELF-GENERATED rule tasks (learn the skill).
# ===================================================================================================
def _lits(X, max_eq=4, nq=9, binary_presence_only=False):
    """Candidate conditions. Categorical (<=max_eq uniques) -> equality; continuous -> nq rounded thresholds.
    nq is a 'think harder' lever: more thresholds = finer search over continuous features.
    binary_presence_only: for one-hot {0,1} columns emit ONLY '==1' (presence). The '==0' literal of a one-hot
    level means 'NOT that level' -- a disjunctive confound that lets greedy search find spurious shortcuts
    instead of true positive-presence conjunctions; drop it when features are one-hot."""
    L = []
    for j in range(X.shape[1]):
        u = np.unique(X[:, j])
        if binary_presence_only and set(u.tolist()) <= {0.0, 1.0}:
            L += [(j, "==", 1.0)]
        elif len(u) <= max_eq:
            L += [(j, "==", float(t)) for t in u]
        else:
            qs = np.quantile(u, np.linspace(0.1, 0.9, nq))
            for t in np.unique(np.round(qs, 2)):
                L += [(j, ">", float(t)), (j, "<=", float(t))]
    return L


def _sat(lit, X):
    j, o, t = lit
    return X[:, j] > t if o == ">" else (X[:, j] <= t if o == "<=" else X[:, j] == t)


def _lit_feats(lit, X, y, mask, covered):
    """Per-candidate features for the proposer (its current effect) -> learned-to-reason scoring."""
    m = mask & _sat(lit, X)
    nm = max(1, m.sum())
    prec = (y[m] == 1).mean() if m.any() else 0.0
    prec0 = (y[mask] == 1).mean() if mask.any() else 0.0
    newpos = (m & (y == 1) & ~covered).sum() / max(1, (~covered & (y == 1)).sum())
    foil = (m & (y == 1)).sum() * (math.log(prec + 1e-6) - math.log(prec0 + 1e-6))
    return np.array([prec, prec - prec0, newpos, m.sum() / len(X), foil / max(1, len(X)),
                     1.0 if lit[1] == "==" else 0.0], np.float32)


class RuleProposer(nn.Module):
    """Learns (via self-play) to score a candidate condition by how well it advances a generalizing rule."""
    def __init__(self, nin=6, d=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(nin, d), nn.GELU(), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, feats):
        return self.net(feats).squeeze(-1)


def reason_rule(X, y, proposer=None, device="cpu", holdout=0.3, topk=12, max_disj=8, max_lit=4, seed=0,
                nq=9, min_precision=0.0, min_support=1, binary_presence_only=False, names=None, trace=False):
    """Verified DNF rule synthesis. Builds disjuncts on a FIT split; accepts each disjunct ONLY if it improves
    accuracy on a held-out VERIFY split AND its own held-out precision >= min_precision (verified = generalizes
    AND is a confident reason). proposer (optional) ranks candidate literals; without it, evaluate all.
    Returns the rule (list of conjunctions of literals). min_precision=1.0 keeps only PURE clauses (so impure
    regions abstain rather than guess); raise it for the safety-critical class.
    max_disj/max_lit/nq/topk are 'think harder' levers (more disjuncts/deeper conjunctions/finer/less pruning).
    trace=True narrates the propose -> verify -> keep/reject reasoning (pass names for readable literals)."""
    def lname(lit):                                                    # readable literal (one-hot aware)
        j, o, t = lit; nm = names[j] if names else f"x{j}"
        return nm if (o == "==" and t == 1) else (f"NOT {nm}" if (o == "==" and t == 0) else f"{nm}{o}{t:g}")

    rng = np.random.default_rng(seed); idx = rng.permutation(len(y)); c = int((1 - holdout) * len(y))
    fit, ver = idx[:c], idx[c:]
    Xf, yf, Xv, yv = X[fit], y[fit], X[ver], y[ver]
    lits = _lits(Xf, nq=nq, binary_presence_only=binary_presence_only)
    rules = []
    base = (apply_rule([], Xv) == yv).mean()
    if trace:
        print(f"  reasoning over {len(fit)} fit / {len(ver)} held-out rows; {len(lits)} candidate conditions; "
              f"start held-out acc {base:.4f}")
    for d in range(max_disj):
        covered = apply_rule(rules, Xf).astype(bool)
        if not ((yf == 1) & ~covered).any():
            if trace:
                print(f"  round {d + 1}: all positives covered -> stop")
            break
        # build ONE conjunction by PRECISION-greedy on the fit split (so conjunctive rules like
        # contract&charges are found -- a literal good only in combination still gets added on its way to purity)
        mask = np.ones(len(yf), bool); conj = []
        for _ in range(max_lit):
            cand = [lit for lit in lits if (mask & _sat(lit, Xf)).sum() not in (0, mask.sum())
                    and (mask & _sat(lit, Xf) & (yf == 1) & ~covered).sum() >= 1]
            if not cand:
                break
            if proposer is not None:                                   # learned-to-reason prunes the candidates
                feats = torch.tensor(np.stack([_lit_feats(l, Xf, yf, mask, covered) for l in cand]), device=device)
                with torch.no_grad():
                    order = torch.argsort(-proposer(feats)).cpu().numpy()
                cand = [cand[i] for i in order[:topk]]
            def q(lit):
                m = mask & _sat(lit, Xf)
                return ((yf[m] == 1).mean(), int((m & (yf == 1) & ~covered).sum()))   # precision, then coverage
            lit = max(cand, key=q)
            mask = mask & _sat(lit, Xf); conj.append(lit)
            if trace:
                m = mask
                print(f"      propose {lname(lit):<32} precision {(yf[m] == 1).mean():.3f} on "
                      f"{int(m.sum())} rows ({int((m & (yf == 1) & ~covered).sum())} new poisonous)")
            if (yf[mask] == 1).all():
                break
        firedv = apply_rule([conj], Xv).astype(bool)                   # this disjunct's OWN held-out evidence
        nfired = int(firedv.sum())
        precv = (yv[firedv] == 1).mean() if nfired else 0.0
        g = (apply_rule(rules + [conj], Xv) == yv).mean()
        # GATE on held-out accuracy gain AND purity AND enough support (pure-on-2-rows is not verified)
        if not conj or g <= base + 1e-9 or precv < min_precision or nfired < min_support:
            if trace:
                why = ("no generalizing gain" if (not conj or g <= base + 1e-9) else
                       (f"only {nfired} held-out rows (<{min_support}) -> unverified" if nfired < min_support
                        else f"impure (held-out precision {precv:.3f} < {min_precision:.3f}) -> abstain region"))
                print(f"  round {d + 1}: disjunct ({' AND '.join(lname(l) for l in conj) or '-'}) -> "
                      f"held-out {g:.4f}  REJECTED ({why}) -> stop")
            break
        rules.append(conj); base = g
        if trace:
            print(f"  round {d + 1}: KEEP ({' AND '.join(lname(l) for l in conj)})  -> "
                  f"held-out {g:.4f}, precision {precv:.3f} ✓")
    return rules


def _conj(c):
    return c["conj"] if isinstance(c, dict) else c                     # clauses may carry stats {conj, prec, support}


def apply_rule(rules, X):
    p = np.zeros(len(X), int)
    for c in rules:
        m = np.ones(len(X), bool)
        for lit in _conj(c):
            m &= _sat(lit, X)
        p[m] = 1
    return p


# ---- selective prediction: reason BOTH classes, abstain when there's no verified reason (or a conflict) ----
def reason_two_sided(X, y, proposer=None, device="cpu", seed=0, min_prec_pos=1.0, min_prec_neg=1.0, **kw):
    """Verified DNF for class 1 AND for class 0. A firing clause = positive evidence; 'nothing fired' is
    absence of evidence, not evidence of absence -- so we need both sides to know when to abstain. The two
    purities can differ: raise the bar for the safety-critical class (a false 'edible' is worse than a false
    'poisonous')."""
    pos = reason_rule(X, y, proposer, device, seed=seed, min_precision=min_prec_pos, **kw)
    neg = reason_rule(X, 1 - y, proposer, device, seed=seed, min_precision=min_prec_neg, **kw)
    return pos, neg


def predict_selective(pos, neg, X):
    """Return per-row decision: 1 / 0 / -1(ABSTAIN). Answer only when exactly one side has a verified reason;
    abstain on no-evidence (neither fires) and on conflict (both fire)."""
    pf = apply_rule(pos, X).astype(bool); nf = apply_rule(neg, X).astype(bool)
    out = np.full(len(X), -1, int)
    out[pf & ~nf] = 1
    out[nf & ~pf] = 0
    return out


def _filter_clauses(rules, X, is_class, min_precision, min_support):
    """Verify-harder: keep only disjuncts that stay pure on an INDEPENDENT validation set (with enough
    support). Each kept clause carries its verified {prec, support} -- that precision IS the confidence we
    report when it fires. Drops clauses pure-by-luck on the build split -- the cure for verifier-overfit."""
    kept = []
    for c in rules:
        conj = _conj(c)
        m = apply_rule([conj], X).astype(bool); n = int(m.sum())
        if n >= min_support and (is_class[m]).mean() >= min_precision:
            kept.append({"conj": conj, "prec": float(is_class[m].mean()), "support": n})
    return kept


def _render_conj(conj, names):
    return " AND ".join((names[j] if (o == "==" and t == 1) else
                         (f"NOT {names[j]}" if o == "==" else f"{names[j]}{o}{t:g}")) for j, o, t in conj)


def explain_selective(pos, neg, X, names, pos_label="class1", neg_label="class0"):
    """Per-row decision WITH a why + confidence. For an answer, confidence = the firing clause's verified
    precision. For an abstention, we say WHICH kind (no-evidence vs conflict) and show the closest near-miss
    on each side, so 'I don't know' is itself explained. Returns a list of dicts."""
    def fired(clauses, row):
        return [c for c in clauses if all(_sat(l, row[None, :])[0] for l in c["conj"])]

    def nearest(clauses, row):                                         # closest clause + which conditions it missed
        best = None
        for c in clauses:
            sat = [bool(_sat(l, row[None, :])[0]) for l in c["conj"]]
            frac = sum(sat) / len(sat)
            if best is None or frac > best[0]:
                miss = [c["conj"][i] for i, s in enumerate(sat) if not s]
                best = (frac, c, miss)
        return best

    out = []
    for i in range(len(X)):
        row = X[i]
        pf, nf = fired(pos, row), fired(neg, row)
        if pf and not nf:
            c = max(pf, key=lambda c: c["prec"])
            out.append({"decision": pos_label, "confidence": c["prec"], "reason_type": "verified",
                        "why": f"fires {pos_label} clause ({_render_conj(c['conj'], names)}) "
                               f"-- verified pure on {c['support']} held-out rows (precision {c['prec']:.3f})"})
        elif nf and not pf:
            c = max(nf, key=lambda c: c["prec"])
            out.append({"decision": neg_label, "confidence": c["prec"], "reason_type": "verified",
                        "why": f"fires {neg_label} clause ({_render_conj(c['conj'], names)}) "
                               f"-- verified pure on {c['support']} held-out rows (precision {c['prec']:.3f})"})
        elif pf and nf:
            cp = max(pf, key=lambda c: c["prec"]); cn = max(nf, key=lambda c: c["prec"])
            out.append({"decision": "ABSTAIN", "confidence": 0.0, "reason_type": "conflict",
                        "why": f"CONFLICT: {pos_label} clause ({_render_conj(cp['conj'], names)}) AND "
                               f"{neg_label} clause ({_render_conj(cn['conj'], names)}) both fire"})
        else:
            npos, nneg = nearest(pos, row), nearest(neg, row)
            lean = pos_label if (npos and (not nneg or npos[0] >= nneg[0])) else neg_label
            near = npos if lean == pos_label else nneg
            hint = (f"closest is a {lean} clause ({_render_conj(near[1]['conj'], names)}) "
                    f"missing only [{_render_conj(near[2], names)}]") if near else "no clause came close"
            out.append({"decision": "ABSTAIN", "confidence": 0.0, "reason_type": "no_evidence",
                        "why": f"no verified condition matches; {hint}"})
    return out


def reason_selective_iter(X, y, proposer=None, device="cpu", seed=0, stages=None, holdout=0.4,
                          min_prec_pos=0.99, min_prec_neg=1.0, min_support=15, acc_floor=0.999, verbose=True):
    """Think harder until confident. Escalate the search budget stage by stage; at each stage BUILD both rules
    liberally on a fit split, then VERIFY HARDER -- keep only disjuncts that stay pure on an INDEPENDENT
    validation fold (the cure for verifier-overfit). A higher-budget stage is ADOPTED only if it keeps held-out
    accuracy >= acc_floor, so thinking harder buys coverage and NEVER trades away safety. Purity is asymmetric:
    the safety-critical class (neg / 'edible') must be perfectly pure. Returns (pos, neg, history)."""
    stages = stages or [dict(max_disj=8, max_lit=1, nq=5, topk=8),
                        dict(max_disj=16, max_lit=2, nq=9, topk=14),
                        dict(max_disj=32, max_lit=4, nq=17, topk=24)]
    rng = np.random.default_rng(seed); idx = rng.permutation(len(y))
    c = int((1 - holdout) * len(y)); h = (len(y) - c) // 2
    fit, val, sel = idx[:c], idx[c:c + h], idx[c + h:]                  # build / verify-clauses / select-stage
    best = None; best_cov = -1.0; fallback = None; fb_acc = -1.0; history = []
    for s, cfg in enumerate(stages):
        pos, neg = reason_two_sided(X[fit], y[fit], proposer, device, seed=seed,            # build liberally
                                    min_prec_pos=0.9, min_prec_neg=0.9, min_support=3, **cfg)
        pos = _filter_clauses(pos, X[val], (y[val] == 1), min_prec_pos, min_support)        # verify harder
        neg = _filter_clauses(neg, X[val], (y[val] == 0), min_prec_neg, min_support)
        dec = predict_selective(pos, neg, X[sel]); ans = dec >= 0
        cov = float(ans.mean()); acc = float((dec[ans] == y[sel][ans]).mean()) if ans.any() else 0.0
        history.append({"stage": s, "effort": cfg, "coverage": cov, "acc_on_answered": acc})
        if verbose:
            adopt = "adopt" if (acc >= acc_floor and cov > best_cov) else "keep previous"
            print(f"  think-harder stage {s} (disj{cfg['max_disj']}/lit{cfg['max_lit']}/nq{cfg['nq']}): "
                  f"coverage {cov:.3f}  accuracy-on-answered {acc:.4f}  abstained {int((~ans).sum())}  -> {adopt}")
        if acc >= acc_floor and cov > best_cov:                         # adopt ONLY if it stays safe
            best, best_cov = (pos, neg), cov
        if acc > fb_acc:
            fallback, fb_acc = (pos, neg), acc
        if best_cov >= 0.999:
            break
    return (best or fallback)[0], (best or fallback)[1], history


def _enum_pure_conjuncts(X, y, lits, max_lit=3, min_precision=1.0, min_support=15):
    """COMPLETE search for short rules where greedy fails (e.g. tic-tac-toe lines): level-wise enumerate
    conjunctions up to max_lit, pruning by support (monotone), and keep the MINIMAL ones that are pure
    (precision >= min_precision). Unlike greedy, this can reach a pure conjunction none of whose sub-clauses
    is predictive. Returns a list of conjunctions (each a list of literals)."""
    yb = (y == 1)
    masks = [_sat(l, X).astype(bool) for l in lits]
    base = [i for i in range(len(lits)) if masks[i].sum() >= min_support]
    current = {frozenset([i]): masks[i] for i in base}                  # support-frequent sets at this level
    pure_sets, pure = [], []
    for L in range(1, max_lit + 1):
        survivors = {}
        for s, m in current.items():
            sup = int(m.sum())
            if sup < min_support:
                continue
            if any(ps <= s for ps in pure_sets):                        # minimal only: a pure subset already covers it
                continue
            if yb[m].mean() >= min_precision:
                pure_sets.append(s); pure.append([lits[i] for i in s])
            elif L < max_lit:
                survivors[s] = m
        nxt = {}
        for s, m in survivors.items():                                  # extend by one frequent literal
            for i in base:
                if i in s:
                    continue
                ns = s | {i}
                if len(ns) != L + 1 or ns in nxt:
                    continue
                nm = m & masks[i]
                if nm.sum() >= min_support:
                    nxt[ns] = nm
        current = nxt
        if not current:
            break
    return pure


def compress_rules(rules, X, is_class):
    """Set-cover compression: from a (possibly huge) pool of verified-pure clauses, greedily keep the SMALLEST
    subset that still fires on every target-class row the full pool fired on. Exhaustive search maximizes
    coverage but sprawls (many redundant pure clauses); this restores a compact, readable rule with the same
    coverage. Clauses are kept in decreasing marginal-coverage order (the most useful reasons first)."""
    if not rules:
        return rules
    masks = [apply_rule([c], X).astype(bool) & is_class for c in rules]
    target = np.zeros(len(X), bool)
    for m in masks:
        target |= m
    chosen, covered = [], np.zeros(len(X), bool)
    order = list(range(len(rules)))
    while True:
        remaining = target & ~covered
        if not remaining.any():
            break
        gains = [(int((masks[i] & remaining).sum()), i) for i in order]
        g, i = max(gains)
        if g == 0:
            break
        chosen.append(rules[i]); covered |= masks[i]; order.remove(i)
    return chosen


def _prefilter_lits(X, y, lits, keep):
    """Speed: keep only the `keep` most informative literals before the combinatorial search (univariate
    |P(y|lit)-P(y)| weighted by sqrt(support)). Symmetric in y vs 1-y, so one ranking serves both classes.
    Caveat: prunes literals with zero marginal signal -- fine when conjunction parts are individually
    informative (chess/mushroom/ttt), but would hurt pure-XOR concepts; pass keep=None to disable."""
    if not keep or len(lits) <= keep:
        return lits
    p = float(y.mean()); score = []
    for l in lits:
        m = _sat(l, X).astype(bool); s = int(m.sum())
        score.append(abs(float(y[m].mean()) - p) * (s ** 0.5) if s else 0.0)
    keep_idx = sorted(np.argsort(score)[::-1][:keep])
    return [lits[i] for i in keep_idx]


def reason_selective_cv(X, y, proposer=None, device="cpu", seed=0, folds=5, build=None,
                        min_prec_pos=0.99, min_prec_neg=1.0, min_support=15, exhaustive=False,
                        compress=False, verbose=True):
    """Raise the safe-coverage ceiling with MULTI-FOLD (bagged) verification. (1) DISCOVER candidate clauses
    from every fold's training portion (union -> more of the space found). (2) VERIFY each candidate across
    ALL folds: keep it only if it stays pure (>= min_precision) in every fold it fires in, fires in >=2 folds,
    and has enough POOLED support. Pooling support is what lifts coverage -- a genuinely-pure-but-rare clause
    that a single 20% fold drops for thin support survives once support is summed across folds; requiring
    purity in every fold preserves safety. Purity is asymmetric (the safety-critical class must be perfect)."""
    build = build or dict(max_disj=32, max_lit=4, nq=17, topk=24)
    rng = np.random.default_rng(seed); idx = rng.permutation(len(y))
    fid = np.empty(len(y), int); fid[idx] = np.arange(len(y)) % folds              # fold id per row
    cand_pos, cand_neg = {}, {}
    if exhaustive:                                  # DISCOVER ONCE on full data (CV-verify below gives robustness)
        lits = _lits(X, nq=build.get("nq", 9), binary_presence_only=build.get("binary_presence_only", False))
        lits = _prefilter_lits(X, y, lits, build.get("max_lits", None))
        mlit = build.get("max_lit", 3)
        for conj in _enum_pure_conjuncts(X, y, lits, mlit, min_prec_pos, min_support):
            cand_pos[tuple(conj)] = conj
        for conj in _enum_pure_conjuncts(X, 1 - y, lits, mlit, min_prec_neg, min_support):
            cand_neg[tuple(conj)] = conj
    else:
        for k in range(folds):
            tr = fid != k
            pos, neg = reason_two_sided(X[tr], y[tr], proposer, device, seed=seed,
                                        min_prec_pos=0.9, min_prec_neg=0.9, min_support=3, **build)
            for c in pos:
                cand_pos[tuple(_conj(c))] = _conj(c)
            for c in neg:
                cand_neg[tuple(_conj(c))] = _conj(c)
    if verbose:
        print(f"  cv-discover ({folds} folds): {len(cand_pos)} candidate poison / {len(cand_neg)} candidate "
              f"edible clauses")

    def verify(conj, is_class, min_precision):
        firedall = apply_rule([conj], X).astype(bool); total = 0; nfolds = 0; worst = 1.0
        for k in range(folds):
            m = firedall & (fid == k); s = int(m.sum())
            if s == 0:
                continue
            worst = min(worst, float(is_class[m].mean())); total += s; nfolds += 1
        ok = total >= min_support and nfolds >= 2 and worst >= min_precision
        prec = float(is_class[firedall].mean()) if firedall.any() else 0.0
        return ok, total, prec

    pos, neg = [], []
    for conj in cand_pos.values():
        ok, sup, prec = verify(conj, (y == 1), min_prec_pos)
        if ok:
            pos.append({"conj": conj, "prec": prec, "support": sup})
    for conj in cand_neg.values():
        ok, sup, prec = verify(conj, (y == 0), min_prec_neg)
        if ok:
            neg.append({"conj": conj, "prec": prec, "support": sup})
    if compress:
        np0, nn0 = len(pos), len(neg)
        pos = compress_rules(pos, X, (y == 1)); neg = compress_rules(neg, X, (y == 0))
        if verbose:
            print(f"  set-cover compression: {np0}->{len(pos)} poison clauses, {nn0}->{len(neg)} edible clauses")
    return pos, neg


def render_rule(rules, names):
    if not rules:
        return "(always 0)"
    return "  OR  ".join("(" + " AND ".join(f"{names[j]}{o}{t:g}" for j, o, t in c) + ")" for c in rules)


# ===================================================================================================
# TREE TECHNIQUE, NATIVE (no library). A decision tree IS a verified-rule discoverer -- a root->leaf path
# is one of azr_win's conjunctions. We take the ONE thing trees do better than precision-greedy literal
# picking: CART split-finding (the feature+threshold that maximally reduces residual variance), found
# in-house by a single sort + cumulative-sum scan. Boosting = azr_win's iterate-on-residual; each added
# term is a SHALLOW, held-out-VERIFIED if-then rule. Stays interpretable (shallow + few + rendered), not a
# black-box forest. This sharpens conditions (optimal thresholds) and captures interactions natively.
# ===================================================================================================
def _best_split(X, r, min_leaf=20):
    """CART split: the (feature, threshold) that maximally reduces residual variance. Pure numpy, O(d n log n)."""
    n = len(r); g0 = r.sum() ** 2 / n; best = (0.0, None, None, None, None)   # (gain, j, thr, left_mean, right_mean)
    for j in range(X.shape[1]):
        o = np.argsort(X[:, j], kind="stable"); xs = X[o, j]; rs = r[o]
        cs = np.cumsum(rs); tot = cs[-1]; i = np.arange(1, n)
        SL = cs[:-1]; nL = i; SR = tot - SL; nR = n - i
        valid = (xs[1:] > xs[:-1]) & (nL >= min_leaf) & (nR >= min_leaf)
        gain = np.where(valid, SL ** 2 / nL + SR ** 2 / nR - g0, -np.inf)
        k = int(gain.argmax())
        if gain[k] > best[0]:
            best = (float(gain[k]), j, float((xs[k] + xs[k + 1]) / 2), float(SL[k] / nL[k]), float(SR[k] / nR[k]))
    return best if best[1] is not None else None


def _fit_tree(X, r, depth=2, min_leaf=20):
    """A shallow regression tree on the residual -> nested (feature, threshold, left, right); leaf = value.
    Each root->leaf path is a verified conjunction; depth>1 captures interactions."""
    if depth == 0 or len(r) < 2 * min_leaf:
        return ("leaf", float(r.mean()))
    s = _best_split(X, r, min_leaf)
    if s is None:
        return ("leaf", float(r.mean()))
    _, j, t, _, _ = s; m = X[:, j] <= t
    return ("node", j, t, _fit_tree(X[m], r[m], depth - 1, min_leaf), _fit_tree(X[~m], r[~m], depth - 1, min_leaf))


def _tree_pred(tree, X):
    if tree[0] == "leaf":
        return np.full(len(X), tree[1])
    _, j, t, lt, rt = tree; out = np.empty(len(X)); m = X[:, j] <= t
    out[m] = _tree_pred(lt, X[m]); out[~m] = _tree_pred(rt, X[~m])
    return out


def reason_boost(X, y, holdout=0.3, lr=0.1, n_terms=400, depth=2, min_leaf=20, patience=25, seed=0):
    """Verified rule-boosting: iterate-on-residual, each term a shallow CART tree kept while it improves the
    HELD-OUT split (early stop = the verifier). Returns a model (base, lr, trees) -- an additive program of
    interpretable if-then rules. In-house: 'azr_win grows small verified decision rules.'"""
    rng = np.random.default_rng(seed); idx = rng.permutation(len(y)); c = int((1 - holdout) * len(y))
    fit, ver = idx[:c], idx[c:]
    Xf, yf, Xv, yv = X[fit], y[fit], X[ver], y[ver]
    base = float(yf.mean()); pf = np.full(len(fit), base); pv = np.full(len(ver), base)
    best = math.sqrt(((pv - yv) ** 2).mean()); trees = []; keep = 0; bad = 0
    for _ in range(n_terms):
        t = _fit_tree(Xf, yf - pf, depth, min_leaf)
        pf = pf + lr * _tree_pred(t, Xf); pv = pv + lr * _tree_pred(t, Xv); trees.append(t)
        r = math.sqrt(((pv - yv) ** 2).mean())
        if r < best - 1e-6:
            best = r; keep = len(trees); bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    return (base, lr, trees[:keep])


def boost_predict(model, X):
    base, lr, trees = model
    return base + lr * sum((_tree_pred(t, X) for t in trees), np.zeros(len(X)))


def render_boost(model, names, k=4):
    """Render the first few boosted rules as if-then text (the explanation)."""
    base, lr, trees = model
    def paths(t, cond):
        if t[0] == "leaf":
            return [(cond, t[1])]
        _, j, th, lt, rt = t
        return paths(lt, cond + [f"{names[j]}<={th:.2f}"]) + paths(rt, cond + [f"{names[j]}>{th:.2f}"])
    out = [f"base {base:.1f} (+{len(trees)} verified rules @ lr {lr})"]
    for t in trees[:k]:
        for cond, val in paths(t, []):
            out.append(f"  IF {' AND '.join(cond)}: {lr * val:+.1f}")
    return "\n".join(out)


def _gen_rule_task(rng, F, n):
    X = np.stack([rng.normal(0, 1, n) if rng.random() < 0.6 else rng.integers(0, 3, n) for _ in range(F)], 1).astype(np.float32)
    nd = int(rng.integers(1, 4)); y = np.zeros(n, bool)
    for _ in range(nd):
        m = np.ones(n, bool)
        for _ in range(int(rng.integers(1, 3))):
            j = int(rng.integers(0, F))
            m &= (X[:, j] > rng.normal()) if rng.random() < 0.7 else (X[:, j] == int(rng.integers(0, 3)))
        y |= m
    return X, y.astype(int)


def selfplay_reason(steps=3000, device="cpu", lr=1e-3, seed=0, verbose=True):
    """Train RuleProposer on SELF-GENERATED rule tasks: learn to score a literal by the held-out gain it
    yields (learn-to-reason). No external data."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    prop = RuleProposer().to(device); opt = torch.optim.AdamW(prop.parameters(), lr=lr)
    for step in range(steps):
        F_ = int(rng.integers(3, 8)); X, y = _gen_rule_task(rng, F_, 200)
        ci = rng.permutation(len(y)); c = int(0.7 * len(y)); fit, ver = ci[:c], ci[c:]
        Xf, yf, Xv, yv = X[fit], y[fit], X[ver], y[ver]
        lits = _lits(Xf); covered = np.zeros(len(yf), bool); mask = np.ones(len(yf), bool)
        cand = [l for l in lits if (mask & _sat(l, Xf)).sum() not in (0, mask.sum())]
        if len(cand) < 4:
            continue
        feats = np.stack([_lit_feats(l, Xf, yf, mask, covered) for l in cand])
        # target = held-out accuracy of the single-literal rule (generalizing gain)
        tgt = np.array([(apply_rule([[l]], Xv) == yv).mean() for l in cand], np.float32)
        prop.train()
        pred = prop(torch.tensor(feats, device=device))
        loss = F.mse_loss(pred, torch.tensor(tgt, device=device))
        opt.zero_grad(); loss.backward(); opt.step()
        if verbose and step % max(1, steps // 6) == 0:
            print(f"  selfplay-reason step {step}: literal-quality MSE {loss.item():.4f}", flush=True)
    return prop


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--A", type=int, default=6); ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--K", type=int, default=4); ap.add_argument("--concepts", type=int, default=6)
    ap.add_argument("--maxdepth", type=int, default=5); ap.add_argument("--d", type=int, default=160)
    ap.add_argument("--layers", type=int, default=3); ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--bs", type=int, default=32, help="tasks per round (massive batch -> feeds the GPU)")
    ap.add_argument("--rounds", type=int, default=3000); ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--max-macros", type=int, default=24); ap.add_argument("--collect-n", type=int, default=6)
    ap.add_argument("--grpo", dest="grpo", action="store_true", default=True,
                    help="enable the GRPO relative update (the combined winner; on by default)")
    ap.add_argument("--no-grpo", dest="grpo", action="store_false", help="SFT-only baseline (for comparison)")
    ap.add_argument("--grpo-mode", choices=["process", "outcome"], default="process",
                    help="process = per-step Dr.GRPO process reward (default); outcome = trajectory-level")
    ap.add_argument("--G", type=int, default=6); ap.add_argument("--grpo-bs", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--save", default=None, help="path to save the trained solver weights")
    ap.add_argument("--ckpt-every", type=int, default=0,
                    help="checkpoint --save every N rounds (resilient to timeouts; 0 = only at end)")
    ap.add_argument("--frontier-thresh", type=float, default=0.7,
                    help="collect solve-rate to advance the curriculum depth (lower -> climbs deeper faster)")
    ap.add_argument("--cont", action="store_true",
                    help="GENERIC continuous frontend (ingests any numeric data, not a fixed alphabet)")
    ap.add_argument("--solve", default=None, help="path to a saved solver -> answer a prompt and exit")
    ap.add_argument("--depth", type=int, default=5, help="prompt depth for --solve")
    ap.add_argument("--prompts", type=int, default=3, help="how many prompts to answer in --solve")
    ap.add_argument("--device", default=None); ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if args.solve:
        dev = args.device or "cpu"
        prop, lib, concepts, A, L, K = load_model(args.solve, dev)
        print(f"  loaded solver: {len(lib.macros)} discovered macros, {len(concepts)} concepts "
              f"(A={A}, L={L}, K={K})", flush=True)
        for i in range(args.prompts):
            solve_prompt(prop, lib, concepts, A, L, K, args.depth, dev, beam=args.beam, seed=100 + i)
        return 0
    res = selfplay(A=args.A, L=args.L, K=args.K, n_concepts=args.concepts, maxdepth=args.maxdepth,
                   d=args.d, layers=args.layers, heads=args.heads, bs=args.bs, rounds=args.rounds,
                   beam=args.beam, max_macros=args.max_macros, collect_n=args.collect_n,
                   use_grpo=args.grpo, grpo_mode=args.grpo_mode, G=args.G, grpo_bs=args.grpo_bs,
                   save_path=args.save, ckpt_every=args.ckpt_every, frontier_thresh=args.frontier_thresh,
                   cont=args.cont, seeds=args.seeds, device=args.device)
    if args.out:
        import json
        with open(args.out, "w") as f:
            json.dump([{"solve_at_1": {f"d{k}": v for k, v in fin.items()},
                        "best_of_16": {f"d{k}": v for k, v in bN.items()}, "concepts": rec, "macros": m,
                        "grpo": args.grpo} for (fin, rec, m, bN) in res], f, indent=2)
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
