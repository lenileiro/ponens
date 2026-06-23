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
import sys

import numpy as np
import torch
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
                "heads": heads, "max_macros": max_macros,
                "macros": {int(k): list(v) for k, v in lib.macros.items()},
                "concepts": [list(c) for c in concepts]}, path)


def load_model(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    lib = Library(max_macros=ck["max_macros"])
    lib.macros = {int(k): list(v) for k, v in ck["macros"].items()}
    prop = StepProposer(ck["A"], ck["L"], lib.vocab, d=ck["d"], h=ck.get("heads", 4),
                        layers=ck.get("layers", 3)).to(device)
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
             ckpt_every=0, device=None, seeds=1, verbose=True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    enable_tf32(device)                                          # H100: free TF32 matmul speedup
    res = []
    for seed in range(seeds):
        torch.manual_seed(seed); rng = np.random.default_rng(seed)
        concepts = make_concepts(rng, n_concepts, A, L)
        lib = Library(max_macros=max_macros)
        prop = StepProposer(A, L, lib.vocab, d=d, h=heads, layers=layers).to(device)
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
            if len(recent) >= 40 and np.mean(recent[-40:]) > 0.7 and frontier_depth < maxdepth:
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
                   save_path=args.save, ckpt_every=args.ckpt_every, seeds=args.seeds, device=args.device)
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
