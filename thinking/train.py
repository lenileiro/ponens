"""Trainer: dense LM + aux attention supervision on packed rows; negatives fine-tune; checkpoints.

A run directory is the unit of work: config.json + model.pt (state_dict + model config + vocab)
+ results.json (+ negatives stats). load_run() rebuilds everything from the directory alone.
"""
import json
import logging
import os
import sys
from contextlib import nullcontext as _nullctx
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scratchpad_model import ScratchpadLM
from device import get_device

from .config import Config
from .world import ChainWorld, RULES, entity_pools
from .trace import (Vocab, render_example, build_vocab, pack_batch, build_rule_vocab,
                    pack_batch_with_meta)
from .verify import StepChecker
from .flow import FlowRuntime

log = logging.getLogger("thinking")
DEV = get_device()


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)


def _encode_examples(examples, vocab, with_meta=False):
    if with_meta:
        return [(vocab.enc(ex.tokens), ex.aux, ex.meta) for ex in examples]
    return [(vocab.enc(ex.tokens), ex.aux) for ex in examples]


def _rule_contrastive_loss(hidden, spans, temp=0.1):
    """Supervised contrastive loss over proof-line embeddings."""
    if len(spans) < 2:
        return None
    embs, labels = [], []
    for r, s, e, label in spans:
        if e <= s:
            continue
        embs.append(hidden[r, s:e].mean(0))
        labels.append(label)
    if len(embs) < 2:
        return None
    z = F.normalize(torch.stack(embs), dim=-1)
    y = torch.tensor(labels, device=hidden.device)
    sim = z @ z.t() / max(temp, 1e-6)
    eye = torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
    pos = y[:, None].eq(y[None, :]) & ~eye
    has_pos = pos.any(1)
    if not bool(has_pos.any()):
        return None
    sim = sim.masked_fill(eye, -1e9)
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    per_anchor = -(logp.masked_fill(~pos, 0.0).sum(1) / pos.sum(1).clamp(min=1))
    return per_anchor[has_pos].mean()


def _model_logits(model, x):
    out = model(x)
    return out[0] if isinstance(out, tuple) else out


def _candidate_line_score(model, vocab, cfg, prefix_ids, words):
    """Differentiable normalized log-probability of one rendered action line."""
    dot = vocab.stoi["."]
    cand = vocab.enc(words) + [dot]
    seq = list(prefix_ids) + cand
    begin = max(0, len(seq) - (cfg.block + 1))
    window = seq[begin:]
    if len(window) < 2:
        return None
    x = torch.tensor([window[:-1]], device=DEV)
    logits = _model_logits(model, x)[0].float()
    pieces = []
    start = len(prefix_ids)
    for orig in range(start, len(seq)):
        wi = orig - begin
        if 1 <= wi < len(window):
            pieces.append(torch.log_softmax(logits[wi - 1], -1)[window[wi]])
    if not pieces:
        return None
    return torch.stack(pieces).mean()


def _choice_key(step):
    if step is None:
        return None
    if step[0] == "answer":
        return ("answer", step[1])
    return (step[0], step[1], tuple(step[2]))


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        if cfg.world == "kinship":
            from .kinship import FamilyWorld, name_pools
            self.train_ents, self.test_ents = name_pools(                 # pool auto-extends
                cfg.n_train_entities, cfg.n_test_entities, cfg.world_seed)
            self.world = FamilyWorld(self.train_ents, seed=cfg.world_seed)
        else:
            self.train_ents, self.test_ents = entity_pools(
                cfg.n_train_entities, cfg.n_test_entities, cfg.world_seed)
            self.world = ChainWorld(self.train_ents, seed=cfg.world_seed)

    def build_examples(self, n, rng, level=None):
        out, dropped = [], 0
        if self.cfg.world == "kinship":
            from .kinship import surfaces, bank_levels
            TPL, QS = surfaces(level or self.cfg.lang_level, "train")
            # writing exercises always train; without a bank the level cue is just 'mix'
            wlevels = ([level] if level else bank_levels()) or ["mix"]
        for _ in range(n):
            k = self.cfg.train_hops[rng.integers(len(self.cfg.train_hops))]
            deep = (self.cfg.world == "kinship" and self.cfg.deep_depth and
                    rng.random() < self.cfg.deep_frac)
            if deep:
                k = 4 + int(rng.integers(self.cfg.deep_depth - 3))   # deep regime: depths 4..N
            u = rng.random()                               # task type fixed BEFORE fit-retries
            for _try in range(20):                         # resample until it fits the block
                if self.cfg.world == "kinship":
                    from .trace import (render_goal_example, render_extraction_example,
                                        render_write_example)
                    from .world import anonymize
                    include = (self.cfg.deep_preds or None) if deep else None
                    p, lines = self.world.sample(k, rng, include=include,
                                                 exclude=self.cfg.holdout_preds)
                    if self.cfg.anonymize:
                        p, lines = anonymize(p, lines, rng)
                    if u < self.cfg.math_frac:                 # MATH drill (compute a - b)
                        from .trace import render_math_example
                        y1 = 1500 + int(rng.integers(900))
                        ex = render_math_example(y1, y1 + int(rng.integers(601)))
                    elif u < self.cfg.math_frac + self.cfg.def_frac:   # VOCABULARY lesson
                        from .kinship import definitions
                        from .trace import render_def_example
                        lv = wlevels[int(rng.integers(len(wlevels)))]
                        dfs = definitions(lv, "train")
                        if not dfs:                            # no bank: use a goal trace
                            ex = render_goal_example(p, lines, TPL, QS, rng)
                        else:
                            w = list(dfs)[int(rng.integers(len(dfs)))]
                            vs = dfs[w]
                            ex = render_def_example(w, vs[int(rng.integers(len(vs)))], lv)
                    elif u < (self.cfg.math_frac + self.cfg.def_frac
                              + self.cfg.novel_frac):      # NOVEL relation (rule in the question)
                        p, lines = self.world.sample_novel(rng, train=True)
                        if self.cfg.anonymize:
                            p, lines = anonymize(p, lines, rng)
                        ex = render_goal_example(p, lines, TPL, p.question, None)
                    elif wlevels and u < (self.cfg.math_frac + self.cfg.def_frac
                                          + self.cfg.novel_frac
                                          + self.cfg.write_frac):            # WRITING
                        from .kinship import surfaces as _sf
                        lv = wlevels[int(rng.integers(len(wlevels)))]
                        tpl_lv, _ = _sf(lv, "train")
                        fact = p.edb[int(rng.integers(len(p.edb)))]
                        vs = tpl_lv.get(fact[0]) or TPL[fact[0]]
                        ex = render_write_example(fact, vs[int(rng.integers(len(vs)))], lv)
                    elif u < (self.cfg.math_frac + self.cfg.def_frac + self.cfg.novel_frac
                              + self.cfg.write_frac + self.cfg.extract_frac):   # READING task
                        ex = render_extraction_example(p, TPL, QS, rng)
                    else:
                        if (not deep and k < 4
                                and rng.random() < self.cfg.contrastive_frac):
                            # CONTRASTIVE: 3 questions about ONE tree -- question-reading
                            # becomes load-bearing instead of ignorable (the k=2 fix)
                            trio = self.world.sample_contrastive(
                                k, rng, exclude=self.cfg.holdout_preds)
                            # one rng seed for the whole trio: identical slot mapping and
                            # surface phrasing, so ONLY the question differs across members
                            seed = int(rng.integers(2 ** 31))
                            exs = []
                            for tp, tl in trio:
                                rr = np.random.default_rng(seed)
                                if self.cfg.anonymize:
                                    tp, tl = anonymize(tp, tl, rr)
                                exs.append(render_goal_example(tp, tl, TPL, QS, rr))
                            if all(len(e.tokens) <= self.cfg.block + 1 for e in exs):
                                out.extend(exs)
                                break
                            dropped += 1
                            continue
                        ex = render_goal_example(p, lines, TPL, QS, rng)
                else:
                    p = self.world.sample(k, rng)
                    steps = self.world.trace_steps(p)
                    if self.cfg.anonymize:
                        from .world import anonymize
                        p, steps = anonymize(p, [("s", h, b) for h, b in steps], rng)
                        steps = [(h, b) for _t, h, b in steps]
                    ex = render_example(p, steps, self.cfg.sup)
                if len(ex.tokens) <= self.cfg.block + 1:
                    out.append(ex)
                    break
                dropped += 1
            else:
                continue                                   # sampling-tail world: skip, don't die
        if dropped:
            log.info("resampled %d oversized examples (> block+1 tokens)", dropped)
        if len(out) < n // 2:
            raise RuntimeError(f"over half the examples don't fit block={self.cfg.block}")
        return out

    def runtime(self, m, vocab):
        """The right FlowRuntime + checker for this world."""
        if self.cfg.world == "kinship":
            from .kinship import RULES as KR, ANSWER_PREDS, AGE_BUILTINS
            from .verify import GoalChecker
            return FlowRuntime(m, vocab, GoalChecker(KR, ANSWER_PREDS, builtins=AGE_BUILTINS),
                               self.cfg, DEV)
        return FlowRuntime(m, vocab, StepChecker(RULES), self.cfg, DEV)

    def _sample_trace_rank_problem(self, rng):
        """Sample a goal-trace problem for verifier-action ranking."""
        cfg = self.cfg
        if cfg.world != "kinship":
            return None
        from .kinship import surfaces
        from .trace import render_prompt
        from .world import anonymize
        templates, question = surfaces(cfg.lang_level, "train")
        deep = cfg.deep_depth and rng.random() < max(cfg.deep_frac, 0.0)
        if deep:
            k = 4 + int(rng.integers(max(1, cfg.deep_depth - 3)))
            include = cfg.deep_preds or None
        else:
            k = cfg.train_hops[int(rng.integers(len(cfg.train_hops)))]
            include = None
        try:
            problem, lines = self.world.sample(k, rng, include=include,
                                               exclude=cfg.holdout_preds)
        except (AssertionError, RuntimeError):
            return None
        if cfg.anonymize:
            problem, lines = anonymize(problem, lines, rng)
        prompt, _, _ = render_prompt(problem, templates, problem.question or question, rng)
        return problem, prompt

    def _trace_rank_item(self, model, vocab, runtime, rng):
        """One prefix, candidate set, and oracle target for next-action ranking."""
        cfg = self.cfg
        sampled = self._sample_trace_rank_problem(rng)
        if sampled is None:
            return None
        problem, prompt = sampled
        if any(tok not in vocab.stoi for tok in prompt):
            return None
        dot = vocab.stoi["."]
        ids = vocab.enc(prompt)
        st = runtime.chk.new_state(problem.goal[1], problem.edb, goal_pred=problem.goal[0],
                                   extra_rules=problem.extra_rules)
        if runtime.chk.support_atoms(st) is None:
            return None
        plan = list(st.get("support_plan") or ())
        if not plan:
            return None

        use_dagger = rng.random() < cfg.trace_dagger_frac
        if use_dagger:
            # Reach a training state through the model's current ranked policy.  The checker still
            # supplies the oracle support frontier at that visited state.
            steps = int(rng.integers(max(1, cfg.trace_rank_states)))
            was_training = model.training
            model.eval()
            with torch.no_grad():
                for _ in range(steps):
                    ranked, _target = runtime._rank_goal_choices(
                        st, ids, None, max_choices=cfg.trace_rank_candidates,
                        goal_pruned=False, relevance_pruned=False)
                    if not ranked:
                        break
                    _score, words, toks, _state, step = ranked[0]
                    if step[0] == "answer" or not runtime.chk.step(st, *step):
                        break
                    ids += toks + [dot]
            if was_training:
                model.train()
        else:
            # Teacher-forced state sampling across the support plan.
            upto = min(len(plan), max(1, cfg.trace_rank_states))
            advance = int(rng.integers(upto))
            from .trace import render_goal_line
            for step in plan[:advance]:
                if not runtime.chk.step(st, *step):
                    return None
                ids += vocab.enc(render_goal_line(*step)) + [dot]

        target = runtime._target_goal_choice(st)
        if target is None:
            return None
        choices = runtime._goal_choices(
            st, target=target, max_choices=cfg.trace_rank_candidates,
            goal_pruned=False, relevance_pruned=False)
        if len(choices) < 2:
            return None
        target_key = _choice_key(target[1])
        target_idx = None
        kept = []
        for words, step in choices:
            if any(tok not in vocab.stoi for tok in words):
                continue
            idx = len(kept)
            kept.append((words, step))
            if _choice_key(step) == target_key:
                target_idx = idx
        if target_idx is None or len(kept) < 2:
            return None
        return ids, kept, target_idx

    def _trace_rank_loss(self, model, vocab, rng):
        cfg = self.cfg
        if cfg.world != "kinship" or cfg.trace_rank_w <= 0 or cfg.trace_rank_batch <= 0:
            return None
        runtime = self.runtime(model, vocab)
        losses = []
        attempts = 0
        while len(losses) < cfg.trace_rank_batch and attempts < cfg.trace_rank_batch * 8:
            attempts += 1
            item = self._trace_rank_item(model, vocab, runtime, rng)
            if item is None:
                continue
            prefix, choices, target_idx = item
            scores, live_target = [], None
            for i, (words, _step) in enumerate(choices):
                score = _candidate_line_score(model, vocab, cfg, prefix, words)
                if score is None:
                    continue
                if i == target_idx:
                    live_target = len(scores)
                scores.append(score)
            if live_target is None or len(scores) < 2:
                continue
            logits = torch.stack(scores)[None, :]
            target = torch.tensor([live_target], device=DEV)
            losses.append(F.cross_entropy(logits, target))
        if not losses:
            return None
        return torch.stack(losses).mean()

    def build_model(self, vocab):
        set_seed(self.cfg.seed)
        m = ScratchpadLM(len(vocab), d=self.cfg.d, layers=self.cfg.layers, heads=self.cfg.heads,
                         pad=vocab.pad, pos_mode=self.cfg.pos_mode, arch=self.cfg.arch,
                         max_len=self.cfg.block, pointer=self.cfg.pointer,
                         loop=self.cfg.loop, loops=self.cfg.loops, mhc=self.cfg.mhc).to(DEV)
        return m

    def train(self):
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        examples = self.build_examples(cfg.n_examples, rng)
        # LANGUAGE CURRICULUM: climb the education ladder (preschool -> ... -> scholar) in
        # cumulative mixtures -- earlier registers stay in distribution -- final phase = full mix
        lvl_sets = []
        if cfg.world == "kinship" and cfg.curriculum and cfg.lang_level == "mix":
            from .kinship import bank_levels
            lvls = bank_levels()
            if lvls:
                per = max(250, cfg.n_examples // (2 * len(lvls)))
                for i, lv in enumerate(lvls):
                    lvl_sets.append(self.build_examples(
                        per, np.random.default_rng(cfg.seed + 31 + i), level=lv))
                log.info("curriculum: %d levels (%s), %d examples each",
                         len(lvls), "->".join(lvls), per)
        from .world import N_SLOTS
        extra = list(self.test_ents) + [f"p{i}" for i in range(N_SLOTS)]
        if cfg.world == "kinship":                      # every surface variant word, guaranteed
            from .kinship import TEMPLATES, QUESTION
            extra += [w for vs in TEMPLATES.values() for v in vs for w in v if "{" not in w]
            extra += [w for v, _ in QUESTION for w in v if "{" not in w]
            extra += [str(y) for y in range(1400, 3001)]   # stable vocab across pool refreshes
            extra += [str(a) for a in range(0, 701)]
        all_examples = examples + [e for s in lvl_sets for e in s]
        vocab = build_vocab(all_examples, extra_tokens=extra)
        use_rule_meta = cfg.rule_w > 0 or cfg.rule_contrast_w > 0
        rule_stoi = build_rule_vocab(all_examples) if use_rule_meta else {}
        self.rule_stoi = rule_stoi
        use_rule_meta = use_rule_meta and bool(rule_stoi)
        if cfg.rule_w > 0 and not rule_stoi:
            log.info("rule auxiliary loss requested, but no trace metadata labels were found")
        if cfg.rule_contrast_w > 0 and not rule_stoi:
            log.info("rule contrastive loss requested, but no trace metadata labels were found")
        if use_rule_meta:
            log.info("rule labels: %d (%s)", len(rule_stoi), ", ".join(sorted(rule_stoi)[:8]))
        seqs = _encode_examples(examples, vocab, with_meta=use_rule_meta)
        cum_seqs, acc = [], []
        for s in lvl_sets:                                 # cum_seqs[i] = ladder levels 0..i
            acc = acc + _encode_examples(s, vocab, with_meta=use_rule_meta)
            cum_seqs.append(list(acc))
        m = self.build_model(vocab)
        rule_head = (torch.nn.Linear(cfg.d, len(rule_stoi)).to(DEV)
                     if cfg.rule_w > 0 and rule_stoi else None)
        if cfg.aux_w > 0:
            for blk in m.blocks[-cfg.aux_layers:]:
                blk.store_attn = True
        opt_params = list(m.parameters()) + (list(rule_head.parameters()) if rule_head else [])
        opt = torch.optim.AdamW(opt_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        m.train()
        import time
        t0 = time.time()
        for st in range(cfg.steps):
            if (cfg.refresh_every and st and st % cfg.refresh_every == 0
                    and (not cum_seqs or st * (len(cum_seqs) + 1) // cfg.steps >= len(cum_seqs))):
                fresh = self.build_examples(max(1000, cfg.n_examples // 2), rng)
                fseqs = _encode_examples(fresh, vocab, with_meta=use_rule_meta)
                seqs = seqs[len(fseqs):] + fseqs           # rolling pool: replace the oldest half
                log.info("refreshed %d/%d pool examples at step %d", len(fseqs), len(seqs), st)
            if cum_seqs:                                   # ladder phases, final phase = full mix
                ph = st * (len(cum_seqs) + 1) // cfg.steps
                cur = cum_seqs[ph] if ph < len(cum_seqs) else seqs
            else:
                cur = seqs
            rule_targets, spans = None, []
            if use_rule_meta:
                x, sup, rule_targets, spans = pack_batch_with_meta(
                    cur, cfg.block, cfg.batch, vocab.pad, rng, rule_stoi)
            else:
                x, sup = pack_batch(cur, cfg.block, cfg.batch, vocab.pad, rng)
            x = x.to(DEV)
            amp = torch.autocast("cuda", torch.bfloat16) if DEV == "cuda" else _nullctx()
            with amp:
                if m.loop:                              # looped default: learned depth allocation
                    logits, per_loop = m(x[:, :-1], return_per_loop=True)
                else:
                    logits, per_loop = m(x[:, :-1]), None
            loss = F.cross_entropy(logits.reshape(-1, len(vocab)), x[:, 1:].reshape(-1),
                                   ignore_index=vocab.pad)
            if per_loop and m._halt_p is not None:
                from scratchpad_model import expected_halt_loss
                loss = loss + cfg.halt_w * expected_halt_loss(
                    per_loop, m._halt_p, x[:, 1:], ignore_index=vocab.pad, ent_w=cfg.ent_w)
            if cfg.aux_w > 0 and sup:
                ri, pi, ci = (torch.tensor(v, device=DEV) for v in zip(*sup))
                for blk in m.blocks[-cfg.aux_layers:]:
                    loss = loss + (cfg.aux_w / cfg.aux_layers) * \
                        (-torch.log(blk._attn[ri, 0, pi, ci] + 1e-9)).mean()
            if use_rule_meta and hasattr(m, "_last_hidden"):
                hidden = m._last_hidden.float()
                if rule_head is not None and rule_targets is not None:
                    rt = rule_targets.to(DEV)
                    if bool(rt.ne(-100).any()):
                        rlogits = rule_head(hidden)
                        loss = loss + cfg.rule_w * F.cross_entropy(
                            rlogits.reshape(-1, rlogits.shape[-1]), rt.reshape(-1),
                            ignore_index=-100)
                if cfg.rule_contrast_w > 0 and spans:
                    closs = _rule_contrastive_loss(hidden, spans, cfg.rule_contrast_temp)
                    if closs is not None:
                        loss = loss + cfg.rule_contrast_w * closs
            if cfg.trace_rank_w > 0:
                rank_amp = torch.autocast("cuda", torch.bfloat16) if DEV == "cuda" else _nullctx()
                with rank_amp:
                    rloss = self._trace_rank_loss(m, vocab, rng)
                if rloss is not None:
                    loss = loss + cfg.trace_rank_w * rloss
            opt.zero_grad()
            loss.backward()
            opt.step()
            if cfg.log_every and (st % cfg.log_every == 0 or st == cfg.steps - 1):
                log.info("step %5d/%d loss %.3f (pool %d, %.2fs/step)",
                         st, cfg.steps, loss.item(), len(cur),
                         (time.time() - t0) / max(1, st))
        for blk in m.blocks[-cfg.aux_layers:]:
            blk.store_attn = False                  # (pointer re-enables its own block)
        return m, vocab

    # ---- failure as signal --------------------------------------------------------------------
    def mine_negatives(self, m, vocab, n=None):
        """Verified decode over training-distribution worlds; checker-rejected greedy lines are
        labeled negatives (the engine supplies the label, not the model's confidence)."""
        cfg = self.cfg
        runtime = self.runtime(m, vocab)
        rng = np.random.default_rng(cfg.seed + 7)
        negs = []
        for _ in range(n or cfg.neg_mine):
            k = cfg.train_hops[rng.integers(len(cfg.train_hops))]
            if cfg.world == "kinship":
                from .kinship import TEMPLATES, QUESTION
                p, _ = self.world.sample(k, rng, exclude=cfg.holdout_preds)
                negs += runtime.run_goal(p, TEMPLATES, QUESTION, verify=True, rng=rng).rejected
            else:
                p = self.world.sample(k, rng)
                negs += runtime.run(p, verify=True).rejected
        log.info("mined %d negatives", len(negs))
        return negs

    def train_negatives(self, m, vocab, negs):
        """Unlikelihood -log(1-p) on each rejected line given its committed prefix, mixed with
        clean dense-LM replay so the policy doesn't collapse."""
        cfg = self.cfg
        if not negs:
            return
        rng = np.random.default_rng(cfg.seed + 13)
        examples = self.build_examples(max(200, cfg.n_examples // 6), rng)
        seqs = [(vocab.enc(ex.tokens), ex.aux) for ex in examples]
        opt = torch.optim.AdamW(m.parameters(), lr=cfg.lr * cfg.neg_lr_scale,
                                weight_decay=cfg.weight_decay)
        m.train()
        for st in range(cfg.neg_steps):
            x, _ = pack_batch(seqs, cfg.block, cfg.batch, vocab.pad, rng)
            x = x.to(DEV)
            ce = F.cross_entropy(m(x[:, :-1]).reshape(-1, len(vocab)), x[:, 1:].reshape(-1),
                                 ignore_index=vocab.pad)
            nb = min(16, len(negs))
            xn = torch.full((nb, cfg.block), vocab.pad, dtype=torch.long)
            mask = torch.zeros((nb, cfg.block - 1), dtype=torch.bool)
            for r in range(nb):
                pre, line = negs[int(rng.integers(0, len(negs)))]
                row = (list(pre) + list(line))[-cfg.block:]
                xn[r, :len(row)] = torch.tensor(row)
                a = max(0, len(row) - len(line) - 1)
                mask[r, a:len(row) - 1] = True
            xn, mask = xn.to(DEV), mask.to(DEV)
            p = m(xn[:, :-1]).softmax(-1).gather(2, xn[:, 1:].unsqueeze(-1)).squeeze(-1)
            ul = (-torch.log(1 - p + 1e-6))[mask].mean()
            loss = ce + cfg.neg_w * ul
            opt.zero_grad()
            loss.backward()
            opt.step()
            if cfg.log_every and st % cfg.log_every == 0:
                log.info("neg-ft %4d/%d ce %.3f ul %.3f", st, cfg.neg_steps, ce.item(), ul.item())

    # ---- artifacts ------------------------------------------------------------------------------
    def save(self, out_dir, m, vocab):
        os.makedirs(out_dir, exist_ok=True)
        self.cfg.save(os.path.join(out_dir, "config.json"))
        torch.save({"state_dict": m.state_dict(), "config": m.config, "itos": vocab.itos},
                   os.path.join(out_dir, "model.pt"))
        if getattr(self, "rule_stoi", None):
            with open(os.path.join(out_dir, "rule_vocab.json"), "w") as f:
                json.dump(self.rule_stoi, f, indent=1)
        log.info("saved run -> %s", out_dir)


def load_run(out_dir):
    """Rebuild (cfg, model, vocab, world, runtime) from a run directory alone."""
    cfg = Config.load(os.path.join(out_dir, "config.json"))
    ck = torch.load(os.path.join(out_dir, "model.pt"), map_location=DEV, weights_only=False)
    m = ScratchpadLM(**ck["config"]).to(DEV)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    vocab = Vocab([])
    vocab.itos = ck["itos"]
    vocab.stoi = {t: i for i, t in enumerate(vocab.itos)}
    trainer = Trainer(cfg)
    runtime = trainer.runtime(m, vocab)
    return cfg, m, vocab, trainer, runtime
