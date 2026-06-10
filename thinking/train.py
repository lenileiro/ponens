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
from .trace import Vocab, render_example, build_vocab, pack_batch
from .verify import StepChecker
from .flow import FlowRuntime

log = logging.getLogger("thinking")
DEV = get_device()


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)


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
            if self.cfg.world == "kinship" and self.cfg.deep_depth and rng.random() < 0.5:
                k = 4 + int(rng.integers(self.cfg.deep_depth - 3))   # deep regime: depths 4..N
            u = rng.random()                               # task type fixed BEFORE fit-retries
            for _try in range(20):                         # resample until it fits the block
                if self.cfg.world == "kinship":
                    from .trace import (render_goal_example, render_extraction_example,
                                        render_write_example)
                    from .world import anonymize
                    p, lines = self.world.sample(k, rng, exclude=self.cfg.holdout_preds)
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
                        if not dfs:                            # no bank: skip to a QA example
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
        vocab = build_vocab(examples + [e for s in lvl_sets for e in s], extra_tokens=extra)
        seqs = [(vocab.enc(ex.tokens), ex.aux) for ex in examples]
        cum_seqs, acc = [], []
        for s in lvl_sets:                                 # cum_seqs[i] = ladder levels 0..i
            acc = acc + [(vocab.enc(ex.tokens), ex.aux) for ex in s]
            cum_seqs.append(list(acc))
        m = self.build_model(vocab)
        if cfg.aux_w > 0:
            for blk in m.blocks[-cfg.aux_layers:]:
                blk.store_attn = True
        opt = torch.optim.AdamW(m.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        m.train()
        import time
        t0 = time.time()
        for st in range(cfg.steps):
            if (cfg.refresh_every and st and st % cfg.refresh_every == 0
                    and (not cum_seqs or st * (len(cum_seqs) + 1) // cfg.steps >= len(cum_seqs))):
                fresh = self.build_examples(max(1000, cfg.n_examples // 2), rng)
                fseqs = [(vocab.enc(ex.tokens), ex.aux) for ex in fresh]
                seqs = seqs[len(fseqs):] + fseqs           # rolling pool: replace the oldest half
                log.info("refreshed %d/%d pool examples at step %d", len(fseqs), len(seqs), st)
            if cum_seqs:                                   # ladder phases, final phase = full mix
                ph = st * (len(cum_seqs) + 1) // cfg.steps
                cur = cum_seqs[ph] if ph < len(cum_seqs) else seqs
            else:
                cur = seqs
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
