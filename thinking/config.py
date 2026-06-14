"""Single typed configuration for the whole pipeline (replaces the env-var sprawl).

Every knob is a dataclass field with its evidence-backed default; runs serialize their config to
config.json so results are always reproducible from the artifact alone.
"""
import json
from dataclasses import dataclass, asdict, fields


@dataclass
class Config:
    # ---- world ----------------------------------------------------------------------------------
    world: str = "chain"              # 'chain' (linear walks) | 'kinship' (nested proof trees,
    #                                   templated-NL surface, goal-directed traces)
    n_train_entities: int = 1500      # small pools -> entity memorization, no generic copy
    n_test_entities: int = 60         # held-out (disjoint by construction)
    train_hops: tuple = (1, 2, 3, 4)  # kinship: derivation depths, e.g. (2, 3)
    test_hops: tuple = (2, 4, 6, 8, 10)
    holdout_preds: tuple = ("great_grandfather", "father_in_law")   # kinship composition holdout:
    #                                   never queried in training; their COMPONENT rules are (via
    #                                   the gender-mirrored targets) -- tests unseen composition
    deep_depth: int = 0               # kinship: >0 mixes deep spines, depths 4..N
    deep_frac: float = 0.3            # deep share of goal-trace examples
    #                                   (0.5 starved shallow at 15k -- C4;
    #                                   0.3 kept shallow alive while preserving deep signal -- C5)
    deep_preds: tuple = ()            # restrict DEEP-regime query types, e.g. ('ancestor',) for
    #                                   pure recursive length-generalization training
    contrastive_frac: float = 0.5     # shallow goal-trace share arriving as same-tree prompts
    #                                   (question-conditioning: the k=2 fix -- C6d)
    extract_frac: float = 0.25        # fraction of training examples that are READING tasks
    #                                   (NL surface -> exhaustive canonical fact list)
    lang_level: str = "mix"           # surface curriculum target: preschool..scholar|mix
    curriculum: bool = True           # climb the education ladder, then the full mix
    write_frac: float = 0.1           # fraction of training that is WRITING exercises
    math_frac: float = 0.1            # fraction of training that is MATH drills (compute a-b)
    def_frac: float = 0.05            # fraction that is VOCABULARY lessons (word definitions)
    novel_frac: float = 0.08          # fraction that is NOVEL-relation problems (rule in question)
    anonymize: bool = True            # rename persons to trained slot tokens per example (rung-0
    #                                   finding: unseen-name embeddings derail structure)
    world_seed: int = 0
    # ---- model (overrides on ScratchpadLM's project defaults) ------------------------------------
    d: int = 128
    layers: int = 4
    heads: int = 4
    pos_mode: str = "rope"            # 'none' = NoPE (position-free lookup)
    arch: str = "standard"            # 'relational' = FER bet
    pointer: bool = True              # architectural copy -- broke the held-out binding floor
    loop: bool = False                # latent recurrence loses 0.93-vs-2.1 on traces (ablation
    #                                   06-10, param-unmatched shared block); opt in for
    #                                   iterated-computation tasks only
    loops: int = 8                    # latent recursion depth (4 = half compute for fast loops)
    mhc: bool = True                  # hyper-connections (zeros-init write-gate bug fixed 06-10)
    block: int = 384                  # context window (fits k=10 traces + distractor headroom)
    # ---- training --------------------------------------------------------------------------------
    sup: str = "steps"                # 'steps' (thinking trace) | 'path' (datalog_reason baseline)
    steps: int = 2500
    batch: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.01
    aux_w: float = 0.5                # attention supervision on lookup positions
    aux_layers: int = 2               # ...applied to head 0 of the last N blocks
    rule_w: float = 0.0               # auxiliary classifier on proof-line rule/action labels
    rule_contrast_w: float = 0.0      # supervised contrastive alignment for same-rule line states
    rule_contrast_temp: float = 0.1
    trace_rank_w: float = 0.0         # rank the next verifier action among legal candidates
    trace_rank_batch: int = 2         # ranking states sampled per optimizer step when enabled
    trace_rank_candidates: int = 64   # cap legal action candidates; oracle action is retained
    trace_rank_states: int = 3        # max gold/on-policy steps before a sampled rank target
    trace_dagger_frac: float = 0.0    # fraction of rank states reached by model-ranked rollout
    halt_w: float = 0.5               # weight on the Ouro-style expected-halt loss (looped models)
    ent_w: float = 0.01               # entropy bonus on the halting distribution
    n_examples: int = 6000
    refresh_every: int = 3000         # regenerate HALF the pool (the generator is infinite;
    #                                   epoch reuse collapsed stair A.2; full-pool regen every
    #                                   1500 steps cost ~5min CPU each -- A.3b's timeout death)
    seed: int = 0
    log_every: int = 250
    # ---- failure-as-signal ------------------------------------------------------------------------
    neg_w: float = 0.1                # unlikelihood weight (0.5 degraded an undertrained policy)
    neg_steps: int = 300
    neg_mine: int = 150               # worlds to mine for checker-rejected lines
    neg_lr_scale: float = 0.3
    # ---- flow / evaluation ------------------------------------------------------------------------
    retry: int = 8                    # resamples per rejected line before trace-grounded repair
    resample_temp: float = 0.5    # gentle: perturb a strong greedy, not noise
    max_line_tokens: int = 18
    n_eval: int = 30                  # worlds per (depth, mode)
    eval_seed: int = 1000

    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=1)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k in names}
        return cls(**kw)

    def asdict(self):
        return asdict(self)
