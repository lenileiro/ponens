"""FlowRuntime: emit the thinking flow line by line, optionally gated by the StepChecker.

The model proposes; the checker accepts or rejects; verified heads join the known set (so later
steps can build on them); rejected lines are resampled up to cfg.retry times, then the flow
repairs from the verified trace. Rejected greedy lines are returned as labeled negatives.
"""
from dataclasses import dataclass, field
import torch


@dataclass
class FlowResult:
    answer: str = None
    lines: list = field(default_factory=list)      # (tokens, status) status in ok/invalid/answer
    n_invalid: int = 0
    n_resampled: int = 0
    rejected: list = field(default_factory=list)   # (prefix_ids, line_ids) labeled negatives
    causes: dict = field(default_factory=dict)     # rejection taxonomy: cause -> count

    def _blame(self, st, parsed):
        c = "syntax" if parsed is None else st.get("why", "unknown")
        self.causes[c] = self.causes.get(c, 0) + 1


class FlowRuntime:
    def __init__(self, model, vocab, checker, cfg, device):
        self.m, self.v, self.chk, self.cfg, self.dev = model, vocab, checker, cfg, device

    @torch.no_grad()
    def _can_cache(self):
        return hasattr(self.m, "supports_kv_cache") and self.m.supports_kv_cache()

    def _state_from_ids(self, ids):
        if not self._can_cache() or not ids or len(ids) + self.cfg.max_line_tokens + 1 > self.cfg.block:
            return None
        cache = logits = None
        for t in ids:
            tok = torch.tensor([[t]], device=self.dev)
            logits, cache = self.m.forward_step(tok, cache)
        return {"cache": cache, "logits": logits}

    def _state_append_raw(self, state, toks):
        if state is None:
            return None
        if state["cache"]["ids"].shape[1] + len(toks) > self.cfg.block:
            return None
        cache, logits = state["cache"], state["logits"]
        for t in toks:
            tok = torch.tensor([[t]], device=self.dev)
            logits, cache = self.m.forward_step(tok, cache)
        return {"cache": cache, "logits": logits}

    def _state_append(self, state, toks):
        state = self._state_append_raw(state, toks)
        if state is None:
            return None
        if state["cache"]["ids"].shape[1] + self.cfg.max_line_tokens + 1 > self.cfg.block:
            return None
        return state

    @torch.no_grad()
    def _line(self, ids, temp=0.0, state=None, return_state=False, max_tokens=None):
        """Generate tokens until '.' (exclusive); returns (words, token_ids)."""
        out, dot = [], self.v.stoi["."]
        limit = max_tokens or self.cfg.max_line_tokens
        state = state or self._state_from_ids(ids)
        if state is not None:
            cur = state
            for _ in range(limit):
                logits = cur["logits"][0]
                if temp <= 0:
                    t = int(logits.argmax())
                else:
                    t = int(torch.multinomial(torch.softmax(logits / temp, -1), 1))
                if t == dot:
                    break
                out.append(t)
                cur = self._state_append_raw(cur, [t])
                if cur is None:
                    break
            ans = (self.v.dec(out), out, cur) if return_state else (self.v.dec(out), out)
            return ans
        for _ in range(limit):
            ctx = torch.tensor([(ids + out)[-self.cfg.block:]], device=self.dev)
            logits = self.m(ctx)[0, -1]
            if temp <= 0:
                t = int(logits.argmax())
            else:
                t = int(torch.multinomial(torch.softmax(logits / temp, -1), 1))
            if t == dot:
                break
            out.append(t)
        ans = (self.v.dec(out), out, None) if return_state else (self.v.dec(out), out)
        return ans

    @torch.no_grad()
    def _score_line(self, ids, words, state=None):
        """Average log-probability of a complete candidate line, including final period."""
        dot = self.v.stoi["."]
        toks = self.v.enc(words) + [dot]
        if state is not None:
            cur, score = state, 0.0
            for t in toks:
                logits = cur["logits"][0].float()
                score += float(torch.log_softmax(logits, -1)[t])
                cur = self._state_append_raw(cur, [t])
                if cur is None:
                    return float("-inf"), toks[:-1], None
            if cur["cache"]["ids"].shape[1] + self.cfg.max_line_tokens + 1 > self.cfg.block:
                cur = None
            return score / max(1, len(toks)), toks[:-1], cur

        ctx, score = list(ids), 0.0
        for t in toks:
            x = torch.tensor([ctx[-self.cfg.block:]], device=self.dev)
            logits = self.m(x)[0, -1].float()
            score += float(torch.log_softmax(logits, -1)[t])
            ctx.append(t)
        return score / max(1, len(toks)), toks[:-1], None

    def _expand_state(self, state, n):
        if state is None:
            return None
        layers = []
        for k, v in state["cache"]["layers"]:
            layers.append((k.repeat(n, 1, 1, 1), v.repeat(n, 1, 1, 1)))
        return {
            "cache": {"ids": state["cache"]["ids"].repeat(n, 1), "layers": layers},
            "logits": state["logits"].repeat(n, 1),
        }

    @torch.no_grad()
    def _score_choices(self, ids, choices, state=None, chunk=None):
        """Score many candidate lines, using the prefix KV cache in small batches when possible."""
        if chunk is None:
            chunk = 64 if self.dev == "cuda" else 16
        if state is None or not self._can_cache():
            out = []
            for words, step in choices:
                score, toks, _ = self._score_line(ids, words, state=state)
                out.append((score, words, toks, None, step))
            return out

        dot = self.v.stoi["."]
        encoded = [(self.v.enc(words) + [dot], words, step) for words, step in choices]
        by_len = {}
        for toks, words, step in encoded:
            by_len.setdefault(len(toks), []).append((toks, words, step))

        scored = []
        for _ln, group in by_len.items():
            for off in range(0, len(group), chunk):
                part = group[off:off + chunk]
                cur = self._expand_state(state, len(part))
                scores = torch.zeros(len(part), device=self.dev)
                for j in range(len(part[0][0])):
                    logits = cur["logits"].float()
                    tok = torch.tensor([p[0][j] for p in part], device=self.dev)
                    scores += torch.log_softmax(logits, -1).gather(1, tok[:, None]).squeeze(1)
                    cur_logits, cur_cache = self.m.forward_step(tok[:, None], cur["cache"])
                    cur = {"cache": cur_cache, "logits": cur_logits}
                scores = scores / len(part[0][0])
                for i, (_toks, words, step) in enumerate(part):
                    scored.append((float(scores[i]), words, _toks[:-1], None, step))
        return scored

    def _best_goal_candidate(self, st, ids, state, preferred_kind=None):
        from .trace import render_goal_line
        choices = [(["answer", ans], None) for ans in self.chk.answer_candidates(st)]
        if not choices:
            steps = self.chk.candidate_steps(st)
            if preferred_kind in ("check", "think"):
                narrowed = [s for s in steps if s[0] == preferred_kind]
                if narrowed:
                    steps = narrowed
            choices = [(render_goal_line(typ, head, body), (typ, head, body))
                       for typ, head, body in steps]
        if not choices:
            return None
        if choices[0][1] is None or st.get("support_atoms") not in (None, "pending"):
            words, step = choices[0]
            return words, self.v.enc(words), None, step
        scored = self._score_choices(ids, choices, state=state)
        _score, words, toks, _new_state, step = max(scored, key=lambda x: x[0])
        _score, toks, new_state = self._score_line(ids, words, state=state)
        return words, toks, new_state, step

    def _frontier_finish(self, st, res, max_steps, status="constrained"):
        """Finish from the generic proof-support frontier without neural scoring.

        Once support_atoms() has found one Datalog proof path, candidate_steps() is already a
        lazily narrowed, support-ranked frontier. Advancing that frontier does not need another
        token-by-token model pass for each forced line.
        """
        from .trace import render_goal_line
        for _ in range(max_steps):
            answers = self.chk.answer_candidates(st)
            if answers:
                ans = answers[0]
                res.answer = ans
                res.lines.append((["answer", str(ans)], "answer"))
                return res
            steps = self.chk.candidate_steps(st)
            if not steps:
                return self._finish_goal(st, res)
            typ, head, body = steps[0]
            if not self.chk.step(st, typ, head, body):
                res.n_invalid += 1
                res._blame(st, (typ, head, body))
                res.lines.append((render_goal_line(typ, head, body), "invalid"))
                continue
            res.lines.append((render_goal_line(typ, head, body), status))
        return self._finish_goal(st, res)

    @torch.no_grad()
    def run(self, problem, verify=True):
        from .trace import parse_line, atom_tokens
        res = FlowResult()
        edb = set(problem.edb)
        known = set(edb)
        goal_pred = problem.goal[0]
        prompt = []
        for pred, (h, t) in problem.edb:
            prompt += [h, pred, t, "."]
        prompt += ["query", goal_pred, problem.head, "?"]
        ids = self.v.enc(prompt)
        dot = self.v.stoi["."]
        last = None                                # deepest verified (goal_pred head .) tail
        for _ in range(problem.k + 4):             # line budget
            words, toks = self._line(ids)
            if words[:1] == ["answer"]:
                ans = words[1] if len(words) > 1 else None
                if not verify or self.chk.valid_answer(goal_pred, problem.head, ans, known, edb):
                    res.answer = ans
                    res.lines.append((words, "answer"))
                    return res
                for _ in range(self.cfg.retry):    # resample the answer line
                    res.n_resampled += 1
                    words, toks = self._line(ids, temp=self.cfg.resample_temp)
                    if (words[:1] == ["answer"] and len(words) > 1
                            and self.chk.valid_answer(goal_pred, problem.head, words[1], known, edb)):
                        res.answer = words[1]
                        res.lines.append((words, "answer"))
                        return res
                res.answer = last or ans           # repair: answer from the verified trace
                res.lines.append((["answer", str(res.answer)], "repair"))
                return res
            ps = parse_line(words)
            ok = ps is not None and self.chk.valid_step(ps[0], ps[1], known)
            if not ok and words:
                res.rejected.append((tuple(ids), tuple(toks)))
            if verify and not ok:
                for _ in range(self.cfg.retry):
                    res.n_resampled += 1
                    words, toks = self._line(ids, temp=self.cfg.resample_temp)
                    ps = parse_line(words)
                    if ps is not None and self.chk.valid_step(ps[0], ps[1], known):
                        ok = True
                        break
                if not ok:                         # drop the line; trace-grounded answer
                    res.answer = last
                    res.lines.append((words, "drop"))
                    return res
            if not ok:
                res.n_invalid += 1
            res.lines.append((words, "ok" if ok else "invalid"))
            if ok or not verify:                   # commit (free mode commits anything parseable)
                if ps is not None:
                    known.add(ps[0])
                    if ok and ps[0][0] == goal_pred and ps[0][1][0] == problem.head:
                        last = ps[0][1][1]
                ids += toks + [dot]
        res.answer = last
        return res

    @torch.no_grad()
    def extract(self, problem, templates, question, rng=None):
        """PHASE-1 READ: emit canonical fact-lines from the NL surface ('extract' cue, 'done' to
        stop). Returns (extracted_facts, {precision, recall, f1}) scored against the gold EDB."""
        from .trace import render_prompt, parse_fact_line
        prompt, _, _ = render_prompt(problem, templates, question, rng)
        while prompt and prompt[-1] != ".":                # reading-only: strip the question
            prompt.pop()
        ids = self.v.enc(prompt + ["extract"])
        dot = self.v.stoi["."]
        got = []
        for _ in range(len(problem.edb) + 8):
            words, toks = self._line(ids)
            if not words or words[:1] == ["done"]:
                break
            f = parse_fact_line(words)
            if f is not None:
                got.append(f)
            ids += toks + [dot]
        gold, gset = set(problem.edb), set(got)
        tp = len(gset & gold)
        prec, rec = tp / max(1, len(gset)), tp / max(1, len(gold))
        return gset, {"precision": prec, "recall": rec,
                      "f1": 2 * prec * rec / max(1e-9, prec + rec)}

    @torch.no_grad()
    def compute(self, y1, y2):
        """MATH exam: emit the difference for a (possibly never-seen) operand pair."""
        ids = self.v.enc(["compute", str(y2), "minus", str(y1), ":"])
        words, _ = self._line(ids)
        return words[0] if words else None

    @torch.no_grad()
    def define(self, word, level):
        """VOCABULARY exam: define a word at an education level. Returns the words."""
        ids = self.v.enc(["define", level, word, ":"])
        words, _ = self._line(ids, max_tokens=34)
        return words

    @torch.no_grad()
    def write(self, fact, level):
        """WRITING exercise: express a canonical fact at an education level. Returns the words."""
        pred, (h, t) = fact
        ids = self.v.enc(["write", level, h, pred, t, ":"])
        words, _ = self._line(ids)
        return words

    @torch.no_grad()
    def run_goal(self, problem, templates, question, verify=True, rng=None, prompt=None,
                 edb=None, decode="sample"):
        """Goal-directed flow: NL facts + NL question prompt (synonym variants via rng, or a
        pre-rendered `prompt`); the model decomposes subgoals ('think H needs ...'), grounds
        leaves ('check F'), then names the relation ('answer p'). The GoalChecker's agenda gates
        every line when verify=True. edb overrides the checker's fact base (SELF mode: validate
        against the model's own extraction instead of the oracle's facts)."""
        from .trace import parse_goal_line, render_prompt
        if decode == "constrained":
            return self.run_goal_constrained(problem, templates, question, rng=rng,
                                             prompt=prompt, edb=edb)
        # builtin-headed goals need compute/extract lines we cannot enumerate (you cannot
        # invert a verifier) -- gate masking to relational goals, like eager literal
        # intersections gate to closed maps
        maskable = problem.goal[0] not in self.chk.builtins
        if decode == "masked-full":
            if maskable:
                return self.run_goal_masked(problem, templates, question, rng=rng,
                                            prompt=prompt, edb=edb)
            decode = "sample"
        masked_repair = decode == "masked" and maskable
        if decode == "masked":
            decode = "sample"
        if decode not in ("sample", "hybrid"):
            raise ValueError(f"unknown goal decode mode {decode!r}")
        res = FlowResult()
        question = problem.question or question            # in-question novel-relation surface
        st = self.chk.new_state(problem.goal[1], edb if edb is not None else problem.edb,
                                goal_pred=problem.goal[0], extra_rules=problem.extra_rules)
        if prompt is None:
            prompt, _, _ = render_prompt(problem, templates, question, rng)
        ids = self.v.enc(prompt)
        dot = self.v.stoi["."]
        state = self._state_from_ids(ids)
        stalls = 0

        def masked_repair_line():
            """Replace ONE rejected line with the model's best fully-valid line (token-masked).
            Returns the (possibly final) updated loop variables, or None to finish."""
            got = self._masked_line(st, ids, state)
            if got is None:
                return None
            w, emitted, st2 = got
            if w[:1] == ["answer"]:
                res.answer = w[1]
                res.lines.append((w, "answer"))
                return "done"
            p2 = parse_goal_line(w)
            assert p2 is not None and self.chk.step(st, *p2), f"masked line invalid: {w}"
            res.lines.append((w, "masked"))
            return emitted, st2

        for _ in range(4 * problem.k + 8):                 # line budget (proof tree, not a walk)
            words, toks, line_state = self._line(ids, state=state, return_state=True)
            if words[:1] == ["answer"]:
                ans = words[1] if len(words) > 1 else None
                if not verify or self.chk.valid_answer(st, ans):
                    res.answer = ans
                    res.lines.append((words, "answer"))
                    return res
                if masked_repair:
                    res.n_invalid += 1
                    res.causes["answer-unsupported"] = res.causes.get("answer-unsupported", 0) + 1
                    fix = masked_repair_line()
                    if fix == "done":
                        return res
                    if fix is None:
                        return self._finish_goal(st, res, fallback=ans)
                    emitted, state = fix
                    ids += emitted + [dot]
                    stalls = 0
                    continue
                if decode == "hybrid" and self.chk.support_atoms(st) is not None:
                    res.n_invalid += 1
                    return self._frontier_finish(st, res, 4 * problem.k + 8)
                for _ in range(self.cfg.retry):
                    res.n_resampled += 1
                    words, toks = self._line(ids, temp=self.cfg.resample_temp)
                    if (words[:1] == ["answer"] and len(words) > 1
                            and self.chk.valid_answer(st, words[1])):
                        res.answer = words[1]
                        res.lines.append((words, "answer"))
                        return res
                if decode == "hybrid":
                    best = self._best_goal_candidate(st, ids, state)
                    if best is not None:
                        words, toks, state, step = best
                        if step is None:
                            ans = words[1] if len(words) > 1 else None
                            res.answer = ans
                            res.lines.append((words, "answer"))
                            return res
                        if self.chk.step(st, *step):
                            ids += toks + [dot]
                            stalls = 0
                            res.n_invalid += 1
                            res.lines.append((words, "constrained"))
                            continue
                # repair: if the decomposition completed, the root names the answer
                return self._finish_goal(st, res, fallback=ans)
            ps = parse_goal_line(words)
            ok = ps is not None and self.chk.step(st, *ps)  # step mutates ONLY when valid
            if not ok and words:
                res.rejected.append((tuple(ids), tuple(toks)))
                res._blame(st, ps)
            if verify and not ok and decode == "hybrid" and self.chk.support_atoms(st) is not None:
                res.n_invalid += 1
                return self._frontier_finish(st, res, 4 * problem.k + 8)
            if verify and not ok and masked_repair:
                res.n_invalid += 1
                fix = masked_repair_line()
                if fix == "done":
                    return res
                if fix is None:
                    return self._finish_goal(st, res)
                emitted, state = fix
                ids += emitted + [dot]
                stalls = 0
                continue
            if verify and not ok:
                trie = self._line_trie(st) if maskable else None
                budget = self.cfg.retry * (4 if trie is not None else 1)
                for _ in range(budget):
                    res.n_resampled += 1
                    if trie is not None:                   # eager prefix pruning: identical
                        words, toks, line_state = self._line_pruned(   # acceptance set, the
                            ids, trie, self.cfg.resample_temp, state=state)  # fail is cheap
                        if words is None:
                            continue
                    else:
                        words, toks, line_state = self._line(ids, temp=self.cfg.resample_temp,
                                                             state=state, return_state=True)
                    ps = parse_goal_line(words)
                    if ps is not None and self.chk.step(st, *ps):
                        ok = True
                        break
                    res._blame(st, ps)
                if not ok:
                    if decode == "hybrid":
                        preferred = words[0] if words and words[0] in ("check", "think") else None
                        best = self._best_goal_candidate(st, ids, state,
                                                         preferred_kind=preferred)
                        if best is not None:
                            words, toks, state, step = best
                            if step is None:
                                ans = words[1] if len(words) > 1 else None
                                res.answer = ans
                                res.lines.append((words, "answer"))
                                return res
                            if self.chk.step(st, *step):
                                ids += toks + [dot]
                                stalls = 0
                                res.n_invalid += 1
                                res.lines.append((words, "constrained"))
                                continue
                    # Verified mode must not poison the prompt with rejected text. A repeated
                    # invalid line stalls at this same clean prefix; repair after a small budget.
                    res.n_invalid += 1
                    stalls += 1
                    res.lines.append((words, "reject"))
                    if stalls >= max(1, self.cfg.retry):
                        return self._finish_goal(st, res)
                    continue
            if not ok:
                res.n_invalid += 1
            res.lines.append((words, "ok" if ok else "invalid"))
            if ok or not verify:
                stalls = 0
                ids += toks + [dot]
                state = self._state_append(line_state, [dot])
        return self._finish_goal(st, res)

    @torch.no_grad()
    def _masked_next(self, ids, state, allowed_ids):
        """Greedy next token under a validity mask. Returns (token, new_state)."""
        if state is not None:
            logits = state["logits"][0]
        else:
            ctx = torch.tensor([ids[-self.cfg.block:]], device=self.dev)
            logits = self.m(ctx)[0, -1]
        if len(allowed_ids) == 1:
            t = next(iter(allowed_ids))
        else:
            mask = torch.full_like(logits, float("-inf"))
            mask[torch.tensor(sorted(allowed_ids), device=logits.device)] = 0.0
            t = int((logits + mask).argmax())
        return t, (self._state_append_raw(state, [t]) if state is not None else None)

    @torch.no_grad()
    def _line_trie(self, st):
        """Prefix trie over ALL valid lines at this checker state (validity only). Used to
        early-abort doomed resamples: same acceptance set as full-line propose-then-reject,
        the failure is just detected at the first impossible token instead of after 18."""
        from .trace import render_goal_line
        cands = [render_goal_line(typ, head, body) + ["."]
                 for typ, head, body in self.chk.candidate_steps(
                     st, goal_pruned=False, relevance_pruned=False)]
        cands += [["answer", a, "."] for a in self.chk.answer_candidates(st)]
        root = {}
        for words in cands:
            if not all(w in self.v.stoi for w in words):
                continue
            node = root
            for t in self.v.enc(words):
                node = node.setdefault(t, {})
        return root or None

    @torch.no_grad()
    def _line_pruned(self, ids, trie, temp, state=None):
        """One resample attempt with eager prefix pruning: sample from the model's OWN
        distribution (no masking -- the distribution over accepted lines is exactly the same
        as plain resampling), but return early the moment the prefix leaves the trie.
        Returns (words, toks, post_line_state); words is None on abort."""
        out, dot = [], self.v.stoi["."]
        node = trie
        cur = state if state is not None else self._state_from_ids(ids)
        for _ in range(self.cfg.max_line_tokens):
            if cur is not None:
                logits = cur["logits"][0]
            else:
                ctx = torch.tensor([(ids + out)[-self.cfg.block:]], device=self.dev)
                logits = self.m(ctx)[0, -1]
            if temp <= 0:
                t = int(logits.argmax())
            else:
                t = int(torch.multinomial(torch.softmax(logits / temp, -1), 1))
            if t not in node:
                return None, None, None                    # doomed prefix: abort, retry
            if t == dot:
                return self.v.dec(out), out, cur
            out.append(t)
            node = node[t]
            if cur is not None:
                cur = self._state_append_raw(cur, [t])
        return None, None, None

    @torch.no_grad()
    def _masked_line(self, st, ids, state):
        """Emit ONE line under the validity mask: at every token the sampler is restricted to
        prefixes of VALID lines (candidate_steps(relevance_pruned=False) -- validity only, so
        choosing the relevant step among all legal ones stays the model's job). Returns
        (words, emitted_ids, post_dot_state) or None when no valid line exists."""
        from .trace import render_goal_line
        dot = self.v.stoi["."]
        cands = [render_goal_line(typ, head, body) + ["."]
                 for typ, head, body in self.chk.candidate_steps(
                     st, goal_pruned=False, relevance_pruned=False)]
        cands += [["answer", a, "."] for a in self.chk.answer_candidates(st)]
        cand_ids = [self.v.enc(words) for words in cands
                    if all(w in self.v.stoi for w in words)]   # OOV atoms can't be emitted
        if not cand_ids:
            return None
        cur, emitted, active, pos = state, [], list(range(len(cand_ids))), 0
        while True:
            allowed = {}
            for i in active:
                allowed.setdefault(cand_ids[i][pos], []).append(i)
            t, cur = self._masked_next(ids + emitted, cur, set(allowed))
            if t == dot and any(len(cand_ids[i]) == pos + 1 for i in allowed[dot]):
                break
            emitted.append(t)
            active, pos = allowed[t], pos + 1
        return self.v.dec(emitted), emitted, (self._state_append(cur, [])
                                              if cur is not None else None)

    @torch.no_grad()
    def run_goal_masked(self, problem, templates, question, rng=None, prompt=None, edb=None):
        """PURE checker-masked decoding: every line token-masked to valid prefixes.

        Per-line validity is 1.0 by construction, but masking is the PRIMARY decode: when the
        model's intent is invalid the mask bends the line into a valid-but-possibly-irrelevant
        one that then enters the context. Measured on the consolidation checkpoint this LOSES
        to verified-resample (0.40 vs 0.95 at k=3): off-distribution forced lines compound.
        Kept as a diagnostic; production masking is the REPAIR path in run_goal (decode=
        'masked'), where the model proposes freely and the mask only replaces rejected lines."""
        from .trace import parse_goal_line, render_prompt
        res = FlowResult()
        question = problem.question or question
        st = self.chk.new_state(problem.goal[1], edb if edb is not None else problem.edb,
                                goal_pred=problem.goal[0], extra_rules=problem.extra_rules)
        if prompt is None:
            prompt, _, _ = render_prompt(problem, templates, question, rng)
        ids = self.v.enc(prompt)
        dot = self.v.stoi["."]
        state = self._state_from_ids(ids)
        for _ in range(4 * problem.k + 8):                 # line budget
            got = self._masked_line(st, ids, state)
            if got is None:
                return self._finish_goal(st, res)
            words, emitted, state = got
            ids += emitted + [dot]
            if words[:1] == ["answer"]:
                res.answer = words[1]
                res.lines.append((words, "answer"))
                return res
            ps = parse_goal_line(words)
            assert ps is not None and self.chk.step(st, *ps), f"masked line invalid: {words}"
            res.lines.append((words, "ok"))
            if state is None and len(ids) + self.cfg.max_line_tokens > self.cfg.block:
                break                                      # context budget: repair from state
        return self._finish_goal(st, res)

    @torch.no_grad()
    def run_goal_constrained(self, problem, templates, question, rng=None, prompt=None,
                             edb=None):
        """Verifier-guided decoding over generic legal local proof actions.

        Candidate steps come from GoalChecker.candidate_steps(), which enumerates unchecked EDB
        facts and rule instances from the loaded Datalog rules. The model only scores candidates;
        it no longer free-forms grammar or relation-specific rule structure.
        """
        from .trace import render_prompt
        res = FlowResult()
        question = problem.question or question
        st = self.chk.new_state(problem.goal[1], edb if edb is not None else problem.edb,
                                goal_pred=problem.goal[0], extra_rules=problem.extra_rules)
        if prompt is None:
            prompt, _, _ = render_prompt(problem, templates, question, rng)
        ids = self.v.enc(prompt)
        dot = self.v.stoi["."]

        if self.chk.support_atoms(st) is not None:
            return self._frontier_finish(st, res, 4 * problem.k + 8, status="ok")

        state = self._state_from_ids(ids)

        for _ in range(4 * problem.k + 8):
            best = self._best_goal_candidate(st, ids, state)
            if best is None:
                return self._finish_goal(st, res)
            words, toks, new_state, step = best

            if step is None:
                ans = words[1] if len(words) > 1 else None
                res.answer = ans
                res.lines.append((words, "answer" if self.chk.valid_answer(st, ans) else "repair"))
                return res
            if not self.chk.step(st, *step):
                res.n_invalid += 1
                res.lines.append((words, "invalid"))
                continue
            res.lines.append((words, "ok"))
            ids += toks + [dot]
            state = new_state
        return self._finish_goal(st, res)

    def _finish_goal(self, st, res, fallback=None):
        """Finish from the verified checker state when it already entails a valid answer."""
        cands = self.chk.answer_candidates(st)
        ans = cands[0] if cands else (self._root_answer(st) or fallback)
        res.answer = ans
        status = "answer" if ans is not None and self.chk.valid_answer(st, ans) else "repair"
        res.lines.append((["answer", str(ans)], status))
        return res

    def _root_answer(self, st):
        """Trace-grounded repair answer from the DERIVED facts (forward semantics)."""
        gp, (x, z) = st["goal_pred"], st["pair"]
        for h in reversed(st["derived"]):
            if gp in getattr(self.chk, "builtins", {}):
                if h[0] == gp and h[1][0] == x:
                    return h[1][1]
            elif h[0] in self.chk.answer_preds and h[1] == (x, z):
                return h[0]
        return None

    @torch.no_grad()
    def path_answer(self, problem):
        """Baseline decode for path-supervised models: emit the path, endpoint = token before 'end'."""
        prompt = []
        for pred, (h, t) in problem.edb:
            prompt += [h, pred, t, "."]
        prompt += ["path", problem.head]
        save = self.cfg.max_line_tokens
        self.cfg.max_line_tokens = problem.k + 4
        words, _ = self._line(self.v.enc(prompt))
        self.cfg.max_line_tokens = save
        for j, t in enumerate(words):
            if t == "end" and j > 0:
                return words[j - 1]
        return words[-1] if words else None
