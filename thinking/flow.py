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
        if decode != "sample":
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
        for _ in range(4 * problem.k + 8):                 # line budget (proof tree, not a walk)
            words, toks, line_state = self._line(ids, state=state, return_state=True)
            if words[:1] == ["answer"]:
                ans = words[1] if len(words) > 1 else None
                if not verify or self.chk.valid_answer(st, ans):
                    res.answer = ans
                    res.lines.append((words, "answer"))
                    return res
                for _ in range(self.cfg.retry):
                    res.n_resampled += 1
                    words, toks = self._line(ids, temp=self.cfg.resample_temp)
                    if (words[:1] == ["answer"] and len(words) > 1
                            and self.chk.valid_answer(st, words[1])):
                        res.answer = words[1]
                        res.lines.append((words, "answer"))
                        return res
                # repair: if the decomposition completed, the root names the answer
                res.answer = self._root_answer(st) or ans
                res.lines.append((["answer", str(res.answer)], "repair"))
                return res
            ps = parse_goal_line(words)
            ok = ps is not None and self.chk.step(st, *ps)  # step mutates ONLY when valid
            if not ok and words:
                res.rejected.append((tuple(ids), tuple(toks)))
            if verify and not ok:
                for _ in range(self.cfg.retry):
                    res.n_resampled += 1
                    words, toks, line_state = self._line(ids, temp=self.cfg.resample_temp,
                                                         state=state, return_state=True)
                    ps = parse_goal_line(words)
                    if ps is not None and self.chk.step(st, *ps):
                        ok = True
                        break
                if not ok:
                    # Verified mode must not poison the prompt with rejected text. A repeated
                    # invalid line stalls at this same clean prefix; repair after a small budget.
                    res.n_invalid += 1
                    stalls += 1
                    res.lines.append((words, "reject"))
                    if stalls >= max(1, self.cfg.retry):
                        res.answer = self._root_answer(st)
                        res.lines.append((["answer", str(res.answer)], "repair"))
                        return res
                    continue
            if not ok:
                res.n_invalid += 1
            res.lines.append((words, "ok" if ok else "invalid"))
            if ok or not verify:
                stalls = 0
                ids += toks + [dot]
                state = self._state_append(line_state, [dot])
        res.answer = self._root_answer(st)
        return res

    @torch.no_grad()
    def run_goal_constrained(self, problem, templates, question, rng=None, prompt=None,
                             edb=None):
        """Verifier-guided decoding over generic legal local proof actions.

        Candidate steps come from GoalChecker.candidate_steps(), which enumerates unchecked EDB
        facts and rule instances from the loaded Datalog rules. The model only scores candidates;
        it no longer free-forms grammar or relation-specific rule structure.
        """
        from .trace import render_goal_line, render_prompt
        res = FlowResult()
        question = problem.question or question
        st = self.chk.new_state(problem.goal[1], edb if edb is not None else problem.edb,
                                goal_pred=problem.goal[0], extra_rules=problem.extra_rules)
        if prompt is None:
            prompt, _, _ = render_prompt(problem, templates, question, rng)
        ids = self.v.enc(prompt)
        dot = self.v.stoi["."]
        state = self._state_from_ids(ids)

        for _ in range(4 * problem.k + 8):
            choices = [(["answer", ans], None) for ans in self.chk.answer_candidates(st)]
            if not choices:
                choices = [(render_goal_line(typ, head, body), (typ, head, body))
                           for typ, head, body in self.chk.candidate_steps(st)]
            if not choices:
                res.answer = self._root_answer(st)
                res.lines.append((["answer", str(res.answer)], "repair"))
                return res

            scored = []
            for words, step in choices:
                score, toks, new_state = self._score_line(ids, words, state=state)
                scored.append((score, words, toks, new_state, step))
            _score, words, toks, new_state, step = max(scored, key=lambda x: x[0])

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
        res.answer = self._root_answer(st)
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
