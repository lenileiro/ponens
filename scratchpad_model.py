"""Causal decoder-only LM for the v6 seq2seq transfer test (+ char-LM / self-train reuse).

Positional encoding is a FACTOR (v6 step 1), because SCAN length-gen is known to be largely a
PE/EOS artifact (Csordás 2021: relative-PE alone takes length-split 0->100%): mandating one PE
would confound the architecture comparison. pos_mode in:
  'none'    : NoPE (no positional signal; the strongest length-gen lever, Kazemnejad 2023)
  'learned' : learned absolute position embeddings
  'rope'    : rotary embeddings applied to q,k per head

arch is a spectrum of relational-bottleneck strength (v6 step 5):
  'standard'   : rel_heads=0 — ordinary attention (values = content). Bit-identical to before.
  'relational' : rel_heads=heads//2 — Dual-Attention: half the heads read LEARNED token-identity
                 symbols as values (relabeling-equivariant relational lever).
  'abstractor' : rel_heads=heads — FULL relational bottleneck (all heads symbolic, no sensory
                 content in the value path). Strong bias; expected to struggle on lexical tasks
                 like SCAN (Kerg 2022) — included as the falsification arm.
  'entity'     : standard attention + an EXPLICIT ENTITY-SLOT MEMORY (content-addressed slots,
                 written on entity mentions, read back at pronouns) — an inductive bias for
                 REFERENTIAL BINDING (antecedent->pronoun/possessive) from few examples, which
                 the symbol path can't do (it discards the antecedent content). For option-2.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

_REL_HEADS = {"standard": 0.0, "relational": 0.5, "abstractor": 1.0, "entity": 0.0}


class EntityMemory(nn.Module):
    """Content-addressed entity slots, updated causally (à la Recurrent Entity Networks).
    At each position: READ the bound entity from slots built so far (so a pronoun retrieves its
    antecedent's representation), then WRITE the current token into a content-addressed slot.
    Gives a few-shot binding bias the dual-attention symbols lack.

    Addressing keys are FIXED learned slot prototypes (`slot_keys`, the canonical REN form), so the
    per-slot update is a GATED LINEAR recurrence  S[t] = (1-a[t])*S[t-1] + a[t]*v[t]  (a = gate*addr).
    That lets us replace the O(L) Python loop with an exact, stable parallel associative scan
    (Hillis-Steele, ~log2(L) steps, no division) — see `_gated_scan`. Vectorized for scale."""
    def __init__(self, d, n_slots=6):
        super().__init__()
        self.K, self.d = n_slots, d
        self.slot_keys = nn.Parameter(torch.randn(n_slots, d) * 0.02)  # FIXED addressing prototypes
        self.rq = nn.Linear(d, d)      # read query
        self.wk = nn.Linear(d, d)      # write/address key
        self.wv = nn.Linear(d, d)      # write value
        self.wg = nn.Linear(d, 1)      # write gate
        self.out = nn.Linear(2 * d, d)
        self.ln = nn.LayerNorm(d)

    @staticmethod
    def _gated_scan(beta, u):
        """Exact inclusive scan of S[t] = beta[t]*S[t-1] + u[t] over dim=1 (length).
        beta: (B,L,K,1) decays in [0,1]; u: (B,L,K,d). Associative op (b,u)x(b',u')=(b*b', b'*u+u'),
        run as a Hillis-Steele log-step scan: division-free => numerically stable even at beta->0."""
        L = beta.shape[1]
        b, s = beta, u
        shift = 1
        while shift < L:
            sb = F.pad(b, (0, 0, 0, 0, shift, 0), value=1.0)[:, :L]   # b[t-shift], identity 1
            su = F.pad(s, (0, 0, 0, 0, shift, 0), value=0.0)[:, :L]   # u[t-shift], identity 0
            s = b * su + s                                            # uses OLD b (pre-update)
            b = b * sb
            shift *= 2
        return s                                                     # (B,L,K,d), S[t]

    def forward(self, x):
        B, L, d = x.shape
        sc = 1.0 / math.sqrt(d)
        kdist = torch.softmax((self.wk(x) @ self.slot_keys.t()) * sc, -1)   # B,L,K write address
        g = torch.sigmoid(self.wg(x))                                       # B,L,1 write gate
        a = (g * kdist).unsqueeze(-1)                                       # B,L,K,1 in [0,1]
        v = self.wv(x).unsqueeze(2)                                         # B,L,1,d (shared over K)
        S = self._gated_scan(1.0 - a, a * v)                               # B,L,K,d slots after t
        S_prev = F.pad(S, (0, 0, 0, 0, 1, 0), value=0.0)[:, :L]            # slots from < t (causal)
        rdist = torch.softmax((self.rq(x) @ self.slot_keys.t()) * sc, -1)   # B,L,K read address
        r = (rdist.unsqueeze(-1) * S_prev).sum(2)                          # B,L,d retrieved entity
        return self.ln(x + self.out(torch.cat([x, r], -1)))                # residual


def _rope_cos_sin(L, hd, device, base=10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, hd, 2, device=device).float() / hd))
    t = torch.arange(L, device=device).float()
    freqs = torch.outer(t, inv_freq)                 # L, hd/2
    emb = torch.cat([freqs, freqs], dim=-1)          # L, hd
    return emb.cos()[None, None], emb.sin()[None, None]   # 1,1,L,hd


def _rope_cos_sin_pos(pos, hd, device, base=10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, hd, 2, device=device).float() / hd))
    t = torch.tensor([pos], device=device).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos()[None, None], emb.sin()[None, None]


def _rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


class RMSNorm(nn.Module):
    """Root-mean-square norm (no mean-subtraction, no bias) — the 2025-26 small-LM standard."""
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.g


class SwiGLU(nn.Module):
    """Gated FFN (w2(silu(w1 x) * w3 x)); hidden ~ 2/3*ff*d to param-match a GELU FFN."""
    def __init__(self, d, ff=4):
        super().__init__()
        hidden = max(8, int(round(ff * d * 2 / 3 / 8)) * 8)   # ~param-matched, multiple of 8
        self.w1 = nn.Linear(d, hidden, bias=False)
        self.w3 = nn.Linear(d, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class ProductKeyMemory(nn.Module):
    """Product-Key Memory (Lample 2019 / Meta 'Memory Layers at Scale' 2024): a huge trainable
    key-value table (N = n_keys^2 slots per head) addressed by LEARNED product keys at O(sqrt N) cost.
    Retrieval lives IN the weights (no external store), trained end-to-end -> knowledge capacity
    decoupled from compute. The 'RAG built into the weights' / FER-from-the-memory-angle component.
    Buys capacity/recall, NOT composition (that stays the relational core's job)."""
    def __init__(self, d, n_keys=256, heads=4, topk=16, dk=32):
        super().__init__()
        self.h, self.topk, self.n_keys, self.dk = heads, topk, n_keys, dk
        self.q = nn.Linear(d, heads * 2 * dk, bias=False)
        self.qn = RMSNorm(dk)
        self.subkeys = nn.Parameter(torch.randn(heads, 2, n_keys, dk) * 0.02)
        self.values = nn.Embedding(heads * n_keys * n_keys, d)   # N=n_keys^2 slots per head
        nn.init.normal_(self.values.weight, std=0.02)

    def forward(self, x):
        B, L, d = x.shape
        q = self.qn(self.q(x).view(B, L, self.h, 2, self.dk))
        s = torch.einsum("blhcd,hckd->blhck", q, self.subkeys)   # (B,L,h,2,n_keys) half-scores
        v1, i1 = s[..., 0, :].topk(self.topk, -1)                # per-half top-k
        v2, i2 = s[..., 1, :].topk(self.topk, -1)
        cand = (v1.unsqueeze(-1) + v2.unsqueeze(-2)).flatten(-2)  # (B,L,h,topk^2) cartesian scores
        ci = (i1.unsqueeze(-1) * self.n_keys + i2.unsqueeze(-2)).flatten(-2)   # global slot ids
        sc, idx = cand.topk(self.topk, -1)                       # final top-k over candidates
        slot = torch.gather(ci, -1, idx)
        off = (torch.arange(self.h, device=x.device) * self.n_keys * self.n_keys).view(1, 1, self.h, 1)
        vemb = self.values(slot + off)                           # (B,L,h,topk,d)
        return (sc.softmax(-1).unsqueeze(-1) * vemb).sum(3).sum(2)   # weighted sum over topk + heads


class CausalBlock(nn.Module):
    def __init__(self, d, heads, ff=4, arch="standard", vocab=None, pos_mode="learned", attn_window=None,
                 causal=True):
        super().__init__()
        self.causal = causal                             # False -> BIDIRECTIONAL attention (diffusion LM)
        self.h, self.hd = heads, d // heads
        self.rh = int(round(_REL_HEADS[arch] * heads))   # relational (symbol-value) heads
        self.pos_mode = pos_mode
        self.attn_window = attn_window                   # sliding-window (local) attention: only attend
        #                                                  to the last `attn_window` positions. Forces a
        #                                                  LOCAL solution (e.g. left-to-right scan) that
        #                                                  length-generalizes (no global gather/indexing).
        self.norm1 = RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.qnorm = RMSNorm(self.hd)                    # QK-norm (per-head) — stabilises attn logits
        self.knorm = RMSNorm(self.hd)
        # learnable per-head logit scale: QK-norm caps the dot-product, blurring softmax so it cannot
        # single out one key among many (the MQAR 'right set, wrong binding' plateau). A trainable
        # temperature lets attention SHARPEN past that cap. Init = the old fixed 1/sqrt(hd) (compat).
        self.logit_scale = nn.Parameter(torch.full((heads,), 1.0 / math.sqrt(self.hd)))
        self.proj = nn.Linear(d, d, bias=False)
        self.norm2 = RMSNorm(d)
        self.ff = SwiGLU(d, ff)
        if self.rh > 0:
            assert vocab is not None, "relational/abstractor arch needs vocab for the symbol table"
            self.symbols = nn.Embedding(vocab, self.rh * self.hd)  # token-identity symbols
            nn.init.normal_(self.symbols.weight, std=0.02)
        self.store_attn = False                          # when set, stash softmax weights for aux supervision
        self._attn = None

    def forward(self, x, ids, pad_mask):
        B, L, D = x.shape
        h = self.norm1(x)
        q, k, v = self.qkv(h).view(B, L, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)  # B,H,L,hd
        q, k = self.qnorm(q), self.knorm(k)              # QK-norm
        if self.pos_mode == "rope":
            cos, sin = _rope_cos_sin(L, self.hd, x.device)
            q = q * cos + _rotate_half(q) * sin
            k = k * cos + _rotate_half(k) * sin
        scores = (q @ k.transpose(-2, -1)) * self.logit_scale.view(1, self.h, 1, 1)
        if self.causal:
            mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), 1)
            if self.attn_window is not None:             # also mask positions > window steps back
                idx = torch.arange(L, device=x.device)
                mask = mask | (idx[None, :] <= idx[:, None] - self.attn_window)
            scores = scores.masked_fill(mask[None, None], float("-inf"))
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask[:, None, None, :], float("-inf"))
        a = scores.softmax(-1)
        if self.store_attn:                               # keep (with grad) for aux attention-supervision
            self._attn = a
        if self.rh > 0:                                   # (dual-)attention symbol value path
            sens_h = self.h - self.rh
            sym = self.symbols(ids).view(B, L, self.rh, self.hd).transpose(1, 2)  # B,rh,L,hd
            parts = []
            if sens_h > 0:
                parts.append(a[:, :sens_h] @ v[:, :sens_h])
            parts.append(a[:, sens_h:] @ sym)
            o = torch.cat(parts, dim=1).transpose(1, 2).reshape(B, L, D)
        else:
            o = (a @ v).transpose(1, 2).reshape(B, L, D)  # standard: bit-identical to before
        x = x + self.proj(o)
        x = x + self.ff(self.norm2(x))
        return x

    def forward_step(self, x, ids, ids_all, cache=None, pad=0):
        """One-token causal decode with KV cache. Training still uses forward(); this path exists
        so very deep proof traces do not rerun the whole prefix for every generated token."""
        B, L, D = x.shape
        assert L == 1, "forward_step expects exactly one token"
        pos = ids_all.shape[1] - 1
        h = self.norm1(x)
        q, k, v = self.qkv(h).view(B, L, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        q, k = self.qnorm(q), self.knorm(k)
        if self.pos_mode == "rope":
            cos, sin = _rope_cos_sin_pos(pos, self.hd, x.device)
            q = q * cos + _rotate_half(q) * sin
            k = k * cos + _rotate_half(k) * sin
        if cache is None:
            k_all, v_all = k, v
        else:
            pk, pv = cache
            k_all, v_all = torch.cat([pk, k], 2), torch.cat([pv, v], 2)
        scores = (q @ k_all.transpose(-2, -1)) * self.logit_scale.view(1, self.h, 1, 1)
        if self.attn_window is not None:
            key_pos = torch.arange(k_all.shape[2], device=x.device)
            scores = scores.masked_fill((key_pos <= pos - self.attn_window)[None, None, None],
                                        float("-inf"))
        pad_mask = ids_all.eq(pad)
        if pad_mask.any():
            scores = scores.masked_fill(pad_mask[:, None, None, :], float("-inf"))
        a = scores.softmax(-1)
        if self.store_attn:
            self._attn = a
        if self.rh > 0:
            sens_h = self.h - self.rh
            sym = self.symbols(ids_all).view(B, ids_all.shape[1], self.rh, self.hd).transpose(1, 2)
            parts = []
            if sens_h > 0:
                parts.append(a[:, :sens_h] @ v_all[:, :sens_h])
            parts.append(a[:, sens_h:] @ sym)
            o = torch.cat(parts, dim=1).transpose(1, 2).reshape(B, L, D)
        else:
            o = (a @ v_all).transpose(1, 2).reshape(B, L, D)
        x = x + self.proj(o)
        x = x + self.ff(self.norm2(x))
        return x, (k_all, v_all)


class ScratchpadLM(nn.Module):
    """PROJECT DEFAULTS = the configuration each lever EARNED (see memory/experiment evidence):
      pos_mode='rope'    fluent generation + context-window extension (teach_english/cotrain)
      tie=True           REQUIRED for unseen-token copy (untied output rows are untrained noise)
      pointer=True       architectural in-context copying -- broke the held-out binding 0.00 floor
                         (thinking_flow); gate is learned, so tasks that don't copy drive g -> 0
      logit_scale        learnable per-head attn temperature (MQAR sharpness), always on
      loop=False         latent recurrence OFF by default: 4-arm ablation 2026-06-10 (kinship
                         traces, identical data/objective/budget) -- distinct-layer stack 0.93
                         vs shared-block loop 2.09-2.27 REGARDLESS of mHC fixed/absent. A shared
                         block has 1/layers the transformer params and underfits language at our
                         scale; Ouro's looped wins needed 7.7T-token pretraining. Recurrence
                         remains the validated choice for ITERATED-COMPUTATION tasks
                         (loop_compute: train k<=20 -> k=40 at 0.98) -- opt in there
      loop_inject=True   decisive anti-drift lever for recurrence (computation-lengthgen)
      mhc=True           manifold-constrained hyper-connections, default-on for looped models
                         (loop_compute); inert when loop=False
      halting            looped models get a LEARNED halt head (Ouro: entropy-regularized depth
                         allocation). forward(..., loops='auto') early-exits when the head says
                         stop; train with expected_halt_loss() over per-loop readouts
      d=256/layers=4/heads=8/max_len=512   the proven small-model working point
    Train looped models with loop_noise=0.03 (error-injection robustness; 0.1 destructive)."""

    def __init__(self, vocab, d=256, layers=4, heads=8, max_len=512, pad=0,
                 pos_mode="rope", loop=False, loops=8, arch="standard",
                 mem=False, mem_keys=256, mem_heads=4, tie=True, loop_steps=0, attn_window=None,
                 loop_inject=True, hier=False, hier_period=4, mhc=True, mhc_n=4, causal=True,
                 pointer=True, trm=False, trm_T=4):
        cfg = {k: v for k, v in locals().items() if k not in ("self", "__class__")}
        super().__init__()
        self.config = cfg                                # complete, serializable -> checkpoints
        #                                                  can always be rebuilt via ScratchpadLM(**config)
        self.causal = causal                             # False -> bidirectional (diffusion LM)
        self.mhc = mhc if (loop and mhc and not trm and not hier) else None   # mHC (DeepSeek-V4);
        #                                                  trm/hier recursion styles override the default
        self.mhc_n = mhc_n                               # N residual streams across loops, identity-preserving mix
        self.loop_inject = loop_inject                   # re-add the input embedding every loop (Huginn/n-RASP-L:
        #                                                  prevents latent drift over many recurrent iterations)
        self.hier = hier                                 # two-timescale recurrence (HRM): fast low-level block
        self.hier_period = hier_period                   # every loop + slow high-level block every hier_period
        self.trm, self.trm_T = trm, trm_T                # TRM (arXiv:2510.04871): ONE shared block recursing
        #                                                  over (answer y, latent z); trm_T latent updates per
        #                                                  answer update; `loops` = supervision segments.
        #                                                  Train with return_per_loop=True and SUM the per-
        #                                                  segment losses (deep supervision) -- states detach
        #                                                  between segments, so a final-only loss trains just
        #                                                  the last segment.
        assert not trm or loop, "trm mode requires loop=True (single shared block)"
        self.tie = tie
        assert arch in _REL_HEADS, f"unknown arch {arch}"
        assert arch == "standard" or heads % 2 == 0, "relational/abstractor need even heads"
        assert pos_mode != "rope" or (d // heads) % 2 == 0, "rope needs even head dim"
        self.pad = pad
        self.pos_mode = pos_mode
        self.loop, self.loops = loop, loops
        # optional per-iteration timestep embedding (Universal-Transformer style): tells the shared block
        # which loop iteration it is, so it can advance one step of the iterated computation per loop.
        self.loop_step = nn.Embedding(loop_steps, d) if (loop and loop_steps > 0) else None
        self.tok = nn.Embedding(vocab, d, padding_idx=pad)
        self.pos = nn.Embedding(max_len, d) if pos_mode == "learned" else None  # rope/none: no add
        nb = 1 if loop else layers
        self.blocks = nn.ModuleList([CausalBlock(d, heads, arch=arch, vocab=vocab, pos_mode=pos_mode,
                                                 attn_window=attn_window, causal=causal) for _ in range(nb)])
        self.blockH = CausalBlock(d, heads, arch=arch, vocab=vocab, pos_mode=pos_mode,
                                  attn_window=attn_window) if (loop and hier) else None  # slow high-level
        if self.mhc is not None:                         # hyper-connection params (read combo, near-identity
            self.hc_read = nn.Parameter(torch.zeros(mhc_n))                     # row-stochastic mix, write)
            # IDENTITY-PRESERVING INIT (HC/ICLR-2025 + mHC): at init the system must be equivalent
            # to a plain residual stack -- the block output IS written (tanh(1.5)~=0.9). zeros here
            # made ww=0: block output never written, gradients to the block gated to ZERO, the model
            # stuck as a bigram LM (the loss~3 plateau of every looped run).
            self.hc_write = nn.Parameter(torch.full((mhc_n,), 1.5))
            self.hc_mix = nn.Parameter(torch.eye(mhc_n) * 4.0)   # softmax(rows) -> diag-dominant ~= identity (mHC)
        # in-weight Product-Key Memory (RAG-in-weights), applied mid-stack; ablatable for the bet test
        self.mem = ProductKeyMemory(d, mem_keys, mem_heads) if mem else None
        self.memnorm = RMSNorm(d) if mem else None
        self.mem_at = nb // 2
        self.entmem = EntityMemory(d) if arch == "entity" else None
        self.lnf = RMSNorm(d)
        # learned halting (Ouro/LoopLM): per-loop halt logit from the pooled state; the halting
        # distribution allocates depth per input (train via expected_halt_loss, infer loops='auto')
        self.halt_head = nn.Linear(d, 1) if loop else None
        self._halt_p = None
        # pointer/copy head: mix the LM distribution with head-0-of-the-final-block's attention
        # SCATTERED onto the context token identities (pointer-generator). Copying becomes
        # architectural ("attend to the right position" -- exactly what aux supervision trains)
        # instead of statistical (drag embedding -> hope the tied head scores it top-1): unseen
        # tokens copy by construction, and binding sharpness no longer fights the softmax logit cap.
        self.pointer = pointer
        if pointer:
            self.ptr_gate = nn.Parameter(torch.tensor(-1.0))   # sigmoid -> g~0.27, learned
        self.head = nn.Linear(d, vocab, bias=False)
        self.apply(self._init)                           # GPT-2-style 0.02 init (needed for tying)
        if tie:                                          # untie for disjoint key/value recall tasks
            self.head.weight = self.tok.weight           # weight tying (embedding <-> output)
        with torch.no_grad():
            self.tok.weight[pad].zero_()

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def _halt(self, state, pad_mask, halts, auto):
        """Append this loop's halt logit (pooled state); True -> early-exit ('auto' inference)."""
        if self.halt_head is None:
            return False
        keep = (~pad_mask).unsqueeze(-1)
        pooled = (state * keep).sum(1) / keep.sum(1).clamp(min=1)
        q = self.halt_head(pooled).squeeze(-1)               # (B,)
        halts.append(q)
        return auto and bool((torch.sigmoid(q) > 0.5).all())

    def forward(self, ids, loops=None, return_per_loop=False, loop_noise=0.0, prefix=None):
        """prefix: optional (B, P, d) continuous embeddings (image patches, audio frames)
        prepended BEFORE the token stream -- the multimodal bridge. Logits are returned for
        ALL positions; callers slice [:, P:] for the token part. Incompatible with the
        pointer head (no token ids to scatter onto) -- use pointer=False models."""
        B, L = ids.shape
        if self.pointer:
            if prefix is not None:
                raise ValueError("prefix embeddings are incompatible with the pointer head")
            self.blocks[-1].store_attn = True                # pointer needs the weights every forward
        pad_mask = ids.eq(self.pad)
        x = self.tok(ids)
        if self.pos is not None:
            x = x + self.pos(torch.arange(L, device=ids.device)[None, :])
        if prefix is not None:
            x = torch.cat([prefix.to(x.dtype), x], dim=1)
            pad_mask = F.pad(pad_mask, (prefix.shape[1], 0), value=False)
            ids = F.pad(ids, (prefix.shape[1], 0), value=self.pad + 1 if self.pad == 0 else 0)
            B, L = ids.shape
        per_loop, halts = [], []
        auto = loops == "auto"
        if auto:
            loops = None                                     # run up to self.loops, may exit early
        if self.loop and self.mhc is not None:
            # mHC: N parallel residual streams + IDENTITY-PRESERVING row-stochastic mixing (manifold
            # constraint) -> healthier gradient flow / residual stream across MANY loops (deep recurrence).
            n = loops if loops is not None else self.loops
            inj = x if self.loop_inject else None
            N = self.mhc_n
            X = x.unsqueeze(1).repeat(1, N, 1, 1)            # (B,N,L,d): init every stream = input embedding
            rw = torch.softmax(self.hc_read, 0)              # read combo (loop-invariant -> length-gen)
            M = torch.softmax(self.hc_mix, dim=1)            # (N,N) ROW-STOCHASTIC, init near-identity (mHC)
            ww = torch.tanh(self.hc_write)                   # per-stream write of the block output
            for i in range(n):
                r = torch.einsum("n,bnld->bld", rw, X)       # read: convex combo of streams
                xin = r + inj if inj is not None else r
                o = self.blocks[0](xin, ids, pad_mask)
                X = torch.einsum("mn,bnld->bmld", M, X) + ww.view(1, N, 1, 1) * o.unsqueeze(1)
                if loop_noise > 0.0 and self.training:
                    X = X + torch.randn_like(X) * loop_noise
                r2 = torch.einsum("n,bnld->bld", rw, X)      # this loop's readout state
                if return_per_loop:
                    per_loop.append(self.head(self.lnf(r2)))
                if self._halt(r2, pad_mask, halts, auto):
                    break
            x = torch.einsum("n,bnld->bld", rw, X)           # final readout = read combo of streams
        elif self.loop and self.trm:
            # TRM: full backprop WITHIN a segment (through all trm_T z-updates + the y-update),
            # graph CUT between segments -- the paper's replacement for HRM's one-step gradient
            # approximation. z sees x+y+z (input injection); the answer update y = f(y+z) does not
            # see x directly (TRM's asymmetry). Readout/deep supervision is on y after each segment.
            n = loops if loops is not None else self.loops   # n = supervision segments
            y, z = torch.zeros_like(x), torch.zeros_like(x)
            for i in range(n):
                for _ in range(self.trm_T):
                    z = self.blocks[0](x + y + z, ids, pad_mask)
                y = self.blocks[0](y + z, ids, pad_mask)
                if loop_noise > 0.0 and self.training:       # error-injection robustness (project lever)
                    z = z + torch.randn_like(z) * loop_noise
                if return_per_loop:
                    per_loop.append(self.head(self.lnf(y)))
                if self._halt(y, pad_mask, halts, auto):
                    break
                if i < n - 1:
                    y, z = y.detach(), z.detach()            # carry state, cut the graph (deep supervision)
            x = y
        elif self.loop:
            n = loops if loops is not None else self.loops   # dynamic loop count (n-RASP-L: loops ~ problem size)
            inj = x if self.loop_inject else None            # the input embedding, re-injected each iteration
            xH = torch.zeros_like(x) if self.blockH is not None else None   # slow high-level state (HRM)
            for i in range(n):
                xin = x + inj if inj is not None else x
                if xH is not None:                           # fast block is conditioned on the slow state
                    xin = xin + xH
                if self.loop_step is not None:               # per-iteration timestep (Universal-Transformer style)
                    xin = xin + self.loop_step(torch.tensor(
                        [min(i, self.loop_step.num_embeddings - 1)], device=ids.device))
                x = self.blocks[0](xin, ids, pad_mask)       # fast low-level: every loop
                if xH is not None and (i + 1) % self.hier_period == 0:
                    xH = self.blockH(xH + x, ids, pad_mask)  # slow high-level: every hier_period loops
                if loop_noise > 0.0 and self.training:       # error-injection robustness: perturb the latent so
                    x = x + torch.randn_like(x) * loop_noise  # the block learns to CORRECT (don't let slips cascade)
                if return_per_loop:                          # readout after EACH loop (for deep supervision)
                    per_loop.append(self.head(self.lnf(x)))
                if self._halt(x, pad_mask, halts, auto):
                    break
        else:
            for i, blk in enumerate(self.blocks):
                x = blk(x, ids, pad_mask)
                if self.mem is not None and i == self.mem_at:     # in-weight memory read (residual)
                    x = x + self.mem(self.memnorm(x))
        self._halt_p = None
        if halts:                                          # halting distribution p_t = q_t * prod(1-q_<t)
            q = torch.sigmoid(torch.stack(halts, 1))       # (B, T)
            cont = torch.cumprod(1 - q, 1)
            p = q * torch.cat([torch.ones_like(q[:, :1]), cont[:, :-1]], 1)
            p = torch.cat([p[:, :-1], p[:, -1:] + (1 - p.sum(1, keepdim=True))], 1)  # absorb tail
            self._halt_p = p
        if self.entmem is not None:                        # explicit entity-binding memory
            x = self.entmem(x)
        hidden = self.lnf(x)
        self._last_hidden = hidden
        out = self.head(hidden)
        if self.pointer:
            if self.blocks[-1]._attn is None:                # seen once: CUDA+autocast LOOPED path
                raise RuntimeError("pointer: blocks[-1]._attn is None despite store_attn -- "
                                   "looped+pointer+autocast combo (v6 kinship crash); "
                                   "loop is opt-in now, investigate before re-enabling")
            a0 = self.blocks[-1]._attn[:, 0]                 # (B,L,L) copy head = aux-supervised head
            # zeros in a0's dtype: under autocast, softmax is fp32 while the head's out is bf16
            ptr = torch.zeros(out.shape, device=out.device, dtype=a0.dtype) \
                       .scatter_add_(2, ids.unsqueeze(1).expand(-1, L, -1), a0)
            ptr[..., self.pad] = 0.0                         # never copy padding
            g = torch.sigmoid(self.ptr_gate)
            # log of a normalized mixture: log_softmax is shift-invariant, so this feeds CE and
            # argmax/temperature decoding unchanged.
            out = torch.log((1 - g) * out.softmax(-1) + g * ptr + 1e-9)
        return (out, per_loop) if return_per_loop else out

    def supports_kv_cache(self):
        return (self.causal and not self.loop and self.mem is None and self.entmem is None)

    @torch.no_grad()
    def forward_step(self, ids, cache=None):
        """One-token inference step. Returns logits for the next token after `ids` and a cache.
        Supported for the standard non-recurrent decoder path; callers should fall back to
        forward() when supports_kv_cache() is false."""
        if not self.supports_kv_cache():
            raise RuntimeError("forward_step is only supported for non-loop causal decoder inference")
        assert ids.ndim == 2 and ids.shape[1] == 1, "forward_step expects shape (B, 1)"
        layers = [None] * len(self.blocks) if cache is None else cache["layers"]
        ids_all = ids if cache is None else torch.cat([cache["ids"], ids], 1)
        pos = ids_all.shape[1] - 1
        x = self.tok(ids)
        if self.pos is not None:
            x = x + self.pos(torch.tensor([[pos]], device=ids.device))
        if self.pointer:
            self.blocks[-1].store_attn = True
        new_layers = []
        for blk, layer_cache in zip(self.blocks, layers):
            x, new_cache = blk.forward_step(x, ids, ids_all, layer_cache, pad=self.pad)
            new_layers.append(new_cache)
        hidden = self.lnf(x)
        self._last_hidden = hidden
        out = self.head(hidden)
        if self.pointer:
            a0 = self.blocks[-1]._attn[:, 0]                 # (B,1,S)
            ptr = torch.zeros(out.shape, device=out.device, dtype=a0.dtype) \
                       .scatter_add_(2, ids_all.unsqueeze(1), a0)
            ptr[..., self.pad] = 0.0
            g = torch.sigmoid(self.ptr_gate)
            out = torch.log((1 - g) * out.softmax(-1) + g * ptr + 1e-9)
        return out[:, -1], {"layers": new_layers, "ids": ids_all}


def expected_halt_loss(per_loop, halt_p, targets, ignore_index=-100, ent_w=0.01):
    """Ouro-style learned depth allocation: expected per-loop CE under the halting distribution,
    minus an entropy BONUS (keeps depth exploration alive; without it the head collapses to one
    depth). Use alongside the final-readout CE:  loss = CE(out) + halt_w * expected_halt_loss(...).
    per_loop = forward(..., return_per_loop=True)[1];  halt_p = model._halt_p."""
    import torch.nn.functional as F
    pbar = halt_p.mean(0)                                  # (T,) batch-mean halting distribution
    ces = torch.stack([F.cross_entropy(l.reshape(-1, l.size(-1)), targets.reshape(-1),
                                       ignore_index=ignore_index) for l in per_loop])
    ent = -(pbar * torch.log(pbar + 1e-9)).sum()
    return (pbar * ces).sum() - ent_w * ent


def count_params(m):
    return sum(p.numel() for p in m.parameters())
