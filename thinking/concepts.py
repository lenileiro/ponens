"""Generic schema-driven concept heads.

This module is intentionally task-agnostic.  A caller supplies fact keys and value
sets from data, and the head learns where each key should read from a source
sequence plus a value-embedding space for that key.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _off_diagonal(x):
    n, m = x.shape
    if n != m:
        raise ValueError("off-diagonal helper expects a square matrix")
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class SchemaConceptMixer(nn.Module):
    """Self-attention workspace over data-defined schema concept states."""

    def __init__(self, n_concepts, d, heads=4, layers=1, gate_init=-2.0):
        super().__init__()
        if n_concepts <= 0:
            raise ValueError("concept mixer requires at least one concept")
        if layers <= 0:
            raise ValueError("concept mixer layers must be positive")
        if d % heads != 0:
            raise ValueError("concept mixer dimension must be divisible by heads")
        self.n_concepts = int(n_concepts)
        self.layers = int(layers)
        self.key_pos = nn.Parameter(torch.randn(self.n_concepts, d) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=4 * d, dropout=0.0,
            activation="gelu", batch_first=True)
        self.enc = nn.TransformerEncoder(
            layer, num_layers=self.layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, states):
        if states.shape[1] == 0:
            return states
        pos = self.key_pos[:states.shape[1]].unsqueeze(0)
        h = states + pos
        mixed = self.ln(self.enc(h))
        gate = torch.sigmoid(self.gate_logit)
        return states + gate * (mixed - h)


class LatentConceptHead(nn.Module):
    """Schema-free concept slots read from a source sequence.

    These slots are not tied to hand-authored keys or values. They are trainable
    queries that must discover reusable factors from whatever source states the
    caller gives them.
    """

    def __init__(self, slots, d, heads=4, mixer_layers=1, init_scale=0.02):
        super().__init__()
        self.slots = int(slots)
        self.d = int(d)
        self.heads = int(heads)
        self.mixer_layers = int(mixer_layers)
        if self.slots <= 0:
            raise ValueError("latent concept slots must be positive")
        if self.heads <= 0:
            raise ValueError("latent concept heads must be positive")
        if self.mixer_layers < 0:
            raise ValueError("latent concept mixer layers must be non-negative")
        if d % self.heads != 0:
            raise ValueError("latent concept dimension must be divisible by heads")
        self.queries = nn.Parameter(torch.randn(self.slots, d) * init_scale)
        self.query_ln = nn.LayerNorm(d)
        self.source_ln = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, self.heads, dropout=0.0, batch_first=True)
        self.mixer = (SchemaConceptMixer(
            self.slots, d, heads=self.heads, layers=self.mixer_layers)
            if self.mixer_layers > 0 else None)
        self.out_ln = nn.LayerNorm(d)
        self.projector = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, d, bias=False),
        )

    def forward(self, source, mask=None, project=False):
        q = self.queries.unsqueeze(0).expand(source.shape[0], -1, -1)
        key_padding_mask = None
        all_masked = None
        if mask is not None:
            key_padding_mask = mask.to(device=source.device, dtype=torch.bool)
            all_masked = key_padding_mask.all(dim=-1)
            if bool(all_masked.any()):
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_masked] = False
        slots, _weights = self.attn(
            self.query_ln(q), self.source_ln(source), self.source_ln(source),
            key_padding_mask=key_padding_mask, need_weights=False)
        if all_masked is not None and bool(all_masked.any()):
            slots = slots.masked_fill(all_masked[:, None, None], 0.0)
        if self.mixer is not None:
            slots = self.mixer(slots)
        slots = self.out_ln(slots)
        if all_masked is not None and bool(all_masked.any()):
            slots = slots.masked_fill(all_masked[:, None, None], 0.0)
        if project:
            slots = self.projector(slots)
            if all_masked is not None and bool(all_masked.any()):
                slots = slots.masked_fill(all_masked[:, None, None], 0.0)
            return slots
        return slots


class LatentConceptMemory(nn.Module):
    """Checkpointed schema-free prototype memory for latent concept slots."""

    def __init__(self, size, d):
        super().__init__()
        self.size = int(size)
        self.d = int(d)
        if self.size <= 0:
            raise ValueError("latent concept memory size must be positive")
        if self.d <= 0:
            raise ValueError("latent concept memory dimension must be positive")
        self.register_buffer("memory", torch.zeros(self.size, self.d))
        self.register_buffer("filled", torch.zeros((), dtype=torch.long))
        self.register_buffer("updates", torch.zeros((), dtype=torch.long))
        self.register_buffer("relations", torch.zeros(self.size, self.size))
        self.register_buffer("relation_updates", torch.zeros((), dtype=torch.long))
        self.register_buffer("transitions", torch.zeros(self.size, self.size))
        self.register_buffer("transition_updates", torch.zeros((), dtype=torch.long))

    def active(self):
        n = int(self.filled.item())
        return self.memory[:n]

    def active_relations(self):
        n = int(self.filled.item())
        return self.relations[:n, :n]

    def active_transitions(self):
        n = int(self.filled.item())
        return self.transitions[:n, :n]

    def active_prediction_relations(self):
        n = int(self.filled.item())
        if n <= 0:
            return self.relations[:0, :0]
        transitions = self.transitions[:n, :n]
        if int(self.transition_updates.item()) > 0 and bool(transitions.gt(0.0).any()):
            return transitions
        return self.relations[:n, :n]

    def forward(self, slots, temperature=0.1, balance_w=0.0):
        return latent_concept_memory_loss(
            slots, self.active(), temperature=temperature, balance_w=balance_w)

    def association_loss(self, slots, temperature=0.1, target_power=1.0,
                         self_loop_w=0.05, transitive_steps=1,
                         transitive_w=0.0):
        return latent_concept_association_loss(
            slots, self.active(), self.active_relations(),
            temperature=temperature, target_power=target_power,
            self_loop_w=self_loop_w, transitive_steps=transitive_steps,
            transitive_w=transitive_w)

    def composition_loss(self, slots, temperature=0.1, self_loop_w=0.0,
                         transitive_steps=2, transitive_w=0.1, margin=0.0):
        return latent_concept_composition_loss(
            slots, self.active(), self.active_relations(),
            temperature=temperature, self_loop_w=self_loop_w,
            transitive_steps=transitive_steps, transitive_w=transitive_w,
            margin=margin)

    def graph_prediction_loss(self, source_slots, target_slots, temperature=0.1,
                              self_loop_w=0.05, transitive_steps=2,
                              transitive_w=0.1, target_power=1.0):
        return latent_concept_graph_prediction_loss(
            source_slots, target_slots, self.active(), self.active_prediction_relations(),
            temperature=temperature, self_loop_w=self_loop_w,
            transitive_steps=transitive_steps, transitive_w=transitive_w,
            target_power=target_power)

    def graph_prediction_scores(self, source_slots, target_slots, temperature=0.1,
                                self_loop_w=0.05, transitive_steps=2,
                                transitive_w=0.1, target_power=1.0):
        return latent_concept_graph_prediction_scores(
            source_slots, target_slots, self.active(), self.active_prediction_relations(),
            temperature=temperature, self_loop_w=self_loop_w,
            transitive_steps=transitive_steps, transitive_w=transitive_w,
            target_power=target_power)

    def graph_cycle_loss(self, source_slots, target_slots, temperature=0.1,
                         self_loop_w=0.05, transitive_steps=2,
                         transitive_w=0.1, target_power=1.0,
                         cycle_w=0.5):
        return latent_concept_graph_cycle_loss(
            source_slots, target_slots, self.active(), self.active_prediction_relations(),
            temperature=temperature, self_loop_w=self_loop_w,
            transitive_steps=transitive_steps, transitive_w=transitive_w,
            target_power=target_power, cycle_w=cycle_w)

    @torch.no_grad()
    def update(self, slots, momentum=0.95, relation_decay=None):
        if slots is None:
            return 0
        if slots.shape[-1] != self.d:
            raise ValueError("latent concept memory update dimension mismatch")
        rows = slots.detach().reshape(-1, self.d)
        if rows.numel() == 0:
            return 0
        rows = F.normalize(rows, dim=-1)
        rows = rows[torch.isfinite(rows).all(-1)]
        if rows.numel() == 0:
            return 0
        filled = int(self.filled.item())
        added = 0
        if filled < self.size:
            n_new = min(self.size - filled, rows.shape[0])
            self.memory[filled:filled + n_new].copy_(rows[:n_new].to(self.memory))
            filled += n_new
            added += n_new
            self.filled.fill_(filled)
            rows = rows[n_new:]
        if rows.shape[0] and filled:
            active = F.normalize(self.memory[:filled], dim=-1)
            nearest = rows.to(active).matmul(active.t()).argmax(-1)
            mom = float(momentum)
            if mom < 0.0 or mom >= 1.0:
                raise ValueError("latent concept memory momentum must be in [0, 1)")
            for idx in nearest.unique():
                selected = rows[nearest.eq(idx)].to(self.memory)
                target = selected.mean(0)
                updated = mom * self.memory[int(idx)] + (1.0 - mom) * target
                self.memory[int(idx)].copy_(F.normalize(updated, dim=0))
            added += int(rows.shape[0])
        if added:
            self.updates.add_(1)
        if relation_decay is not None:
            self.update_relations(slots, decay=relation_decay)
        return int(added)

    @torch.no_grad()
    def update_relations(self, slots, decay=0.99):
        if slots is None:
            return 0
        if slots.ndim != 3:
            raise ValueError("latent concept relation update expects [batch, slots, dim]")
        if slots.shape[-1] != self.d:
            raise ValueError("latent concept relation update dimension mismatch")
        filled = int(self.filled.item())
        if filled <= 0:
            return 0
        dec = float(decay)
        if dec < 0.0 or dec >= 1.0:
            raise ValueError("latent concept relation decay must be in [0, 1)")
        rows = F.normalize(slots.detach(), dim=-1)
        valid = torch.isfinite(rows).all(-1)
        if not bool(valid.any()):
            return 0
        active = F.normalize(self.memory[:filled], dim=-1)
        nearest = rows.to(active).matmul(active.t()).argmax(-1)
        one_hot = F.one_hot(nearest, num_classes=filled).to(dtype=active.dtype)
        one_hot = one_hot * valid.to(dtype=active.dtype).unsqueeze(-1)
        present = one_hot.sum(1).gt(0).to(dtype=active.dtype)
        if not bool(present.any()):
            return 0
        rel = present[:, :, None] * present[:, None, :]
        eye = torch.eye(filled, dtype=torch.bool, device=rel.device)
        rel = rel.masked_fill(eye[None], 0.0)
        batch_rel = rel.mean(0)
        if float(batch_rel.sum()) <= 0.0:
            return 0
        target = self.relations[:filled, :filled]
        target.mul_(dec).add_(batch_rel.to(target), alpha=1.0 - dec)
        self.relation_updates.add_(1)
        return int(present.shape[0])

    @torch.no_grad()
    def update_transitions(self, source_slots, target_slots, decay=0.99):
        if source_slots is None or target_slots is None:
            return 0
        if source_slots.ndim != 3 or target_slots.ndim != 3:
            raise ValueError("latent concept transition update expects [batch, slots, dim]")
        if source_slots.shape[0] != target_slots.shape[0]:
            raise ValueError("latent concept transition update batch mismatch")
        if source_slots.shape[-1] != self.d or target_slots.shape[-1] != self.d:
            raise ValueError("latent concept transition update dimension mismatch")
        filled = int(self.filled.item())
        if filled <= 0:
            return 0
        dec = float(decay)
        if dec < 0.0 or dec >= 1.0:
            raise ValueError("latent concept transition decay must be in [0, 1)")
        source = F.normalize(source_slots.detach(), dim=-1)
        target = F.normalize(target_slots.detach(), dim=-1)
        source_valid = torch.isfinite(source).all(-1)
        target_valid = torch.isfinite(target).all(-1)
        if not bool(source_valid.any()) or not bool(target_valid.any()):
            return 0
        active = F.normalize(self.memory[:filled], dim=-1)
        source_nearest = source.to(active).matmul(active.t()).argmax(-1)
        target_nearest = target.to(active).matmul(active.t()).argmax(-1)
        source_hot = F.one_hot(source_nearest, num_classes=filled).to(dtype=active.dtype)
        target_hot = F.one_hot(target_nearest, num_classes=filled).to(dtype=active.dtype)
        source_hot = source_hot * source_valid.to(dtype=active.dtype).unsqueeze(-1)
        target_hot = target_hot * target_valid.to(dtype=active.dtype).unsqueeze(-1)
        source_present = source_hot.sum(1).gt(0).to(dtype=active.dtype)
        target_present = target_hot.sum(1).gt(0).to(dtype=active.dtype)
        usable = source_present.sum(-1).gt(0) & target_present.sum(-1).gt(0)
        if not bool(usable.any()):
            return 0
        edge = source_present[usable, :, None] * target_present[usable, None, :]
        batch_transitions = edge.mean(0)
        if float(batch_transitions.sum()) <= 0.0:
            return 0
        target_buffer = self.transitions[:filled, :filled]
        target_buffer.mul_(dec).add_(batch_transitions.to(target_buffer), alpha=1.0 - dec)
        self.transition_updates.add_(1)
        return int(usable.sum().item())


def latent_concept_memory_loss(slots, memory, temperature=0.1, balance_w=0.0):
    """Align latent slots to a persistent self-mined prototype memory.

    The memory rows are discovered from previous batches and stored in the
    checkpoint. The loss uses nearest current prototypes as detached targets,
    so it does not need labels, schema fields, or language-specific rules.
    """
    if slots is None:
        return torch.tensor(0.0)
    if memory is None or memory.numel() == 0:
        return slots.sum() * 0.0
    if slots.shape[-1] != memory.shape[-1]:
        raise ValueError("latent concept memory dimension mismatch")
    rows = F.normalize(slots.reshape(-1, slots.shape[-1]), dim=-1)
    mem = F.normalize(memory.to(device=rows.device, dtype=rows.dtype), dim=-1)
    if mem.shape[0] <= 0:
        return slots.sum() * 0.0
    temp = max(float(temperature), 1e-6)
    logits = rows.matmul(mem.t()) / temp
    targets = logits.detach().argmax(-1)
    nearest_sim = rows.matmul(mem.t()).detach().max(-1).values
    losses = [F.cross_entropy(logits, targets), (1.0 - nearest_sim).mean()]
    if float(balance_w) and mem.shape[0] > 1:
        probs = logits.softmax(-1).mean(0)
        uniform = torch.full_like(probs, 1.0 / probs.numel())
        losses.append(float(balance_w) * F.kl_div(
            probs.clamp_min(1e-8).log(), uniform, reduction="batchmean"))
    return torch.stack(losses).mean()


def latent_concept_association_loss(slots, memory, relations, temperature=0.1,
                                    target_power=1.0, self_loop_w=0.05,
                                    transitive_steps=1, transitive_w=0.0):
    """Train current slots against a self-mined concept association graph.

    Memory rows are persistent latent concepts; relation rows are discovered
    co-activations between those concepts across previous examples. The target
    distribution is produced from the graph itself, so the objective can connect
    concepts without hand-authored labels, schemas, or language-specific rules.
    """
    if slots is None:
        return torch.tensor(0.0)
    if memory is None or memory.numel() == 0:
        return slots.sum() * 0.0
    if relations is None or relations.numel() == 0:
        return slots.sum() * 0.0
    if slots.ndim != 3:
        raise ValueError("latent concept association expects [batch, slots, dim]")
    if slots.shape[-1] != memory.shape[-1]:
        raise ValueError("latent concept association dimension mismatch")
    n = int(memory.shape[0])
    rel = relations[:n, :n].to(device=slots.device, dtype=slots.dtype)
    if rel.shape != (n, n):
        raise ValueError("latent concept association relation shape mismatch")
    rel = rel.clamp_min(0.0)
    if float(self_loop_w):
        rel = rel + float(self_loop_w) * torch.eye(
            n, dtype=rel.dtype, device=rel.device)
    row_sum = rel.sum(-1, keepdim=True)
    active_rows = row_sum.squeeze(-1).gt(0.0)
    if not bool(active_rows.any()):
        return slots.sum() * 0.0
    rel = rel / row_sum.clamp_min(1e-8)
    steps = int(transitive_steps)
    if steps < 1:
        raise ValueError("latent concept association transitive steps must be positive")
    trans_w = float(transitive_w)
    if trans_w < 0.0:
        raise ValueError("latent concept association transitive weight must be non-negative")
    if steps > 1 and trans_w:
        walk = rel
        inferred = torch.zeros_like(rel)
        for _hop in range(2, steps + 1):
            walk = walk.matmul(rel)
            inferred = inferred + walk
        inferred = inferred / max(1, steps - 1)
        rel = rel + trans_w * inferred
        rel = rel / rel.sum(-1, keepdim=True).clamp_min(1e-8)
    rows = F.normalize(slots, dim=-1)
    mem = F.normalize(memory.to(device=slots.device, dtype=slots.dtype), dim=-1)
    temp = max(float(temperature), 1e-6)
    logits = rows.matmul(mem.t()) / temp
    pred = logits.softmax(-1).mean(1)
    nearest = logits.detach().argmax(-1)
    target = rel[nearest].mean(1).detach()
    power = float(target_power)
    if power <= 0.0:
        raise ValueError("latent concept association target power must be positive")
    if power != 1.0:
        target = target.clamp_min(1e-8).pow(power)
    target = target / target.sum(-1, keepdim=True).clamp_min(1e-8)
    return F.kl_div(pred.clamp_min(1e-8).log(), target, reduction="batchmean")


def _latent_relation_targets(relations, n, device, dtype, self_loop_w=0.05,
                             transitive_steps=1, transitive_w=0.0):
    rel = relations[:n, :n].to(device=device, dtype=dtype).clamp_min(0.0)
    if rel.shape != (n, n):
        raise ValueError("latent concept relation shape mismatch")
    if float(self_loop_w):
        rel = rel + float(self_loop_w) * torch.eye(n, dtype=dtype, device=device)
    row_sum = rel.sum(-1, keepdim=True)
    if not bool(row_sum.squeeze(-1).gt(0.0).any()):
        return None
    rel = rel / row_sum.clamp_min(1e-8)
    steps = int(transitive_steps)
    if steps < 1:
        raise ValueError("latent concept transitive steps must be positive")
    trans_w = float(transitive_w)
    if trans_w < 0.0:
        raise ValueError("latent concept transitive weight must be non-negative")
    if steps > 1 and trans_w:
        walk = rel
        inferred = torch.zeros_like(rel)
        for _hop in range(2, steps + 1):
            walk = walk.matmul(rel)
            inferred = inferred + walk
        inferred = inferred / max(1, steps - 1)
        rel = rel + trans_w * inferred
        rel = rel / rel.sum(-1, keepdim=True).clamp_min(1e-8)
    return rel


def latent_concept_graph_curiosity_scores(slots, memory, relations=None,
                                          temperature=0.1, self_loop_w=0.05,
                                          transitive_steps=1, transitive_w=0.0,
                                          novelty_w=1.0, association_w=1.0):
    """Score examples for self-study from latent novelty and graph mismatch.

    Higher scores mean the current example is either far from the persistent
    concept prototypes or poorly explained by the self-mined association graph.
    The score is label-free and can be used as a curriculum signal.
    """
    if slots is None:
        return torch.zeros(0), {}
    if memory is None or memory.numel() == 0:
        return slots.new_zeros((slots.shape[0],)), {
            "novelty": slots.new_zeros((slots.shape[0],)),
            "association": slots.new_zeros((slots.shape[0],)),
        }
    if slots.ndim != 3:
        raise ValueError("latent concept curiosity expects [batch, slots, dim]")
    if slots.shape[-1] != memory.shape[-1]:
        raise ValueError("latent concept curiosity dimension mismatch")
    rows = F.normalize(slots, dim=-1)
    mem = F.normalize(memory.to(device=slots.device, dtype=slots.dtype), dim=-1)
    sim = rows.matmul(mem.t())
    nearest_sim, nearest = sim.max(-1)
    novelty = (1.0 - nearest_sim).clamp_min(0.0).mean(-1)
    association = torch.zeros_like(novelty)
    if relations is not None and relations.numel() and float(association_w):
        rel = _latent_relation_targets(
            relations, mem.shape[0], slots.device, slots.dtype,
            self_loop_w=self_loop_w, transitive_steps=transitive_steps,
            transitive_w=transitive_w)
        if rel is not None:
            temp = max(float(temperature), 1e-6)
            pred = (sim / temp).softmax(-1).mean(1)
            target = rel[nearest].mean(1).detach()
            target = target / target.sum(-1, keepdim=True).clamp_min(1e-8)
            association = F.kl_div(
                pred.clamp_min(1e-8).log(), target,
                reduction="none").sum(-1)
    score = float(novelty_w) * novelty + float(association_w) * association
    return score, {"novelty": novelty, "association": association}


def latent_concept_composition_loss(slots, memory, relations, temperature=0.1,
                                    self_loop_w=0.0, transitive_steps=2,
                                    transitive_w=0.1, margin=0.0):
    """Use the self-mined graph as a reusable concept transformation space.

    For each current latent slot, find its nearest persistent concept A. The
    graph predicts a related concept distribution B. The loss teaches the slot
    that applying the learned A->B relation delta should land near B. This is
    label-free: only the model's own prototypes and co-activation graph define
    the relation.
    """
    if slots is None:
        return torch.tensor(0.0)
    if memory is None or memory.numel() == 0:
        return slots.sum() * 0.0
    if relations is None or relations.numel() == 0:
        return slots.sum() * 0.0
    if slots.ndim != 3:
        raise ValueError("latent concept composition expects [batch, slots, dim]")
    if slots.shape[-1] != memory.shape[-1]:
        raise ValueError("latent concept composition dimension mismatch")
    mem = F.normalize(memory.to(device=slots.device, dtype=slots.dtype), dim=-1)
    rel = _latent_relation_targets(
        relations, mem.shape[0], slots.device, slots.dtype,
        self_loop_w=self_loop_w, transitive_steps=transitive_steps,
        transitive_w=transitive_w)
    if rel is None:
        return slots.sum() * 0.0
    rows = F.normalize(slots.reshape(-1, slots.shape[-1]), dim=-1)
    nearest = rows.matmul(mem.t()).detach().argmax(-1)
    target = rel[nearest].detach()
    active = target.sum(-1).gt(0.0)
    if not bool(active.any()):
        return slots.sum() * 0.0
    rows = rows[active]
    nearest = nearest[active]
    target = target[active]
    target = target / target.sum(-1, keepdim=True).clamp_min(1e-8)
    source = mem[nearest].detach()
    target_center = F.normalize(target.matmul(mem).detach(), dim=-1)
    composed = F.normalize(rows + (target_center - source), dim=-1)
    temp = max(float(temperature), 1e-6)
    logits = composed.matmul(mem.t()) / temp
    losses = [
        F.kl_div(F.log_softmax(logits, dim=-1), target, reduction="batchmean"),
        (1.0 - (composed * target_center).sum(-1)).mean(),
    ]
    margin_t = float(margin)
    if margin_t and mem.shape[0] > 1:
        target_sim = (composed * target_center).sum(-1)
        negative_mask = target.le(0.0)
        if bool(negative_mask.any(-1).all()):
            negative_sim = composed.matmul(mem.t()).masked_fill(
                ~negative_mask, -float("inf")).max(-1).values
            finite = torch.isfinite(negative_sim)
            if bool(finite.any()):
                losses.append(
                    F.relu(negative_sim[finite] + margin_t - target_sim[finite]).mean())
    return torch.stack(losses).mean()


def _latent_graph_zero_scores(source_slots, target_slots):
    slots = source_slots if source_slots is not None else target_slots
    if slots is None:
        return torch.zeros(0)
    if slots.ndim == 0:
        return slots.reshape(1) * 0.0
    return slots.reshape(slots.shape[0], -1).sum(-1) * 0.0


def latent_concept_graph_prediction_scores(source_slots, target_slots, memory,
                                           relations, temperature=0.1,
                                           self_loop_w=0.05,
                                           transitive_steps=2,
                                           transitive_w=0.1,
                                           target_power=1.0):
    """Infer held-out/full latent concepts through the self-mined graph.

    Source slots form a distribution over persistent latent concepts. The
    relation graph propagates that distribution into a predicted concept closure,
    which is trained to match target slots from held-out text or fuller modality
    evidence. This is a schema-free predictive-coding objective over concepts,
    not a token prediction or task-label objective.
    """
    if source_slots is None or target_slots is None:
        zero = _latent_graph_zero_scores(source_slots, target_slots)
        return zero, {"kl": zero, "cosine": zero}
    if memory is None or memory.numel() == 0:
        zero = _latent_graph_zero_scores(source_slots, target_slots)
        return zero, {"kl": zero, "cosine": zero}
    if relations is None or relations.numel() == 0:
        zero = _latent_graph_zero_scores(source_slots, target_slots)
        return zero, {"kl": zero, "cosine": zero}
    if source_slots.ndim != 3 or target_slots.ndim != 3:
        raise ValueError("latent graph prediction expects [batch, slots, dim]")
    if source_slots.shape[0] != target_slots.shape[0]:
        raise ValueError("latent graph prediction batch mismatch")
    if source_slots.shape[-1] != target_slots.shape[-1]:
        raise ValueError("latent graph prediction source/target dimension mismatch")
    if source_slots.shape[-1] != memory.shape[-1]:
        raise ValueError("latent graph prediction memory dimension mismatch")
    mem = F.normalize(
        memory.to(device=source_slots.device, dtype=source_slots.dtype), dim=-1)
    rel = _latent_relation_targets(
        relations, mem.shape[0], source_slots.device, source_slots.dtype,
        self_loop_w=self_loop_w, transitive_steps=transitive_steps,
        transitive_w=transitive_w)
    if rel is None:
        zero = _latent_graph_zero_scores(source_slots, target_slots)
        return zero, {"kl": zero, "cosine": zero}
    temp = max(float(temperature), 1e-6)
    source = F.normalize(source_slots, dim=-1)
    target = F.normalize(target_slots.to(source_slots), dim=-1)
    source_dist = (source.matmul(mem.t()) / temp).softmax(-1).mean(1)
    pred = source_dist.matmul(rel)
    pred = pred / pred.sum(-1, keepdim=True).clamp_min(1e-8)
    with torch.no_grad():
        target_dist = (target.matmul(mem.t()) / temp).softmax(-1).mean(1)
        power = float(target_power)
        if power <= 0.0:
            raise ValueError("latent graph prediction target power must be positive")
        if power != 1.0:
            target_dist = target_dist.clamp_min(1e-8).pow(power)
        target_dist = target_dist / target_dist.sum(-1, keepdim=True).clamp_min(1e-8)
        target_center = F.normalize(target_dist.matmul(mem), dim=-1)
    pred_center = F.normalize(pred.matmul(mem), dim=-1)
    kl = F.kl_div(
        pred.clamp_min(1e-8).log(), target_dist, reduction="none").sum(-1)
    cosine = 1.0 - (pred_center * target_center).sum(-1)
    score = 0.5 * (kl + cosine)
    return score, {"kl": kl, "cosine": cosine}


def latent_concept_graph_prediction_loss(source_slots, target_slots, memory,
                                         relations, temperature=0.1,
                                         self_loop_w=0.05,
                                         transitive_steps=2,
                                         transitive_w=0.1,
                                         target_power=1.0):
    scores, _parts = latent_concept_graph_prediction_scores(
        source_slots, target_slots, memory, relations,
        temperature=temperature, self_loop_w=self_loop_w,
        transitive_steps=transitive_steps, transitive_w=transitive_w,
        target_power=target_power)
    if scores.numel() == 0:
        return torch.tensor(0.0)
    return scores.mean()


def latent_concept_graph_cycle_loss(source_slots, target_slots, memory, relations,
                                    temperature=0.1, self_loop_w=0.05,
                                    transitive_steps=2, transitive_w=0.1,
                                    target_power=1.0, cycle_w=0.5):
    """Make learned concept transitions reversible enough to support discovery.

    Source slots and target slots are two views of the same example: a context and
    its held-out continuation, or a partial modality and the fuller multimodal
    observation.  The learned graph should predict target concepts from source
    concepts and also reconstruct each side after a forward-then-reverse walk.
    The objective is entirely defined by the model's own prototype memory and
    graph; it does not need labels, answer choices, schemas, or hand-authored
    language rules.
    """
    if source_slots is None or target_slots is None:
        return _latent_graph_zero_scores(source_slots, target_slots).sum()
    if memory is None or memory.numel() == 0:
        return source_slots.sum() * 0.0
    if relations is None or relations.numel() == 0:
        return source_slots.sum() * 0.0
    if source_slots.ndim != 3 or target_slots.ndim != 3:
        raise ValueError("latent graph cycle expects [batch, slots, dim]")
    if source_slots.shape[0] != target_slots.shape[0]:
        raise ValueError("latent graph cycle batch mismatch")
    if source_slots.shape[-1] != target_slots.shape[-1]:
        raise ValueError("latent graph cycle source/target dimension mismatch")
    if source_slots.shape[-1] != memory.shape[-1]:
        raise ValueError("latent graph cycle memory dimension mismatch")
    mem = F.normalize(
        memory.to(device=source_slots.device, dtype=source_slots.dtype), dim=-1)
    rel = _latent_relation_targets(
        relations, mem.shape[0], source_slots.device, source_slots.dtype,
        self_loop_w=self_loop_w, transitive_steps=transitive_steps,
        transitive_w=transitive_w)
    rev = _latent_relation_targets(
        relations.t(), mem.shape[0], source_slots.device, source_slots.dtype,
        self_loop_w=self_loop_w, transitive_steps=transitive_steps,
        transitive_w=transitive_w)
    if rel is None or rev is None:
        return source_slots.sum() * 0.0
    temp = max(float(temperature), 1e-6)
    power = float(target_power)
    if power <= 0.0:
        raise ValueError("latent graph cycle target power must be positive")
    cyc_w = float(cycle_w)
    if cyc_w < 0.0:
        raise ValueError("latent graph cycle weight must be non-negative")

    source = F.normalize(source_slots, dim=-1)
    target = F.normalize(target_slots.to(source_slots), dim=-1)
    source_dist = (source.matmul(mem.t()) / temp).softmax(-1).mean(1)
    target_dist = (target.matmul(mem.t()) / temp).softmax(-1).mean(1)
    source_target = source_dist.detach()
    target_target = target_dist.detach()
    if power != 1.0:
        source_target = source_target.clamp_min(1e-8).pow(power)
        target_target = target_target.clamp_min(1e-8).pow(power)
    source_target = source_target / source_target.sum(-1, keepdim=True).clamp_min(1e-8)
    target_target = target_target / target_target.sum(-1, keepdim=True).clamp_min(1e-8)

    forward = source_dist.matmul(rel)
    forward = forward / forward.sum(-1, keepdim=True).clamp_min(1e-8)
    reverse = target_dist.matmul(rev)
    reverse = reverse / reverse.sum(-1, keepdim=True).clamp_min(1e-8)
    losses = [
        F.kl_div(forward.clamp_min(1e-8).log(), target_target, reduction="batchmean"),
        F.kl_div(reverse.clamp_min(1e-8).log(), source_target, reduction="batchmean"),
    ]
    if cyc_w:
        source_cycle = forward.matmul(rev)
        source_cycle = source_cycle / source_cycle.sum(-1, keepdim=True).clamp_min(1e-8)
        target_cycle = reverse.matmul(rel)
        target_cycle = target_cycle / target_cycle.sum(-1, keepdim=True).clamp_min(1e-8)
        losses.extend([
            cyc_w * F.kl_div(
                source_cycle.clamp_min(1e-8).log(), source_target,
                reduction="batchmean"),
            cyc_w * F.kl_div(
                target_cycle.clamp_min(1e-8).log(), target_target,
                reduction="batchmean"),
        ])
    return torch.stack(losses).mean()


class SchemaConceptHead(nn.Module):
    """Attend over source states and score values for each schema key."""

    def __init__(self, keys, values, d, init_scale=0.02, mixer_layers=0,
                 mixer_heads=4, mixer_gate_init=-2.0):
        super().__init__()
        self.keys = tuple(tuple(k) if isinstance(k, (list, tuple)) else (str(k),)
                          for k in keys)
        self.values = tuple(tuple(vs) for vs in values)
        if len(self.keys) != len(self.values):
            raise ValueError("concept keys and values must have the same length")
        self.key_query = nn.Parameter(torch.randn(len(self.keys), d) * init_scale)
        self.value_embeds = nn.ParameterList([
            nn.Parameter(torch.randn(len(vs), d) * init_scale) for vs in self.values
        ])
        self.value_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(len(vs))) for vs in self.values
        ])
        self.state_projectors = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d, bias=False))
            for _ in self.keys
        ])
        self.geometry_prototypes = nn.ParameterList([
            nn.Parameter(torch.randn(len(vs), d) * init_scale) for vs in self.values
        ])
        self.mixer_layers = int(mixer_layers)
        self.mixer_gate_init = float(mixer_gate_init)
        if self.mixer_layers < 0:
            raise ValueError("concept mixer layers must be non-negative")
        self.mixer = (SchemaConceptMixer(
            len(self.keys), d, heads=mixer_heads, layers=self.mixer_layers,
            gate_init=self.mixer_gate_init)
            if self.mixer_layers > 0 and len(self.keys) > 0 else None)

    def enable_mixer(self, heads=4, layers=1, gate_init=-2.0):
        if int(layers) <= 0:
            raise ValueError("concept mixer layers must be positive")
        if not self.keys:
            self.mixer = None
            self.mixer_layers = 0
            return self
        if self.mixer is None or int(layers) != self.mixer.layers:
            self.mixer = SchemaConceptMixer(
                len(self.keys), self.key_query.shape[-1], heads=heads,
                layers=int(layers), gate_init=gate_init)
        self.mixer_layers = int(layers)
        self.mixer_gate_init = float(gate_init)
        return self

    def state_tensor(self, source, mask=None):
        if not self.keys:
            return source.new_zeros((source.shape[0], 0, source.shape[-1]))
        scale = source.shape[-1] ** -0.5
        scores = torch.einsum("btd,kd->bkt", source, self.key_query) * scale
        if mask is not None:
            mask = mask.to(device=source.device, dtype=torch.bool)
            scores = scores.masked_fill(mask[:, None, :], torch.finfo(scores.dtype).min)
        weights = scores.softmax(-1)
        if mask is not None:
            weights = weights.masked_fill(mask[:, None, :], 0.0)
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        states = torch.einsum("bkt,btd->bkd", weights, source)
        if self.mixer is not None:
            states = self.mixer(states)
        return states

    def states(self, source, mask=None):
        state_tensor = self.state_tensor(source, mask=mask)
        return {key: state_tensor[:, i] for i, key in enumerate(self.keys)}

    def geometry_state_tensor_from_states(self, state_tensor):
        if not self.keys:
            return state_tensor
        return torch.stack([
            projector(state_tensor[:, i]) for i, projector in enumerate(self.state_projectors)
        ], dim=1)

    def geometry_state_tensor(self, source, mask=None):
        return self.geometry_state_tensor_from_states(self.state_tensor(source, mask=mask))

    def geometry_states(self, source, mask=None):
        state_tensor = self.geometry_state_tensor(source, mask=mask)
        return {key: state_tensor[:, i] for i, key in enumerate(self.keys)}

    def geometry_logits_from_states(self, states, temperature=0.1):
        out = {}
        temp = max(float(temperature), 1e-6)
        for i, key in enumerate(self.keys):
            state = F.normalize(states[key], dim=-1)
            prototypes = F.normalize(self.geometry_prototypes[i], dim=-1)
            out[key] = state.matmul(prototypes.t()) / temp
        return out

    def geometry_logits(self, source, mask=None, temperature=0.1):
        return self.geometry_logits_from_states(
            self.geometry_states(source, mask=mask), temperature=temperature)

    def logits_from_state_tensor(self, state_tensor):
        out = {}
        scale = state_tensor.shape[-1] ** -0.5
        for i, key in enumerate(self.keys):
            out[key] = (state_tensor[:, i].matmul(self.value_embeds[i].t()) * scale
                        + self.value_biases[i])
        return out

    def logits_from_states(self, states):
        out = {}
        scale = self.key_query.shape[-1] ** -0.5
        for i, key in enumerate(self.keys):
            state = states[key]
            out[key] = state.matmul(self.value_embeds[i].t()) * scale + self.value_biases[i]
        return out

    def forward(self, source, mask=None):
        return self.logits_from_state_tensor(self.state_tensor(source, mask=mask))


class SchemaConceptRefiner(nn.Module):
    """Feed learned schema concept state back into source token states.

    The module is schema-agnostic: callers provide concept states from data-defined
    schema heads, and the refiner learns how much those states should reorganize
    the upstream representation before downstream heads read from it.
    """

    def __init__(self, d, heads=4, hidden_mult=4, gate_init=-2.0):
        super().__init__()
        if d % heads != 0:
            raise ValueError("refiner dimension must be divisible by heads")
        hidden = int(hidden_mult) * d
        self.source_ln = nn.LayerNorm(d)
        self.concept_ln = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=0.0, batch_first=True)
        self.attn_out = nn.Linear(d, d, bias=False)
        self.ff = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Linear(hidden, d, bias=False),
        )
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, source, concept_states, mask=None):
        if concept_states is None or concept_states.shape[1] == 0:
            return source.masked_fill(mask.unsqueeze(-1), 0.0) if mask is not None else source
        q = self.source_ln(source)
        kv = self.concept_ln(concept_states)
        attn, _weights = self.attn(q, kv, kv, need_weights=False)
        gate = torch.sigmoid(self.gate_logit)
        h = source + gate * self.attn_out(attn)
        h = h + gate * self.ff(h)
        if mask is not None:
            h = h.masked_fill(mask.unsqueeze(-1), 0.0)
        return h


def _zero_from_states(states_by_key):
    for states in states_by_key.values():
        return states.sum() * 0.0
    return torch.tensor(0.0)


def latent_concept_vicreg_loss(slots_a, slots_b, invariance_weight=25.0,
                               variance_weight=25.0, covariance_weight=1.0,
                               variance_target=1.0):
    """Self-supervised latent-concept invariance plus anti-collapse loss.

    Two stochastic or cross-modal views of the same observation should keep the
    same latent concept slots, while batch variance and covariance terms prevent
    the slots from collapsing to a constant representation.
    """
    if slots_a.shape != slots_b.shape:
        raise ValueError("latent concept views must have matching shapes")
    x = slots_a.reshape(-1, slots_a.shape[-1])
    y = slots_b.reshape(-1, slots_b.shape[-1])
    inv = F.mse_loss(x, y)
    if x.shape[0] < 2:
        zero = (x.sum() + y.sum()) * 0.0
        return float(invariance_weight) * inv + zero
    var_target = float(variance_target)
    x_std = torch.sqrt(x.var(dim=0, unbiased=False) + 1e-4)
    y_std = torch.sqrt(y.var(dim=0, unbiased=False) + 1e-4)
    var = F.relu(var_target - x_std).mean() + F.relu(var_target - y_std).mean()
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    cov_x = x.t().matmul(x) / (x.shape[0] - 1)
    cov_y = y.t().matmul(y) / (y.shape[0] - 1)
    cov = (_off_diagonal(cov_x).pow(2).sum() / x.shape[-1]
           + _off_diagonal(cov_y).pow(2).sum() / y.shape[-1])
    return (float(invariance_weight) * inv
            + float(variance_weight) * var
            + float(covariance_weight) * cov)


def latent_concept_neighborhood_loss(anchor_slots, positive_slots,
                                     temperature=0.1, margin=0.0):
    """Align self-mined latent concept neighbors while using the batch as negatives.

    The caller decides how neighbors were mined. This loss only sees two batches
    of schema-free latent slots: each row in `positive_slots` is treated as the
    discovered neighbor of the corresponding row in `anchor_slots`. No labels,
    task ids, or language-specific rules are required.
    """
    if anchor_slots is None or positive_slots is None:
        if anchor_slots is not None:
            return anchor_slots.sum() * 0.0
        if positive_slots is not None:
            return positive_slots.sum() * 0.0
        return torch.tensor(0.0)
    if anchor_slots.shape != positive_slots.shape:
        raise ValueError("latent concept neighbors must have matching shapes")
    anchor = F.normalize(anchor_slots.reshape(anchor_slots.shape[0], -1), dim=-1)
    positive = F.normalize(positive_slots.reshape(positive_slots.shape[0], -1), dim=-1)
    pos_sim = (anchor * positive).sum(-1)
    losses = [(1.0 - pos_sim).mean()]
    if anchor.shape[0] > 1:
        temp = max(float(temperature), 1e-6)
        logits = anchor.matmul(positive.t()) / temp
        labels = torch.arange(logits.shape[0], device=logits.device)
        losses.append(0.5 * (
            F.cross_entropy(logits, labels)
            + F.cross_entropy(logits.t(), labels)))
        margin_t = float(margin)
        if margin_t:
            eye = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
            sim = anchor.matmul(positive.t())
            hardest_other = sim.masked_fill(eye, -float("inf")).max(-1).values
            hardest_other_t = sim.t().masked_fill(eye, -float("inf")).max(-1).values
            losses.append(0.5 * (
                F.relu(hardest_other + margin_t - pos_sim).mean()
                + F.relu(hardest_other_t + margin_t - pos_sim).mean()))
    return torch.stack(losses).mean()


def latent_concept_transition_consistency_loss(anchor_a_slots, positive_a_slots,
                                               anchor_b_slots, positive_b_slots,
                                               temperature=0.1, margin=0.0):
    """Align discovered latent transitions across two views of related records.

    The caller mines related pairs. This loss only sees the latent slots for an
    anchor and its discovered neighbor under two stochastic or modality views.
    It makes the slot-space delta reusable without requiring task labels.
    """
    if (anchor_a_slots is None or positive_a_slots is None
            or anchor_b_slots is None or positive_b_slots is None):
        for slots in (anchor_a_slots, positive_a_slots, anchor_b_slots, positive_b_slots):
            if slots is not None:
                return slots.sum() * 0.0
        return torch.tensor(0.0)
    if not (anchor_a_slots.shape == positive_a_slots.shape
            == anchor_b_slots.shape == positive_b_slots.shape):
        raise ValueError("latent transition views must have matching shapes")
    delta_a = F.normalize(
        (positive_a_slots - anchor_a_slots).reshape(anchor_a_slots.shape[0], -1),
        dim=-1)
    delta_b = F.normalize(
        (positive_b_slots - anchor_b_slots).reshape(anchor_b_slots.shape[0], -1),
        dim=-1)
    pos_sim = (delta_a * delta_b).sum(-1)
    losses = [(1.0 - pos_sim).mean()]
    if delta_a.shape[0] > 1:
        temp = max(float(temperature), 1e-6)
        logits = delta_a.matmul(delta_b.t()) / temp
        labels = torch.arange(logits.shape[0], device=logits.device)
        losses.append(0.5 * (
            F.cross_entropy(logits, labels)
            + F.cross_entropy(logits.t(), labels)))
        margin_t = float(margin)
        if margin_t:
            eye = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
            sim = delta_a.matmul(delta_b.t())
            hardest_other = sim.masked_fill(eye, -float("inf")).max(-1).values
            hardest_other_t = sim.t().masked_fill(eye, -float("inf")).max(-1).values
            losses.append(0.5 * (
                F.relu(hardest_other + margin_t - pos_sim).mean()
                + F.relu(hardest_other_t + margin_t - pos_sim).mean()))
    return torch.stack(losses).mean()


def latent_concept_cluster_prototype_loss(slots, cluster_ids, temperature=0.1,
                                          margin=0.0, min_cluster_size=2):
    """Consolidate self-mined latent clusters into reusable prototypes.

    `cluster_ids` are discovered outside this loss, usually from current latent
    nearest-neighbor structure. The objective is task-agnostic: records with the
    same discovered id move toward their current detached cluster prototype, and
    observed prototypes compete as batch negatives when more than one exists.
    """
    if slots is None:
        return torch.tensor(0.0)
    if not torch.is_tensor(cluster_ids):
        cluster_ids = torch.tensor(cluster_ids, dtype=torch.long, device=slots.device)
    else:
        cluster_ids = cluster_ids.to(device=slots.device, dtype=torch.long)
    if cluster_ids.shape[0] != slots.shape[0]:
        raise ValueError("latent cluster ids must match batch rows")
    valid = cluster_ids.ge(0)
    if int(valid.sum()) < max(1, int(min_cluster_size)):
        return slots.sum() * 0.0
    z = F.normalize(slots.reshape(slots.shape[0], -1), dim=-1)
    labels = cluster_ids[valid]
    valid_z = z[valid]
    centers = []
    center_labels = []
    for label in labels.unique(sorted=True):
        rows = labels.eq(label)
        if int(rows.sum()) >= int(min_cluster_size):
            centers.append(F.normalize(valid_z[rows].mean(0), dim=0))
            center_labels.append(int(label.item()))
    if not centers:
        return slots.sum() * 0.0
    centers = torch.stack(centers, dim=0).detach()
    label_to_center = {label: i for i, label in enumerate(center_labels)}
    keep = torch.tensor([int(label.item()) in label_to_center for label in labels],
                        dtype=torch.bool, device=slots.device)
    if not bool(keep.any()):
        return slots.sum() * 0.0
    state = valid_z[keep]
    mapped = torch.tensor([label_to_center[int(label.item())] for label in labels[keep]],
                          dtype=torch.long, device=slots.device)
    sim = state.matmul(centers.t())
    target_sim = sim.gather(1, mapped[:, None]).squeeze(1)
    losses = [(1.0 - target_sim).mean()]
    if centers.shape[0] > 1:
        temp = max(float(temperature), 1e-6)
        losses.append(F.cross_entropy(sim / temp, mapped))
        margin_t = float(margin)
        if margin_t:
            other_sim = sim.masked_fill(
                F.one_hot(mapped, num_classes=centers.shape[0]).bool(),
                -float("inf")).max(-1).values
            losses.append(F.relu(other_sim + margin_t - target_sim).mean())
    return torch.stack(losses).mean()


def latent_concept_slot_factorization_loss(slots, variance_target=0.05,
                                           separation_margin=0.2,
                                           covariance_weight=0.05):
    """Encourage schema-free latent slots to become distinct reusable factors.

    This loss is label-free: it only sees the slot tensor. Slots in the same
    observation are pushed away from duplicate geometry, each slot is kept active
    across the batch, and dimensions inside each slot are lightly decorrelated.
    """
    if slots is None:
        return torch.tensor(0.0)
    if slots.ndim != 3:
        raise ValueError("latent concept factorization expects [batch, slots, dim]")
    if slots.shape[0] == 0 or slots.shape[1] == 0:
        return slots.sum() * 0.0
    losses = []
    if slots.shape[1] > 1:
        z = F.normalize(slots, dim=-1)
        sim = z.matmul(z.transpose(1, 2))
        eye = torch.eye(slots.shape[1], dtype=torch.bool, device=slots.device)
        off = sim.masked_select(~eye[None])
        losses.append(F.relu(off.abs() - float(separation_margin)).pow(2).mean())
    if slots.shape[0] > 1:
        centered = slots - slots.mean(dim=0, keepdim=True)
        std = torch.sqrt(centered.var(dim=0, unbiased=False) + 1e-4)
        losses.append(F.relu(float(variance_target) - std).mean())
        cov_losses = []
        for slot_i in range(slots.shape[1]):
            x = centered[:, slot_i]
            if x.shape[0] > 1 and x.shape[1] > 1:
                cov = x.t().matmul(x) / max(1, x.shape[0] - 1)
                cov_losses.append(_off_diagonal(cov).pow(2).sum() / x.shape[-1])
        if cov_losses and float(covariance_weight):
            losses.append(float(covariance_weight) * torch.stack(cov_losses).mean())
    if not losses:
        return slots.sum() * 0.0
    return torch.stack(losses).mean()


def schema_concept_contrastive_loss(states_by_key, target_ids_by_key, temperature=0.1):
    """Cluster same-value concept states and separate other values for the same key.

    `states_by_key` maps an arbitrary schema key to a `[batch, d]` state tensor.
    `target_ids_by_key` maps the same key to integer ids, using negative ids for
    missing labels. The objective is schema-generic: callers decide what the keys
    and ids mean from data, while the loss only sees repeated ids inside a key.
    """
    temp = max(float(temperature), 1e-6)
    losses = []
    for key, states in states_by_key.items():
        targets = target_ids_by_key.get(key)
        if targets is None:
            continue
        if not torch.is_tensor(targets):
            targets = torch.tensor(targets, dtype=torch.long, device=states.device)
        else:
            targets = targets.to(device=states.device, dtype=torch.long)
        valid = targets.ge(0)
        if int(valid.sum()) < 2:
            continue
        z = F.normalize(states[valid], dim=-1)
        labels = targets[valid]
        n = labels.shape[0]
        eye = torch.eye(n, dtype=torch.bool, device=states.device)
        positives = labels[:, None].eq(labels[None, :]) & ~eye
        row_has_positive = positives.any(-1)
        if not bool(row_has_positive.any()):
            continue
        logits = z.matmul(z.t()) / temp
        logits = logits.masked_fill(eye, -float("inf"))
        log_prob = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        positive_counts = positives.sum(-1).clamp(min=1)
        positive_log_prob = log_prob.masked_fill(~positives, 0.0).sum(-1) / positive_counts
        losses.append(-positive_log_prob[row_has_positive].mean())
    if not losses:
        return _zero_from_states(states_by_key)
    return torch.stack(losses).mean()


def schema_concept_batch_centroid_loss(states_by_key, target_ids_by_key, temperature=0.1,
                                       margin=0.0):
    """Align concept states to value centroids discovered inside the current batch.

    Unlike learned prototype rows, these centroids are computed from repeated observations in
    the data. The objective is still schema-generic: it only knows that matching ids inside a
    key should share a reusable representation, while different ids for the same key should be
    separated by a margin.
    """
    temp = max(float(temperature), 1e-6)
    margin_t = float(margin)
    losses = []
    for key, states in states_by_key.items():
        targets = target_ids_by_key.get(key)
        if targets is None:
            continue
        targets = targets.to(device=states.device, dtype=torch.long)
        valid = targets.ge(0)
        if int(valid.sum()) < 2:
            continue
        z = F.normalize(states[valid], dim=-1)
        labels = targets[valid]
        unique = labels.unique(sorted=True)
        if unique.numel() < 2:
            continue
        centers = []
        center_labels = []
        for label in unique:
            rows = labels.eq(label)
            if int(rows.sum()) < 1:
                continue
            centers.append(F.normalize(z[rows].mean(dim=0), dim=0))
            center_labels.append(int(label.item()))
        if len(centers) < 2:
            continue
        centers = torch.stack(centers, dim=0)
        label_to_center = {label: i for i, label in enumerate(center_labels)}
        mapped = torch.tensor([label_to_center[int(label.item())] for label in labels],
                              dtype=torch.long, device=states.device)
        sim = z.matmul(centers.t())
        losses.append(F.cross_entropy(sim / temp, mapped))
        target_sim = sim.gather(1, mapped[:, None]).squeeze(1)
        other_sim = sim.masked_fill(
            F.one_hot(mapped, num_classes=centers.shape[0]).bool(),
            -float("inf")).max(-1).values
        losses.append(F.relu(other_sim + margin_t - target_sim).mean())
    if not losses:
        return _zero_from_states(states_by_key)
    return torch.stack(losses).mean()


def schema_concept_prototype_loss(logits_by_key, target_ids_by_key):
    """Classify projected concept states against learned prototypes for each schema key."""
    losses = []
    for key, logits in logits_by_key.items():
        targets = target_ids_by_key.get(key)
        if targets is None:
            continue
        targets = targets.to(device=logits.device, dtype=torch.long)
        if targets.ge(0).any() and logits.shape[-1] > 1:
            losses.append(F.cross_entropy(logits, targets, ignore_index=-1))
    if not losses:
        for logits in logits_by_key.values():
            return logits.sum() * 0.0
        return torch.tensor(0.0)
    return torch.stack(losses).mean()


def schema_concept_prototype_alignment_loss(states_by_key, target_ids_by_key,
                                            prototypes_by_key, temperature=0.1,
                                            margin=0.2):
    """Pull states to their target prototype and rank it above other prototypes."""
    temp = max(float(temperature), 1e-6)
    margin_t = float(margin)
    losses = []
    for key, states in states_by_key.items():
        targets = target_ids_by_key.get(key)
        prototypes = prototypes_by_key.get(key)
        if targets is None or prototypes is None:
            continue
        targets = targets.to(device=states.device, dtype=torch.long)
        valid = targets.ge(0)
        if not bool(valid.any()) or prototypes.shape[0] < 2:
            continue
        state = F.normalize(states[valid], dim=-1)
        labels = targets[valid]
        proto = F.normalize(prototypes.to(device=states.device), dim=-1)
        sim = state.matmul(proto.t())
        losses.append(F.cross_entropy(sim / temp, labels))
        target_sim = sim.gather(1, labels[:, None]).squeeze(1)
        losses.append((1.0 - target_sim).mean())
        other_sim = sim.masked_fill(
            F.one_hot(labels, num_classes=proto.shape[0]).bool(),
            -float("inf")).max(-1).values
        losses.append(F.relu(other_sim + margin_t - target_sim).mean())
    if not losses:
        return _zero_from_states(states_by_key)
    return torch.stack(losses).mean()


def schema_concept_state_spread_loss(states_by_key, target_ids_by_key,
                                     variance_target=0.05,
                                     centroid_margin=0.2,
                                     covariance_weight=0.05):
    """Anti-collapse regularizer for schema concept geometry states.

    The regularizer is task-agnostic. It uses only repeated data-supplied ids
    inside each schema key to keep projected states from becoming a single
    direction: dimensions need non-trivial batch variance, dimensions should not
    all carry the same signal, and observed value centroids should separate.
    """
    var_target = float(variance_target)
    centroid_margin = float(centroid_margin)
    covariance_weight = float(covariance_weight)
    losses = []
    for key, states in states_by_key.items():
        targets = target_ids_by_key.get(key)
        if targets is None:
            continue
        targets = targets.to(device=states.device, dtype=torch.long)
        valid = targets.ge(0)
        if int(valid.sum()) < 2:
            continue
        z = F.normalize(states[valid], dim=-1)
        labels = targets[valid]
        std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-4)
        losses.append(F.relu(var_target - std).mean())
        if z.shape[0] > 2:
            centered = z - z.mean(dim=0, keepdim=True)
            cov = centered.t().matmul(centered) / (z.shape[0] - 1)
            eye = torch.eye(cov.shape[0], dtype=torch.bool, device=cov.device)
            losses.append(cov.masked_select(~eye).pow(2).mean() * covariance_weight)
        centers = []
        for label in labels.unique(sorted=True):
            rows = labels.eq(label)
            if bool(rows.any()):
                centers.append(F.normalize(z[rows].mean(dim=0), dim=0))
        if len(centers) > 1:
            centers = torch.stack(centers, dim=0)
            sim = centers.matmul(centers.t())
            eye = torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
            losses.append(F.relu(sim.masked_select(~eye) - centroid_margin).mean())
    if not losses:
        return _zero_from_states(states_by_key)
    return torch.stack(losses).mean()


def schema_concept_prototype_spread_loss(prototypes_by_key, margin=0.2):
    """Keep value prototypes for the same schema key from collapsing together."""
    margin = float(margin)
    losses = []
    for prototypes in prototypes_by_key.values():
        if prototypes.shape[0] < 2:
            continue
        z = F.normalize(prototypes, dim=-1)
        sim = z.matmul(z.t())
        eye = torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
        losses.append(F.relu(sim.masked_select(~eye) - margin).mean())
    if not losses:
        for prototypes in prototypes_by_key.values():
            return prototypes.sum() * 0.0
        return torch.tensor(0.0)
    return torch.stack(losses).mean()
