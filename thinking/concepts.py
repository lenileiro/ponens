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
