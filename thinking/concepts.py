"""Generic schema-driven concept heads.

This module is intentionally task-agnostic.  A caller supplies fact keys and value
sets from data, and the head learns where each key should read from a source
sequence plus a value-embedding space for that key.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SchemaConceptHead(nn.Module):
    """Attend over source states and score values for each schema key."""

    def __init__(self, keys, values, d, init_scale=0.02):
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

    def state_tensor(self, source, mask=None):
        if not self.keys:
            return source.new_zeros((source.shape[0], 0, source.shape[-1]))
        scale = source.shape[-1] ** -0.5
        scores = torch.einsum("btd,kd->bkt", source, self.key_query) * scale
        if mask is not None:
            scores = scores.masked_fill(mask[:, None, :], -float("inf"))
        weights = scores.softmax(-1)
        return torch.einsum("bkt,btd->bkd", weights, source)

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


def _zero_from_states(states_by_key):
    for states in states_by_key.values():
        return states.sum() * 0.0
    return torch.tensor(0.0)


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
