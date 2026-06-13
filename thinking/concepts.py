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
