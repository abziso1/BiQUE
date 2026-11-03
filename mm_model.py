"""Structure-anchored, gain-supervised multimodal residual correction."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from models import KBCModel, KGC, BiQUEScorer


class ModalityExpert(nn.Module):
    def __init__(self, features, nrelations, rank, dropout):
        super().__init__()
        if features.ndim != 2 or not torch.isfinite(features).all():
            raise ValueError("Features must be a finite [entity, feature] matrix")
        self.rank = rank
        self.register_buffer("available", features.norm(dim=1) > 0)
        self.register_buffer("features", F.normalize(features.float(), dim=-1))
        self.encoder = nn.Sequential(
            nn.Linear(features.shape[1], rank * 8), nn.LayerNorm(rank * 8),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(rank * 8, rank * 8),
        )
        self.relations = nn.Embedding(nrelations, rank * 16)
        nn.init.xavier_uniform_(self.relations.weight)
        self.scorer = BiQUEScorer(rank)

    def forward(self, queries):
        # Mask AFTER biased projections: missing input must never become evidence.
        entities = self.encoder(self.features) * self.available[:, None]
        scores = self.scorer(entities[queries[:, 0]].clone(),
                             self.relations(queries[:, 1]), entities)
        return scores


class SelectiveBiQUE(KBCModel):
    def __init__(self, sizes, rank, modal_rank, visual, textual, dropout=0.1,
                 alpha=0.2, gate_dim=32, fusion="selective"):
        super().__init__()
        if visual.shape[0] != sizes[0] or textual.shape[0] != sizes[0]:
            raise ValueError("Feature rows do not match entity count")
        self.backbone = KGC(sizes, rank)
        self.experts = nn.ModuleList([
            ModalityExpert(visual, sizes[1], modal_rank, dropout),
            ModalityExpert(textual, sizes[1], modal_rank, dropout),
        ])
        self.head_gate = nn.Linear(rank * 8, gate_dim)
        self.relation_gate = nn.Embedding(sizes[1], gate_dim)
        # Three distributions x four statistics, two agreements, two missing masks.
        self.router = nn.Sequential(nn.Linear(2 * gate_dim + 16, gate_dim),
                                    nn.GELU(), nn.Linear(gate_dim, 3))
        nn.init.zeros_(self.router[-1].weight)
        with torch.no_grad():
            self.router[-1].bias.copy_(torch.tensor([2., 0., 0.]))
        self.sacrf_gate = nn.Embedding(sizes[1], 2)
        nn.init.zeros_(self.sacrf_gate.weight)
        self.alpha = alpha
        self.fusion = fusion
        self.stage = "structure"

    def set_stage(self, stage):
        self.stage = stage
        self.requires_grad_(False)
        for parameter in self.parameters():
            parameter.grad = None
        if stage == "structure":
            self.backbone.requires_grad_(True)
        elif stage == "experts":
            self.experts.requires_grad_(True)
        elif stage == "router":
            for module in (self.head_gate, self.relation_gate, self.router, self.sacrf_gate):
                module.requires_grad_(True)
        else:
            raise ValueError(stage)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.stage != "structure":
            self.backbone.eval()
        if self.stage != "experts":
            self.experts.eval()
        return self

    @staticmethod
    def distribution_stats(scores):
        scores = scores.detach()
        logp = F.log_softmax(scores, dim=-1)
        p = logp.exp()
        top = scores.topk(2, dim=-1).values
        std = scores.std(dim=-1).clamp_min(1e-6)
        return torch.stack((-(p * logp).sum(-1) / math.log(scores.shape[1]),
                            p.max(-1).values, (top[:, 0] - top[:, 1]) / std,
                            (top[:, 0] - scores.mean(-1)) / std), dim=-1)

    def components(self, queries):
        structural, factors = self.backbone(queries)
        raw = [expert(queries) for expert in self.experts]
        scale = structural.detach().std(-1, keepdim=True).clamp_min(1e-6)
        deltas, availability = [], []
        for scores, expert in zip(raw, self.experts):
            present = expert.available[queries[:, 0]]
            mask = present[:, None] & expert.available[None, :]
            normalized = ((scores - scores.mean(-1, keepdim=True)) /
                          scores.std(-1, keepdim=True).clamp_min(1e-6))
            deltas.append(self.alpha * scale * normalized * mask)
            availability.append(present)
        return structural, raw, torch.stack(deltas, 1), torch.stack(availability, 1), factors

    def route(self, queries, structural, raw, available):
        head = self.backbone.entity_embeddings(queries[:, 0]).detach()
        stats = [self.distribution_stats(s) for s in [structural] + raw]
        agreement = torch.stack([(s.argmax(-1) == structural.argmax(-1)).float()
                                 for s in raw], -1)
        inputs = torch.cat([self.head_gate(F.normalize(head, dim=-1)),
                            self.relation_gate(queries[:, 1]), *stats,
                            agreement, available.float()], -1)
        logits = self.router(inputs)
        logits = torch.cat([logits[:, :1], logits[:, 1:].masked_fill(~available, -1e9)], -1)
        return logits

    def predict(self, queries, return_details=False):
        structural, raw, deltas, available, factors = self.components(queries)
        logits = self.route(queries, structural, raw, available)
        weights = logits.softmax(-1)
        if self.fusion == "structure":
            scores = structural
        elif self.fusion == "fixed":
            scores = structural + deltas.mean(1)
        elif self.fusion == "sacrf":
            gates = self.sacrf_gate(queries[:, 1]).sigmoid() * available
            scores = structural + (gates[:, :, None] * deltas).sum(1)
        else:
            scores = structural + (weights[:, 1:, None] * deltas).sum(1)
        if return_details:
            return {"scores": scores, "structural": structural, "raw": raw,
                    "deltas": deltas, "available": available, "logits": logits,
                    "weights": weights, "factors": factors}
        return scores, factors

    def forward(self, queries):
        if self.stage == "structure":
            return self.backbone(queries)
        return self.predict(queries)
