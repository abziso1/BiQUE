"""Explicit data audits, held-out training facts and train-only filters."""

import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def audit_splits(data):
    sets = {name: set(map(tuple, rows)) for name, rows in data.items()}
    result = {"counts": {s: len(a) for s, a in data.items()},
              "unique": {s: len(a) for s, a in sets.items()}}
    for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]:
        result[a + "_" + b] = len(sets[a] & sets[b])
    result["clean"] = (all(result[a + "_" + b] == 0 for a, b in
                           [("train", "valid"), ("train", "test"), ("valid", "test")])
                       and all(len(data[s]) == len(sets[s]) for s in sets))
    return result


def reciprocal(rows, nrelations):
    rows = np.asarray(rows, dtype=np.int64)
    reverse = rows[:, [2, 1, 0]].copy()
    reverse[:, 1] += nrelations
    return np.concatenate([rows, reverse])


def split_training(rows, fraction, seed):
    if not 0 < fraction < 0.5:
        raise ValueError("calibration_fraction must lie in (0, 0.5)")
    # Unique original facts are split BEFORE constructing inverse triples.
    rows = np.unique(rows.astype(np.int64), axis=0)
    order = np.random.default_rng(seed).permutation(len(rows))
    count = max(1, round(len(rows) * fraction))
    return rows[order[count:]], rows[order[:count]]


def training_filters(rows):
    result = defaultdict(set)
    for h, r, t in rows:
        result[int(h), int(r)].add(int(t))
    return {key: sorted(values) for key, values in result.items()}


def mask_known(scores, queries, filters):
    # Always work on a clone; never mutate logits subsequently used by CE.
    result = scores.clone()
    for i, (h, r, _) in enumerate(queries.detach().cpu().tolist()):
        known = filters.get((h, r), [])
        if known:
            result[i, known] = -torch.inf
    return result


def hard_candidates(structural, queries, filters, k):
    masked = mask_known(structural.detach(), queries, filters)
    masked.scatter_(1, queries[:, 2:3], -torch.inf)
    values, indices = masked.topk(min(k, masked.shape[1]), dim=-1)
    return indices, values.isfinite()


def filtered_ranks(scores, queries, filters):
    target = scores.gather(1, queries[:, 2:3])
    filtered = mask_known(scores, queries, filters)
    filtered.scatter_(1, queries[:, 2:3], -torch.inf)
    return 1 + (filtered >= target).sum(-1)


def feature_manifest(feature_dir, dataset, nentities, mapping_path):
    result, tensors = {}, []
    for modal in ["visual", "textual"]:
        path = Path(feature_dir) / f"{dataset}-{modal}.pth"
        features = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
            raise ValueError(f"{path}: expected a 2D tensor")
        if features.shape[0] != nentities or not torch.isfinite(features).all():
            raise ValueError(f"{path}: wrong entity count or nonfinite values")
        result[modal] = {"path": str(path.resolve()), "shape": list(features.shape),
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "zero_rows": int((features.norm(dim=-1) == 0).sum())}
        tensors.append(features.float())
    result["entity_mapping_sha256"] = hashlib.sha256(Path(mapping_path).read_bytes()).hexdigest()
    return tensors, result
