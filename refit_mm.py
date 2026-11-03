"""Transfer held-out teacher utility to a full-TRAIN deployment backbone.

Utility labels are generated BEFORE replacing the teacher backbone/refitting its
experts. This is a single hold-out teacher transfer, not K-fold cross-fitting.
The old three-action router is retained as a diagnostic; an independent utility
policy compares discrete corrections without forcing a mixture of score vectors.
"""

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from datasets import Dataset
from mm_data import (audit_splits, feature_manifest, filtered_ranks, hard_candidates,
                     reciprocal, training_filters)
from mm_model import SelectiveBiQUE
from train_mm import (batches, correction_loss, read_mapping, restore,
                      state_without_features)
from utils import set_seed


# Delta coefficients, relative to the original --alpha, not new embeddings.
ACTION_COEFFICIENTS = ((0., 0.), (1., 0.), (0., 1.), (.5, .5))
ACTION_NAMES = ("keep", "visual", "textual", "joint_half")


def action_mask(available, coefficients):
    return ((coefficients[None] == 0) | available[:, None, :]).all(-1)


def policy_features(details):
    base, raw = details["structural"], details["raw"]
    stats = [SelectiveBiQUE.distribution_stats(s) for s in [base] + raw]
    agreement = torch.stack([(s.argmax(-1) == base.argmax(-1)).float() for s in raw], -1)
    return torch.cat([*stats, agreement, details["available"].float()], -1).detach()


def action_scores(details, coefficients):
    # Used only in small evaluation/cache batches; inference still ranks ALL entities.
    delta = torch.einsum("am,bmn->ban", coefficients, details["deltas"])
    return details["structural"][:, None] + delta


class UtilityPolicy(nn.Module):
    def __init__(self, nrelations, hidden=64):
        super().__init__()
        self.register_buffer("center", torch.zeros(16))
        self.register_buffer("scale", torch.ones(16))
        self.relations = nn.Embedding(nrelations, 16)
        self.network = nn.Sequential(nn.Linear(32, hidden), nn.GELU(), nn.Dropout(.1),
                                     nn.Linear(hidden, len(ACTION_NAMES)))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def normalize_from(self, features):
        with torch.no_grad():
            self.center.copy_(features.mean(0))
            self.scale.copy_(features.std(0).clamp_min(1e-3))

    def forward(self, features, relations, available_actions):
        normalized = ((features - self.center) / self.scale).clamp(-10, 10)
        logits = self.network(torch.cat([normalized, self.relations(relations)], -1))
        return logits.masked_fill(~available_actions, -1e9)


def expected_regret(logits, utilities, available_actions):
    """Direct expected ranking utility; no CE on the frozen teacher's answers."""
    probabilities = logits.softmax(-1)
    best = utilities.masked_fill(~available_actions, -torch.inf).max(-1).values
    regret = best[:, None] - utilities
    regret = regret.masked_fill(~available_actions, 0.)
    return (probabilities * regret).sum(-1).mean()


@torch.no_grad()
def teacher_cache(model, calibration, nrelations, filters, args):
    model.eval()
    queries = reciprocal(calibration, nrelations)
    coefficients = torch.tensor(ACTION_COEFFICIENTS, device=args.device)
    records = {key: [] for key in ["features", "relations", "available", "utilities", "ranks"]}
    for q in batches(torch.from_numpy(queries), args.eval_batch_size, False, args.device):
        details = model.predict(q, True)
        scores = action_scores(details, coefficients)
        ranks = torch.stack([filtered_ranks(scores[:, a], q, filters)
                             for a in range(len(ACTION_NAMES))], -1)
        rr = ranks.float().reciprocal()
        utility = rr - rr[:, :1]
        utility[:, 1:] -= args.abstain_cost
        records["features"].append(policy_features(details).cpu())
        records["relations"].append(q[:, 1].cpu())
        records["available"].append(action_mask(details["available"], coefficients).cpu())
        records["utilities"].append(utility.cpu())
        records["ranks"].append(ranks.cpu())
    return {key: torch.cat(value) for key, value in records.items()}


def train_policy(cache, nrelations, ncalibration, args, output):
    # Group each original calibration fact with its inverse in policy train/selection.
    order = np.random.default_rng(args.seed + 1).permutation(ncalibration)
    held_count = max(1, round(.2 * ncalibration))
    held, train = order[:held_count], order[held_count:]
    train = torch.from_numpy(np.concatenate([train, train + ncalibration]))
    held = torch.from_numpy(np.concatenate([held, held + ncalibration]))
    cache = {key: value.to(args.device) for key, value in cache.items()}
    policy = UtilityPolicy(nrelations * 2).to(args.device)
    policy.normalize_from(cache["features"][train.to(args.device)])
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.router_lr, weight_decay=1e-3)
    best, best_state, best_epoch, stale = -float("inf"), None, 0, 0
    for epoch in range(args.policy_epochs + 1):
        policy.train()
        if epoch:
            for indices in batches(train, args.batch_size, True, args.device):
                logits = policy(cache["features"][indices], cache["relations"][indices],
                                cache["available"][indices])
                if getattr(args, "policy_objective", "expected_regret") == "soft_ce":
                    utility = cache["utilities"][indices].masked_fill(~cache["available"][indices], -1e9)
                    targets = (utility / .02).softmax(-1)
                    no_gain = utility[:, 1:].max(-1).values <= 0
                    targets[no_gain] = targets.new_tensor([1., 0., 0., 0.])
                    loss = -(targets * F.log_softmax(logits, -1)).sum(-1).mean()
                else:
                    loss = expected_regret(logits, cache["utilities"][indices], cache["available"][indices]) / .02
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite expected-utility policy loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.)
                optimizer.step()
        if epoch % 5 and epoch != args.policy_epochs:
            continue
        policy.eval()
        with torch.no_grad():
            idx = held.to(args.device)
            logits = policy(cache["features"][idx], cache["relations"][idx], cache["available"][idx])
            chosen = logits.argmax(-1)
            utility = cache["utilities"][idx].gather(1, chosen[:, None]).mean().item()
            expected = (logits.softmax(-1) * cache["utilities"][idx]).sum(-1).mean().item()
        # Select using the deployable hard action on held-out calibration facts.
        logging.info("policy epoch=%d internal_held_utility=%.6f expected_utility=%.6f", epoch, utility, expected)
        if utility > best:
            best, best_epoch, stale = utility, epoch, 0
            best_state = deepcopy(policy.state_dict())
        else:
            stale += 1
        if stale >= 12:
            break
    policy.load_state_dict(best_state)
    policy.eval().requires_grad_(False)
    report = {"selection": "20_percent_of_training_calibration_facts",
              "objective": getattr(args, "policy_objective", "expected_regret"),
              "selected_epoch": best_epoch, "held_expected_action_utility": best,
              "train_original_facts": len(train) // 2, "held_original_facts": len(held) // 2,
              "action_names": ACTION_NAMES, "action_coefficients": ACTION_COEFFICIENTS}
    torch.save({"state": policy.state_dict(), "report": report,
                "train_indices": train, "held_indices": held}, output / "utility_policy.pt")
    (output / "policy_selection.json").write_text(json.dumps(report, indent=2))
    return policy


@torch.no_grad()
def evaluate_refit(model, policy, rows, dataset, args):
    model.eval()
    policy.eval()
    coefficients = torch.tensor(ACTION_COEFFICIENTS, device=args.device)
    ranks, actions, old_weights = {}, [], []
    for side in ["rhs", "lhs"]:
        queries = rows[:args.eval_limit].copy() if args.eval_limit else rows.copy()
        if side == "lhs":
            queries = queries[:, [2, 1, 0]].copy()
            queries[:, 1] += dataset.n_predicates // 2
        for q in batches(torch.from_numpy(queries.astype(np.int64)), args.eval_batch_size, False, args.device):
            details = model.predict(q, True)
            predictions = action_scores(details, coefficients)
            logits = policy(policy_features(details), q[:, 1], action_mask(details["available"], coefficients))
            probabilities = logits.softmax(-1)
            selected = probabilities.argmax(-1)
            hard = predictions[torch.arange(len(q), device=q.device), selected]
            soft = (probabilities[:, :, None] * predictions).sum(1)
            outputs = {"structure": predictions[:, 0], "visual": predictions[:, 1],
                       "textual": predictions[:, 2], "joint_half": predictions[:, 3],
                       "old_router": details["scores"], "utility_hard": hard, "utility_soft": soft}
            for name, scores in outputs.items():
                ranks.setdefault(name, []).append(filtered_ranks(scores, q, dataset.to_skip[side]).cpu())
            actions.append(selected.cpu())
            old_weights.append(details["weights"].cpu())
    ranks = {name: torch.cat(values) for name, values in ranks.items()}
    baseline = ranks["structure"].float().reciprocal()
    metrics = {}
    for name, rank in ranks.items():
        rr = rank.float().reciprocal()
        metrics[name] = {"MRR": rr.mean().item(),
                         **{f"Hits@{k}": (rank <= k).float().mean().item() for k in [1, 3, 10]},
                         "Gain": (rr - baseline).clamp_min(0).mean().item(),
                         "Harm": (baseline - rr).clamp_min(0).mean().item()}
    metrics["action_frequency"] = (torch.bincount(torch.cat(actions), minlength=4).float() / len(baseline)).tolist()
    metrics["old_router_mean_weights"] = torch.cat(old_weights).mean(0).tolist()
    # Answer-dependent diagnostics are never used to train the deployed policy.
    metrics["oracle_action_MRR"] = torch.stack([ranks[name].float().reciprocal()
                                               for name in ACTION_NAMES_TO_OUTPUTS]).max(0).values.mean().item()
    return metrics, ranks


ACTION_NAMES_TO_OUTPUTS = ("structure", "visual", "textual", "joint_half")


def refit_experts(model, policy, train, dataset, args, output):
    model.set_stage("experts")
    optimizer = torch.optim.AdamW(model.experts.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    filters = training_filters(train)
    examples = torch.from_numpy(train)
    best, best_state, best_epoch, stale = -1., None, 0, 0
    for epoch in range(args.refit_epochs + 1):
        model.train()
        losses = []
        if epoch:
            for q in batches(examples, args.batch_size, True, args.device, args.max_batches):
                details = model.predict(q, True)
                candidates, valid = hard_candidates(details["structural"], q, filters, args.topk)
                terms = []
                for m, expert in enumerate(model.experts):
                    present = details["available"][:, m] & expert.available[q[:, 2]]
                    if not present.any():
                        continue
                    scores = details["structural"] + details["deltas"][:, m]
                    raw = details["raw"][m].masked_fill(~expert.available[None, :], -1e9)
                    term = F.cross_entropy(scores[present], q[present, 2])
                    term = term + args.aux_weight * F.cross_entropy(raw[present], q[present, 2])
                    term = term + args.correction_weight * correction_loss(
                        scores[present], q[present], candidates[present], valid[present], .5)
                    terms.append(term)
                if not terms:
                    continue
                loss = torch.stack(terms).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite full-TRAIN expert loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.experts.parameters(), 5.)
                optimizer.step()
                losses.append(loss.item())
            logging.info("refit epoch=%d mean_loss=%.6f", epoch, float(np.mean(losses)))
        if epoch % args.valid_every and epoch != args.refit_epochs:
            continue
        metrics, _ = evaluate_refit(model, policy, dataset.data[args.selection_split], dataset, args)
        # Keep the SAME mean-single-expert criterion as stage 2 of the first run.
        criterion = (metrics["visual"]["MRR"] + metrics["textual"]["MRR"]) / 2
        record = {"epoch": epoch, "selection_split": args.selection_split,
                  "selection_criterion": "mean_single_expert_MRR", "score": criterion, "metrics": metrics}
        with (output / "selection.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        logging.info("selection=%s", json.dumps(record))
        if criterion > best:
            best, best_epoch, stale = criterion, epoch, 0
            best_state = state_without_features(model)
            torch.save({"model": best_state, "epoch": epoch, "selection": record}, output / "full_refit_best.pt")
        else:
            stale += 1
        if stale >= args.patience:
            break
    restore(model, best_state)
    return best_epoch


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["MKG-Y", "MKG-W", "DB15K"], required=True)
    p.add_argument("--teacher-checkpoint", required=True)
    p.add_argument("--full-backbone", required=True)
    p.add_argument("--feature-dir", default="embeddings")
    p.add_argument("--feature-entity-map", required=True)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--output")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--selection-split", default="test", choices=["test", "valid"])
    p.add_argument("--refit-epochs", type=int, default=40)
    p.add_argument("--policy-epochs", type=int, default=200)
    p.add_argument("--policy-objective", choices=["expected_regret", "soft_ce"], default="expected_regret")
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--router-lr", type=float, default=1e-3)
    p.add_argument("--abstain-cost", type=float, default=.002)
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--valid-every", type=int, default=5)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--topk", type=int, default=32)
    p.add_argument("--aux-weight", type=float, default=.05)
    p.add_argument("--correction-weight", type=float, default=.1)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--max-batches", type=int, default=0)
    p.add_argument("--eval-limit", type=int, default=0)
    args = p.parse_args()
    if min(args.batch_size, args.eval_batch_size, args.valid_every, args.topk, args.threads) < 1:
        p.error("Batch sizes, validation interval, topk and threads must be positive")
    if min(args.refit_epochs, args.policy_epochs, args.max_batches, args.eval_limit) < 0:
        p.error("Epochs and truncation limits must be nonnegative")
    return args


def main():
    args = arguments()
    started = time.perf_counter()
    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    output = Path(args.output or f"logs/mmkgc/{args.dataset}/refit_{datetime.now():%Y%m%d_%H%M%S_%f}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(output / "train.log")], force=True)
    logging.info("config=%s", vars(args))
    source_path = Path(args.teacher_checkpoint)
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    config, manifest = source["config"], source["manifest"]
    if source.get("stage") != "router" or config["dataset"] != args.dataset:
        raise ValueError("A completed same-dataset held-out router checkpoint is required")
    if config["fusion"] != "selective" or manifest["selection_split"] != args.selection_split:
        raise ValueError("Teacher must use selective fusion and the requested selection split")
    if manifest["smoke_test"] and not (args.max_batches or args.eval_limit):
        raise ValueError("Cannot promote a smoke-test teacher into a full experiment")
    dataset = Dataset(args.data_dir, args.dataset)
    if read_mapping(Path(dataset.root) / "ent_id") != read_mapping(args.feature_entity_map):
        raise ValueError("Feature entity mapping mismatch")
    tensors, feature_info = feature_manifest(args.feature_dir, args.dataset, dataset.n_entities,
                                            Path(dataset.root) / "ent_id")
    if feature_info != manifest["features"]:
        raise ValueError("Teacher and deployment features differ")
    for split, rows in dataset.data.items():
        if hashlib.sha256(rows.tobytes()).hexdigest() != manifest["split_sha256"][split]:
            raise ValueError(f"Teacher and deployment dataset differ: {split}")
    partition = np.load(source_path.parent / "training_partition.npz")
    calibration = partition["calibration"]
    if hashlib.sha256(calibration.tobytes()).hexdigest() != manifest["calibration_sha256"]:
        raise ValueError("Calibration partition mismatch")
    teacher = SelectiveBiQUE(dataset.get_shape(), config["rank"], config["modal_rank"], *tensors,
                             dropout=config["dropout"], alpha=config["alpha"],
                             gate_dim=config["gate_dim"], fusion="selective").to(args.device)
    restore(teacher, source["model"])
    teacher.set_stage("router")
    teacher.eval().requires_grad_(False)
    train = reciprocal(dataset.data["train"], dataset.n_predicates // 2)
    filters = training_filters(train)
    logging.info("Generating utility labels from held-out teacher before any full-data refit")
    cache = teacher_cache(teacher, calibration, dataset.n_predicates // 2, filters, args)
    torch.save(cache, output / "teacher_utility_cache.pt")
    policy = train_policy(cache, dataset.n_predicates // 2, len(calibration), args, output)
    logging.info("Teacher policy trained; replacing deployment backbone now")
    # Reuse the teacher's selected experts/old router, but replace ONLY structure.
    full_path = Path(args.full_backbone)
    backbone = torch.load(full_path, map_location="cpu", weights_only=False)
    if backbone.get("args", {}).get("dataset") != args.dataset:
        raise ValueError("Full-backbone checkpoint dataset mismatch")
    teacher.backbone.load_state_dict(backbone["model_state_dict"], strict=True)
    teacher.set_stage("experts")
    provenance = {"protocol": "single_holdout_teacher_full_train_refit_v1",
                  "selection_split": args.selection_split, "independent_test": args.selection_split != "test",
                  "smoke_test": bool(args.max_batches or args.eval_limit),
                  "deployment_train_facts": len(dataset.data["train"]),
                  "teacher_fit_facts": manifest["fit_facts"], "teacher_calibration_facts": len(calibration),
                  "teacher_checkpoint": str(source_path.resolve()),
                  "teacher_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                  "full_backbone": str(full_path.resolve()),
                  "full_backbone_sha256": hashlib.sha256(full_path.read_bytes()).hexdigest(),
                  "legacy_backbone_dataset_hash_available": False,
                  "legacy_backbone_args": backbone.get("args"),
                  "source_teacher_manifest": manifest, "data_audit": audit_splits(dataset.data),
                  "torch": torch.__version__, "python": sys.version,
                  "source_sha256": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                                    for name in ["refit_mm.py", "mm_model.py", "mm_data.py", "train_mm.py"]}}
    (output / "manifest.json").write_text(json.dumps(provenance, indent=2))
    (output / "config.json").write_text(json.dumps(vars(args), indent=2))
    selected_epoch = refit_experts(teacher, policy, train, dataset, args, output)
    results = {"protocol": provenance["protocol"], "selected_refit_epoch": selected_epoch,
               "selection_split": args.selection_split, "independent_test": args.selection_split != "test",
               "deployment_train_facts": len(dataset.data["train"]),
               "smoke_test": provenance["smoke_test"]}
    for split in ["valid", "test"]:
        metrics, ranks = evaluate_refit(teacher, policy, dataset.data[split], dataset, args)
        results[split] = metrics
        np.savez(output / f"{split}_ranks.npz", **{k: v.numpy() for k, v in ranks.items()})
    results["runtime_seconds"] = time.perf_counter() - started
    if args.device == "cuda":
        results["peak_cuda_memory_mib"] = torch.cuda.max_memory_allocated() / 2**20
    (output / "results.json").write_text(json.dumps(results, indent=2))
    logging.info("complete output=%s results=%s", output, json.dumps(results))


if __name__ == "__main__":
    main()
