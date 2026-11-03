"""Three-stage training with held-out gain supervision and matched evaluation.

No legacy checkpoint is loaded implicitly. The selection split is explicit;
the default follows the user's test-selected exploratory protocol.
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
from torch.nn import functional as F

from datasets import Dataset
from mm_data import (audit_splits, feature_manifest, filtered_ranks, hard_candidates,
                     reciprocal, split_training, training_filters)
from mm_model import SelectiveBiQUE
from regularizers import wN3
from utils import set_seed


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["DB15K", "MKG-W", "MKG-Y"], required=True)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--feature-dir", default="embeddings")
    p.add_argument("--feature-entity-map", required=True,
                   help="Entity-ID manifest associated with the feature files")
    p.add_argument("--output", default=None)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--rank", type=int, default=128)
    p.add_argument("--modal-rank", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--structure-epochs", type=int, default=300)
    p.add_argument("--expert-epochs", type=int, default=100)
    p.add_argument("--router-epochs", type=int, default=50)
    p.add_argument("--valid-every", type=int, default=5)
    p.add_argument("--selection-split", choices=["valid", "test"], default="test")
    p.add_argument("--patience", type=int, default=8, help="Validation checks without improvement")
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--router-lr", type=float, default=1e-3)
    p.add_argument("--reg", type=float, default=0.07)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--gate-dim", type=int, default=32)
    p.add_argument("--topk", type=int, default=32)
    p.add_argument("--margin", type=float, default=0.5)
    p.add_argument("--correction-weight", type=float, default=0.1)
    p.add_argument("--aux-weight", type=float, default=0.05)
    p.add_argument("--router-supervision", type=float, default=1.0)
    p.add_argument("--utility-temperature", type=float, default=0.02)
    p.add_argument("--abstain-cost", type=float, default=0.002,
                   help="Subtract this cost from both correction utilities")
    p.add_argument("--calibration-fraction", type=float, default=0.1)
    p.add_argument("--fusion", choices=["selective", "sacrf", "fixed"], default="selective")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--max-batches", type=int, default=0, help="Smoke-test limit; 0 means full epochs")
    p.add_argument("--eval-limit", type=int, default=0, help="Smoke-test limit per direction")
    p.add_argument("--skip-test", action="store_true")
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--checkpoint", help="A checkpoint from THIS runner with identical data/config")
    p.add_argument("--start-from", choices=["structure", "experts", "router"], default="structure")
    args = p.parse_args()
    for key in ["rank", "modal_rank", "batch_size", "eval_batch_size", "valid_every", "topk", "threads"]:
        if getattr(args, key) < 1:
            p.error(f"{key} must be positive")
    if min(args.structure_epochs, args.expert_epochs, args.router_epochs) < 1:
        p.error("Each stage requires at least one epoch")
    if args.utility_temperature <= 0 or not 0 < args.alpha <= 1:
        p.error("utility-temperature must be positive and alpha must lie in (0, 1]")
    if args.start_from != "structure" and not args.checkpoint:
        p.error("--start-from requires --checkpoint")
    return args


def read_mapping(path):
    mapping = {}
    for line in Path(path).read_text().splitlines():
        fields = line.rsplit(maxsplit=1)
        if len(fields) == 2:
            mapping[fields[0]] = int(fields[1])
    return mapping


def batches(examples, batch_size, shuffle, device, limit=0):
    indices = torch.randperm(len(examples)) if shuffle else torch.arange(len(examples))
    for number, begin in enumerate(range(0, len(indices), batch_size)):
        if limit and number >= limit:
            break
        yield examples[indices[begin:begin + batch_size]].to(device)


def correction_loss(scores, queries, candidates, valid, margin):
    positive = scores.gather(1, queries[:, 2:3])
    negative = scores.gather(1, candidates)
    losses = F.softplus(margin - positive + negative)
    return (losses * valid).sum() / valid.sum().clamp_min(1)


@torch.no_grad()
def utility_targets(details, queries, filters, temperature, cost):
    base = details["structural"]
    baseline_rr = filtered_ranks(base, queries, filters).float().reciprocal()
    gains = [torch.zeros_like(baseline_rr)]
    for m in range(2):
        rr = filtered_ranks(base + details["deltas"][:, m], queries, filters).float().reciprocal()
        gains.append(rr - baseline_rr - cost)
    utility = torch.stack(gains, -1)
    utility[:, 1:] = utility[:, 1:].masked_fill(~details["available"], -1e9)
    targets = (utility / temperature).softmax(-1)
    no_gain = utility[:, 1:].max(-1).values <= 0
    targets[no_gain] = targets.new_tensor([1., 0., 0.])
    return targets, utility


@torch.no_grad()
def evaluate(model, rows, nrelations, filters, args, structure_only=False):
    model.eval()
    ranks = {}
    weights = []
    for side in ["rhs", "lhs"]:
        queries = rows.copy()
        if args.eval_limit:
            queries = queries[:args.eval_limit]
        if side == "lhs":
            queries = queries[:, [2, 1, 0]].copy()
            queries[:, 1] += nrelations
        tensor = torch.from_numpy(queries.astype(np.int64))
        for batch in batches(tensor, args.eval_batch_size, False, args.device):
            if structure_only:
                predictions = {"structure": model.backbone(batch)[0]}
            else:
                details = model.predict(batch, return_details=True)
                predictions = {"structure": details["structural"], "fused": details["scores"],
                               "visual": details["structural"] + details["deltas"][:, 0],
                               "textual": details["structural"] + details["deltas"][:, 1]}
                weights.append(details["weights"].cpu())
            for key, scores in predictions.items():
                ranks.setdefault(key, []).append(filtered_ranks(scores, batch, filters[side]).cpu())
    ranks = {key: torch.cat(values) for key, values in ranks.items()}
    result = {}
    base_rr = ranks["structure"].float().reciprocal()
    for key, values in ranks.items():
        rr = values.float().reciprocal()
        result[key] = {"MRR": rr.mean().item(),
                       **{f"Hits@{k}": (values <= k).float().mean().item() for k in [1, 3, 10]},
                       "Gain": (rr - base_rr).clamp_min(0).mean().item(),
                       "Harm": (base_rr - rr).clamp_min(0).mean().item()}
    if not structure_only:
        oracle_rr = torch.stack([r.float().reciprocal() for r in ranks.values()]).max(0).values
        result["oracle_diagnostic_MRR"] = oracle_rr.mean().item()
        result["mean_router_weights"] = torch.cat(weights).mean(0).tolist()
        if model.fusion == "selective":
            intervention = 1 - torch.cat(weights)[:, 0]
            order = intervention.argsort(descending=True)
            fused_rr = ranks["fused"].float().reciprocal()
            curve = []
            for fraction in [0., .1, .25, .5, .75, 1.]:
                count = round(len(base_rr) * fraction)
                rr = base_rr.clone()
                rr[order[:count]] = fused_rr[order[:count]]
                curve.append({"fraction": fraction, "MRR": rr.mean().item()})
            result["intervention_curve"] = curve
    return result, ranks


def state_without_features(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            if not key.endswith(".features")}


def restore(model, state):
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not key.endswith(".features") for key in missing):
        raise ValueError(f"Invalid checkpoint keys: missing={missing}, unexpected={unexpected}")


def run_stage(model, stage, examples, fit_filters, train_filters, dataset, args, output, manifest):
    model.set_stage(stage)
    parameters = [p for p in model.parameters() if p.requires_grad]
    lr = args.router_lr if stage == "router" else args.learning_rate
    # Match original Adam backbone; apply weight decay only to auxiliary modules.
    optimizer = torch.optim.Adam(parameters, lr=lr) if stage == "structure" else \
        torch.optim.AdamW(parameters, lr=lr, weight_decay=args.weight_decay)
    regularizer = wN3(args.reg)
    epochs = {"structure": args.structure_epochs, "experts": args.expert_epochs,
              "router": args.router_epochs}[stage]
    if stage == "router" and args.fusion == "fixed":
        epochs = 1
    best, best_state, best_epoch, stale = -1., None, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        total, count = {}, 0
        for batch in batches(examples, args.batch_size, True, args.device, args.max_batches):
            truth = batch[:, 2]
            if stage == "structure":
                scores, factors = model(batch)
                terms = {"ce": F.cross_entropy(scores, truth), "reg": regularizer(factors)}
            else:
                details = model.predict(batch, return_details=True)
                candidates, valid = hard_candidates(details["structural"], batch,
                                                     fit_filters if stage == "experts" else train_filters,
                                                     args.topk)
                if stage == "experts":
                    ce, aux, corr = [], [], []
                    for m, expert in enumerate(model.experts):
                        present = details["available"][:, m] & expert.available[truth]
                        if present.any():
                            scores = details["structural"] + details["deltas"][:, m]
                            ce.append(F.cross_entropy(scores[present], truth[present]))
                            # Missing candidates cannot act as synthetic learned modality entities.
                            raw = details["raw"][m].masked_fill(~expert.available[None, :], -1e9)
                            aux.append(F.cross_entropy(raw[present], truth[present]))
                            corr.append(correction_loss(scores[present], batch[present],
                                                        candidates[present], valid[present], args.margin))
                    if not ce:
                        continue
                    terms = {"ce": torch.stack(ce).mean(),
                             "aux": args.aux_weight * torch.stack(aux).mean(),
                             "correction": args.correction_weight * torch.stack(corr).mean()}
                else:
                    targets, utility = utility_targets(details, batch, train_filters,
                                                       args.utility_temperature, args.abstain_cost)
                    supervised = -(targets * F.log_softmax(details["logits"], -1)).sum(-1).mean()
                    terms = {"ce": F.cross_entropy(details["scores"], truth),
                             "correction": args.correction_weight * correction_loss(
                                 details["scores"], batch, candidates, valid, args.margin)}
                    if args.fusion == "selective":
                        terms["utility"] = args.router_supervision * supervised
                    if args.fusion == "fixed":
                        # Fixed fusion has no trainable router; evaluate without meaningless updates.
                        terms = {}
            if terms:
                loss = sum(terms.values())
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss in {stage} epoch {epoch}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 5.)
                optimizer.step()
                for name, value in terms.items():
                    total[name] = total.get(name, 0.) + value.item() * len(batch)
                count += len(batch)
        logging.info("stage=%s epoch=%d losses=%s", stage, epoch,
                     {name: round(value / max(count, 1), 6) for name, value in total.items()})
        if epoch % args.valid_every and epoch != epochs:
            continue
        metrics, _ = evaluate(model, dataset.data[args.selection_split], dataset.n_predicates // 2,
                              dataset.to_skip, args, structure_only=stage == "structure")
        if stage == "structure":
            score = metrics["structure"]["MRR"]
        elif stage == "experts":
            score = (metrics["visual"]["MRR"] + metrics["textual"]["MRR"]) / 2
        else:
            score = metrics["fused"]["MRR"]
        record = {"stage": stage, "epoch": epoch, "selection_split": args.selection_split,
                  "selection_score": score, "metrics": metrics}
        with (output / "validation.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        logging.info("validation=%s", json.dumps(record))
        if score > best:
            best, best_epoch, stale = score, epoch, 0
            best_state = state_without_features(model)
            torch.save({"model": best_state, "stage": stage, "epoch": epoch,
                        "config": vars(args), "manifest": manifest, "validation": metrics},
                       output / f"{stage}_best.pt")
        else:
            stale += 1
        if args.patience and stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError(f"Stage {stage} did not produce a checkpoint")
    restore(model, best_state)
    logging.info("stage=%s selected_epoch=%d score=%.6f", stage, best_epoch, best)


def main():
    started = time.perf_counter()
    args = arguments()
    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    output = Path(args.output or f"logs/mmkgc/{args.dataset}/{datetime.now():%Y%m%d_%H%M%S_%f}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty run directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(output / "train.log")], force=True)
    logging.info("config=%s", vars(args))
    dataset = Dataset(args.data_dir, args.dataset)
    audit = audit_splits(dataset.data)
    (output / "data_audit.json").write_text(json.dumps(audit, indent=2))
    logging.info("data_audit=%s", audit)
    if not audit["clean"] and args.selection_split == "valid":
        raise ValueError("Overlapping or duplicate dataset splits. Inspect data_audit.json; source data was not changed.")
    if not audit["clean"]:
        logging.warning("Split overlaps recorded in audit; proceeding with explicitly requested test selection")
    local_mapping = read_mapping(Path(dataset.root) / "ent_id")
    feature_mapping = read_mapping(args.feature_entity_map)
    if local_mapping != feature_mapping or set(local_mapping.values()) != set(range(dataset.n_entities)):
        raise ValueError("Feature entity-ID manifest does not exactly match this dataset")
    tensors, features = feature_manifest(args.feature_dir, args.dataset, dataset.n_entities,
                                         Path(dataset.root) / "ent_id")
    fit, calibration = split_training(dataset.data["train"], args.calibration_fraction, args.seed)
    nrelations = dataset.n_predicates // 2
    fit_aug, calibration_aug = reciprocal(fit, nrelations), reciprocal(calibration, nrelations)
    fit_filters = training_filters(fit_aug)
    train_filters = training_filters(reciprocal(dataset.data["train"], nrelations))
    manifest = {"protocol": f"heldout_gain_v1_select_{args.selection_split}",
                "selection_split": args.selection_split,
                "independent_test": args.selection_split != "test",
                "audit": audit, "features": features,
                "fit_facts": len(fit), "calibration_facts": len(calibration),
                "fit_sha256": hashlib.sha256(fit.tobytes()).hexdigest(),
                "calibration_sha256": hashlib.sha256(calibration.tobytes()).hexdigest(),
                "split_sha256": {s: hashlib.sha256(a.tobytes()).hexdigest() for s, a in dataset.data.items()},
                "smoke_test": bool(args.max_batches or args.eval_limit),
                "source_sha256": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                                  for name in ["mm_model.py", "mm_data.py", "train_mm.py", "models.py", "qutils.py"]},
                "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (output / "config.json").write_text(json.dumps(vars(args), indent=2))
    np.savez(output / "training_partition.npz", fit=fit, calibration=calibration)
    if args.audit_only:
        logging.info("Audit complete: %s", output)
        return
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu for CPU diagnostics")
    if args.device == "cuda":
        logging.info("GPU=%s", torch.cuda.get_device_name(0))
    model = SelectiveBiQUE(dataset.get_shape(), args.rank, args.modal_rank, *tensors,
                           dropout=args.dropout, alpha=args.alpha, gate_dim=args.gate_dim,
                           fusion=args.fusion).to(args.device)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        previous = checkpoint.get("manifest", {})
        for key in ["protocol", "features", "fit_sha256", "calibration_sha256", "split_sha256"]:
            if previous.get(key) != manifest[key]:
                raise ValueError(f"Checkpoint protocol/data mismatch: {key}")
        for key in ["rank", "modal_rank", "alpha", "gate_dim", "fusion", "dropout"]:
            if checkpoint["config"][key] != getattr(args, key):
                raise ValueError(f"Checkpoint model configuration mismatch: {key}")
        needed = {"experts": "structure", "router": "experts"}
        if args.start_from in needed and checkpoint["stage"] != needed[args.start_from]:
            raise ValueError("Checkpoint must be the best result of the preceding stage")
        if previous.get("smoke_test") and not manifest["smoke_test"]:
            raise ValueError("Cannot use a smoke-test checkpoint for a full research run")
        restore(model, checkpoint["model"])
    stages = ["structure", "experts", "router"]
    for stage in stages[stages.index(args.start_from):]:
        examples = torch.from_numpy(calibration_aug if stage == "router" else fit_aug)
        run_stage(model, stage, examples, fit_filters, train_filters, dataset, args, output, manifest)
    validation, valid_ranks = evaluate(model, dataset.data["valid"], nrelations, dataset.to_skip, args)
    results = {"validation": validation, "protocol": manifest["protocol"],
               "selection_split": args.selection_split,
               "independent_test": args.selection_split != "test",
               "smoke_test": manifest["smoke_test"], "fit_facts": len(fit),
               "calibration_facts": len(calibration)}
    if not args.skip_test:
        test, test_ranks = evaluate(model, dataset.data["test"], nrelations, dataset.to_skip, args)
        results["test"] = test
        np.savez(output / "test_ranks.npz", **{k: v.numpy() for k, v in test_ranks.items()})
    np.savez(output / "valid_ranks.npz", **{k: v.numpy() for k, v in valid_ranks.items()})
    results["runtime_seconds"] = time.perf_counter() - started
    if args.device == "cuda":
        results["peak_cuda_memory_mib"] = torch.cuda.max_memory_allocated() / 2**20
    (output / "results.json").write_text(json.dumps(results, indent=2))
    logging.info("complete output=%s results=%s", output.resolve(), json.dumps(results))


if __name__ == "__main__":
    main()
