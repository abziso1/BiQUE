"""Matched policy-objective ablation on one frozen full-refit expert checkpoint."""

import argparse
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from datasets import Dataset
from mm_data import feature_manifest
from mm_model import SelectiveBiQUE
from refit_mm import evaluate_refit, train_policy
from train_mm import restore
from utils import set_seed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--output")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    cli = p.parse_args()
    root = Path(cli.run_dir)
    output = Path(cli.output or root / f"objective_control_{datetime.now():%Y%m%d_%H%M%S_%f}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(output / "analysis.log")])
    args = SimpleNamespace(**json.loads((root / "config.json").read_text()))
    args.device = cli.device
    torch.set_num_threads(args.threads)
    manifest = json.loads((root / "manifest.json").read_text())
    teacher = torch.load(manifest["teacher_checkpoint"], map_location="cpu", weights_only=False)
    config = teacher["config"]
    dataset = Dataset(args.data_dir, args.dataset)
    tensors, feature_info = feature_manifest(args.feature_dir, args.dataset, dataset.n_entities,
                                            Path(dataset.root) / "ent_id")
    if feature_info != manifest["source_teacher_manifest"]["features"]:
        raise ValueError("Feature provenance mismatch")
    for split, rows in dataset.data.items():
        if hashlib.sha256(rows.tobytes()).hexdigest() != manifest["source_teacher_manifest"]["split_sha256"][split]:
            raise ValueError("Dataset provenance mismatch")
    model = SelectiveBiQUE(dataset.get_shape(), config["rank"], config["modal_rank"], *tensors,
                           dropout=config["dropout"], alpha=config["alpha"], gate_dim=config["gate_dim"]).to(args.device)
    selected = torch.load(root / "full_refit_best.pt", map_location="cpu", weights_only=False)
    restore(model, selected["model"])
    model.set_stage("router")
    model.eval().requires_grad_(False)
    cache = torch.load(root / "teacher_utility_cache.pt", map_location="cpu", weights_only=True)
    results = {"selected_expert_epoch": selected["epoch"],
               "matched_initialization_seed": args.seed + 123,
               "selection_split": args.selection_split, "independent_test": args.selection_split != "test",
               "smoke_test": manifest["smoke_test"],
               "cache_sha256": hashlib.sha256((root / "teacher_utility_cache.pt").read_bytes()).hexdigest(),
               "expert_checkpoint_sha256": hashlib.sha256((root / "full_refit_best.pt").read_bytes()).hexdigest(),
               "source_sha256": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                                  for name in ["compare_policies.py", "refit_mm.py"]}}
    for objective in ["expected_regret", "soft_ce"]:
        # Both policies start from identical weights, normalized inputs and folds.
        set_seed(args.seed + 123)
        args.policy_objective = objective
        destination = output / objective
        destination.mkdir()
        policy = train_policy(cache, dataset.n_predicates // 2,
                              manifest["teacher_calibration_facts"], args, destination)
        record = {"policy_selection": json.loads((destination / "policy_selection.json").read_text())}
        for split in ["valid", "test"]:
            metrics, ranks = evaluate_refit(model, policy, dataset.data[split], dataset, args)
            record[split] = metrics
            np.savez(destination / f"{split}_ranks.npz", **{key: value.numpy() for key, value in ranks.items()})
        results[objective] = record
        logging.info("objective=%s test=%s", objective, json.dumps(record["test"]))
    (output / "results.json").write_text(json.dumps(results, indent=2))
    logging.info("complete output=%s", output)


if __name__ == "__main__":
    main()
