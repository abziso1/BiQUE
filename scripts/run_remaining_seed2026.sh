#!/usr/bin/env bash
set -euo pipefail

# Same primary settings as the completed MKG-Y seed2026 experiment.
# Run from the repository root with the existing mhyper environment active.
suite_dir="${1:-logs/mmkgc/suites/remaining_seed2026}"
if [[ -e "$suite_dir" ]]; then
  echo "Refusing to overwrite suite directory: $suite_dir" >&2
  exit 2
fi
for dataset_name in MKG-W DB15K; do
  if [[ -e "logs/mmkgc/$dataset_name/seed2026_first" || -e "logs/mmkgc/$dataset_name/full_refit_seed2026" ]]; then
    echo "Existing run directory for $dataset_name; inspect before starting." >&2
    exit 2
  fi
  test -f "ckpt/$dataset_name/best_checkpoint.pt"
  test -f "embeddings/$dataset_name-visual.pth"
  test -f "embeddings/$dataset_name-textual.pth"
done
mkdir -p "$suite_dir"
for dataset_name in MKG-W DB15K; do
  printf '%s %s teacher starting\n' "$(date -Is)" "$dataset_name"
  python -u train_mm.py --dataset "$dataset_name" \
    --feature-dir embeddings \
    --feature-entity-map "src_data/$dataset_name/entity2id.txt" \
    --device cuda --selection-split test --seed 2026 \
    --rank 128 --modal-rank 32 --batch-size 500 --eval-batch-size 256 \
    --structure-epochs 300 --expert-epochs 100 --router-epochs 50 \
    --valid-every 5 --patience 8 \
    --output "logs/mmkgc/$dataset_name/seed2026_first" \
    > "$suite_dir/$dataset_name-teacher.console.log" 2>&1
  printf '%s %s teacher complete; full refit starting\n' "$(date -Is)" "$dataset_name"
  python -u refit_mm.py --dataset "$dataset_name" \
    --teacher-checkpoint "logs/mmkgc/$dataset_name/seed2026_first/router_best.pt" \
    --full-backbone "ckpt/$dataset_name/best_checkpoint.pt" \
    --feature-entity-map "src_data/$dataset_name/entity2id.txt" \
    --device cuda --selection-split test --seed 2026 \
    --batch-size 500 --eval-batch-size 256 \
    --refit-epochs 40 --policy-epochs 200 --policy-objective expected_regret \
    --valid-every 5 --patience 4 \
    --output "logs/mmkgc/$dataset_name/full_refit_seed2026" \
    > "$suite_dir/$dataset_name-refit.console.log" 2>&1
  printf '%s %s full refit complete\n' "$(date -Is)" "$dataset_name"
done
printf '%s MKG-W and DB15K seed2026 complete\n' "$(date -Is)"
