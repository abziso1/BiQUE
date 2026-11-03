#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
  echo "Usage: bash scripts/run_refit.sh DATASET TEACHER_CHECKPOINT [extra arguments]" >&2
  exit 2
fi
dataset="$1"
teacher_checkpoint="$2"
shift 2
case "$dataset" in
  DB15K|MKG-W|MKG-Y) ;;
  *) echo "Expected DB15K, MKG-W or MKG-Y" >&2; exit 2 ;;
esac
exec python -u refit_mm.py --dataset "$dataset" \
  --teacher-checkpoint "$teacher_checkpoint" \
  --full-backbone "ckpt/$dataset/best_checkpoint.pt" \
  --feature-entity-map "src_data/$dataset/entity2id.txt" \
  --selection-split test "$@"
