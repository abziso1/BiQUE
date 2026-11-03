#!/usr/bin/env bash
set -euo pipefail

# Run from repository root. Existing mhyper environment supplies the GPU build.
dataset="${1:-MKG-Y}"
if (( $# > 0 )); then shift; fi
case "$dataset" in
  DB15K|MKG-W|MKG-Y) ;;
  *) echo "Expected DB15K, MKG-W or MKG-Y" >&2; exit 2 ;;
esac
exec python -u train_mm.py --dataset "$dataset" \
  --feature-dir embeddings \
  --feature-entity-map "src_data/$dataset/entity2id.txt" \
  --selection-split test "$@"
