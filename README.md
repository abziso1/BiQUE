# BiQUE and selective multimodal residual correction

The original structural baseline runs through `main.py` / `run.sh`.

The multimodal extension adds structure-confusion correction training and
held-out training-utility supervision for a query router with an abstention action.
See [implementation and experiment protocol](MMKG_IMPLEMENTATION.md).

With the existing `mhyper` environment active:

```bash
python -m unittest -v test_mm.py
bash scripts/run_mm.sh MKG-Y
```

Input features are read from `embeddings/`; original dataset mappings are in
`src_data/<dataset>/entity2id.txt`. Model selection uses the test split by default
at the user's request. Run manifests record this explicitly. New checkpoints and
metrics are stored in a separate timestamped `logs/mmkgc/` directory.

To retain the existing full-TRAIN structural backbone after training a held-out
teacher, run the full-data refit and direct-utility routing experiment:

```bash
bash scripts/run_refit.sh MKG-Y \
  logs/mmkgc/MKG-Y/seed2026_first/router_best.pt
```

This generates utility labels with the held-out teacher first, then loads the
existing `ckpt/MKG-Y/best_checkpoint.pt` and refits modality experts on all TRAIN
facts. See the full-data refit section of the implementation notes for its
single-holdout transfer assumptions and matched routing controls.
