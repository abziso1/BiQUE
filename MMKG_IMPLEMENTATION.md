# Selective multimodal residual correction

This implementation extends the SACRF idea with two mechanisms:

1. Auxiliary residuals learn to resolve the frozen structure model's top-scoring
   incorrect candidates, in addition to full-entity cross-entropy.
2. A query router learns measured reciprocal-rank utility from held-out training
   facts, with three actions: keep structure, visual correction, text correction.

This is an experimental implementation, not evidence that the proposed mechanisms
improve MRR or that they constitute an independently novel contribution.

## Files and dependencies

- `mm_model.py`: frozen input features, separate BiQUE experts, candidate missing
  masks, query-conditioned router, and SACRF-style score calibration.
- `mm_data.py`: split audits, feature hashes, reciprocal splitting, hard-candidate
  selection and filtered ranks.
- `train_mm.py`: three stages, model selection, checkpointing, evaluation.
- `refit_mm.py`: held-out teacher utility transfer, direct expected-utility
  policy, and full-TRAIN deployment-backbone/expert refit.
- `test_mm.py`: semantic and leakage-prevention invariants.
- `requirements-mm.txt`: dependencies verified with the existing `mhyper` Conda
  environment (PyTorch 2.9.0+cu130). No dependency installation was necessary.

The old `main.py` continues to run the original pure structural model. Its
test-selected protocol is retained at the user's explicit request.

## Data and selection protocol

Default `--selection-split test` intentionally uses test MRR for checkpoint
selection, as requested. Manifests identify this as
`heldout_gain_v1_select_test`, with `independent_test: false`. Results from this
protocol must not be described as an independent held-out test evaluation.
`--selection-split valid` is available for a conventional validation protocol.

Original TRAIN facts are deduplicated and split deterministically into 90% fit
and 10% calibration **before** creating reciprocal triples. Neither structure nor
modality experts train on calibration facts. All router parameters train only on
calibration facts after both experts are frozen, using frozen predictions for
utility labels. No valid/test labels are used in gradient updates or utility
targets. Selecting checkpoints using test metrics remains intentional.

The reported `structure` baseline uses exactly the same 90% fit partition as the
multimodal backbone. It is not directly equivalent to a BiQUE trained on 100% of
TRAIN. The saved `training_partition.npz` makes the difference inspectable.

Hard candidates exclude known FIT positives while training experts; router
training and utility targets filter known original TRAIN positives. Evaluation
uses the existing standard all-split filtered dictionaries. There is no shortlist
restriction during inference: every entity is scored.

`--feature-entity-map` must identify the entity-ID mapping associated with the
downloaded features. The runner checks equality with `data/<dataset>/ent_id`.
Shape/hash checks do not independently prove the feature producer's row order;
the supplied manifest must genuinely describe the feature source. The current
source mappings match the inspected AdaMF/MMRNS-format mappings.

DB15K's current validation split contains duplicates and overlaps. These are
recorded in `data_audit.json`; raw/processed data are never silently changed.
Test-selected runs proceed with a warning. Valid-selected runs fail this audit.

## Run

From the repository root, with the `mhyper` environment active:

```bash
python -m unittest -v test_mm.py
python -u train_mm.py --dataset MKG-Y \
  --feature-dir embeddings \
  --feature-entity-map src_data/MKG-Y/entity2id.txt \
  --selection-split test
```

Replace both occurrences of `MKG-Y` for another dataset. Default runs use rank
128, modality rank 32, alpha 0.2, 300/100/50 epochs for structure/experts/router,
and patience of eight selection checks. All stages restore their best checkpoint.
Stage-2 selection averages the MRR of the two single-correction variants; it does
not use the ground-truth oracle to select examples or predictions.

CPU smoke run, with intentionally truncated training/evaluation:

```bash
python -u train_mm.py --dataset MKG-Y \
  --feature-entity-map src_data/MKG-Y/entity2id.txt --device cpu \
  --rank 4 --modal-rank 2 --structure-epochs 1 --expert-epochs 1 \
  --router-epochs 1 --valid-every 1 --max-batches 2 --eval-limit 32 \
  --batch-size 16 --eval-batch-size 16
```

Smoke artifacts are marked explicitly and cannot initialize a full research run.
Output directories default to `logs/mmkgc/<dataset>/<timestamp>/`, separate from
the user's original checkpoints. Nonempty output directories are never overwritten.

To warm-start the next stage, use `--start-from experts --checkpoint <structure_best.pt>`
or `--start-from router --checkpoint <experts_best.pt>`. Dataset hashes, feature
hashes, partitions, selection protocol and model configuration must match.
These are stage warm starts, not exact optimizer/RNG resumes mid-stage.

## Ablations and measurements

- `--correction-weight 0`: remove structural-confusion loss.
- `--router-supervision 0`: query router trained by fused CE/correction only.
- `--fusion fixed`: equal auxiliary weights, no router learning.
- `--fusion sacrf`: relation-only sigmoid gates instead of the query router.

The last option is a matched-protocol fusion ablation, **not** a reproduction of
the original SACRF training recipe: it still uses this runner's held-out split,
frozen structure and expert phases. Original SACRF should also be compared with
its own documented recipe before making research claims.

`results.json` contains MRR, Hits@1/3/10, positive reciprocal-rank Gain and Harm,
mean action probabilities and an intervention-fraction/MRR curve. For the same
queries, `Gain - Harm = fused MRR - structure MRR`. The curve is descriptive,
not an independently tuned deployment policy. Per-query ranks are saved as NPZ.
`oracle_diagnostic_MRR` uses answers to choose the best prediction per query and
must never be reported as deployable performance.

Utility targets compare each single-modal correction with structure on calibration
queries. A small action cost favors abstention for negligible gains. Queries with
no positive corrected utility receive an explicit keep-structure target. On other
queries, temperature-softened utilities supervise the router. Feature availability,
relation, structure embedding, prediction entropy/margins and branch agreement are
router inputs; the true answer is never an input to inference.

Follow-up research checks: at least three seeds, randomized-feature controls,
pure BiQUE with the same hard-candidate loss, matched parameter/runtime reporting,
and comparison to the full-TRAIN backbone. These are not replaced by smoke tests.

## First completed run (2026-11-03)

Artifacts: `logs/mmkgc/MKG-Y/seed2026_first/`. This is a single-seed exploratory
run using test-selected checkpoints, not a conventional independent test estimate.
Project-local features were compared byte-for-byte against the previously inspected
AdaMF/MMRNS-format copies; all six files match.

The run used 19,179 fit facts and 2,131 router-calibration facts, batch size 500,
structure rank 128, modality rank 32. Best epochs were structure 70, experts 15,
router 5. Total elapsed time was 457 seconds; peak PyTorch CUDA allocated memory
was 857 MiB (this excludes other processes, CUDA context and reserved memory).

| Prediction | Test-selected MRR |
| --- | ---: |
| Matched 90%-TRAIN structure | 0.351103 |
| Structure + visual correction | 0.351039 |
| Structure + textual correction | 0.356758 |
| Gain-supervised fused correction | 0.354741 |

Fused Gain = 0.004589, Harm = 0.000951; the net MRR change is +0.003637.
The fused model has less harm than full textual correction but lower net MRR.
It has **not** surpassed the user's full-TRAIN BiQUE baseline. Holding out 10% of
this sparse graph imposes a substantial structure-performance cost, and the router
does not yet beat the simple textual-correction control. Further experiments
should address calibration-data efficiency and compare against properly generated
out-of-fold utility supervision before claiming a stronger model. DB15K and MKG-W
passed feature/manifest audits; their full training runs have not been launched.

## Full-data refit and direct utility routing

`refit_mm.py` addresses the permanent 10%-TRAIN withholding cost of the first
prototype. Its order of operations is significant:

1. Restore the original held-out teacher (90% backbone and experts).
2. Generate and freeze action-utility labels on its untouched 10% calibration
   facts, using only original TRAIN positives for filtered ranks.
3. Train an independent query policy from those labels. Its 80/20 training/internal
   selection split groups each original fact with its reciprocal. Feature
   normalization is fitted only on policy-training examples.
4. Replace the deployment backbone with the existing full-TRAIN BiQUE checkpoint.
   Keep that backbone frozen and refit the auxiliary experts using all TRAIN.
5. Compare all variants using identical deployment backbone and expert weights.

This is **single-holdout teacher transfer**, not full K-fold cross-fitting. The
policy uses teacher-generated held-out labels; no new labels are generated from
the in-sample full-TRAIN deployment model. There can still be distribution shift
between the teacher and full-data deployment experts/prediction statistics.
Teacher/source hashes and the explicit operation order are recorded. A legacy
backbone checkpoint lacks dataset hashes; dataset-name metadata and tensor shapes
are checked, but its exact historical dataset version cannot be proved from that
checkpoint alone. The original checkpoint is read, never overwritten.

The new policy has four discrete actions: no correction, full visual residual,
full textual residual, and half of each residual. The residual scale is still
the teacher's original alpha. Unavailable actions are masked. Its input contains
query prediction statistics, branch agreement, modality availability and learned
relation features; it does not use the true candidate as a feature or depend on
the old teacher's structural-embedding coordinate system.

Instead of soft classification targets that frequently favor abstention, train
the policy to minimize expected action regret:

`mean_q sum_a p(a|q) [max_b u(q,b) - u(q,a)]`.

Utilities are measured reciprocal-rank changes minus the correction action cost.
Hard action selection is evaluated on the internal 20% calibration split to
choose the policy checkpoint. The result also reports soft-mixture inference as
a separate control; expected utility over actions and the utility of mixing
their score vectors are different quantities.

Deployment expert refit includes epoch zero, so simply transferring the existing
teacher experts is an explicit control. Expert checkpoints are selected using the
same average visual/textual single-correction MRR as the first run. Routing variants
are evaluated on the SAME selected expert checkpoint. Thus each router is not
individually optimized over deployment expert checkpoints.

```bash
python -u refit_mm.py --dataset MKG-Y \
  --teacher-checkpoint logs/mmkgc/MKG-Y/seed2026_first/router_best.pt \
  --full-backbone ckpt/MKG-Y/best_checkpoint.pt \
  --feature-entity-map src_data/MKG-Y/entity2id.txt \
  --selection-split test
```

Artifacts include `teacher_utility_cache.pt`, `utility_policy.pt`,
`full_refit_best.pt`, `policy_selection.json`, all checkpoint-selection evaluations,
per-query ranks and final results. Source/feature/teacher/full-backbone hashes are
recorded. Model selection still uses TEST as explicitly requested; neither this
refit nor the new policy makes that an independent test estimate.

## Completed full-data refit (2026-11-03)

Run: `logs/mmkgc/MKG-Y/full_refit_seed2026/`. The deployment backbone is exactly
the existing full-TRAIN checkpoint, with 21,310 original training facts and MRR
0.3891438. The old 90% teacher is used only to generate utility supervision before
full-data refit. The selected expert refit checkpoint is epoch 5; the policy was
selected at epoch 40 on the internal calibration holdout. Runtime was 146 seconds.

| Prediction on the same selected experts | Test-selected MRR |
| --- | ---: |
| Full-TRAIN structure | 0.389144 |
| Visual correction | 0.389646 |
| Textual correction | 0.392990 |
| Fixed half-visual, half-textual correction | 0.393095 |
| Original prototype router transferred to full backbone | 0.392820 |
| Direct-utility policy, hard action | 0.394386 |
| Direct-utility policy, soft score mixture | 0.394944 |

The soft policy's Gain is 0.011108 and Harm is 0.005308, for a net MRR increase
of 0.005800 (+0.580 percentage points) over the identical structural checkpoint.
Hard routing has a net gain of 0.005242. The new method has larger absolute Harm
than the conservative old router, as well as larger Gain; this is improved net
ranking performance, not a claim of uniformly lower intervention risk. Its hard
action frequencies are 23.0% keep, 33.5% visual, 43.5% textual, and 0% joint-half.

### Matched objective control

`compare_policies.py` restores that exact selected expert checkpoint and trains
both policies on the same frozen teacher utility cache. Architecture, four actions,
initialization seed 2149, normalization inputs and calibration split are identical.
Only the policy loss differs. The soft-CE control uses temperature-softened action
utilities and a keep-structure label whenever no action has positive utility.
Both policy checkpoints are selected using internal calibration utility, not by
selecting the better policy epoch on the evaluation split.

```bash
python -u compare_policies.py \
  --run-dir logs/mmkgc/MKG-Y/full_refit_seed2026
```

Artifacts: `logs/mmkgc/MKG-Y/policy_objective_control_seed2026/`.

| Matched objective | Hard-action MRR | Soft-mixture MRR |
| --- | ---: | ---: |
| Direct expected regret | 0.393686 | 0.394020 |
| Soft action classification | 0.389182 | 0.391752 |

The classification policy chooses keep-structure on 99.42% of evaluation queries.
This single matched initialization supports the diagnosis that frequent no-gain
classification targets can suppress intervention excessively. It does not prove
that every classification objective or class-balancing scheme will fail.

The first full-refit policy and this matched control use different policy
initializations; their numbers are not substituted for each other. All comparisons
above still use one structural/expert training seed and test-selected expert
checkpoints. Multi-seed replication, full K-fold utility generation, other datasets,
feature-randomization controls and a hard-negative-only structural control remain
necessary before a broad research claim. The implemented teacher transfer removes
permanent data withholding, but does not remove teacher-to-deployment shift.

## Other datasets, same code and seed 2026 (2026-11-03)

MKG-W and DB15K completed the same primary teacher/full-refit pipeline. No
dataset-specific hyperparameter tuning was performed. Teacher settings were rank
128, modality rank 32, batch 500, up to 300/100/50 epochs, alpha 0.2, and seed 2026.
Full refit used batch 500, learning rate 0.0003, up to 40 expert epochs, 200 policy
epochs, expected-regret policy loss, and the same seed. Both runs used TEST for
checkpoint selection as requested. Their recorded model/training source hashes
are identical; original full-backbone checkpoint hashes remain unchanged.

| Dataset | Full structure | Text correction | Fixed joint-half | Old router | Utility hard | Utility soft |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DB15K | 0.414211 | 0.418376 | 0.417965 | 0.417690 | 0.417958 | 0.418847 |
| MKG-W | 0.376883 | 0.391546 | 0.385403 | 0.385625 | 0.390832 | 0.387745 |
| MKG-Y (previous run) | 0.389144 | 0.392990 | 0.393095 | 0.392820 | 0.394386 | 0.394944 |

Utility-soft net gains over the identical full structural checkpoint are +0.464,
+1.086 and +0.580 percentage points for DB15K, MKG-W and MKG-Y respectively.
Utility-hard gains are +0.375, +1.395 and +0.524 percentage points. Both inference
modes are reported consistently; the better mode is not chosen separately for
each dataset to form a single headline method.

Both new runs selected expert-refit epoch 10 using the predefined average
single-expert MRR criterion. All routing variants in each row use that same expert
checkpoint. In DB15K, a later epoch had a higher observed utility-soft MRR but was
not selected by this criterion; the table uses final `results.json`, not the
largest value observed for each individual inference variant.

The utility policy beats structure in all three single-seed runs. Text-only
correction remains stronger on MKG-W, and also beats hard routing on DB15K;
soft routing slightly beats text on DB15K. Thus this experiment does not establish
that routing consistently outperforms a strong text-correction baseline.

Artifacts:

- `logs/mmkgc/MKG-W/seed2026_first/` and `full_refit_seed2026/`.
- `logs/mmkgc/DB15K/seed2026_first/` and `full_refit_seed2026/`.
- `logs/mmkgc/suites/remaining_seed2026/`: console logs and combined summary.
- `scripts/run_remaining_seed2026.sh`: exact serial orchestration settings.

The full suite exited successfully. MKG-W teacher/full-refit runtimes were about
1082/362 seconds; DB15K about 3117/950 seconds. These are complete, untruncated
runs, not smoke tests. The DB15K validation overlap remains recorded in the audit
and data files were not changed. Multi-seed replication and the remaining
feature/structural controls are still outstanding research checks.
