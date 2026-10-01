# Reproducing the CoRL experiments

Use the fixed splits and configuration files in this release. The recorded experiments are separate tasks with distinct execution interfaces; their scores should be compared within each task.

## 1. Dependency prediction

Follow the training and evaluation commands in the [README](README.md). The release contains 27 models for nine learned configurations at seeds 7/11/23, plus two additional Ours models at seeds 31/43 for ordered retrieval. Geometry, center-distance and random-score baselines have configs and validation calibrations but no learned checkpoint.

Install the dependency versions recorded in `environment-reference.json` when matching the original environment. A matching CUDA-enabled PyTorch build is needed for GPU training. Prediction, metric and graph operations can also run on CPU.

Each evaluation uses the entire fixed test split. All label thresholds come from validation. Average precision uses the tie-aware scikit-learn definition; the public metrics module incorporates the exact replacement that was active in the experiment wrapper. The release does not use a new architecture or new weights.

## 2. Stored-label planning

Evaluate a predictor first to produce `test_predictions.npz`, then run:

```bash
python scripts/evaluate_planning.py \
  --config configs/planning/main_ours_calibrated.json \
  --predictions outputs/ours_seed7_test/test_predictions.npz \
  --dataset data/benchmark --output outputs/planning_main
```

The config specifies planner types, open/closed-loop modes and validation-selected parameters. Run each of the 42 configs with its matching predictor (`source` in the config). For `oracle`, the command uses `y_true` in the score file. Stored-label closed loop updates the remaining-object set; it does not run fresh physics. The frozen validation tuning grid and selected configuration are in `results/planning/`.

## 3. Physical simulation

This path requires Linux, MuJoCo 3.2.6, the original object assets and licensed/native grasp backends. See [THIRD_PARTY.md](THIRD_PARTY.md). Install the simulation extras with `python -m pip install -e '.[simulation]'`. Configure `configs/simulation/provider.yaml` to point to your separately installed AnyGrasp and SuctionNet packages.

The frozen single-target and ordered configuration files are in `configs/physical/` and `configs/ordered/`. Both use 9,000 settling steps at 0.002 s, multi-contact collision support, a 0.04 m/s stability threshold and a five-intervention budget including fallback. All methods use the same numerical backend and proposal protocol. Single-target trials remove non-target objects; ordered trials relocate them within the bin.

Native provider errors must fail explicitly. They must not be converted into an empty proposal set. The common `provider_guard.py`, `numerical_backend.py` and `safe_placement_floor.py` preserve the evaluated behavior. The normal-std SuctionNet-format backend is the frozen suction configuration used for these experiments.

The release includes complete recorded episode outcomes for auditing even when the native SDK is unavailable. After supplying the external assets under `assets/` and configuring the native SDK, run a single episode with:

```bash
python scripts/verify_assets.py
python scripts/evaluate_physical.py --method ours --scene-index 0 --target obj_00 --output outputs/physical_ours_0_obj00
python scripts/evaluate_ordered.py --condition ours_floor_seed11 --index 0 --order original --output outputs/ordered_ours11_0_original
```

`benchmark/physical_roots.json` enumerates all ten physical roots and their target lists. `benchmark/ordered/orders.json` enumerates all sixteen ordered trials. Use a fresh output directory for every episode. Run from the repository root so the relative asset paths resolve. The release checks graph construction, stored planning and checkpoint inference; live native simulation additionally depends on the separately licensed SDK and exact external assets.

## 4. Recount recorded results

```bash
python scripts/recount_ordered.py
```

`results/ordered/episodes.csv` contains all 144 episodes: nine conditions × sixteen sequences. The representative seed 11 and all four other seeds are retained. `results/ordered/full_episodes.json.gz` contains the phase-level records; `results/physical/episodes.json.gz` contains all 1,280 physical single-target records. The 42 stored planner groups each include per-target outcomes and aggregated metrics.

Hardware counts come from the three existing batches of twenty trials per method in `results/hardware/counts.json`.

## Release transformations

Scientific numbers and model tensors are preserved. Public files relocate local machine paths, compact JSON, add portable command-line interfaces and embed the evaluated tie-aware AP definition. Checkpoint manifests retain the source hash, public hash and selected epoch. No training, new scene generation or performance-driven seed selection is part of packaging this release.

## Independent release check (2026-10-01)

The public repository, benchmark downloads and model archive were checked in a fresh Linux Python 3.11 environment with PyTorch 2.11.0 on CPU and the dependency versions in `environment-reference.json`. All 39,455 benchmark files passed their published checksums, all 29 checkpoints loaded strictly and produced finite predictions, and all seven release tests passed.

All 36 prediction configurations were evaluated on the full 1,945-target test split. The 98 numeric cells in the five prediction tables matched at the published precision. All 42 stored-label planning configurations were rerun: their aggregate metrics, per-target retrieval outcomes and removal sequences matched across 103,085 records. Execution times were excluded from this comparison.

CPU/GPU rounding produced a maximum AP difference of 0.000002416. Edge-MLP seed 7 and generic OO/GG seed 11 each had one sufficient-label edge cross its threshold. The no-grasp-score planning ablation selected a different grasp on one target whose candidate scores differed by approximately 0.0000000002; its retrieval outcome and removal sequence were unchanged. Original published results are retained.

The 1,280 recorded physical episodes, 144 ordered episodes and existing hardware counts were also recounted. This check did not retrain the models or rerun native physical simulation or robot trials; native simulation requires the external assets and licensed backends listed above. The evaluation command now exports minimal-blocker-set metrics and target/edge counts alongside edge metrics.
