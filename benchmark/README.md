# GCDG dense-scene benchmark

The benchmark supports grasp-conditioned dependency prediction, stored-label minimal-intervention planning and physical target retrieval. Its 500 dense scenes each contain 25–27 objects. Training, validation and test are separated by scene, not by target.

| Split | Scenes | Targets | Parallel-jaw grasps | Suction grasps |
|---|---:|---:|---:|---:|
| Train | 350 | 9,090 | 112,521 | 106,428 |
| Validation | 75 | 1,950 | 24,462 | 22,792 |
| Test | 75 | 1,945 | 24,321 | 22,704 |
| Total | 500 | 12,985 | 161,304 | 151,924 |

## Download and verify

Use [release v1.0.0](https://huggingface.co/datasets/WanhaoX25/GCDG-Benchmark), or run:

```bash
python scripts/download_benchmark.py --split all --output data --extract
```

The download script checks SHA-256 before extraction. Shards contain up to 50 scenes and can be extracted into the same directory. The full uncompressed source annotation set is approximately 17.6 GB; released JSON is compacted. Download only the test shards to evaluate an existing checkpoint. Download train and validation as well to reproduce training.

## Schema

```text
benchmark/
  splits.json
  class_map.json
  scenes/scene_<split>_<id>/
    scene.json
    targets/obj_<id>/
      manifest.json
      proposals.json
      labels.json
```

`scene.json` stores object identities, physical specifications and poses. `manifest.json` provides observation-derived graph support and artifact metadata. `proposals.json` stores parallel-jaw and suction candidates with their pose and modality. `labels.json` stores staged validation outcomes, single-object intervention labels and bounded minimal blocker sets.

One sample is one scene–target pair, with a 36-dimensional feature vector per non-target object, a 28-dimensional vector per candidate grasp and a 22-dimensional vector per object–grasp edge. The predictor outputs four logits per edge:

| Output | Meaning | Stored annotation |
|---|---|---|
| `dep_progress_any` | Approach or lift progress | Union of the two stage labels |
| `dep_sufficient` | Removing this object alone makes the fixed grasp feasible | `dep_any` |
| `dep_approach` | Approach-stage progress after removal | `dep_collision_approach` |
| `dep_lift` | Lift-stage progress after removal | `dep_collision_lift` |

The all-zero vector is the no-dependency state. The supervision mask identifies actual modeled edges. Targets with no proposals remain in the benchmark; they are not discarded. A target with no proposals or no solution within the depth-three blocker search is not treated as a zero-blocker success.

The fixed feature settings disable privileged scene geometry. Ground-truth poses and intervention outcomes are supplied for labeling, physics and oracle evaluation; they must not be introduced as learned-predictor inputs. Exact feature names are in `src/grasp_dependency_dataset/hetero_gnn/graph_features.py`.

## Evaluation tasks

1. **Prediction:** 75 test scenes / 1,945 targets; per-label AP, validation-calibrated F1 and blocker IoU. Report mean and sample SD over seeds 7/11/23. AP groups tied scores using scikit-learn's definition.
2. **Stored-label planning:** the same 75 test scenes / 1,945 targets, with 42 frozen configurations. Validation selection uses success minus 0.08 times removals. Keep the oracle separate from learned policies.
3. **Single-target physical retrieval:** the first ten specified test roots, 256 targets, five methods, 1,280 episodes. Reset to the initial scene for each target. Allow at most five non-target removals, including shared fallback.
4. **Ordered retrieval:** four additional development scenes with 15 objects, ten targets per order and four orders per scene. Each of nine conditions has 16 sequences / 160 target trials. These are repeated orders of four scenes. Continue after ordinary grasp failures; stop on invalid physical states and retain unattempted targets in the denominator.
5. **Hardware:** three existing batches of twenty trials per method. The release reaggregates recorded counts; it does not add robot trials.

See [protocol.json](protocol.json) and [all tables](../results/tables/). Ordered retrieval uses all five Ours seeds. The accepted representative seed 11 was selected after observing the original-order development results and fixed before the additional orders.

## Assets and portability

The annotation release contains the stored graph inputs needed to train and evaluate dependency predictors without running AnyGrasp or MuJoCo. It does not contain third-party mesh/texture files or native grasp SDK binaries. See [third-party dependencies](../THIRD_PARTY.md) for asset acquisition and simulation setup. Public JSON relocates machine-specific paths while preserving all numeric values. The source and released SHA-256 values are recorded in the release manifest.

The objects derive from the CEPB object collection. The local mesh asset provenance and redistribution terms need to be matched to their original providers before redistribution; a paper's open-access license does not itself license all underlying mesh files. The release therefore keeps external asset acquisition separate from the self-owned annotations.

## License

Original annotations, scene configurations and results: [CC BY 4.0](LICENSE.md). Cite GCDG and acknowledge the original object collections when using their assets. Code: [MIT](../LICENSE).
