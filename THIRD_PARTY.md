# Third-party components and object assets

The MIT license covers self-owned code. The dataset's CC BY 4.0 grant covers original scene configurations, annotations and result records. Neither grant changes the terms of third-party software or assets.

| Component | Use in GCDG | Distribution |
|---|---|---|
| PyTorch, NumPy, scikit-learn, SciPy, Pillow, PyYAML | Prediction, training and metrics | Install from their upstream distributions; retain their licenses. |
| MuJoCo 3.2.6 | Physical validation and closed-loop simulation | Install the upstream package. |
| AnyGrasp SDK | Parallel-jaw grasp proposals | Obtain the SDK, model and a license for your machine from [the official project](https://github.com/graspnet/anygrasp_sdk). Native binaries, weights and machine licenses are not included here. |
| SuctionNet baseline utilities | Suction-format normal-std proposal generation in the frozen configuration | Obtain from [the official repository](https://github.com/graspnet/suctionnet-baseline); preserve its original terms. |
| CoACD 1.0.10 / trimesh | Convex collision decomposition and mesh processing | Install upstream packages. |
| CEPB-style object meshes and textures | Rendering and physics of dense scenes | External asset files are not included in the annotation release. Original sources and per-file identities are tracked separately. |

## AnyGrasp

Follow the upstream [license registration instructions](https://github.com/graspnet/anygrasp_sdk/tree/main/license_registration). Install a binary matching your Python/CUDA environment. The evaluated wrapper uses the `AnyGrasp` interface. Newer SDKs may expose a different interface; using one does not by itself reproduce the frozen proposal backend. Configure paths explicitly and test the provider before running a benchmark rollout.

## Object models

The object names and collection derive from the [Cluttered Environment Picking Benchmark](http://cepbbenchmark.eu/), described by [D'Avella et al.](https://doi.org/10.1109/MRA.2023.3310861) and the [CEPB dataset paper](https://doi.org/10.3389/frobt.2024.1222465). The collection includes assets originating in several object datasets.

The local experiment used normalized visual meshes, textures and CoACD collision pieces. Exact asset identities and geometric transformations matter for physical reproduction. Public annotations preserve those transformations. Do not substitute an approximate shape and call the resulting physical result a reproduction of the released benchmark.

Redistribution terms for the exact local mesh copies have not been established by the collection's paper license. They are therefore acquired separately. The downloadable annotation benchmark remains usable for graph construction, dependency learning and stored-label planning without loading these mesh files.

## Baselines

The released G2N2-style, Mechanical Search and X-Ray comparisons are the adapters used in the paper's common input/action interface. They are not redistributions of the complete upstream systems. Their definitions and citations are in the paper and frozen configurations.
