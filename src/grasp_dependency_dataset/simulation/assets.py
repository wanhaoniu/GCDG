"""Asset catalog helpers for CEPB-ready scene generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from grasp_dependency_dataset.common.io import load_yaml
from grasp_dependency_dataset.common.types import ObjectSpec


@dataclass(frozen=True)
class AssetCatalog:
    """Collection of scene object specifications."""

    name: str
    object_specs: tuple[ObjectSpec, ...]

    def sampleable_specs(self) -> tuple[ObjectSpec, ...]:
        """Return objects available for scene sampling."""

        return self.object_specs


def _tuple_rgba(values: list[float] | tuple[float, ...] | None) -> tuple[float, float, float, float]:
    rgba = values or [0.5, 0.5, 0.5, 1.0]
    if len(rgba) != 4:
        raise ValueError("RGBA values must contain exactly 4 floats.")
    return tuple(float(v) for v in rgba)


def _build_demo_catalog() -> AssetCatalog:
    """Create primitive demo proxies used before CEPB assets are imported."""

    demo_specs = (
        ObjectSpec("demo_box_small", "box", (0.018, 0.020, 0.025), 0.10, (0.82, 0.52, 0.40, 1.0)),
        ObjectSpec("demo_box_tall", "box", (0.016, 0.018, 0.035), 0.11, (0.48, 0.67, 0.77, 1.0)),
        ObjectSpec("demo_cylinder_short", "cylinder", (0.018, 0.020), 0.08, (0.62, 0.75, 0.44, 1.0)),
        ObjectSpec("demo_cylinder_tall", "cylinder", (0.015, 0.032), 0.09, (0.83, 0.72, 0.39, 1.0)),
        ObjectSpec("demo_capsule", "capsule", (0.014, 0.020), 0.07, (0.71, 0.47, 0.73, 1.0)),
        ObjectSpec("demo_sphere", "sphere", (0.020,), 0.06, (0.93, 0.53, 0.58, 1.0)),
        ObjectSpec("demo_box_flat", "box", (0.026, 0.018, 0.012), 0.09, (0.36, 0.56, 0.83, 1.0)),
        ObjectSpec("demo_box_wide", "box", (0.028, 0.015, 0.018), 0.09, (0.52, 0.81, 0.71, 1.0)),
        ObjectSpec("demo_cylinder_wide", "cylinder", (0.022, 0.015), 0.08, (0.90, 0.63, 0.39, 1.0)),
        ObjectSpec("demo_capsule_long", "capsule", (0.012, 0.028), 0.07, (0.64, 0.58, 0.78, 1.0)),
    )
    return AssetCatalog(name="demo_primitives", object_specs=demo_specs)


def load_asset_catalog(object_source: str, catalog_manifest: str | Path) -> AssetCatalog:
    """Load a CEPB-ready catalog or fall back to the demo primitives."""

    if object_source == "demo_primitives":
        return _build_demo_catalog()

    manifest_path = Path(catalog_manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Catalog manifest '{manifest_path}' does not exist for source '{object_source}'."
        )

    payload = load_yaml(manifest_path)
    object_specs = []
    for raw in payload.get("objects", []):
        object_specs.append(
            ObjectSpec(
                name=str(raw["name"]),
                primitive_type=str(raw["primitive_type"]),
                size=tuple(float(v) for v in raw["size"]),
                mass=float(raw["mass"]),
                rgba=_tuple_rgba(raw.get("rgba")),
                mesh_path=(
                    None
                    if not raw.get("mesh_path")
                    else str((manifest_path.parent / raw["mesh_path"]).resolve())
                    if not Path(str(raw["mesh_path"])).is_absolute()
                    else str(raw["mesh_path"])
                ),
                collision_mesh_paths=(
                    None
                    if not raw.get("collision_mesh_paths")
                    else tuple(
                        str((manifest_path.parent / path_value).resolve())
                        if not Path(str(path_value)).is_absolute()
                        else str(path_value)
                        for path_value in raw["collision_mesh_paths"]
                    )
                ),
                texture_path=(
                    None
                    if not raw.get("texture_path")
                    else str((manifest_path.parent / raw["texture_path"]).resolve())
                    if not Path(str(raw["texture_path"])).is_absolute()
                    else str(raw["texture_path"])
                ),
                mesh_scale=(
                    None
                    if not raw.get("mesh_scale")
                    else tuple(float(v) for v in raw["mesh_scale"])
                ),
                mesh_offset=(
                    None
                    if not raw.get("mesh_offset")
                    else tuple(float(v) for v in raw["mesh_offset"])
                ),
                proxy_quaternion_wxyz=(
                    None
                    if not raw.get("proxy_quaternion_wxyz")
                    else tuple(float(v) for v in raw["proxy_quaternion_wxyz"])
                ),
                notes=raw.get("notes"),
            )
        )

    if not object_specs:
        raise ValueError(f"Catalog manifest '{manifest_path}' did not define any objects.")

    return AssetCatalog(name=object_source, object_specs=tuple(object_specs))
