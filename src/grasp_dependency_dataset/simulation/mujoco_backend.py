"""MuJoCo-based clutter settling backend."""

from __future__ import annotations

import math

from grasp_dependency_dataset.common.types import Pose

from .backend import SettledSceneState, SimulationBackend

try:
    import mujoco
except ImportError as exc:  # pragma: no cover - import error exercised only when env is missing
    mujoco = None
    _MUJOCO_IMPORT_ERROR = exc
else:
    _MUJOCO_IMPORT_ERROR = None


class MujocoBackend(SimulationBackend):
    """Settle bin clutter scenes with MuJoCo physics."""

    def __init__(self) -> None:
        if mujoco is None:
            raise RuntimeError(
                "MuJoCo is not installed in the active environment."
            ) from _MUJOCO_IMPORT_ERROR

    def settle_scene(
        self,
        scene_xml: str,
        body_names: list[str],
        settle_steps: int,
        stability_window: int,
        velocity_threshold: float,
    ) -> SettledSceneState:
        """Simulate the scene and return the final body poses."""

        model = mujoco.MjModel.from_xml_string(scene_xml)
        data = mujoco.MjData(model)

        velocity_history: list[float] = []
        for _ in range(settle_steps):
            mujoco.mj_step(model, data)
            velocity_history.append(self._max_body_velocity(model, data, body_names))

        recent = velocity_history[-stability_window:] if stability_window > 0 else velocity_history
        recent_max = max(recent) if recent else math.inf
        stable = recent_max <= velocity_threshold

        body_poses = {}
        for body_name in body_names:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            body_poses[body_name] = Pose(
                position=tuple(float(v) for v in data.xpos[body_id]),
                quaternion_wxyz=tuple(float(v) for v in data.xquat[body_id]),
            )

        return SettledSceneState(
            body_poses=body_poses,
            stable=stable,
            metadata={
                "recent_max_velocity": recent_max,
                "settle_steps": settle_steps,
                "stability_window": stability_window,
                "velocity_threshold": velocity_threshold,
                "timestep": model.opt.timestep,
            },
        )

    @staticmethod
    def _max_body_velocity(model: "mujoco.MjModel", data: "mujoco.MjData", body_names: list[str]) -> float:
        max_velocity = 0.0
        for body_name in body_names:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{body_name}_joint")
            dof_addr = model.jnt_dofadr[joint_id]
            linear_velocity = data.qvel[dof_addr : dof_addr + 3]
            speed = float(sum(component * component for component in linear_velocity) ** 0.5)
            max_velocity = max(max_velocity, speed)
        return max_velocity
