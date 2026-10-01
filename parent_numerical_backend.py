"""Common numerical repair: mesh multi-contact support; synchronized final poses."""
import math
import numpy as np
import mujoco
from grasp_dependency_dataset.common.types import Pose
from grasp_dependency_dataset.simulation.backend import SettledSceneState
from grasp_dependency_dataset.simulation.mujoco_backend import MujocoBackend as OriginalBackend

POLICY=dict(version='mesh_multiccd_v4',enable='MULTICCD',integrator='unchanged Euler',timestep='unchanged0.002s',
    duration='unchanged settle_steps*timestep',stability='unchanged0.04m/s maxlinear over last250steps',
    pose_export='mj_kinematics after final integration, no additional dynamics or velocity reset',
    acceleration='bulk mj_step before final monitoring window; exact replay checked',
    material_changes=False,geometry_changes=False,model_changes=False)

def configure_model(model):
    model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_MULTICCD)

class MultiContactBackend(OriginalBackend):
    def settle_scene(self,scene_xml,body_names,settle_steps,stability_window,velocity_threshold):
        model=mujoco.MjModel.from_xml_string(scene_xml);configure_model(model)
        data=mujoco.MjData(model)
        n=int(settle_steps);window=min(n,int(stability_window)) if stability_window>0 else n
        prefix=max(0,n-window)
        if prefix:mujoco.mj_step(model,data,nstep=prefix)
        history=[]
        for _ in range(window):
            mujoco.mj_step(model,data)
            history.append(self._max_body_velocity(model,data,body_names))
        recent=max(history) if history else math.inf
        # mj_step updates qpos after computing xpos. Refresh kinematics ONLY so
        # exported poses represent the same timestamp as the checked velocities.
        mujoco.mj_kinematics(model,data)
        bodies={name:mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,name) for name in body_names}
        poses={name:Pose(tuple(float(v) for v in data.xpos[b]),tuple(float(v) for v in data.xquat[b])) for name,b in bodies.items()}
        warning_counts={k:int(data.warning[int(v)].number) for k,v in mujoco.mjtWarning.__members__.items() if k!='mjNWARNING'}
        bad_warnings=any(warning_counts.get(k,0)>0 for k in ['mjWARN_BADQPOS','mjWARN_BADQVEL','mjWARN_BADQACC'])
        finite=bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
        stable=bool(recent<=velocity_threshold and finite and not bad_warnings)
        return SettledSceneState(body_poses=poses,stable=stable,metadata=dict(recent_max_velocity=recent,settle_steps=n,
            stability_window=int(stability_window),velocity_threshold=velocity_threshold,timestep=float(model.opt.timestep),
            simulated_seconds=float(data.time),numerical_policy=POLICY,warning_counts=warning_counts,
            finite=finite,final_kinematics_synchronized=True))

def install():
    import grasp_dependency_dataset.simulation.mujoco_backend as module
    module.MujocoBackend=MultiContactBackend
