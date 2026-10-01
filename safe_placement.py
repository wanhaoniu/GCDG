"""Deterministic native-geometry placement shared by all ordered policies.

Only placement feasibility is changed. Native physical previews test candidate
poses before one real intervention; they never call grasp proposals or planners.
"""
from __future__ import annotations
import json, time
from pathlib import Path
from dataclasses import replace
import numpy as np
import mujoco
from grasp_dependency_dataset.common.types import Pose
from grasp_dependency_dataset.simulation.scene_builder import build_bin_scene_xml
from grasp_dependency_dataset.simulation.scene_generator import BinClutterSceneGenerator
from grasp_dependency_dataset.simulation.mujoco_backend import MujocoBackend

POLICY = dict(version="native_preview_placement_v2", grid=[9, 7],
              wall_margin_m=.003, min_xy_displacement_m=.025,
              downward_step_m=.001, support_probe_m=.00005,
              support_com_tolerance_m=.001, release_clearance_m=.0001,
              max_initial_penetration_m=.00025,
              orientations="current, world-yaw 90/180/270, upright, upright-yaw90",
              search_order="descending squared distance from active target, deterministic grid/orientation ties",
              max_physical_previews=32, preview_settle_steps=9000,
              final_geometry_tolerance_m=.00025, final_penetration_limit_m=.003,
              settle_selection="first native-preview stable, contained, nonpenetrating relocation; no target-grasp/planner outcome used; one actual execution",
              static_com_test="diagnostic only; infinitesimal first-contact hull is not a necessary condition for a stable settled pose")

class PlacementRejected(RuntimeError):
    def __init__(self, record):
        super().__init__('placement_no_feasible_pose')
        self.record = record

def native_model(scene, cfg):
    xml = build_bin_scene_xml(scene_objects=[(o.object_id,o.spec,o.pose) for o in scene.objects],
                             bin_config=cfg.bin, simulation_config=cfg.simulation)
    model=mujoco.MjModel.from_xml_string(xml); data=mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    return model,data

def geom_bounds(model,data,g):
    r=data.geom_xmat[g].reshape(3,3); p=data.geom_xpos[g]
    typ=int(model.geom_type[g]); size=model.geom_size[g]
    if typ==int(mujoco.mjtGeom.mjGEOM_MESH):
        mid=int(model.geom_dataid[g]); start=int(model.mesh_vertadr[mid]); n=int(model.mesh_vertnum[mid])
        pts=model.mesh_vert[start:start+n] @ r.T + p
        return pts.min(axis=0),pts.max(axis=0)
    if typ==int(mujoco.mjtGeom.mjGEOM_SPHERE): e=np.full(3,size[0])
    elif typ==int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        e=size[0]*np.sqrt(np.maximum(0,1-r[:,2]**2))+size[1]*np.abs(r[:,2])
    elif typ==int(mujoco.mjtGeom.mjGEOM_CAPSULE): e=size[0]+size[1]*np.abs(r[:,2])
    elif typ==int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): e=np.sqrt((r*r)@(size*size))
    elif typ==int(mujoco.mjtGeom.mjGEOM_BOX): e=np.abs(r)@size
    else: raise ValueError(f'unsupported native collision geometry {typ}')
    return p-e,p+e

def body_bounds(model,data,b):
    geoms=[g for g in range(model.ngeom) if model.geom_bodyid[g]==b and (model.geom_contype[g] or model.geom_conaffinity[g])]
    bounds=[geom_bounds(model,data,g) for g in geoms]
    return np.min([x[0] for x in bounds],axis=0),np.max([x[1] for x in bounds],axis=0)

def moving_contacts(model,data,body):
    out=[]
    for c in data.contact:
        b1=int(model.geom_bodyid[c.geom1]);b2=int(model.geom_bodyid[c.geom2])
        if body not in (b1,b2):continue
        normal=np.asarray(c.frame[:3])*(1 if b2==body else -1)
        out.append(dict(distance=float(c.dist),position=np.asarray(c.pos).tolist(),
                        upward=float(normal[2]),other_body=int(b1 if b2==body else b2)))
    return out

def hull(points):
    points=sorted(set(tuple(float(v) for v in p) for p in points))
    if len(points)<=1:return points
    def cross(o,a,b):return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    lo=[];hi=[]
    for p in points:
        while len(lo)>=2 and cross(lo[-2],lo[-1],p)<=0:lo.pop()
        lo.append(p)
    for p in reversed(points):
        while len(hi)>=2 and cross(hi[-2],hi[-1],p)<=0:hi.pop()
        hi.append(p)
    return lo[:-1]+hi[:-1]

def support_distance(points,com):
    poly=np.asarray(hull(points),dtype=float)
    if len(poly)==0:return float('inf')
    if len(poly)==1:return float(np.linalg.norm(poly[0]-com))
    p=np.asarray(com);best=float('inf');inside=True
    for i,a in enumerate(poly):
        b=poly[(i+1)%len(poly)];v=b-a;w=p-a
        if v[0]*w[1]-v[1]*w[0]<-1e-12:inside=False
        t=np.clip(np.dot(w,v)/max(np.dot(v,v),1e-20),0,1)
        best=min(best,float(np.linalg.norm(p-(a+t*v))))
    return 0. if len(poly)>=3 and inside else best

def physical_preview(scene, cfg, object_id, original_position):
    """Use precisely the execution backend/settings, starting from reset velocities.

    Recompile the scene through the original XML serializer, including its six
    decimal pose rounding, so planning and execution see the same physical input.
    """
    gen=BinClutterSceneGenerator(cfg,None,MujocoBackend())
    result=gen._settle_states(list(scene.objects),settle_steps=POLICY['preview_settle_steps'])
    objects=gen._apply_settled_poses(list(scene.objects),result.body_poses)
    final=replace(scene,objects=tuple(objects))
    finite=all(np.all(np.isfinite(o.pose.position)) and np.all(np.isfinite(o.pose.quaternion_wxyz)) for o in objects)
    bad=gen._out_of_bin_object_ids(objects)
    audit=dict(stable=bool(result.stable),finite=bool(finite),out_of_bin_object_ids=bad,settle=result.metadata)
    if not finite:
        audit.update(accepted=False,reason='nonfinite');return audit
    m,d=native_model(final,cfg)
    excess={}
    for o in objects:
        b=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,o.object_id);lo,hi=body_bounds(m,d,b)
        excess[o.object_id]=float(max(-scene.bin_size[0]/2-lo[0],hi[0]-scene.bin_size[0]/2,
            -scene.bin_size[1]/2-lo[1],hi[1]-scene.bin_size[1]/2,-lo[2],hi[2]-scene.bin_size[2]))
    pen=max([max(0.,-float(c.dist)) for c in d.contact],default=0.)
    displacement=float(np.linalg.norm(np.array(final.get_object(object_id).pose.position)[:2]-np.array(original_position)[:2]))
    audit.update(max_geometry_excess_m=max(excess.values()),geometry_excess_by_object=excess,
                 max_final_penetration_m=pen,actual_xy_displacement_m=displacement,
                 settled_poses={o.object_id:o.pose.to_dict() for o in objects})
    reason=('unstable' if not result.stable else 'out_of_bin' if bad else
            'geometry_outside_bin' if max(excess.values())>POLICY['final_geometry_tolerance_m'] else
            'penetration' if pen>POLICY['final_penetration_limit_m'] else
            'no_relocation_progress' if displacement<POLICY['min_xy_displacement_m'] else 'accepted')
    audit.update(accepted=reason=='accepted',reason=reason)
    return audit

class SafePlacement:
    def __init__(self,cfg,audit_dir):
        self.cfg=cfg;self.audit_dir=Path(audit_dir);self.audit_dir.mkdir(parents=True,exist_ok=True)
        self.calls=0
        self.last_preview=None

    def __call__(self,scene,object_id,target_id,relocation_index):
        started=time.monotonic();self.calls+=1
        model,data=native_model(scene,self.cfg)
        body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,object_id)
        joint=int(model.body_jntadr[body]);addr=int(model.jnt_qposadr[joint])
        obj=scene.get_object(object_id);target=scene.get_object(target_id)
        original=np.array(obj.pose.position);q0=np.array(obj.pose.quaternion_wxyz)
        orientations=[]
        for k in range(4):
            yaw=np.array([np.cos(k*np.pi/4),0,0,np.sin(k*np.pi/4)])
            q=np.empty(4);mujoco.mju_mulQuat(q,yaw,q0);orientations.append(q)
        orientations += [np.array([1.,0,0,0]),np.array([2**-.5,0,0,2**-.5])]
        unique=[]
        for q in orientations:
            q=q/np.linalg.norm(q)
            if not any(abs(float(np.dot(q,r)))>1-1e-10 for r in unique):unique.append(q)
        fixed_bounds={b:body_bounds(model,data,b) for b in range(1,model.nbody) if b!=body and model.body_geomnum[b]}
        def forward(x,y,z,q):
            data.qpos[addr:addr+3]=[x,y,z];data.qpos[addr+3:addr+7]=q
            mujoco.mj_forward(model,data)
        def penetration():
            return max([-c['distance'] for c in moving_contacts(model,data,body)],default=0.)
        ranked=[];margin=POLICY['wall_margin_m']
        for qi,q in enumerate(unique):
            forward(0,0,0,q);lo,hi=body_bounds(model,data,body)
            xmin=-scene.bin_size[0]/2+margin-lo[0];xmax=scene.bin_size[0]/2-margin-hi[0]
            ymin=-scene.bin_size[1]/2+margin-lo[1];ymax=scene.bin_size[1]/2-margin-hi[1]
            if xmin>xmax or ymin>ymax:continue
            for x in np.linspace(xmin,xmax,POLICY['grid'][0]):
                for y in np.linspace(ymin,ymax,POLICY['grid'][1]):
                    if np.linalg.norm(np.array([x,y])-original[:2])<POLICY['min_xy_displacement_m']:continue
                    distance=(x-target.pose.position[0])**2+(y-target.pose.position[1])**2
                    ranked.append((-float(distance),qi,float(x),float(y),lo.copy(),hi.copy()))
        ranked.sort(key=lambda r:r[:4]);counts={};chosen=None;previews=[]
        for rank,(_,qi,x,y,lo,hi) in enumerate(ranked):
            q=unique[qi]
            # Begin above every overlapping native collision AABB, then descend
            # to the FIRST native contact. Never insert below an overhang.
            support_z=0.
            for blo,bhi in fixed_bounds.values():
                if x+hi[0]>=blo[0] and x+lo[0]<=bhi[0] and y+hi[1]>=blo[1] and y+lo[1]<=bhi[1]:
                    support_z=max(support_z,float(bhi[2]))
            z=support_z-lo[2]+.0005
            forward(x,y,z,q)
            if penetration()>0.000001:
                counts['unexpected_start_contact']=counts.get('unexpected_start_contact',0)+1;continue
            upper=z;lower=None
            # At most the full bin height plus object height, independent of outcomes.
            for _ in range(400):
                probe=z-POLICY['downward_step_m'];forward(x,y,probe,q)
                if penetration()>0:
                    lower=probe;upper=z;break
                z=probe
                if z+lo[2]<-.002:break
            if lower is None:
                counts['no_contact']=counts.get('no_contact',0)+1;continue
            for _ in range(14):
                mid=(lower+upper)/2;forward(x,y,mid,q)
                if penetration()>0:lower=mid
                else:upper=mid
            contact_z=upper
            forward(x,y,contact_z-POLICY['support_probe_m'],q)
            contacts=moving_contacts(model,data,body)
            points=[c['position'][:2] for c in contacts if c['upward']>=.5 and c['distance']<=0]
            com=data.xipos[body,:2].copy();dist=support_distance(points,com)
            if dist>POLICY['support_com_tolerance_m']:
                counts['first_contact_com_outside']=counts.get('first_contact_com_outside',0)+1
            release_z=contact_z+POLICY['release_clearance_m'];forward(x,y,release_z,q)
            blo,bhi=body_bounds(model,data,body);pen=penetration()
            inside=bool(blo[0]>=-scene.bin_size[0]/2+margin-1e-8 and bhi[0]<=scene.bin_size[0]/2-margin+1e-8 and
                        blo[1]>=-scene.bin_size[1]/2+margin-1e-8 and bhi[1]<=scene.bin_size[1]/2-margin+1e-8 and
                        blo[2]>=-.00025 and bhi[2]<=scene.bin_size[2]-.001)
            if not inside:
                counts['outside_bin']=counts.get('outside_bin',0)+1;continue
            if pen>POLICY['max_initial_penetration_m']:
                counts['penetration']=counts.get('penetration',0)+1;continue
            candidate=dict(position=[x,y,release_z],quaternion_wxyz=q.tolist(),rank=rank,
                        support_points_xy=points,com_xy=com.tolist(),support_distance_m=dist if np.isfinite(dist) else None,
                        release_bounds=[blo.tolist(),bhi.tolist()],max_initial_penetration_m=pen)
            moved=replace(obj,pose=Pose(tuple(candidate['position']),tuple(candidate['quaternion_wxyz'])))
            trial=replace(scene,objects=tuple(moved if o.object_id==object_id else o for o in scene.objects))
            # Validate the serialized input too, not only a high-precision qpos.
            pm,pd=native_model(trial,self.cfg)
            pb=mujoco.mj_name2id(pm,mujoco.mjtObj.mjOBJ_BODY,object_id)
            rounded_pen=max([max(0.,-c['distance']) for c in moving_contacts(pm,pd,pb)],default=0.)
            if rounded_pen>POLICY['max_initial_penetration_m']:
                counts['serialized_penetration']=counts.get('serialized_penetration',0)+1;continue
            candidate['serialized_initial_penetration_m']=rounded_pen
            preview=physical_preview(trial,self.cfg,object_id,original)
            previews.append(dict(candidate=candidate,physics=preview))
            if not preview['accepted']:
                key='preview_'+preview['reason'];counts[key]=counts.get(key,0)+1
                if len(previews)>=POLICY['max_physical_previews']:break
                continue
            chosen={**candidate,'preview_index':len(previews)-1}
            self.last_preview=preview
            break
        record=dict(scene_id=scene.scene_id,object_id=object_id,target_id=target_id,
                    relocation_index=relocation_index,policy=POLICY,from_pose=obj.pose.to_dict(),
                    num_candidates=len(ranked),rejected_counts=counts,chosen=chosen,physical_previews=previews,
                    elapsed_seconds=time.monotonic()-started,status='ACCEPTED' if chosen else 'NO_FEASIBLE_PLACEMENT')
        dest=self.audit_dir/f'{self.calls:04d}_{scene.scene_id}_{object_id}.json'
        dest.write_text(json.dumps(record,indent=2))
        if chosen is None:raise PlacementRejected(record)
        updated=replace(obj,pose=Pose(tuple(chosen['position']),tuple(chosen['quaternion_wxyz'])))
        metadata=dict(scene.metadata);events=list(metadata.get('ordered_sequence_relocations') or [])
        events.append({k:v for k,v in record.items() if k!='physical_previews'})
        events[-1]['physical_preview_audit']=str(dest)
        metadata['ordered_sequence_relocations']=events
        return replace(scene,objects=tuple(updated if o.object_id==object_id else o for o in scene.objects),metadata=metadata)
