from __future__ import annotations
import os,sys,json,time,hashlib,argparse,traceback,importlib.util
from pathlib import Path
from dataclasses import replace
os.environ['MUJOCO_GL']='egl';os.environ['PYOPENGL_PLATFORM']='egl';os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
W=Path(__file__).resolve().parents[1]
CONDITIONS={}
CONDITIONS.update({f'ours_floor_seed{s}':('ours','control','new',s) for s in [7,11,23,31,43]})
CONDITIONS.update({f'{m}_floor':(m,'control','new',7) for m in ['direct','nearest','mechanical','xray']})
CONDITION=sys.argv[sys.argv.index('--condition')+1]
METHOD,VARIANT,MODEL_VERSION,MODEL_SEED=CONDITIONS[CONDITION]
FLOOR_REPAIR='_floor' in CONDITION
sys.path[:0]=[str(W),str(W/'src')]
import numerical_backend
numerical_backend.install()
if FLOOR_REPAIR:
    import safe_placement_floor
    sys.modules['safe_placement']=safe_placement_floor
from safe_placement import SafePlacement,native_model,body_bounds
from fresh_mujoco_closed_loop.run_closed_loop import (load_scene_json,FreshClosedLoopEpisodeRunner,
    write_transient_shared_observations,invalid_resettle_terminal_reason)
from grasp_dependency_dataset.common.config import SceneGenerationConfig
from grasp_dependency_dataset.simulation.scene_generator import BinClutterSceneGenerator
from grasp_dependency_dataset.simulation.mujoco_backend import MujocoBackend
import numpy as np
_original_invalid_resettle=invalid_resettle_terminal_reason
def invalid_resettle_terminal_reason(record):
    if record and record.get('geometry_accepted') is False:return 'resettle_geometry_invalid'
    return _original_invalid_resettle(record)

def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(v,indent=2));tmp.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def heartbeat(out,phase,**kw):write(out/'STATUS.json',dict(phase=phase,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw))

def run(args):
    roots=json.loads((W/'benchmark/ordered/orders.json').read_text())
    root=next(r for r in roots if r['root_index']==args.index and str(r['order_seed'])==args.order)
    root={**root,'path':str(W/f"benchmark/ordered/scenes/root_{args.index:02d}.json")}
    plan={'scene_config':str(W/'configs/simulation/ordered.yaml')}
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);os.chdir(W)
    heartbeat(out,'initializing')
    from simulation_runtime import build
    rt,loop=build(W/f'configs/ordered/{args.condition}.json',out,'ordered')
    write(out/'CONDITION.json',dict(condition=args.condition,method=args.method,variant=VARIANT,model_version=MODEL_VERSION,
        model_seed=MODEL_SEED,floor_repair=FLOOR_REPAIR,numerical_policy=numerical_backend.POLICY,checkpoint_path=str(rt.checkpoint),checkpoint_sha256=sha(rt.checkpoint) if rt.checkpoint else None))
    cfg=SceneGenerationConfig.from_yaml(Path(plan['scene_config']))
    gen=BinClutterSceneGenerator(cfg,None,MujocoBackend());placer=SafePlacement(cfg,out/'placements')
    settle_counter=0;frame_counter=0
    def settle(scene,object_id,step_index):
        nonlocal settle_counter
        settle_counter+=1;heartbeat(out,'native_settle',scene_id=scene.scene_id,object_id=object_id,settle_count=settle_counter)
        result=gen._settle_states(list(scene.objects),settle_steps=9000)
        objects=gen._apply_settled_poses(list(scene.objects),result.body_poses)
        bad=gen._out_of_bin_object_ids(objects)
        finite=all(np.all(np.isfinite(o.pose.position)) and np.all(np.isfinite(o.pose.quaternion_wxyz)) for o in objects)
        audit={**result.metadata,'removed_object_id':object_id,'step_index':step_index,
               'stable':bool(result.stable),'accepted':bool(result.stable and not bad and finite),
               'out_of_bin_object_ids':bad,'finite':finite,'state_saved':'actual native settled poses'}
        if FLOOR_REPAIR and finite:
            final_scene=replace(scene,objects=tuple(objects))
            gm,gd=native_model(final_scene,cfg)
            geometry=safe_placement_floor.containment_audit(gm,gd,final_scene)
            audit.update(geometry)
            audit['accepted']=bool(audit['accepted'] and geometry['geometry_accepted'])
        if not finite:audit['stable']=False
        metadata={**scene.metadata,'fresh_resettle':audit}
        final=replace(scene,objects=tuple(objects),metadata=metadata)
        write(out/'settles'/f'{settle_counter:04d}_input.json',scene.to_dict())
        write(out/'settles'/f'{settle_counter:04d}_actual.json',final.to_dict())
        return final
    def refresh(scene,target):
        heartbeat(out,'refresh',scene_id=scene.scene_id,target_id=target)
        final=rt.refresh(scene,target);p=out/'states'/scene.scene_id
        write(p/'scene.json',final.to_dict())
        write_transient_shared_observations(runner=rt.runner,scene=final,target_id=target,scene_dir=p)
        return final
    def propose(scene,target):
        heartbeat(out,'proposals',scene_id=scene.scene_id,target_id=target)
        value=rt.proposals(scene,target)
        import torch
        if torch.cuda.is_available():torch.cuda.empty_cache()
        return value
    def predict(scene,target,proposals,budget):
        heartbeat(out,'planner',scene_id=scene.scene_id,target_id=target)
        return rt.plan(scene,target,proposals,budget)
    def frame(scene,target_index,target_id,status):
        nonlocal frame_counter
        p=out/'frames'/f'{frame_counter:02d}';frame_counter+=1
        write(p/'scene.json',scene.to_dict())
        # Renderer target ID is only used for observation metadata; no missing target.
        render_target=target_id if target_id in [o.object_id for o in scene.objects] else scene.objects[0].object_id
        rendered=write_transient_shared_observations(runner=rt.runner,scene=scene,target_id=render_target,scene_dir=p)
        row=dict(index=frame_counter-1,target_index=target_index,target_id=target_id,status=status,scene_path=str(p/'scene.json'),**rendered)
        write(p/'FRAME.json',row);return row
    scene=load_scene_json(Path(root['path']));current=replace(scene,target_ids=tuple(root['target_order']))
    orders=root['target_order'];retrieved=[];failed=[];relocations=[];phases=[];frames=[];terminal='sequence_complete';relocation_index=0
    frames.append(frame(current,-1,orders[0],'initial'))
    begin=time.time()
    for ti,target in enumerate(orders):
        if target not in [o.object_id for o in current.objects]:terminal='target_missing_before_phase';break
        metadata={**current.metadata,'ordered_sequence_base_scene_id':scene.scene_id,'ordered_sequence_target_index':ti,'ordered_sequence_target_id':target}
        phase_scene=replace(current,scene_id=f'{scene.scene_id}__target_{ti:02d}',metadata=metadata)
        latest={'scene':phase_scene}
        def intervene(step_scene,active,object_id,step):
            nonlocal relocation_index
            heartbeat(out,'placement_search',scene_id=step_scene.scene_id,target_id=active,object_id=object_id)
            moved=placer(step_scene,object_id,active,relocation_index)
            relocation_index+=1;relocations.append(object_id)
            if loop.resettle_after_removal:
                moved=settle(moved,object_id,step)
                expected=placer.last_preview['settled_poses']
                assert all(o.pose.to_dict()==expected[o.object_id] for o in moved.objects),'Placement preview/execution mismatch'
                moved.metadata['fresh_resettle']['placement_preview_execution_exact']=True
            latest['scene']=moved
            return moved
        er=FreshClosedLoopEpisodeRunner(propose=propose,plan=predict,validate_target_grasp=rt.validate,
             resettle_after_removal=settle,refresh_observation_metadata=refresh,proposal_trace=rt.proposal_trace,
             apply_non_target_intervention=intervene)
        episode=er.run_episode(phase_scene,target,loop);current=latest['scene']
        invalid=''
        if episode['success']:
            retrieved.append(target);current=settle(current.without_objects({target}),target,ti)
            episode['post_target_resettle']=current.metadata['fresh_resettle']
            invalid=invalid_resettle_terminal_reason(current.metadata['fresh_resettle'])
        else:
            failed.append(target)
            if str(episode['terminal_reason']).startswith('resettle_'):invalid=episode['terminal_reason']
        episode['physical_terminal_reason']=invalid or None
        assert {o.object_id for o in current.objects}=={o.object_id for o in scene.objects}-set(retrieved),'Unexpected non-target disposal'
        episode['target_order_index']=ti;episode['remaining_object_ids_after_phase']=[o.object_id for o in current.objects]
        phases.append(episode)
        status='retrieved' if episode['success'] else episode['terminal_reason']
        if invalid:status=invalid
        frames.append(frame(current,ti,target,status))
        write(out/'PHASE_RESULTS.json',phases)
        print('PHASE',args.mode,args.method,root['scene_id'],ti,status,len(relocations),flush=True)
        if invalid:terminal=invalid;break
        terminal='sequence_complete' if not failed else 'all_targets_attempted_with_failures'
    fb=[o for p in phases for o in p.get('fallback_removal_sequence',[])];pr=[o for p in phases for o in p.get('planner_removal_sequence',[])]
    assert len(relocations)==len(fb)+len(pr)
    prefix=0
    for phase in phases:
        if not phase['success']:break
        prefix+=1
        if phase.get('physical_terminal_reason'):break
    assert len(phases)==len(orders) or terminal.startswith('resettle_'),'Ordinary failure incorrectly stopped continuous sequence'
    result=dict(scene_id=scene.scene_id,scene_index=args.index,planner_type=loop.planner_type,method=args.method,
        protocol_mode=args.mode,condition=args.condition,interface_variant=VARIANT,numerical_policy=numerical_backend.POLICY,model_version=MODEL_VERSION,development_only=True,target_order=orders,success=len(retrieved)==len(orders) and terminal=='sequence_complete',
        completed_count=len(retrieved),attempted_targets=len(phases),strict_success_prefix=prefix,total_targets=len(orders),completion_fraction=len(retrieved)/len(orders),
        num_relocations=len(relocations),relocation_sequence=relocations,num_fallback_relocations=len(fb),
        fallback_relocation_sequence=fb,num_planner_relocations=len(pr),planner_relocation_sequence=pr,
        retrieved_target_sequence=retrieved,failed_target_ids=failed,terminal_reason=terminal,
        target_phase_results=phases,remaining_object_ids=[o.object_id for o in current.objects],frames=frames,
        elapsed_seconds=time.time()-begin,source_manifest_sha256=sha(W/'benchmark/ordered/orders.json'))
    write(out/'RESULT.json',result);heartbeat(out,'COMPLETE',completed_count=len(retrieved),relocations=len(relocations),terminal_reason=terminal)
    print('COMPLETE',args.mode,args.method,args.index,len(retrieved),len(relocations),terminal,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--condition',required=True,choices=list(CONDITIONS));p.add_argument('--index',type=int,required=True)
    p.add_argument('--mode',choices=['continuous'],default='continuous');p.add_argument('--order',choices=['original','101','211','307'],default='original');p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.method=METHOD
    try:run(args)
    except Exception:
        dest=args.output.resolve();write(dest/'FAILURE.json',{'traceback':traceback.format_exc(),'time':time.time()});raise
