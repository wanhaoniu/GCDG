"""Evaluate one fresh physical target with the frozen CoRL protocol (Linux)."""
from pathlib import Path
import argparse,os,sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=['ours','direct','nearest','mechanical','xray'],required=True)
    p.add_argument('--scene-index',type=int,required=True);p.add_argument('--target',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);os.chdir(ROOT)
    os.environ.update(MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',OMP_NUM_THREADS='1')
    import numerical_backend
    numerical_backend.install()
    from simulation_runtime import build
    from fresh_mujoco_closed_loop.run_closed_loop import FreshClosedLoopEpisodeRunner,load_scene_json
    roots=json.loads((ROOT/'benchmark/physical_roots.json').read_text());r=roots[a.scene_index]
    assert a.target in r['targets']
    scene=load_scene_json(ROOT/r['path']);rt,loop=build(ROOT/f'configs/physical/{a.method}.json',out,'physical')
    runner=FreshClosedLoopEpisodeRunner(propose=rt.proposals,plan=rt.plan,validate_target_grasp=rt.validate,
        resettle_after_removal=rt.resettle,refresh_observation_metadata=rt.refresh,proposal_trace=rt.proposal_trace)
    result=runner.run_episode(scene,a.target,loop)
    assert 0<=result['num_removals']<=5
    assert result['num_removals']==result['num_planner_removals']+result['num_fallback_removals']
    result.update(method=a.method,scene_alias=r['alias'],split='test',model_seed=7,numerical_policy=numerical_backend.POLICY)
    (out/'RESULT.json').write_text(json.dumps(result,indent=2));print(out)
if __name__=='__main__':main()
