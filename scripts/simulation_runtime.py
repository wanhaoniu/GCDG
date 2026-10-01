"""Build the frozen CoRL runtime from a portable experiment configuration."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]

def build(config,output,task):
    from provider_guard import install
    install(output)
    import torch
    torch.set_num_threads(1)
    from fresh_mujoco_closed_loop.run_closed_loop import RealFreshMujocoRuntime,FreshLoopConfig
    from fresh_mujoco_closed_loop.proposal_protocol import FreshProposalProtocolConfig
    from grasp_dependency_dataset.pipeline.runner import DatasetPipelineRunner
    from planners.planner_utils import PlannerParams
    c=json.loads(Path(config).read_text());seed=c.get('model_seed',7)
    graph_path=ROOT/f'configs/prediction/ours_seed{seed}.json'
    if not graph_path.exists():graph_path=ROOT/'configs/prediction/ours_seed7.json'
    graph=json.loads(graph_path.read_text())
    scene_config=ROOT/f'configs/simulation/{task}.yaml'
    runner=DatasetPipelineRunner.from_configs(config_dir=ROOT/'configs/simulation',output_root=output/'pipeline_cache',
        proposal_config_path=ROOT/'configs/simulation/proposal_sources.yaml',scene_config_path=scene_config)
    backend=c.get('backend','torch' if c.get('method')=='ours' else 'geometry_heuristic')
    cp=ROOT/c['checkpoint'] if backend=='torch' else None
    rt=RealFreshMujocoRuntime(runner=runner,output_dir=output,graph_cfg=graph,checkpoint=cp,
        prediction_backend=backend,device='cpu',feature_config=c['features'],
        class_map=json.loads((ROOT/'benchmark/class_map.json').read_text()),
        planner_params=PlannerParams.from_dict(c['planner']),planner_seed=7,planner_type=c['loop']['planner_type'],
        proposal_protocol_config=FreshProposalProtocolConfig(**c['proposal']))
    if backend=='torch':
        payload=torch.load(cp,map_location='cpu',weights_only=True)
        rt.dependency_predictor._torch_model.load_state_dict(payload['model_state'],strict=True)
    rt.resettle_steps=9000;rt.reject_unstable_resettle=True
    (output/'CONFIG.json').write_text(json.dumps(c,indent=2))
    return rt,FreshLoopConfig(**c['loop'])
