"""Use the already-repaired frozen provider runtime; errors are not empty grasps."""
from pathlib import Path
import os,sys,json,hashlib,random,time
def install(output):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    import grasp_pose_generator.adapters.anygrasp_adapter
    import numpy as np,torch
    from grasp_dependency_dataset.pipeline.runner import DatasetPipelineRunner
    original=DatasetPipelineRunner.from_configs.__func__
    if getattr(original,'_corl_checked',False):return
    def construct(cls,*args,**kwargs):
        runner=original(cls,*args,**kwargs)
        for provider in [runner.parallel_provider,runner.suction_provider]:
            generation=provider.generate
            def generate(scene,target_id,top_k,_fn=generation):
                seed=int(hashlib.sha256(f'{scene.scene_id}|{target_id}'.encode()).hexdigest()[:8],16)
                random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                return _fn(scene,target_id,top_k)
            provider.generate=generate
            for name in ['generate','generate_from_preprocessed']:
                fn=getattr(provider._manager,name)
                def checked(*args,_fn=fn,_provider=provider.source_name,**kwargs):
                    result=_fn(*args,**kwargs)
                    row=dict(time=time.time(),provider=_provider,status=str(result.status_code),success=bool(result.success),count=len(result.candidates),message=str(result.message))
                    with (output/'PROVIDERS.jsonl').open('a') as h:h.write(json.dumps(row)+'\n')
                    if row['status'] in {'config_error','invalid_input','adapter_not_found','adapter_unavailable','native_runtime_error','not_implemented'}:
                        raise RuntimeError('Native provider failure: '+json.dumps(row))
                    return result
                setattr(provider._manager,name,checked)
        return runner
    construct._corl_checked=True
    DatasetPipelineRunner.from_configs=classmethod(construct)
