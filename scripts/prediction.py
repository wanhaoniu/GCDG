"""Train or evaluate a frozen-split CoRL dependency predictor."""
from pathlib import Path
import argparse, json, random, sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from grasp_dependency_dataset.hetero_gnn.graph_dataset import HeteroGraphDataset, discover_sample_refs
from grasp_dependency_dataset.hetero_gnn.train_hetero_gnn import train_torch_backend, calibration_thresholds_from_metrics
from grasp_dependency_dataset.hetero_gnn.eval_hetero_gnn import collect_torch_predictions
from run_stage2_simple_baselines import collect_predictions, metrics_from_prediction, metrics_with_validation_thresholds

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x):Path(p).write_text(json.dumps(x,indent=2),encoding='utf-8')
def save_predictions(p,pred):
    np.savez_compressed(p,y_true=pred['y_true'],y_score=pred['y_score'],mask=pred['mask'],
        sample_id=np.asarray(pred['edge_sample_ids']),object_id=np.asarray(pred['edge_object_ids']),grasp_id=np.asarray(pred['edge_grasp_ids']))
def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('mode',choices=['train','evaluate'])
    a.add_argument('--config',type=Path,default=ROOT/'configs/prediction/ours_seed7.json')
    a.add_argument('--dataset',type=Path,default=ROOT/'data/benchmark')
    a.add_argument('--output',type=Path,required=True)
    a.add_argument('--checkpoint',type=Path)
    a.add_argument('--calibration',type=Path)
    a.add_argument('--device',default='cpu');a.add_argument('--threads',type=int,default=2)
    args=a.parse_args();c=read(args.config);splits=read(ROOT/'benchmark/splits.json');cm=read(ROOT/'benchmark/class_map.json')
    seed=int(c['training']['seed']);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(args.threads);torch.backends.cudnn.benchmark=False
    c['dataset']={'root':str(args.dataset.resolve())};c['output']={'dir':str(args.output.resolve())};c['training']['device']=args.device
    refs=discover_sample_refs(args.dataset);by={r.sample_id:r for r in refs}
    needed=['train','val'] if args.mode=='train' else ['test']
    for split in needed:
        missing=set(splits[split])-by.keys()
        if missing:raise SystemExit(f'{split}: missing {len(missing)} samples; download the complete split first.')
    def dataset(split):
        allowed=set(splits[split])
        return HeteroGraphDataset(args.dataset,refs=[r for r in refs if r.sample_id in allowed],feature_config=c['features'],class_map=cm)
    if args.output.exists() and any(args.output.iterdir()):raise SystemExit('Choose an empty output directory; existing results are preserved.')
    args.output.mkdir(parents=True,exist_ok=True);save(args.output/'config.json',c)
    simple=c['backend'] in {'random_score','geometry_heuristic'}
    if args.mode=='train':
        train,val=dataset('train'),dataset('val')
        if not simple:train_torch_backend(c,train,val,[],cm,backend_name=c['backend'])
        pred=collect_predictions(c,val) if simple else collect_torch_predictions(args.output/'checkpoints/best.pt',val,batch_size=32,device=args.device)
        metrics=metrics_from_prediction(pred,threshold=.5)
        save(args.output/'calibration_thresholds.json',calibration_thresholds_from_metrics(metrics))
        save(args.output/'validation_metrics.json',metrics);save_predictions(args.output/'validation_predictions.npz',pred)
    else:
        if args.calibration is None:raise SystemExit('--calibration must point to validation-selected thresholds.')
        if not simple and args.checkpoint is None:raise SystemExit('--checkpoint is required for learned models.')
        test=dataset('test')
        pred=collect_predictions(c,test) if simple else collect_torch_predictions(args.checkpoint,test,batch_size=32,device=args.device)
        save(args.output/'test_metrics.json',metrics_with_validation_thresholds(pred,read(args.calibration)))
        save_predictions(args.output/'test_predictions.npz',pred)
    print(args.output.resolve())
if __name__=='__main__':main()
