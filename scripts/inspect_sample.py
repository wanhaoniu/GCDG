"""Inspect a released graph without native proposal or simulation dependencies."""
from pathlib import Path
import sys,json,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from grasp_dependency_dataset.hetero_gnn.graph_dataset import HeteroGraphDataset,FEATURE_SCHEMA
def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--dataset',type=Path,default=ROOT/'examples/benchmark');a.add_argument('--index',type=int,default=0);args=a.parse_args()
    c=json.loads((ROOT/'configs/prediction/ours_seed7.json').read_text());cm=json.loads((ROOT/'benchmark/class_map.json').read_text())
    d=HeteroGraphDataset(args.dataset,feature_config=c['features'],class_map=cm);g=d[args.index]
    print(json.dumps({'sample_id':g.sample_id,'targets_in_directory':len(d),'objects':g.num_objects,'grasps':g.num_grasps,'edges':g.num_og_edges,'object_features':g.x_obj.shape[1],'grasp_features':g.x_grasp.shape[1],'edge_features':g.edge_attr_og.shape[1],'labels':FEATURE_SCHEMA['labels']},indent=2))
if __name__=='__main__':main()
