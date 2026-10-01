from pathlib import Path
import sys,json,unittest,csv
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from grasp_dependency_dataset.hetero_gnn.graph_dataset import HeteroGraphDataset,dependency_label_lookup
from grasp_dependency_dataset.hetero_gnn.metrics import average_precision

class ReleaseTests(unittest.TestCase):
    def test_scene_disjoint_splits(self):
        s=json.loads((ROOT/'benchmark/splits.json').read_text())
        self.assertEqual({k:len(v) for k,v in s.items()},{'train':9090,'val':1950,'test':1945})
        scenes={k:{x.rsplit('__',1)[0] for x in v} for k,v in s.items()}
        self.assertEqual({k:len(v) for k,v in scenes.items()},{'train':350,'val':75,'test':75})
        for a,b in [('train','val'),('train','test'),('val','test')]:self.assertFalse(scenes[a]&scenes[b])

    def test_tied_ap_is_order_invariant(self):
        y=np.array([1,0,0,1]);s=np.array([.9,.9,.5,.1]);p=[1,0,2,3]
        self.assertAlmostEqual(average_precision(y,s),.5)
        self.assertAlmostEqual(average_precision(y[p],s[p]),.5)

    def test_sufficient_label_does_not_use_restored_metadata(self):
        rows=[dict(object_id='o',grasp_id='g',dep_collision_approach=True,dep_collision_lift=False,dep_any=False,metadata={'restored_if_removed':True})]
        label,valid=dependency_label_lookup({'dependencies':rows})[('o','g')]
        np.testing.assert_array_equal(label,[1,0,1,0]);self.assertTrue(valid)

    def test_fixture_dimensions_and_labels(self):
        c=json.loads((ROOT/'configs/prediction/ours_seed7.json').read_text());cm=json.loads((ROOT/'benchmark/class_map.json').read_text())
        self.assertFalse(c['features']['allow_privileged_scene_geometry'])
        g=HeteroGraphDataset(ROOT/'examples/benchmark',feature_config=c['features'],class_map=cm)[0]
        self.assertEqual((g.x_obj.shape[1],g.x_grasp.shape[1],g.edge_attr_og.shape[1],g.edge_label_og.shape[1]),(36,28,22,4))
        self.assertEqual(g.num_og_edges,g.num_objects*g.num_grasps)
        for a in [g.x_obj,g.x_grasp,g.edge_attr_og,g.edge_label_og]:self.assertTrue(np.isfinite(a).all())
        self.assertTrue(np.all(g.edge_label_mask_og==1))

    def test_shared_physical_protocol(self):
        configs=[json.loads(p.read_text()) for p in (ROOT/'configs/physical').glob('*.json')]
        self.assertEqual(len(configs),5)
        for c in configs:
            self.assertEqual(c['proposal'],configs[0]['proposal'])
            self.assertEqual(c['loop']['max_steps'],5)
            self.assertEqual(c['loop']['resettle_steps'],9000)
            self.assertEqual(c['loop']['max_shared_fallback_removals'],5)

    def test_evaluation_exports_minimal_blocker_metrics(self):
        from scripts.prediction import test_metrics
        pred=dict(edge_sample_ids=['s','s'],edge_object_ids=['a','b'],edge_grasp_ids=['g','g'],
                  y_true=np.array([[1,1,1,1],[0,0,0,0]],dtype=np.float32),
                  y_score=np.array([[.9,.9,.9,.9],[.1,.1,.1,.1]],dtype=np.float32),
                  mask=np.ones(2,dtype=bool),planning_labels_by_sample={'s':[
                      dict(grasp_id='g',status='solved_within_depth',minimal_blocker_set=['a'])]})
        metrics=test_metrics(pred,{},1)
        self.assertEqual(metrics['target_count'],1)
        self.assertEqual(metrics['modeled_edges'],2)
        self.assertEqual(metrics['minimal_blocker_sets']['all_solved']['mean_exact_match'],1.)

    def test_ordered_denominators(self):
        from scripts.recount_ordered import recount
        result=recount(ROOT/'results/ordered/episodes.csv')
        self.assertEqual(len(result),9)
        for row in result.values():self.assertEqual(row['target_trials'],160)

if __name__=='__main__':unittest.main()
