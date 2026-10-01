"""Recompute every ordered-retrieval condition from the 144 released episodes."""
from pathlib import Path
import csv,json,argparse
from collections import defaultdict
def recount(path):
    groups=defaultdict(list)
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):groups[r['condition']].append(r)
    result={}
    for name,rows in sorted(groups.items()):
        keys={(r['order_seed'],r['root']) for r in rows}
        assert len(rows)==len(keys)==16,(name,len(rows),len(keys))
        n=sum(int(r['targets']) for r in rows);assert n==160
        retrieved=sum(int(r['retrieved']) for r in rows);moves=sum(int(r['moves']) for r in rows)
        result[name]={'target_trials':n,'retrieved':retrieved,'retrieval_rate':retrieved/n,'relocations':moves,
            'relocations_per_retrieved':moves/retrieved if retrieved else None,
            'attempted':sum(int(r['attempted']) for r in rows),'full_sequences':sum(r['full_sequence'].lower()=='true' for r in rows),
            'sequences':16,'scene_roots':4}
    assert len(groups)==9
    return result
if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--input',type=Path,default=Path(__file__).resolve().parents[1]/'results/ordered/episodes.csv');args=a.parse_args()
    print(json.dumps(recount(args.input),indent=2))
