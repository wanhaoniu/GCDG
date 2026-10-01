"""Check separately acquired object assets against the evaluated file hashes."""
from pathlib import Path
import json,hashlib,sys
ROOT=Path(__file__).resolve().parents[1]
def main():
    rows=json.loads((ROOT/'benchmark/asset_checksums.json').read_text())['files'];missing=[];bad=[]
    for row in rows:
        p=ROOT/row['path']
        if not p.exists():missing.append(row['path']);continue
        h=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
        if h.hexdigest()!=row['sha256']:bad.append(row['path'])
    print(json.dumps({'checked':len(rows),'missing':missing,'mismatched':bad},indent=2))
    if missing or bad:raise SystemExit('Supply the external assets listed in THIRD_PARTY.md before native simulation.')
if __name__=='__main__':main()
