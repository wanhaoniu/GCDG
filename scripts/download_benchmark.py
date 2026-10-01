"""Download versioned benchmark packages with SHA-256 verification."""
from pathlib import Path
import argparse,json,hashlib,urllib.request,tarfile
BASE='https://huggingface.co/datasets/WanhaoX25/GCDG-Benchmark/resolve/v1.0.0/'
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--split',choices=['train','val','test','all'],default='all');a.add_argument('--output',type=Path,default=Path('data'));a.add_argument('--extract',action='store_true');args=a.parse_args()
    manifest=Path(__file__).resolve().parents[1]/'benchmark/packages.json'
    packages=json.loads(manifest.read_text());args.output.mkdir(parents=True,exist_ok=True)
    for p in packages:
        if args.split!='all' and p['split']!=args.split:continue
        dest=args.output/p['name']
        if not dest.exists():
            tmp=dest.with_suffix(dest.suffix+'.partial');urllib.request.urlretrieve(BASE+p['name'],tmp)
            if digest(tmp)!=p['sha256']:raise ValueError(f'Checksum mismatch: {tmp}')
            tmp.replace(dest)
        if digest(dest)!=p['sha256']:raise ValueError(f'Checksum mismatch: {dest}')
        if args.extract:
            with tarfile.open(dest) as t:
                root=args.output.resolve()
                for member in t:
                    target=(root/member.name).resolve()
                    if not target.is_relative_to(root) or not (member.isfile() or member.isdir()):raise ValueError('Unsafe archive entry')
                    if member.isdir():target.mkdir(parents=True,exist_ok=True);continue
                    target.parent.mkdir(parents=True,exist_ok=True)
                    import shutil
                    with t.extractfile(member) as src,target.open('wb') as out:shutil.copyfileobj(src,out)
        print(f'Verified {p["name"]}')
if __name__=='__main__':main()
