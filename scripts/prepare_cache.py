"""Cache development features, or all pairs after the locked assessment is recorded."""
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import argparse
import json
import numpy as np
import cv2
from common import SCRIPTS,public_root,read_rgb,load_split,write_json,sha256
from classical import features

def prepare(image_id,rebuild=False):
    target=SCRIPTS/'cache'/'features_v1'/f'{image_id}.npy'
    if target.exists() and not rebuild:return image_id
    y=read_rgb(public_root()/'noisy'/f'{image_id}_noise.png')
    x=read_rgb(public_root()/'ground_truth'/f'{image_id}.png')
    f=features(y);f[...,6:9]/=.3
    arr=np.rint(np.clip(np.concatenate([f,x],axis=-1),0,1)*255).astype('uint8')
    target.parent.mkdir(parents=True,exist_ok=True)
    tmp=target.with_suffix('.tmp.npy');np.save(tmp,arr);tmp.replace(target)
    return image_id

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--include-locked-test',action='store_true')
    ap.add_argument('--rebuild',action='store_true');args=ap.parse_args()
    cv2.setNumThreads(1)
    manifest_path=SCRIPTS/'cache'/'features_v1'/'manifest.json'
    if manifest_path.exists():
        old=json.loads(manifest_path.read_text())
        if old['classical_sha256']!=sha256(SCRIPTS/'classical.py') and not args.rebuild:
            raise RuntimeError('Preprocessing changed. Use --rebuild to replace every cached feature explicitly.')
    if args.include_locked_test and not (SCRIPTS/'runs'/'final_preparation'/'locked_test'/'metrics.json').exists():
        raise RuntimeError('Record the one-time locked-test evaluation before preparing all-data training')
    ids=load_split('all') if args.include_locked_test else load_split('train')+load_split('val')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i,_ in enumerate(pool.map(partial(prepare,rebuild=args.rebuild),ids)):
            if (i+1)%40==0:print(f'Cached {i+1}/{len(ids)}',flush=True)
    write_json(SCRIPTS/'cache'/'features_v1'/'manifest.json',dict(ids=ids,
        classical_sha256=sha256(SCRIPTS/'classical.py'),split_sha256=sha256(SCRIPTS/'data'/'split.json')))

if __name__=='__main__':main()
