"""Cache train/validation features only; never reads the locked test set."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import numpy as np
import cv2
from common import SCRIPTS,public_root,read_rgb,load_split,write_json,sha256
from classical import features

def prepare(image_id):
    target=SCRIPTS/'cache'/'features_v1'/f'{image_id}.npy'
    if target.exists():return image_id
    y=read_rgb(public_root()/'noisy'/f'{image_id}_noise.png')
    x=read_rgb(public_root()/'ground_truth'/f'{image_id}.png')
    f=features(y);f[...,6:9]/=.3
    arr=np.rint(np.clip(np.concatenate([f,x],axis=-1),0,1)*255).astype('uint8')
    target.parent.mkdir(parents=True,exist_ok=True)
    tmp=target.with_suffix('.tmp.npy');np.save(tmp,arr);tmp.replace(target)
    return image_id

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    cv2.setNumThreads(1)
    ids=load_split('train')+load_split('val')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i,_ in enumerate(pool.map(prepare,ids)):
            if (i+1)%40==0:print(f'Cached {i+1}/{len(ids)}',flush=True)
    write_json(SCRIPTS/'cache'/'features_v1'/'manifest.json',dict(ids=ids,
        classical_sha256=sha256(SCRIPTS/'classical.py'),split_sha256=sha256(SCRIPTS/'data'/'split.json')))

if __name__=='__main__':main()
