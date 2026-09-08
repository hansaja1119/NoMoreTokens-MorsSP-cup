"""Complete-coverage public split benchmarks. Locked test requires explicit flag."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import time
from types import SimpleNamespace
import cv2
import numpy as np
from common import SCRIPTS,ROOT,public_root,load_split,read_rgb,save_rgb,write_json,to_uint8
from metrics import load_module,score_arrays,summarize
from classical import wavelet_estimate

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--method',choices=['baseline','wavelet'],required=True)
    ap.add_argument('--split',choices=['train','val','test'],default='val')
    ap.add_argument('--allow-locked-test',action='store_true')
    ap.add_argument('--limit',type=int)
    ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--strength',type=float,default=1.)
    args=ap.parse_args()
    if args.split=='test' and not args.allow_locked_test:ap.error('Locked test has not been authorized for model selection')
    cv2.setNumThreads(1)
    ids=load_split(args.split)
    if args.limit:ids=ids[:args.limit]
    tag=f'{args.method}_{args.split}'+(f'_{args.strength:g}' if args.method=='wavelet' else '')+(f'_n{args.limit}' if args.limit else '')
    out=SCRIPTS/'runs'/tag
    if (out/'metrics.json').exists():raise SystemExit(f'Already benchmarked: {out}')
    baseline=load_module('starter_baseline',ROOT/'baseline'/'denoise.py')
    def run(id_):
        y=read_rgb(public_root()/'noisy'/f'{id_}_noise.png');x=read_rgb(public_root()/'ground_truth'/f'{id_}.png')
        t=time.perf_counter()
        if args.method=='baseline':
            z=baseline.process_image(y,SimpleNamespace(skip_defect_correction=False,denoise_method='nlm',nlm_h=10.))
            # Match official baseline truncation exactly.
            z=(np.clip(z,0,1)*255).astype('uint8').astype('float32')/255
        else:z=to_uint8(wavelet_estimate(y,strength=args.strength)).astype('float32')/255
        elapsed=time.perf_counter()-t
        save_rgb(out/'images'/f'{id_}.png',z)
        return dict(id=id_,seconds=elapsed,**score_arrays(y,z,x))
    rows=[];started=time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(run,ids):
            rows.append(row)
            if len(rows)%10==0: print(f'{tag}: {len(rows)}/{len(ids)} composite={np.mean([r["composite_score"] for r in rows]):.6f}',flush=True)
    summary=summarize(rows);summary['wall_seconds']=time.perf_counter()-started
    write_json(out/'metrics.json',dict(config=vars(args),summary=summary,rows=rows))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
