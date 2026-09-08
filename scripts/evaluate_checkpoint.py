"""Evaluate one checkpoint and record performance and reproducible output hashes."""
import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from common import SCRIPTS,public_root,load_split,read_rgb,save_rgb,to_uint8,write_json,sha256
from inference import choose_device,load_model,predict
from metrics import score_arrays,summarize

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--split',choices=['train','val','test'],default='val')
    ap.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    ap.add_argument('--limit',type=int)
    ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--tile',type=int,default=512)
    ap.add_argument('--overlap',type=int,default=64)
    ap.add_argument('--mild-threshold',type=float,default=None)
    ap.add_argument('--allow-locked-test',action='store_true')
    args=ap.parse_args()
    if args.split=='test' and not args.allow_locked_test:ap.error('Do not use locked test for model selection')
    ids=load_split(args.split)
    if args.limit:ids=ids[:args.limit]
    if args.output.exists():ap.error('Output already exists; use a distinct experiment directory')
    cv2.setNumThreads(args.threads);torch.set_num_threads(args.threads)
    device=choose_device(args.device);model,obj=load_model(args.checkpoint,device)
    rows=[];start=time.perf_counter()
    for id_ in ids:
        y=read_rgb(public_root()/'noisy'/f'{id_}_noise.png');gt=read_rgb(public_root()/'ground_truth'/f'{id_}.png')
        t=time.perf_counter();pred=predict(model,y,device,args.tile,args.overlap,args.mild_threshold)
        seconds=time.perf_counter()-t
        path=args.output/'images'/f'{id_}.png';save_rgb(path,pred)
        rows.append(dict(id=id_,seconds=seconds,sha256=sha256(path),**score_arrays(y,read_rgb(path),gt)))
        if len(rows)%10==0:print(f'{len(rows)}/{len(ids)}',flush=True)
    write_json(args.output/'metrics.json',dict(checkpoint=str(args.checkpoint),checkpoint_sha256=sha256(args.checkpoint),
        step=obj.get('step'),device=str(device),threads=args.threads,tile=args.tile,overlap=args.overlap,mild_threshold=args.mild_threshold,
        split=args.split,wall_seconds=time.perf_counter()-start,summary=summarize(rows),rows=rows))
    print(json.dumps(summarize(rows),indent=2),flush=True)

if __name__=='__main__':main()
