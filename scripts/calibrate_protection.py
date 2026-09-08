"""Validate a small fixed mild-noise protection grid using one network pass/image."""
import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from common import SCRIPTS,load_split,public_root,read_rgb,to_uint8,write_json,sha256
from inference import load_model,choose_device,predict,protect_mild
from classical import noise_maps
from metrics import score_arrays,summarize

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path);ap.add_argument('--device',default='cpu')
    args=ap.parse_args()
    if args.output.exists():ap.error('Output exists')
    torch.set_num_threads(4);cv2.setNumThreads(1)
    device=choose_device(args.device);model,_=load_model(args.checkpoint,device)
    thresholds=[0.,.02,.03,.04];rows={str(t):[] for t in thresholds}
    for n,id_ in enumerate(load_split('val')):
        y=read_rgb(public_root()/'noisy'/f'{id_}_noise.png');gt=read_rgb(public_root()/'ground_truth'/f'{id_}.png')
        start=time.perf_counter();pred=predict(model,y,device,mild_threshold=0)
        elapsed=time.perf_counter()-start;sigma=noise_maps(y)
        for threshold in thresholds:
            result=to_uint8(protect_mild(y,pred,sigma,threshold)).astype('float32')/255
            rows[str(threshold)].append(dict(id=id_,seconds=elapsed,**score_arrays(y,result,gt)))
        if (n+1)%10==0:print(f'Protection grid {n+1}/60',flush=True)
    output=dict(checkpoint_sha256=sha256(args.checkpoint),device=str(device),results={k:dict(summary=summarize(v),rows=v) for k,v in rows.items()})
    write_json(args.output,output)
    print(json.dumps({k:v['summary']['composite_score'] for k,v in output['results'].items()}),flush=True)

if __name__=='__main__':main()
