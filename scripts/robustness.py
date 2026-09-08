"""Repeatable broad-noise stress tests on validation scenes, never locked test."""
import argparse
from pathlib import Path
import json
import time
import cv2
import numpy as np
import torch
from common import SCRIPTS,public_root,load_split,read_rgb,to_uint8,write_json,sha256
from inference import choose_device,load_model,predict
from classical import wavelet_estimate
from metrics import official

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=Path)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--device',default='cpu')
    ap.add_argument('--count',type=int,default=8)
    ap.add_argument('--mild-threshold',type=float,default=0.);args=ap.parse_args()
    torch.set_num_threads(4);cv2.setNumThreads(1)
    if args.output.exists():ap.error('Use a new output path')
    model=None
    if args.checkpoint:
        device=choose_device(args.device);model,_=load_model(args.checkpoint,device)
    ids=load_split('val');rng=np.random.default_rng(914)
    ids=sorted(rng.choice(ids,size=min(args.count,len(ids)),replace=False).tolist())
    rows=[]
    for image_id in ids:
        full=read_rgb(public_root()/'ground_truth'/f'{image_id}.png')
        # Fixed central 256px crops allow repeatable small independent diagnostics.
        source=full[368:624,368:624]
        for kind in ['clean','mild_gaussian','strong_gaussian','signal_dependent','spatial_mixture','gray_clean','gray_noise','correlated_color']:
            x=np.repeat(source.mean(-1,keepdims=True),3,axis=-1) if kind.startswith('gray') else source
            x=to_uint8(x).astype('float32')/255
            noise=rng.normal(size=x.shape).astype('float32')
            if kind in ['clean','gray_clean']:y=x.copy()
            elif kind in ['gray_noise','correlated_color']:y=x+.10*noise[...,:1]
            elif kind=='mild_gaussian':y=x+.01*noise
            elif kind=='strong_gaussian':y=x+.10*noise
            elif kind=='signal_dependent':y=x+np.sqrt(.01**2+.025*x)*noise
            else:
                sigma=np.linspace(.01,.12,256,dtype='float32')[None,:,None]
                y=x+sigma*noise;mask=rng.random(x.shape)<.002
                y=np.where(mask,rng.integers(0,2,size=x.shape),y)
            y=to_uint8(y).astype('float32')/255
            t=time.perf_counter()
            p=predict(model,y,device,mild_threshold=args.mild_threshold) if model is not None else wavelet_estimate(y)
            p=to_uint8(p).astype('float32')/255
            mse=float(np.mean((p-x)**2))
            rows.append(dict(id=image_id,condition=kind,psnr=float(-10*np.log10(max(mse,1e-12))),
                ssim=official.ssim_value(x,p),mae=float(np.mean(np.abs(p-x))),
                input_mse=float(np.mean((y-x)**2)),output_mse=mse,seconds=time.perf_counter()-t))
        print(f'Stress-tested {image_id}',flush=True)
    summary={}
    for kind in sorted({r['condition'] for r in rows}):
        part=[r for r in rows if r['condition']==kind]
        summary[kind]={k:float(np.mean([r[k] for r in part])) for k in ['psnr','ssim','mae','input_mse','output_mse']}
    write_json(args.output,dict(checkpoint_sha256=sha256(args.checkpoint) if args.checkpoint else None,mild_threshold=args.mild_threshold,
                               ids=ids,seed=914,scope='central 256px validation crops; synthetic diagnostic, not real-camera generalization',summary=summary,rows=rows))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
