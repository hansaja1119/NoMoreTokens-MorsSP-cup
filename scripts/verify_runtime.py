"""Trained-checkpoint repeatability, CPU/CUDA consistency and tile comparison."""
import argparse
from pathlib import Path
import json
import time
import cv2
import numpy as np
import torch
from common import SCRIPTS,public_root,read_rgb,load_split,write_json,sha256,to_uint8
from inference import predict,load_model
from metrics import official

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path);args=ap.parse_args()
    torch.set_num_threads(4);cv2.setNumThreads(1)
    id_=load_split('val')[0];y=read_rgb(public_root()/'noisy'/f'{id_}_noise.png')
    gt=read_rgb(public_root()/'ground_truth'/f'{id_}.png')
    cpu=torch.device('cpu');model,_=load_model(args.checkpoint,cpu)
    t=time.perf_counter();a=predict(model,y,cpu);elapsed=time.perf_counter()-t
    b=predict(model,y,cpu)
    result=dict(id=id_,checkpoint_sha256=sha256(args.checkpoint),cpu_seconds=elapsed,
                cpu_repeat_max_abs=float(np.max(np.abs(a-b))),cpu_repeat_png_exact=bool(np.array_equal(to_uint8(a),to_uint8(b))))
    if not result['cpu_repeat_png_exact']:raise AssertionError('CPU repeat is not reproducible')
    if torch.cuda.is_available():
        gpu=torch.device('cuda');model=model.to(gpu)
        c=predict(model,y,gpu);d=predict(model,y,gpu,tile=1024)
        result.update(cpu_gpu_max_abs=float(np.max(np.abs(a-c))),
                      cpu_gpu_max_png_difference=int(np.max(np.abs(to_uint8(a).astype('int16')-to_uint8(c).astype('int16')))),
                      tiled_psnr=float(official.sk_psnr(gt,to_uint8(c).astype('float32')/255,data_range=1)),
                      full_frame_psnr=float(official.sk_psnr(gt,to_uint8(d).astype('float32')/255,data_range=1)))
        if result['cpu_gpu_max_png_difference']>1:raise AssertionError('CPU/GPU output mismatch beyond one quantization level')
    write_json(args.output,result);print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
