"""Offline competition entry point; neural and explicit classical inference."""
from __future__ import annotations
import argparse
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image
from common import SCRIPTS,to_uint8,write_json,sha256
from classical import wavelet_estimate

def output_name(path):
    stem=path.stem
    if stem.lower().endswith('_noise'):stem=stem[:-6]
    if not stem:raise ValueError(f'Empty output name: {path.name}')
    return stem+'.png'

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--noise_dir','--input_dir',dest='noise_dir',required=True,type=Path)
    ap.add_argument('--denoised_dir','--output_dir',dest='denoised_dir',required=True,type=Path)
    ap.add_argument('--checkpoint',type=Path,default=SCRIPTS/'checkpoints'/'selected.pt')
    ap.add_argument('--method',choices=['neural','wavelet'],default='neural')
    ap.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--tile',type=int,default=512)
    ap.add_argument('--overlap',type=int,default=64)
    ap.add_argument('--overwrite',action='store_true')
    ap.add_argument('--manifest',type=Path)
    args=ap.parse_args()
    src=args.noise_dir.resolve();dest=args.denoised_dir.resolve()
    if not src.is_dir():ap.error('Input directory does not exist')
    if src==dest or src in dest.parents or dest in src.parents:ap.error('Input/output directories must not overlap')
    if args.threads<1:ap.error('--threads must be positive')
    if args.tile<16 or not 0<=args.overlap<args.tile:ap.error('Invalid tile/overlap')
    files=sorted(p for p in src.iterdir() if p.is_file() and p.suffix.lower() in {'.png','.jpg','.jpeg'})
    if not files:ap.error('No supported images found')
    names=[output_name(p) for p in files]
    if len({n.lower() for n in names})!=len(names):ap.error('Output filenames collide')
    if not args.overwrite and any((dest/n).exists() for n in names):ap.error('Output exists; choose another directory or use --overwrite')
    cv2.setNumThreads(args.threads)
    if args.method=='neural':
        import torch
        from inference import choose_device,load_model,predict
        torch.set_num_threads(args.threads)
        device=choose_device(args.device);model,metadata=load_model(args.checkpoint,device)
    else:device='cpu'
    dest.mkdir(parents=True,exist_ok=True)
    records=[];start=time.perf_counter()
    for path,name in zip(files,names):
        t=time.perf_counter()
        with Image.open(path) as im:
            alpha=im.getchannel('A').copy() if 'A' in im.getbands() else None
            rgb=np.asarray(im.convert('RGB'),dtype=np.float32)/255
        result=predict(model,rgb,device,args.tile,args.overlap) if args.method=='neural' else wavelet_estimate(rgb)
        out=Image.fromarray(to_uint8(result))
        if alpha is not None:out.putalpha(alpha)
        temporary=dest/(name+'.tmp')
        out.save(temporary,format='PNG');temporary.replace(dest/name)
        elapsed=time.perf_counter()-t
        records.append(dict(input=path.name,output=name,seconds=elapsed,sha256=sha256(dest/name)))
        print(f'{path.name} -> {name} ({elapsed:.2f}s, {device})',flush=True)
    manifest=dict(method=args.method,device=str(device),tile=args.tile,overlap=args.overlap,
                  checkpoint_sha256=sha256(args.checkpoint) if args.method=='neural' else None,
                  total_seconds=time.perf_counter()-start,images=records)
    if args.manifest:write_json(args.manifest,manifest)
    print(f'Completed {len(files)} images in {manifest["total_seconds"]:.2f}s',flush=True)

if __name__=='__main__':main()
