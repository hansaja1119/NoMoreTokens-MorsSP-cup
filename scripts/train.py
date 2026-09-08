"""Resumable offline GPU/CPU training; full-image validation and exact score selection."""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
import random
import shutil
import time
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from common import SCRIPTS,public_root,read_rgb,load_split,to_uint8,write_json,sha256
from inference import choose_device,predict
from model import Denoiser
from metrics import score_arrays,summarize
from training_data import TrainingCrops

def ssim_loss(x,y):
    ux=F.avg_pool2d(x,7,1);uy=F.avg_pool2d(y,7,1)
    vx=(F.avg_pool2d(x*x,7,1)-ux*ux)*(49/48)
    vy=(F.avg_pool2d(y*y,7,1)-uy*uy)*(49/48)
    cov=(F.avg_pool2d(x*y,7,1)-ux*uy)*(49/48)
    s=((2*ux*uy+.01**2)*(2*cov+.03**2))/((ux*ux+uy*uy+.01**2)*(vx+vy+.03**2))
    return 1-s.mean()

def save_checkpoint(path,obj):
    temp=path.with_suffix('.tmp');torch.save(obj,temp);temp.replace(path)

def evaluate(model,device,ids):
    rows=[]
    model.eval()
    for i,id_ in enumerate(ids):
        y=read_rgb(public_root()/'noisy'/f'{id_}_noise.png');gt=read_rgb(public_root()/'ground_truth'/f'{id_}.png')
        t=time.perf_counter();pred=predict(model,y,device)
        elapsed=time.perf_counter()-t;pred=to_uint8(pred).astype('float32')/255
        rows.append(dict(id=id_,seconds=elapsed,**score_arrays(y,pred,gt)))
        if (i+1)%20==0:print(f'Validation {i+1}/{len(ids)}',flush=True)
    return dict(summary=summarize(rows),rows=rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',required=True)
    ap.add_argument('--steps',type=int,default=1000)
    ap.add_argument('--schedule-steps',type=int,default=30000)
    ap.add_argument('--width',type=int,default=16)
    ap.add_argument('--rgb-only',action='store_true')
    ap.add_argument('--no-synthetic',action='store_true')
    ap.add_argument('--batch',type=int,default=8)
    ap.add_argument('--accumulate',type=int,default=2)
    ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--warmup-steps',type=int,default=1000)
    ap.add_argument('--val-every',type=int,default=1000)
    ap.add_argument('--lr',type=float,default=2e-4)
    ap.add_argument('--ssim-weight',type=float,default=0.)
    ap.add_argument('--device',default='auto',choices=['auto','cuda','cpu'])
    ap.add_argument('--resume',type=Path)
    ap.add_argument('--init',type=Path)
    ap.add_argument('--seed',type=int,default=20260908)
    ap.add_argument('--final-training',action='store_true',help='All 460 public pairs; disables validation and selection')
    args=ap.parse_args()
    if Path(args.run).name!=args.run:ap.error('--run must be a single directory name')
    if args.resume and args.init:ap.error('Use only one of --resume / --init')
    out=SCRIPTS/'runs'/args.run;out.mkdir(parents=True,exist_ok=True)
    if (out/'last.pt').exists() and not args.resume:ap.error('Existing run; use --resume or a new name')
    cache=SCRIPTS/'cache'/'features_v1'/'manifest.json'
    manifest=json.loads(cache.read_text())
    if manifest['classical_sha256']!=sha256(SCRIPTS/'classical.py') or manifest['split_sha256']!=sha256(SCRIPTS/'data'/'split.json'):
        raise RuntimeError('Stale feature cache: rebuild it explicitly before training')
    if args.final_training:
        if not (SCRIPTS/'runs'/'final_preparation'/'locked_test'/'metrics.json').exists():
            raise RuntimeError('Final all-data training requires the recorded one-time locked test')
        if set(manifest['ids'])!=set(load_split('all')):raise RuntimeError('Prepare the all-data cache first')
    cv2.setNumThreads(1);torch.set_num_threads(4)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    device=choose_device(args.device)
    if device.type=='cuda':torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    model=Denoiser(args.width,not args.rgb_only).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    ema=copy.deepcopy(model).eval()
    for p in ema.parameters():p.requires_grad_(False)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    first=0;best=-1.;elapsed_before=0.
    if args.resume or args.init:
        obj=torch.load(args.resume or args.init,map_location='cpu',weights_only=True)
        if obj['model_config']!=model.config:raise ValueError('Checkpoint architecture mismatch')
        model.load_state_dict(obj['model'] if args.resume else obj.get('ema',obj['model']))
        ema.load_state_dict(obj['ema'])
        if args.resume:
            optimizer.load_state_dict(obj['optimizer']);scaler.load_state_dict(obj['scaler'])
            first=obj['step'];best=obj['best_score'];elapsed_before=obj.get('elapsed_seconds',0.)
    if first>=args.steps:ap.error('--steps must exceed the resumed step')
    effective=args.batch*args.accumulate
    dataset=TrainingCrops(args.steps,effective,not args.rgb_only,not args.no_synthetic,args.seed,first,args.warmup_steps,args.final_training)
    loader=DataLoader(dataset,batch_size=args.batch,num_workers=args.workers,pin_memory=device.type=='cuda',
                      persistent_workers=args.workers>0,shuffle=False)
    iterator=iter(loader)
    source_hashes={p.name:sha256(p) for p in SCRIPTS.glob('*.py')}
    archive=out/f'source_from_step_{first:06d}'
    archive.mkdir(exist_ok=True)
    for name in source_hashes:shutil.copy2(SCRIPTS/name,archive/name)
    config=dict(**{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                model=model.config,source_hashes=source_hashes,split_sha256=sha256(SCRIPTS/'data'/'split.json'))
    write_json(out/f'config_from_step_{first:06d}.json',config)
    if first==0:write_json(out/'config.json',config)
    print(f'Training {args.run} on {device}: {sum(p.numel() for p in model.parameters()):,} parameters; effective batch {effective}',flush=True)
    started=time.perf_counter();running=[]
    for step in range(first+1,args.steps+1):
        model.train();optimizer.zero_grad(set_to_none=True)
        lr=args.lr*(.05+.95*.5*(1+math.cos(math.pi*min(step,args.schedule_steps)/args.schedule_steps)))
        for group in optimizer.param_groups:group['lr']=lr
        loss_value=0.
        for _ in range(args.accumulate):
            inp,target=next(iterator);inp=inp.to(device,non_blocking=True);target=target.to(device,non_blocking=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                pred=model(inp)
                loss=F.mse_loss(pred.float(),target)
                if args.ssim_weight:loss=loss+args.ssim_weight*ssim_loss(pred.float(),target)
            scaler.scale(loss/args.accumulate).backward();loss_value+=loss.detach().item()/args.accumulate
        scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        scaler.step(optimizer);scaler.update()
        decay=min(.999,1-1/(step+1))
        with torch.no_grad():
            for ep,p in zip(ema.parameters(),model.parameters()):ep.lerp_(p,1-decay)
        running.append(loss_value)
        elapsed=time.perf_counter()-started+elapsed_before
        if step%50==0 or step==first+1:
            record=dict(step=step,loss=float(np.mean(running)),lr=lr,elapsed_seconds=elapsed,
                        gpu_peak_mb=torch.cuda.max_memory_allocated()/1024**2 if device.type=='cuda' else 0.)
            with (out/'train.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True);running=[]
            write_json(out/'status.json',dict(state='training',**record,best_score=best))
        obj=dict(model_config=model.config,model=model.state_dict(),ema=ema.state_dict(),optimizer=optimizer.state_dict(),
                 scaler=scaler.state_dict(),step=step,best_score=best,elapsed_seconds=elapsed,
                 split_sha256=sha256(SCRIPTS/'data'/'split.json'))
        if step%250==0:save_checkpoint(out/'last.pt',obj)
        if not args.final_training and (step%args.val_every==0 or step==args.steps):
            write_json(out/'status.json',dict(state='validating',step=step,best_score=best))
            result=evaluate(ema,device,load_split('val'))
            result.update(step=step);write_json(out/f'val_{step:06d}.json',result)
            score=result['summary']['composite_score'];print(f'VALIDATION step={step} '+json.dumps(result['summary']),flush=True)
            if score>best:
                best=score;obj['best_score']=best;save_checkpoint(out/'best.pt',obj)
            obj['best_score']=best;save_checkpoint(out/'last.pt',obj)
        elif args.final_training and step==args.steps:
            obj['best_score']=None;obj['validation_enabled']=False
            save_checkpoint(out/'last.pt',obj);save_checkpoint(out/'final.pt',obj)
    write_json(out/'status.json',dict(state='complete',step=args.steps,best_score=best,
                                     elapsed_seconds=time.perf_counter()-started+elapsed_before,final_training=args.final_training))

if __name__=='__main__':main()
