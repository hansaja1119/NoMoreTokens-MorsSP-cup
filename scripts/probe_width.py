"""Measure real optimizer-step VRAM at the largest training crop, without data access."""
import argparse
import copy
import time
import torch
from model import Denoiser
from common import write_json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--width',type=int,required=True)
    ap.add_argument('--batch',type=int,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    torch.set_num_threads(4);torch.manual_seed(20260908)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    result=dict(width=args.width,batch=args.batch,crop=256,accumulate=16//args.batch)
    try:
        model=Denoiser(args.width).cuda();ema=copy.deepcopy(model).eval()
        for p in ema.parameters():p.requires_grad_(False)
        optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
        scaler=torch.amp.GradScaler('cuda');inp=torch.rand(args.batch,9,256,256,device='cuda')
        target=torch.rand(args.batch,3,256,256,device='cuda')
        torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            for _ in range(result['accumulate']):
                with torch.autocast('cuda',dtype=torch.float16):
                    loss=torch.nn.functional.mse_loss(model(inp).float(),target)/result['accumulate']
                scaler.scale(loss).backward()
            scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            scaler.step(optimizer);scaler.update()
            with torch.no_grad():
                for ep,p in zip(ema.parameters(),model.parameters()):ep.lerp_(p,.001)
        torch.cuda.synchronize()
        total=torch.cuda.get_device_properties(0).total_memory
        peak=torch.cuda.max_memory_reserved()
        result.update(state='ok',parameters=sum(p.numel() for p in model.parameters()),
                      peak_allocated_mb=torch.cuda.max_memory_allocated()/1024**2,
                      peak_reserved_mb=peak/1024**2,total_mb=total/1024**2,
                      headroom_pass=peak<=min(total*.86,total-1024**3),seconds_per_step=(time.perf_counter()-started)/2)
    except torch.cuda.OutOfMemoryError:
        result.update(state='out_of_memory',headroom_pass=False)
    write_json(args.output,result);print(result,flush=True)


if __name__=='__main__':main()
