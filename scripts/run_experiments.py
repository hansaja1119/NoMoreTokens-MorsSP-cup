"""Sequential local experiment queue. No upload, locked-test use, or code freeze.

Can safely be resumed: completed training stages and diagnostics are skipped.
One queue per workspace. Logs and status remain on disk if the UI closes.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import torch
from common import SCRIPTS,ROOT,write_json,sha256

QUEUE=SCRIPTS/'runs'/'experiment_queue'

def run_command(label,script,*args):
    command=[sys.executable,str(SCRIPTS/script),*map(str,args)]
    log=QUEUE/f'{label}.log'
    write_json(QUEUE/'status.json',dict(state='running',stage=label,command=command,log=str(log),updated_unix=time.time()))
    print('START',label,flush=True)
    with log.open('a',encoding='utf-8') as out:
        subprocess.run(command,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=True)
    print('DONE',label,flush=True)

def train_stage(run,steps,*,rgb=False,width=16,no_synthetic=False,init=None,ssim=0.,schedule=30000):
    folder=SCRIPTS/'runs'/run
    config=['--run',run,'--steps',steps,'--width',width,'--schedule-steps',schedule,'--val-every',1000]
    if rgb:config+=['--rgb-only']
    if no_synthetic:config+=['--no-synthetic']
    if init:config+=['--init',init,'--warmup-steps',0,'--lr',5e-5]
    if ssim:config+=['--ssim-weight',ssim]
    last=folder/'last.pt'
    if last.exists():
        obj=torch.load(last,map_location='cpu',weights_only=True)
        if obj['step']>=steps and (folder/f'val_{steps:06d}.json').exists():return
        if init:
            i=config.index('--init');del config[i:i+2]
        config+=['--resume',last]
    run_command(f'{run}_{steps}', 'train.py',*config)

def best_score(run):
    obj=torch.load(SCRIPTS/'runs'/run/'best.pt',map_location='cpu',weights_only=True)
    return float(obj['best_score'])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--comparison-steps',type=int,default=4000)
    ap.add_argument('--refine-steps',type=int,default=20000)
    ap.add_argument('--fine-steps',type=int,default=2000)
    args=ap.parse_args()
    QUEUE.mkdir(parents=True,exist_ok=True)
    lock=QUEUE/'queue.lock'
    try:
        lock_handle=lock.open('x')
    except FileExistsError:raise SystemExit('Queue lock exists. Verify no queue is running before removing a stale lock.')
    import os
    lock_handle.write(str(os.getpid()));lock_handle.close()
    write_json(QUEUE/'plan.json',dict(**vars(args),created_unix=time.time(),note='Development experiment queue; no locked test or final freeze'))
    try:
        candidates=[('hybrid16',False,16,False),('rgb16',True,16,False),
                    ('hybrid32',False,32,False),('hybrid16_real_only',False,16,True)]
        for name,rgb,width,no_synthetic in candidates:
            train_stage(name,args.comparison_steps,rgb=rgb,width=width,no_synthetic=no_synthetic)
        scored=[]
        for name,rgb,width,no_synthetic in candidates:
            diag=SCRIPTS/'runs'/name/'queue_cpu'
            if not (diag/'metrics.json').exists():
                run_command(name+'_cpu','evaluate_checkpoint.py','--checkpoint',SCRIPTS/'runs'/name/'best.pt',
                            '--output',diag,'--device','cpu','--limit',3)
            cpu=json.loads((diag/'metrics.json').read_text())['summary']['median_seconds']
            robust=SCRIPTS/'runs'/name/'queue_robustness.json'
            if not robust.exists():run_command(name+'_robustness','robustness.py','--checkpoint',SCRIPTS/'runs'/name/'best.pt',
                                               '--output',robust,'--device','cuda','--mild-threshold',.03)
            stress=json.loads(robust.read_text())['summary']
            noise_conditions=['mild_gaussian','strong_gaussian','signal_dependent','spatial_mixture','gray_noise','correlated_color']
            stress_psnr=sum(stress[c]['psnr'] for c in noise_conditions)/len(noise_conditions)
            scored.append(dict(run=name,score=best_score(name),cpu_seconds=cpu,stress_psnr=stress_psnr,
                               clean_mae=stress['clean']['mae'],rgb=rgb,width=width,no_synthetic=no_synthetic))
        reference=next(r for r in scored if r['run']=='rgb16')['stress_psnr']
        eligible=[r for r in scored if r['cpu_seconds']<=30 and r['stress_psnr']>=reference-.2 and r['clean_mae']<=.003]
        if not eligible:raise RuntimeError('No candidate meets the CPU target; no automatic promotion')
        maximum=max(r['score'] for r in eligible)
        winner=min([r for r in eligible if r['score']>=maximum-.003],key=lambda r:r['cpu_seconds'])
        write_json(QUEUE/'architecture_comparison.json',dict(candidates=scored,provisional_winner=winner))
        train_stage(winner['run'],args.refine_steps,rgb=winner['rgb'],width=winner['width'],no_synthetic=winner['no_synthetic'])
        anchor=SCRIPTS/'runs'/winner['run']/'best.pt'
        finalists=[winner['run']]
        for weight in [.05,.1]:
            name=f'{winner["run"]}_ssim{weight:g}'
            train_stage(name,args.fine_steps,rgb=winner['rgb'],width=winner['width'],
                        no_synthetic=winner['no_synthetic'],init=anchor,ssim=weight,schedule=args.fine_steps)
            finalists.append(name)
        selected=max(finalists,key=best_score)
        source=SCRIPTS/'runs'/selected/'best.pt'
        grid=SCRIPTS/'runs'/selected/'protection_grid.json'
        if not grid.exists():run_command(selected+'_protection','calibrate_protection.py','--checkpoint',source,'--output',grid,'--device','cuda')
        tests=[]
        for threshold in [0.,.02,.03,.04]:
            path=SCRIPTS/'runs'/selected/f'robustness_threshold{threshold:g}.json'
            if not path.exists():run_command(f'{selected}_robustness{threshold:g}','robustness.py','--checkpoint',source,'--output',path,'--device','cuda','--mild-threshold',threshold)
            diagnostics=json.loads(path.read_text())['summary']
            # Clean and mildly noisy photos must not be materially damaged.
            passes=diagnostics['clean']['mae']<=.003 and diagnostics['mild_gaussian']['output_mse']<=1.1*diagnostics['mild_gaussian']['input_mse']
            tests.append(dict(threshold=threshold,passes=passes,diagnostics=diagnostics))
        write_json(QUEUE/'robustness_selection.json',tests)
        accepted=[t for t in tests if t['passes']]
        if not accepted:raise RuntimeError('No mild-noise protection passes robustness checks; review required before promoting')
        results=json.loads(grid.read_text())['results']
        protection=max(accepted,key=lambda t:results[str(t['threshold'])]['summary']['composite_score'])['threshold']
        checkpoint=torch.load(source,map_location='cpu',weights_only=True)
        export={k:checkpoint[k] for k in ['model_config','ema','step','split_sha256']}
        export['inference_config']=dict(mild_threshold=protection)
        export['development_only']=True
        target=SCRIPTS/'checkpoints'/'selected.pt';target.parent.mkdir(exist_ok=True)
        temp=target.with_suffix('.tmp');torch.save(export,temp);temp.replace(target)
        write_json(SCRIPTS/'checkpoints'/'selection.json',dict(run=selected,checkpoint_sha256=sha256(target),
            validation_composite=results[str(protection)]['summary']['composite_score'],mild_threshold=protection,
            status='development candidate; locked test, full-data training, report and submission freeze still required'))
        runtime=QUEUE/'selected_runtime.json'
        if not runtime.exists():run_command('selected_runtime','verify_runtime.py','--checkpoint',target,'--output',runtime)
        write_json(QUEUE/'status.json',dict(state='complete',selected_run=selected,checkpoint=str(target),updated_unix=time.time(),
            next='Review diagnostics before opening locked test or preparing frozen submission'))
    except BaseException as exc:
        previous=json.loads((QUEUE/'status.json').read_text()) if (QUEUE/'status.json').exists() else {}
        write_json(QUEUE/'status.json',dict(state='failed',error=str(exc),stage=previous.get('stage'),updated_unix=time.time()))
        raise
    finally:
        lock.unlink(missing_ok=True)

if __name__=='__main__':main()
