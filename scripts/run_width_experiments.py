"""Compare widths 48/64 at 4k and extend each strict winner to 20k.

Keeps the previous selected model, ZIP and held-out assessment unchanged.
Only the original 340-image train and 60-image validation partitions are used.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import torch
from common import ROOT,SCRIPTS,write_json,sha256

QUEUE=SCRIPTS/'runs'/'width_experiments'


def qualifies(score,reference):return score>reference


def command(label,script,*args):
    argv=[sys.executable,str(SCRIPTS/script),*map(str,args)];log=QUEUE/f'{label}.log'
    write_json(QUEUE/'status.json',dict(state='running',stage=label,command=argv,log=str(log),updated_unix=time.time()))
    print('START',label,flush=True)
    with log.open('a',encoding='utf-8') as stream:
        subprocess.run(argv,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    print('DONE',label,flush=True)


def train(width,batch,steps):
    run=f'hybrid{width}';folder=SCRIPTS/'runs'/run
    args=['--run',run,'--width',width,'--steps',steps,'--schedule-steps',30000,
          '--batch',batch,'--accumulate',16//batch,'--val-every',1000,'--device','cuda']
    last=folder/'last.pt'
    if last.exists():
        obj=torch.load(last,map_location='cpu',weights_only=True)
        old=json.loads((folder/'config.json').read_text())
        if old['batch']!=batch or old['accumulate']!=16//batch or old['final_training']:
            raise RuntimeError(f'Existing {run} has a different training recipe')
        if obj['step']>=steps and (folder/f'val_{steps:06d}.json').exists():return
        args+=['--resume',last]
    command(f'{run}_{steps}','train.py',*args)


def main():
    QUEUE.mkdir(parents=True,exist_ok=True)
    with (QUEUE/'queue.lock').open('x') as stream:stream.write(str(os.getpid()))
    try:
        baseline=SCRIPTS/'runs'/'hybrid32'/'val_004000.json'
        reference=json.loads(baseline.read_text())['summary']['composite_score']
        plan=dict(widths=[48,64],comparison_steps=4000,refine_steps=20000,
                  reference_score=reference,reference_sha256=sha256(baseline),effective_batch=16,
                  rule='Both widths run to 4000 first. Extend EACH width with exact step-4000 score strictly above width 32 to 20000.',
                  data='Original train/validation split only. Earlier locked assessment is not reused for selection.')
        if (QUEUE/'plan.json').exists() and json.loads((QUEUE/'plan.json').read_text())!=plan:
            raise RuntimeError('Existing width experiment plan differs')
        write_json(QUEUE/'plan.json',plan);rows=[]
        for width in plan['widths']:
            memory=QUEUE/f'width{width}_memory.json'
            if not memory.exists():
                chosen=None
                for batch in [8,4,2,1]:
                    probe=QUEUE/f'probe_w{width}_b{batch}.json'
                    if not probe.exists():command(f'probe_w{width}_b{batch}','probe_width.py','--width',width,'--batch',batch,'--output',probe)
                    measured=json.loads(probe.read_text())
                    if measured['state']=='ok' and measured['headroom_pass']:chosen=measured;break
                if chosen is None:raise RuntimeError(f'Width {width} cannot fit with safe memory headroom')
                write_json(memory,chosen)
            batch=json.loads(memory.read_text())['batch'];train(width,batch,4000)
            folder=SCRIPTS/'runs'/f'hybrid{width}'
            checkpoint=folder/'comparison_4000.pt'
            if not checkpoint.exists():
                if torch.load(folder/'last.pt',map_location='cpu',weights_only=True)['step']!=4000:
                    raise RuntimeError('Cannot reconstruct the exact 4000-step checkpoint')
                shutil.copy2(folder/'last.pt',checkpoint)
            score=json.loads((folder/'val_004000.json').read_text())['summary']['composite_score']
            rows.append(dict(width=width,run=f'hybrid{width}',batch=batch,accumulate=16//batch,
                             score_4000=score,delta_vs_width32=score-reference,extend_to_20000=qualifies(score,reference),
                             checkpoint_4000_sha256=sha256(checkpoint)))
            write_json(QUEUE/'comparison.json',dict(reference_score=reference,rows=rows))
        for row in rows:
            if row['extend_to_20000']:train(row['width'],row['batch'],20000)
            folder=SCRIPTS/'runs'/row['run'];best=folder/'best.pt'
            obj=torch.load(best,map_location='cpu',weights_only=True)
            row.update(best_step=obj['step'],best_score=obj['best_score'],best_checkpoint_sha256=sha256(best))
            output=folder/'width_cpu'
            if not (output/'metrics.json').exists():
                command(row['run']+'_cpu','evaluate_checkpoint.py','--checkpoint',best,'--output',output,'--device','cpu','--limit',3)
            row['cpu_median_seconds']=json.loads((output/'metrics.json').read_text())['summary']['median_seconds']
            robust=folder/'width_robustness.json'
            if not robust.exists():
                command(row['run']+'_robustness','robustness.py','--checkpoint',best,'--output',robust,'--device','cuda','--mild-threshold',.03)
            row['robustness_file']=str(robust)
            write_json(QUEUE/'comparison.json',dict(reference_score=reference,rows=rows))
        write_json(QUEUE/'status.json',dict(state='complete',updated_unix=time.time(),
                   results=str(QUEUE/'comparison.json'),next='Review new width results before replacing the preserved width-32 candidate.'))
    except BaseException as exc:
        old=json.loads((QUEUE/'status.json').read_text()) if (QUEUE/'status.json').exists() else {}
        write_json(QUEUE/'status.json',dict(state='failed',stage=old.get('stage'),error=str(exc),updated_unix=time.time()))
        raise
    finally:(QUEUE/'queue.lock').unlink(missing_ok=True)


if __name__=='__main__':main()
