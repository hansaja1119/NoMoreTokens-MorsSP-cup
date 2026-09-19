"""Run the fixed final recipe after development selection, without publishing/freezing.

The one-time locked test precedes all-data training. Its score is recorded for
the development checkpoint, not falsely attributed to the all-data checkpoint.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import torch
from common import SCRIPTS,ROOT,write_json,sha256,load_split

WORK=SCRIPTS/'runs'/'final_preparation'

def verify_selection(selection,selected,queue):
    """Recheck recorded promotion gates before allowing held-out assessment."""
    if json.loads((queue/'status.json').read_text())['state']!='complete':
        raise RuntimeError('Development selection is not complete')
    if sha256(selected)!=selection['checkpoint_sha256']:
        raise RuntimeError('Selected checkpoint checksum changed')
    runtime=json.loads((queue/'selected_runtime.json').read_text())
    if runtime['checkpoint_sha256']!=selection['checkpoint_sha256']:
        raise RuntimeError('Runtime checks belong to a different checkpoint')
    if not runtime['cpu_repeat_png_exact'] or runtime.get('cpu_gpu_max_png_difference',0)>1:
        raise RuntimeError('Selected checkpoint failed repeatability checks')
    checks=json.loads((queue/'robustness_selection.json').read_text())
    accepted=[check for check in checks if check['threshold']==selection['mild_threshold'] and check['passes']]
    if len(accepted)!=1:raise RuntimeError('Selected protection has no passing robustness record')
    diagnostics=accepted[0]['diagnostics']
    if diagnostics['clean']['mae']>.003 or diagnostics['mild_gaussian']['output_mse']>1.1*diagnostics['mild_gaussian']['input_mse']:
        raise RuntimeError('Recorded robustness measurements do not pass the selection criteria')

def command(label,script,*args):
    argv=[sys.executable,str(SCRIPTS/script),*map(str,args)]
    write_json(WORK/'status.json',dict(state='running',stage=label,command=argv,updated_unix=time.time()))
    print('START',label,flush=True)
    with (WORK/f'{label}.log').open('a',encoding='utf-8') as log:
        subprocess.run(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('DONE',label,flush=True)

def final_train(run,steps,width,rgb,real_only,init=None,ssim=0.,schedule=30000):
    folder=SCRIPTS/'runs'/run
    if (folder/'final.pt').exists():
        if torch.load(folder/'final.pt',map_location='cpu',weights_only=True)['step']==steps:return folder/'final.pt'
        raise RuntimeError('Existing final checkpoint has a different training duration')
    args=['--run',run,'--steps',steps,'--width',width,'--final-training','--schedule-steps',schedule]
    if rgb:args+=['--rgb-only']
    if real_only:args+=['--no-synthetic']
    if init:args+=['--init',init,'--warmup-steps',0,'--lr',5e-5]
    if ssim:args+=['--ssim-weight',ssim]
    if (folder/'last.pt').exists():
        if init:
            i=args.index('--init');del args[i:i+2]
        args+=['--resume',folder/'last.pt']
    command(run,'train.py',*args)
    return folder/'final.pt'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--wait-for-queue',action='store_true');args=ap.parse_args()
    WORK.mkdir(parents=True,exist_ok=True)
    while True:
        status=json.loads((SCRIPTS/'runs'/'experiment_queue'/'status.json').read_text())
        if status['state']=='complete':break
        if status['state']=='failed':raise RuntimeError('Experiment queue failed; do not open the locked test')
        if not args.wait_for_queue:raise RuntimeError('Development experiments are not complete')
        write_json(WORK/'status.json',dict(state='waiting',stage='development_selection',updated_unix=time.time()))
        time.sleep(30)
    lock=WORK/'preparation.lock'
    with lock.open('x') as f:
        import os
        f.write(str(os.getpid()))
    try:
        selection=json.loads((SCRIPTS/'checkpoints'/'selection.json').read_text())
        selected=SCRIPTS/'checkpoints'/'selected.pt'
        verify_selection(selection,selected,SCRIPTS/'runs'/'experiment_queue')
        architecture=json.loads((SCRIPTS/'runs'/'experiment_queue'/'architecture_comparison.json').read_text())['provisional_winner']
        selected_config=json.loads((SCRIPTS/'runs'/selection['run']/'config.json').read_text())
        base_checkpoint=torch.load(SCRIPTS/'runs'/architecture['run']/'best.pt',map_location='cpu',weights_only=True)
        chosen_checkpoint=torch.load(selected,map_location='cpu',weights_only=True)
        if chosen_checkpoint['split_sha256']!=sha256(SCRIPTS/'data'/'split.json'):
            raise RuntimeError('Selected model belongs to another data split')
        if chosen_checkpoint['model_config']!=base_checkpoint['model_config']:
            raise RuntimeError('Selected model and base recipe have different architectures')
        recipe=dict(architecture=architecture,selection=selection,base_steps=base_checkpoint['step'],
                    base_schedule_steps=30000,fine_schedule_steps=selected_config['schedule_steps'],
                    fine_steps=chosen_checkpoint['step'] if selected_config['ssim_weight'] else 0,
                    ssim_weight=selected_config['ssim_weight'],
                    declaration='Recipe fixed before locked test. No tuning on locked-test results.')
        recipe_path=WORK/'fixed_recipe.json'
        if recipe_path.exists() and json.loads(recipe_path.read_text())!=recipe:
            raise RuntimeError('Refusing to change a recipe after the locked assessment has been prepared')
        write_json(recipe_path,recipe)
        test=WORK/'locked_test'/'metrics.json'
        if not test.exists():
            command('locked_test','evaluate_checkpoint.py','--checkpoint',selected,'--output',test.parent,
                    '--split','test','--allow-locked-test','--device','cuda')
        assessment=json.loads(test.read_text())
        if assessment['checkpoint_sha256']!=selection['checkpoint_sha256']:
            raise RuntimeError('Locked test belongs to another checkpoint; do not evaluate another selection')
        if assessment['split']!='test' or sorted(row['id'] for row in assessment['rows'])!=sorted(load_split('test')):
            raise RuntimeError('Locked assessment must cover the complete held-out partition')
        command('all_data_cache','prepare_cache.py','--include-locked-test')
        weights=final_train('final_all_mse',recipe['base_steps'],architecture['width'],architecture['rgb'],
                            architecture['no_synthetic'],schedule=recipe['base_schedule_steps'])
        if recipe['fine_steps']:
            weights=final_train('final_all_ssim',recipe['fine_steps'],architecture['width'],architecture['rgb'],
                                architecture['no_synthetic'],init=weights,ssim=recipe['ssim_weight'],schedule=recipe['fine_schedule_steps'])
        checkpoint=SCRIPTS/'checkpoints'/'NoMoreTokens_final_candidate.pt'
        if not checkpoint.exists():
            command('export','export_checkpoint.py','--source',weights,'--output',checkpoint,'--mild-threshold',selection['mild_threshold'])
        runtime=WORK/'final_runtime.json'
        if not runtime.exists():command('runtime','verify_runtime.py','--checkpoint',checkpoint,'--output',runtime)
        out=SCRIPTS/'outputs'/'final_candidate'
        manifest=out/'inference_manifest.json'
        noisy=ROOT/'submissions'/'noisy'
        if not noisy.is_dir():noisy=ROOT/'competition_data'/'submissions'/'noisy'
        if not manifest.exists():
            command('preliminary_images','denoise.py','--noise_dir',noisy,'--denoised_dir',out/'images',
                    '--checkpoint',checkpoint,'--device','cpu','--manifest',manifest)
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        if not (out/'NoMoreTokens.zip').exists():
            command('package','package_submission.py','--images',out/'images','--output-dir',out,
                    '--inference-manifest',manifest,'--checkpoint',checkpoint,'--commit',revision)
        write_json(WORK/'status.json',dict(state='complete',checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),
                  locked_test_score=json.loads(test.read_text())['summary']['composite_score'],
                  locked_test_checkpoint=selection['checkpoint_sha256'],output=str(out),updated_unix=time.time(),
                  next='Render and review the final report, verify code provenance, then arrange repository access and submission freeze.'))
    except BaseException as exc:
        old=json.loads((WORK/'status.json').read_text()) if (WORK/'status.json').exists() else {}
        write_json(WORK/'status.json',dict(state='failed',stage=old.get('stage'),error=str(exc),updated_unix=time.time()))
        raise
    finally:
        lock.unlink(missing_ok=True)

if __name__=='__main__':main()
