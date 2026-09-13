"""User-requested width 96/128 experiments: each gets 20,000 steps."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import torch
from common import ROOT, SCRIPTS, write_json, sha256

QUEUE = SCRIPTS / 'runs' / 'large_width_experiments'
WIDTHS = (96, 128)
STEPS = 20000


def command(label, script, *args):
    argv = [sys.executable, str(SCRIPTS / script), *map(str, args)]
    log = QUEUE / f'{label}.log'
    write_json(QUEUE / 'status.json', dict(state='running', stage=label,
        command=argv, log=str(log), updated_unix=time.time()))
    print('START', label, flush=True)
    with log.open('a', encoding='utf-8') as stream:
        subprocess.run(argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    print('DONE', label, flush=True)


def main():
    QUEUE.mkdir(parents=True, exist_ok=True)
    with (QUEUE / 'queue.lock').open('x') as stream:
        stream.write(str(os.getpid()))
    try:
        plan = dict(widths=list(WIDTHS), steps=STEPS, schedule_steps=30000,
            validation_every=1000, effective_batch=16, crop=256,
            rule='Train BOTH widths to 20000; no 4000-step qualification gate.',
            split_sha256=sha256(SCRIPTS / 'data' / 'split.json'),
            data='Original 340 training / 60 validation images. No held-out test reuse.')
        plan_path = QUEUE / 'plan.json'
        if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
            raise RuntimeError('Existing large-width experiment plan differs')
        write_json(plan_path, plan)
        # Probe both widths before the long jobs, without competing GPU processes.
        for width in WIDTHS:
            memory = QUEUE / f'width{width}_memory.json'
            if memory.exists():
                continue
            chosen = None
            for batch in ([2, 1] if width == 96 else [1]):
                probe = QUEUE / f'probe_w{width}_b{batch}.json'
                if not probe.exists():
                    command(f'probe_w{width}_b{batch}', 'probe_width.py',
                        '--width', width, '--batch', batch, '--output', probe)
                measured = json.loads(probe.read_text())
                if measured.get('state') == 'ok' and measured.get('headroom_pass'):
                    chosen = measured
                    break
            if chosen is None:
                raise RuntimeError(f'Width {width} did not fit with memory headroom; review probe logs before changing the recipe.')
            write_json(memory, chosen)
            print(f'Width {width}: batch {chosen["batch"]}, accumulation {16 // chosen["batch"]}, peak reserved {chosen["peak_reserved_mb"]:.0f} MiB', flush=True)
        rows = []
        for width in WIDTHS:
            run = f'hybrid{width}'
            folder = SCRIPTS / 'runs' / run
            batch = json.loads((QUEUE / f'width{width}_memory.json').read_text())['batch']
            args = ['--run', run, '--width', width, '--steps', STEPS,
                '--schedule-steps', 30000, '--batch', batch, '--accumulate', 16 // batch,
                '--val-every', 1000, '--device', 'cuda']
            last = folder / 'last.pt'
            done = False
            if last.exists():
                obj = torch.load(last, map_location='cpu', weights_only=True, mmap=True)
                old = json.loads((folder / 'config.json').read_text())
                if (old['batch'], old['accumulate'], old['final_training'], old['width']) != (batch, 16 // batch, False, width):
                    raise RuntimeError(f'Existing {run} has a different recipe')
                if obj['split_sha256'] != plan['split_sha256']:
                    raise RuntimeError('Checkpoint split mismatch')
                done = obj['step'] >= STEPS and (folder / f'val_{STEPS:06d}.json').exists()
                args += ['--resume', last]
                del obj
            if not done:
                if shutil.disk_usage(ROOT).free < 10 * 1024**3:
                    raise RuntimeError('Less than 10 GiB free disk space; make room before resuming.')
                command(f'{run}_{STEPS}', 'train.py', *args)
            best = folder / 'best.pt'
            obj = torch.load(best, map_location='cpu', weights_only=True, mmap=True)
            row = dict(width=width, run=run, batch=batch, accumulate=16 // batch,
                best_step=obj['step'], best_score=obj['best_score'],
                best_checkpoint_sha256=sha256(best))
            del obj
            output = folder / 'width_cpu'
            if not (output / 'metrics.json').exists():
                command(run + '_cpu', 'evaluate_checkpoint.py', '--checkpoint', best,
                    '--output', output, '--device', 'cpu', '--limit', 3)
            row['cpu_median_seconds'] = json.loads((output / 'metrics.json').read_text())['summary']['median_seconds']
            robust = folder / 'width_robustness.json'
            if not robust.exists():
                command(run + '_robustness', 'robustness.py', '--checkpoint', best,
                    '--output', robust, '--device', 'cuda', '--mild-threshold', .03)
            row['robustness_file'] = str(robust)
            rows.append(row)
            write_json(QUEUE / 'comparison.json', dict(rows=rows))
        write_json(QUEUE / 'status.json', dict(state='complete', updated_unix=time.time(),
            results=str(QUEUE / 'comparison.json'), next='Compare validation and runtime before replacing the existing submission candidate.'))
    except BaseException as exc:
        old = json.loads((QUEUE / 'status.json').read_text()) if (QUEUE / 'status.json').exists() else {}
        write_json(QUEUE / 'status.json', dict(state='failed', stage=old.get('stage'),
            error=str(exc), updated_unix=time.time()))
        raise
    finally:
        (QUEUE / 'queue.lock').unlink(missing_ok=True)


if __name__ == '__main__':
    main()
