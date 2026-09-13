"""Read-only live monitor for development and final preparation."""
from datetime import datetime
import argparse
import json
import os
from pathlib import Path
import sys
import time
from validation_progress import collect, terminal_lines, export_history

SCRIPTS=Path(__file__).resolve().parent


def read_json(path):
    try:return json.loads(path.read_text(encoding='utf-8'))
    except (OSError,ValueError):return {}


def process_alive(pid):
    """Read Windows process status without sending any signal."""
    if sys.platform!='win32':return None
    import ctypes
    from ctypes import wintypes
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
    kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=kernel.OpenProcess(0x1000,False,pid)
    if not handle:return False if ctypes.get_last_error()==87 else None
    try:
        code=wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle,ctypes.byref(code)):return None
        return code.value==259
    finally:kernel.CloseHandle(handle)


def completion_state(queue,final,runs):
    large=read_json(runs/'large_width_experiments'/'status.json')
    if large.get('state')=='paused':return 'paused','TRAINING PAUSED - checkpoint saved. You can turn off the laptop.'
    if large.get('state')=='failed':return 'attention',f'Width 96/128 experiments failed: {large.get("error","see log")}'
    if large and large.get('state')!='complete':
        try:pid=int((runs/'large_width_experiments'/'queue.lock').read_text())
        except (OSError,ValueError):pid=None
        if pid is not None and process_alive(pid) is False:return 'attention','Width 96/128 experiment process stopped unexpectedly.'
        return 'running','WIDTH 96 AND 128 EXPERIMENTS RUNNING - keep the laptop awake.'
    extra=read_json(runs/'width_experiments'/'status.json')
    if extra.get('state')=='failed':return 'attention',f'Width experiments failed: {extra.get("error","see log")}'
    if extra and extra.get('state')!='complete':
        try:pid=int((runs/'width_experiments'/'queue.lock').read_text())
        except (OSError,ValueError):pid=None
        if pid is not None and process_alive(pid) is False:return 'attention','Width experiment process stopped unexpectedly.'
        return 'running','WIDTH 48 AND 64 EXPERIMENTS RUNNING - keep the laptop awake.'
    for label,state in [('Development',queue),('Final preparation',final)]:
        if state.get('state')=='failed':return 'attention',f'{label} failed: {state.get("error","see its log")}'
    if queue.get('state')=='complete' and final.get('state')=='complete':
        output=Path(final.get('output',''));checkpoint=Path(final.get('checkpoint',''))
        if (output/'VISION_HUNTERS.zip').is_file() and checkpoint.is_file():
            return 'complete','ALL QUEUED WORK FINISHED - YOU CAN TURN OFF THE LAPTOP.'
        return 'attention','Completion was recorded, but the final checkpoint or ZIP is missing.'
    for folder,state,lockname in [('experiment_queue',queue,'queue.lock'),('final_preparation',final,'preparation.lock')]:
        if state.get('state')!='running':continue
        try:pid=int((runs/folder/lockname).read_text())
        except (OSError,ValueError):continue
        if process_alive(pid) is False:return 'attention',f'{folder} process stopped unexpectedly. Send me this status.'
    if not queue:return 'attention','Cannot read development status. Completion is not confirmed.'
    if final.get('state')=='waiting' and time.time()-final.get('updated_unix',0)>180:
        return 'attention','Final preparation has stopped updating. Completion is not confirmed.'
    return 'running','WORK IN PROGRESS - keep the laptop plugged in and awake.'


def argument(command,name):
    try:return command[command.index(name)+1]
    except (ValueError,IndexError):return None


def stage_eta(folder,step,target):
    try:
        with (folder/'train.jsonl').open('rb') as stream:
            stream.seek(0,2);size=stream.tell();stream.seek(max(0,size-12000));records=[]
            for line in stream.read().decode('utf-8',errors='replace').splitlines():
                try:records.append(json.loads(line))
                except ValueError:pass
        a,b=records[-8],records[-1]
        if b['step']<=a['step'] or b['elapsed_seconds']<=a['elapsed_seconds'] or step>=target:return None
        seconds=(target-step)*(b['elapsed_seconds']-a['elapsed_seconds'])/(b['step']-a['step'])
        minutes=max(1,int(seconds/60+.5))
        return f'{minutes//60}h {minutes%60:02d}m' if minutes>=60 else f'{minutes}m'
    except (OSError,KeyError,IndexError,TypeError):return None


def snapshot(runs, export_dir=None):
    queue=read_json(runs/'experiment_queue'/'status.json');final=read_json(runs/'final_preparation'/'status.json')
    extra=read_json(runs/'width_experiments'/'status.json')
    large=read_json(runs/'large_width_experiments'/'status.json')
    state,message=completion_state(queue,final,runs)
    lines=['VISION_HUNTERS | LIVE TRAINING PROGRESS',datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
           '='*78,message,'='*78,'',
           f'Development:       {queue.get("state","unknown")} | {queue.get("stage",queue.get("selected_run","-"))}',
           f'Final preparation: {final.get("state","not started")} | {final.get("stage","-")}']
    active=final if final.get('state')=='running' else queue
    if extra:
        lines.append(f'Width 48 and 64:   {extra.get("state")} | {extra.get("stage","-")}')
        if extra.get('state')!='complete':active=extra
    if large:
        lines.append(f'Width 96 and 128:  {large.get("state")} | {large.get("stage","-")}')
        if large.get('state')!='complete':active=large
    command=active.get('command',[]);run=argument(command,'--run');target=argument(command,'--steps')
    if run and target and state=='running':
        current=read_json(runs/run/'status.json');step=current.get('step',0);target=int(target)
        count=int(32*min(step/target,1));bar='#'*count+'-'*(32-count)
        lines+=['',f'Current stage: {run} - {current.get("state","starting")}',
                f'[{bar}] {step:,} / {target:,} steps ({min(100*step/target,100):.1f}%)']
        eta=stage_eta(runs/run,step,target)
        if eta:lines.append(f'Approximate time left in THIS training stage: {eta} + validation.')
        if current.get('state')=='validating':lines.append('Checking 60 validation images; training steps pause during this check.')
    histories=collect(runs)
    lines+=terminal_lines(histories,run)
    if export_dir is not None:
        try:
            export_history(histories,export_dir)
            lines.append(f'Full validation data: {export_dir / "validation_history.csv"}')
        except OSError as exc:
            lines.append(f'Validation export unavailable (training continues): {exc}')
    lines+=['','Recorded runs (one completed run does not mean the whole queue is finished):',
            f'{"Run":<29} {"State":<12} {"Step":>7} {"Best val score":>16}']
    for path in sorted(runs.glob('*/status.json')):
        if path.parent.name in ('experiment_queue','final_preparation','width_experiments','large_width_experiments'):continue
        record=read_json(path);score=record.get('best_score')
        score=f'{score:.5f}' if isinstance(score,(int,float)) and score>=0 else '-'
        lines.append(f'{path.parent.name:<29} {record.get("state","unknown"):<12} {str(record.get("step","-")):>7} {score:>16}')
    if state=='paused':
        lines+=['',f'Paused at saved step {large.get("saved_step","-")} in {large.get("run","-")}.',
                'Training will resume only when requested. All earlier results are preserved.']
    elif state=='complete':
        lines+=['',f'Final candidate files: {final["output"]}',
                'Training and local packaging are complete. Report review and submission remain.']
        if extra or large:lines.append('New width experiments finished. Their results need review before replacing the earlier ZIP.')
    elif large and large.get('state')!='complete':
        lines+=['','Width queue: memory probes -> width 96 to 20,000 -> width 128 to 20,000.',
                'BOTH run for 20,000 steps. Best checkpoint selected by 60-image validation.',
                'CPU and robustness checks follow each run. Earlier checkpoints and ZIP are preserved.']
    elif extra and extra.get('state')!='complete':
        lines+=['','Width queue: 48 and 64 to 4,000 steps each, then each qualifying model to 20,000.',
                'Qualifying means step-4,000 score strictly above width 32 at step 4,000.',
                'CPU and robustness checks follow. The earlier candidate ZIP is preserved.']
    else:
        lines+=['','Pipeline: comparisons -> checks -> longer training -> SSIM fine-tuning',
                '-> selection -> held-out assessment -> all-pair training -> CPU checks -> ZIP.',
                'No reliable whole-job finish time yet. Wait for ALL QUEUED WORK FINISHED.']
    log=active.get('log')
    if not log and active is final and final.get('stage'):log=str(runs/'final_preparation'/f'{final["stage"]}.log')
    if log and state!='complete':lines+=['',f'Current log: {log}']
    lines+=['','Watch refreshes every 5 seconds. Closing THIS monitor does not stop training.']
    return state,'\n'.join(lines)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--watch',action='store_true');args=ap.parse_args();previous=None
    try:
        while True:
            state,text=snapshot(SCRIPTS/'runs',SCRIPTS/'outputs'/'validation_progress')
            if args.watch and sys.stdout.isatty():os.system('cls' if os.name=='nt' else 'clear')
            print(text,flush=True)
            if args.watch and state=='complete' and previous!='complete' and sys.platform=='win32':
                import winsound
                winsound.MessageBeep()
            if not args.watch:break
            previous=state;time.sleep(5)
    except KeyboardInterrupt:print('\nMonitor closed. Training was not interrupted.')


if __name__=='__main__':main()
