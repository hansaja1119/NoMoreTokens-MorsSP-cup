"""Validation history from completed metric files; no GPU or training mutations."""
import csv
import io
import json
import math
from pathlib import Path

BASE_RUNS = ('hybrid32', 'hybrid48', 'hybrid64', 'hybrid96', 'hybrid128')


def read_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def collect(runs):
    histories = {}
    for run in BASE_RUNS:
        folder = runs / run
        records = []
        losses = {}
        try:
            for line in (folder / 'train.jsonl').read_text().splitlines():
                try:
                    item = json.loads(line)
                    losses[int(item['step'])] = float(item['loss'])
                except (ValueError, KeyError, TypeError):
                    pass
        except OSError:
            pass
        for path in sorted(folder.glob('val_*.json')):
            data = read_json(path)
            try:
                summary = data['summary']
                score = float(summary['composite_score'])
                step = int(data['step'])
                if not math.isfinite(score) or summary['count'] != 60:
                    continue
                records.append(dict(run=run, width=int(run.removeprefix('hybrid')),
                    step=step, score=score, psnr=float(summary['psnr']),
                    ssim=float(summary['ssim']), validation_images=summary['count']))
            except (KeyError, TypeError, ValueError):
                continue  # A writer may still be completing this file.
        records.sort(key=lambda row: row['step'])
        best = None
        previous = None
        checks = 0
        for row in records:
            if best is None or row['score'] > best['score']:
                best = row
                checks = 0
            else:
                checks += 1
            eligible = [step for step in losses if step <= row['step']]
            loss_step = max(eligible) if eligible else None
            row.update(delta_previous=None if previous is None else row['score']-previous['score'],
                best_score_so_far=best['score'], best_step_so_far=best['step'],
                delta_best=row['score']-best['score'], checks_since_best=checks,
                logged_training_loss=losses.get(loss_step), training_loss_step=loss_step)
            previous = row
        if records:
            for row in records:
                row['best_checkpoint_for_run'] = row['step'] == best['step']
                row['checkpoint_path'] = str(folder / 'best.pt') if row['best_checkpoint_for_run'] else ''
            histories[run] = records
    return histories


def signed(value):
    return '-' if value is None else f'{value:+.5f}'


def terminal_lines(histories, active_run=None):
    lines = ['', 'VALIDATION | original 60 images | higher score is better',
        f'{"Run":<10} {"Latest step/score":>20} {"Best step/score":>20} {"Change":>9} {"No gain":>8}']
    best_rows = []
    for run, records in histories.items():
        latest = records[-1]
        best = next(row for row in records if row['best_checkpoint_for_run'])
        best_rows.append(best)
        latest_text = f'{latest["step"]:,} / {latest["score"]:.5f}'
        best_text = f'{best["step"]:,} / {best["score"]:.5f}'
        lines.append(f'{run:<10} {latest_text:>20} {best_text:>20} {signed(latest["delta_previous"]):>9} {latest["checks_since_best"]:>8}')
    if best_rows:
        winner = max(best_rows, key=lambda row: row['score'])
        lines.append(f'Best measured BASE run so far: width {winner["width"]}, step {winner["step"]:,}, score {winner["score"]:.5f}.')
        lines.append(f'Saved winning weights: runs/{winner["run"]}/best.pt (EMA weights).')
    records = histories.get(active_run, [])
    if records:
        lines += ['', f'Recent {active_run} validation checks:',
            f'{"Step":>7} {"Score":>9} {"PSNR":>8} {"SSIM":>9} {"Change":>9} {"From best":>10}']
        for row in records[-5:]:
            lines.append(f'{row["step"]:>7,} {row["score"]:>9.5f} {row["psnr"]:>8.3f} {row["ssim"]:>9.5f} {signed(row["delta_previous"]):>9} {signed(row["delta_best"]):>10}')
    lines += ['Change = score minus previous check. No gain = checks since a new best.',
        'A dip is not proof of overfitting. Checks are 1,000 steps apart, not every step.',
        'Base runs only; SSIM fine-tuning/protection results are separate comparisons.']
    return lines


def export_history(histories, output):
    rows = [row for records in histories.values() for row in records]
    if not rows:
        return
    output.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    content = buffer.getvalue()
    path = output / 'validation_history.csv'
    try:
        if path.read_text(encoding='utf-8-sig') == content.replace('\r\n', '\n'):
            return
    except OSError:
        pass
    temporary = path.with_suffix('.tmp')
    temporary.write_text(content, encoding='utf-8-sig', newline='')
    temporary.replace(path)
    best_rows = [row for row in rows if row['best_checkpoint_for_run']]
    (output / 'best_checkpoints.json').write_text(json.dumps(best_rows, indent=2), encoding='utf-8')
    (output / 'README.txt').write_text(
        'NoMoreTokens validation history\n\n'
        'Updated by Watch Training when new completed validation results arrive.\n'
        'All rows use the original 60 validation images. Higher composite score is better.\n'
        'These are base MSE runs; later SSIM fine-tuning and protection are separate.\n'
        'best_checkpoint_for_run identifies the highest measured score within each run.\n'
        'Ties retain the earlier checkpoint, matching the trainer. best.pt is updated on strict improvement.\n'
        'Load its EMA weights for inference. The last checkpoint is for resuming training.\n'
        'Widths with incomplete training have provisional best checkpoints.\n\n'
        'A negative delta_previous is a drop from the previous validation check.\n'
        'delta_best is the difference from the best score seen up to that step.\n'
        'checks_since_best counts completed checks without a strict new best.\n'
        'Several drops alongside falling training loss suggest overfitting; they do not prove its exact onset.\n'
        'logged_training_loss is the latest logged batch-loss average at or before validation,\n'
        'not a full training-set evaluation. Random crops and synthetic noise make it fluctuate.\n'
        'Checkpoint spacing is 1,000 steps: the best unmeasured step cannot be identified.\n'
        'Repeated selection on this validation set can itself overfit it.\n'
        'The earlier held-out assessment must not be reused to tune these runs.\n'
        'No validation result guarantees perfect denoising on every unseen image.\n', encoding='utf-8')
