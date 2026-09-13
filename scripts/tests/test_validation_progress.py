import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validation_progress import collect, export_history, terminal_lines


def test_drop_recovery_tie_and_incomplete_validation(tmp_path):
    folder = tmp_path / 'hybrid48'
    folder.mkdir()
    for step, score in [(1000, .5), (2000, .48), (3000, .51), (4000, .51)]:
        (folder / f'val_{step:06d}.json').write_text(json.dumps(dict(step=step,
            summary=dict(composite_score=score, count=60, psnr=30, ssim=.9))))
    (folder / 'val_005000.json').write_text('{')
    (folder / 'train.jsonl').write_text('\n'.join(json.dumps(dict(step=s, loss=l))
        for s,l in [(1000,.1),(2000,.09),(2000,.08),(4500,.01)]))
    histories = collect(tmp_path)
    rows = histories['hybrid48']
    assert len(rows) == 4
    assert rows[1]['delta_previous'] < 0
    assert rows[1]['checks_since_best'] == 1
    assert rows[1]['logged_training_loss'] == .08
    assert rows[2]['checks_since_best'] == 0
    assert rows[3]['best_step_so_far'] == 3000
    assert [r['step'] for r in rows if r['best_checkpoint_for_run']] == [3000]
    assert rows[3]['training_loss_step'] == 2000
    assert 'step 3,000' in '\n'.join(terminal_lines(histories, 'hybrid48'))
    output = tmp_path / 'output'
    export_history(histories, output)
    stamp = (output / 'validation_history.csv').stat().st_mtime_ns
    export_history(histories, output)
    assert (output / 'validation_history.csv').stat().st_mtime_ns == stamp
