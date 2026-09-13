"""Build an auditable handoff from saved measurements; never train or upload."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import numpy as np
from PIL import Image, ImageDraw
from common import ROOT, SCRIPTS, sha256, write_json
from package_submission import validate_images

REPORT = SCRIPTS / 'reports' / 'final'
DELIVERY = ROOT / 'final_output'
CHECKPOINT = SCRIPTS / 'checkpoints' / 'VISION_HUNTERS_w128_s20000.pt'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def csv_write(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    split = read(SCRIPTS / 'data' / 'split.json')
    val_ids = set(split['val'])
    assert len(val_ids) == 60 and len(split['train']) == 340 and len(split['test']) == 60
    assert not (set(split['train']) & val_ids or set(split['test']) & val_ids or set(split['train']) & set(split['test']))
    audit = {r['id']: r for r in read(SCRIPTS / 'reports' / 'audit.json')['rows']}
    cuts = np.quantile([audit[i]['residual_std'] for i in split['train']], [1/3, 2/3])
    names = ['Low', 'Medium', 'High']
    levels = {i: names[int(np.searchsorted(cuts, audit[i]['residual_std'], side='right'))] for i in val_ids}
    entries, level_rows, image_rows, evidence = [], [], [], []
    configs, traces = {}, {}
    for folder in sorted((SCRIPTS / 'runs').iterdir()):
        paths = sorted(folder.glob('val_*.json'))
        if folder.name in ['baseline_val', 'wavelet_val_1']:
            paths = [folder / 'metrics.json']
        if not paths:
            continue
        config = read(folder / 'config.json') if (folder / 'config.json').exists() else {}
        configs[folder.name] = config
        log = folder / 'train.jsonl'
        trace = [json.loads(s) for s in log.read_text().splitlines() if s.strip()] if log.exists() else []
        traces[folder.name] = trace
        best, previous, missed = -float('inf'), None, 0
        for path in paths:
            obj = read(path)
            if len(obj['rows']) != 60 or {r['id'] for r in obj['rows']} != val_ids:
                raise ValueError(f'Incomplete or wrong validation partition: {path}')
            summary = obj['summary']
            for key in ['psnr', 'ssim', 'composite_score']:
                assert abs(statistics.mean(r[key] for r in obj['rows']) - summary[key]) < 1e-9
            score = summary['composite_score']
            step = obj.get('step', 0)
            new_best = score > best
            missed = 0 if new_best else missed + 1
            status = 'First check' if previous is None else 'New best; no measured decline' if new_best else 'Single/checkpoint dip; inconclusive' if missed < 3 else 'Persistent deficit; investigate'
            losses = [r['loss'] for r in trace if step - 1000 < r['step'] <= step]
            entry = dict(run=folder.name, width=config.get('width', ''), step=step,
                         total_updates=step + (20000 if config.get('init') else 0),
                         psnr=summary['psnr'], ssim=summary['ssim'], composite_score=score,
                         delta_previous='' if previous is None else score-previous,
                         delta_best=score-max(best, score), checks_without_new_best=missed,
                         logged_train_loss=statistics.mean(losses) if losses else '',
                         overfitting_evidence=status, source=path.relative_to(ROOT).as_posix())
            entry['level_metrics'] = {}
            for level in names:
                part = [r for r in obj['rows'] if levels[r['id']] == level]
                row = dict(run=folder.name, width=config.get('width', ''), step=step, level=level, count=len(part),
                           **{k: statistics.mean(r[k] for r in part) for k in ['psnr', 'ssim', 'composite_score']})
                level_rows.append(row)
                entry['level_metrics'][level] = row
            for row in obj['rows']:
                image_rows.append(dict(run=folder.name, width=config.get('width', ''), step=step,
                                       level=levels[row['id']], **row))
            entries.append(entry)
            evidence.append(path)
            best, previous = max(best, score), score
        evidence.extend(p for p in folder.glob('config*.json'))
        if log.exists(): evidence.append(log)
    csv_write(REPORT / 'validation_steps.csv', [{k: v for k, v in e.items() if k != 'level_metrics'} for e in entries])
    csv_write(REPORT / 'validation_levels.csv', level_rows)
    csv_write(REPORT / 'validation_images.csv', image_rows)
    write_json(REPORT / 'level_definition.json', dict(source='Training-only residual standard deviation terciles; analytical bins, not organizer labels',
        boundaries=cuts.tolist(), validation_levels=levels))
    selected_path = SCRIPTS / 'outputs' / 'final_delivery' / 'validation' / 'metrics.json'
    selected = read(selected_path)
    assert selected['checkpoint_sha256'] == sha256(CHECKPOINT)
    assert {r['id'] for r in selected['rows']} == val_ids
    evidence += [selected_path, CHECKPOINT.with_suffix('.json'),
                 SCRIPTS / 'runs' / 'final_preparation' / 'locked_test' / 'metrics.json',
                 SCRIPTS / 'runs' / 'final_preparation' / 'fixed_recipe.json',
                 SCRIPTS / 'runs' / 'final_preparation' / 'final_runtime.json']
    for width in [48, 64, 96, 128]:
        evidence += [SCRIPTS / 'runs' / f'hybrid{width}' / 'width_robustness.json',
                     SCRIPTS / 'runs' / f'hybrid{width}' / 'width_cpu' / 'metrics.json']
    for folder in ['experiment_queue', 'width_experiments', 'large_width_experiments']:
        evidence += list((SCRIPTS / 'runs' / folder).glob('*.json'))
    for folder in ['final_all_mse', 'final_all_ssim']:
        evidence += [SCRIPTS / 'runs' / folder / 'config.json', SCRIPTS / 'runs' / folder / 'status.json']
    runtime_path = DELIVERY / 'runtime_checks.json'
    if runtime_path.exists(): evidence.append(runtime_path)
    hashes = []
    for path in sorted(set(evidence)):
        dest = REPORT / 'evidence' / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        hashes.append(dict(path=dest.relative_to(REPORT).as_posix(), sha256=sha256(dest)))
    write_json(REPORT / 'evidence_manifest.json', hashes)
    ranking = sorted([max([e for e in entries if e['run'] == run], key=lambda e: e['composite_score']) for run in configs], key=lambda e: e['composite_score'], reverse=True)
    winner = ranking[0]
    assert winner['run'] == 'hybrid128' and winner['step'] == 20000
    counts = {level: sum(x == level for x in levels.values()) for level in names}
    lines = ['# Final denoising results and model improvement guide', '',
        'Prepared 2026-09-13 for VISION_HUNTERS. This report supersedes the early development scorecard. All values are measured local results, not competition leaderboard scores.', '',
        '## What to submit', '',
        f'The highest measured base model is **hybrid128, width 128, step 20,000**, EMA weights: PSNR {winner["psnr"]:.6f} dB, SSIM {winner["ssim"]:.6f}, composite {winner["composite_score"]:.9f} on the original 60 validation images. The delivered inference model uses the already stress-tested mild-protection threshold 0.03. Its separately measured full-validation score is **{selected["summary"]["composite_score"]:.9f}**, PSNR **{selected["summary"]["psnr"]:.6f} dB**, SSIM **{selected["summary"]["ssim"]:.6f}**.', '',
        'Use [the 20 final PNGs](../../../final_output/images) or the image-only `final_output/VISION_HUNTERS.zip`. [Image list and checksums](../../../final_output/IMAGE_LIST.md). The checkpoint is `scripts/checkpoints/VISION_HUNTERS_w128_s20000.pt`; download it from the GitHub release if it is not present in a clone. This is the trained 340-image development model, not a new all-460 refit. No new long training run was needed for this handoff.', '',
        'This is the strongest measured choice among the completed runs when quality is the priority and runtime permits it. It is not a guarantee of the best hidden-test result. Width 96 is a speed/size compromise; width 32 is the compact fallback. The earlier all-data width-32 output remains in `scripts/outputs/final_candidate`; its score cannot be compared as held-out validation because its training includes that partition. There is no all-data width-128 checkpoint or width-128 SSIM fine-tune in these experiments.', '',
        '## Dataset, validation, and leakage boundaries', '',
        '- Public data: 460 clean/noisy RGB pairs, IDs 001–460, each 992 × 992. Current paths: `public/ground_truth` and `public/noisy`; the code also recognizes `competition_data/public`.',
        '- Development training: 340 image IDs. Their clean targets and noisy inputs may drive gradients and training statistics.',
        '- Validation: 60 image IDs used repeatedly for architecture, step, and protection selection. Full images, never random validation crops, determine the selection score.',
        '- Original locked test: 60 image IDs, evaluated once for the earlier selected width-32 model before all-data training. That assessment is historical; it is not an unopened test available for fresh width-128 model selection.',
        '- Preliminary submission: 461–480, noisy images only, from `submissions/noisy`. No clean targets are supplied, so no honest PSNR/SSIM/competition score can be reported for these final outputs. They did not drive training or selection.',
        '- Final-round IDs 481–500 are hidden according to the supplied challenge README; they are not present here.', '',
        'The fixed split is [split.json](../../data/split.json), seed 20260908. Related views from visual scene review and the duplicate heuristic stay in one partition; the split allocator interleaves noisy-brightness and estimated-noise strata. No exact duplicate pairs were found in the recorded audit. Grouping reduces obvious leakage but cannot certify independence of every scene. Splitting occurs before crops. The actual loader samples **image IDs uniformly**, then random crops; despite its source comment, it does not sample scene groups uniformly.', '',
        '**Validation IDs:** ' + ', '.join(split['val']) + '.', '',
        '**Original locked-test IDs:** ' + ', '.join(split['test']) + '.', '',
        'Training IDs are the remaining public IDs, enumerated in split.json. Caches may contain validation/test targets after the previous all-data run, but development TrainingCrops indexes only the 340 training IDs. Cache presence is not evidence of gradient use. The all-image audit is descriptive; fitted noise statistics and the diagnostic thresholds below use training IDs only. No external pretrained weights or datasets are recorded.', '',
        '## Scores and the meaning of noise levels', '',
        'The source of truth is the unmodified [official evaluator](../../../evaluation/evaluate.py). RGB is normalized to [0,1]. Predictions are clipped, rounded to uint8 PNG, then read back for evaluation. SSIM uses a 7 × 7 uniform window, sample covariance, channel_axis=-1, data_range=1, K1=0.01 and K2=0.03. For each image:', '',
        '```text\ndelta_PSNR = PSNR(output, clean) - PSNR(noisy, clean)\ndelta_SSIM = SSIM(output, clean) - SSIM(noisy, clean)\nscore = 0.6 * clip(delta_PSNR / 15, 0, 1) + 0.4 * max(delta_SSIM, 0)\nvalidation score = arithmetic mean of the 60 per-image scores\n```', '',
        'No authoritative per-image organizer severity labels were found in the provided metadata. Low/Medium/High below are explicitly **analytical noise bins**, not invented official levels. We take the per-image standard deviation of noisy-minus-clean residuals from the existing audit, calculate its 1/3 and 2/3 quantiles on the 340 training images, and apply those frozen boundaries to validation images. Ground truth is used only for this retrospective diagnostic; this is not a deployable severity classifier or routing rule.', '',
        f'Low: residual std < {cuts[0]:.9f}; Medium: {cuts[0]:.9f} ≤ std < {cuts[1]:.9f}; High: std ≥ {cuts[1]:.9f}. Validation counts: {counts}. Units are normalized pixel intensities. Boundaries, memberships, and counts are exported in [level_definition.json](level_definition.json). These bins are not brightness levels or network stages.', '',
        '## Architecture and why widths were chosen', '',
        'The classical front end estimates local RGB noise maps, conservatively repairs isolated outliers, and uses opponent-color db2 wavelet shrinkage (up to three scales). The hybrid network receives nine channels: original RGB, classical RGB estimate, and three noise maps. A NAF-style encoder-decoder predicts a residual added to the classical estimate. Its zero-initialized output layer begins at that anchor. The RGB-only control instead uses three channels and anchors to the noisy input.', '',
        'Width is the base feature-channel count, not image resolution. Encoder channels are w, 2w, 4w, 8w; the bottleneck has 16w channels. Encoder block counts are 1/1/2/4, middle 4, decoder one block per scale, with skip additions and pixel-shuffle upsampling. Four downsamples require padding to multiples of 16. Standard PyTorch implementation in [model.py](../../model.py), based on the NAF block design attributed there.', '',
        'The first controlled comparison trained hybrid16, rgb16, hybrid32, and real-pairs-only hybrid16 to 4,000 updates. Width 32 won and was extended to 20,000. Widths 48 and 64 were later tested at exactly 4,000 updates against width 32’s 0.564605450; both improved and qualified for extension. The requested 96/128 queue trained both to 20,000 without a 4,000-step gate. Therefore width 16 has only 4,000-step evidence, and wider-versus-16 final scores mix capacity and duration. Compare equal-step rows for capacity claims. Parameter count and memory grow roughly quadratically in width.', '',
        '| Run | Width | Best local step | Total updates | PSNR dB | SSIM | Score |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for e in ranking:
        lines.append(f'| {e["run"]} | {e["width"] or "—"} | {e["step"] or "—"} | {e["total_updates"] or "—"} | {e["psnr"]:.6f} | {e["ssim"]:.6f} | {e["composite_score"]:.9f} |')
    lines += ['', 'Scores above are base/raw validation scores unless a run name denotes SSIM fine-tuning; mild-protection results are separate below. Baseline and wavelet have no neural width or optimizer steps.', '']
    groups = [sorted(set(g) & val_ids) for g in split['groups'] if set(g) & val_ids]
    grouped = {i for g in groups for i in g}
    groups += [[i] for i in sorted(val_ids-grouped)]
    rng = np.random.default_rng(20260913)
    samples = rng.integers(0, len(groups), (10000, len(groups)))
    group_counts = np.array([len(g) for g in groups])
    selected_scores = {r['id']: r['composite_score'] for r in selected['rows']}
    lines += ['Paired uncertainty check: delivered protected width 128 versus saved base models on the same validation images. Resample the ' + str(len(groups)) + ' scene groups (including singleton images) with replacement 10,000 times, seed 20260913; weight each bootstrap sample by its image count. These percentile intervals describe this dataset only and do not correct for repeated model selection.', '',
        '| Comparator | Mean score gain | Paired group bootstrap 95% interval |', '|---|---:|---|']
    for w in [32, 64, 96]:
        comparator = read(SCRIPTS/'runs'/f'hybrid{w}'/'val_020000.json')
        scores = {r['id']: r['composite_score'] for r in comparator['rows']}
        sums = np.array([sum(selected_scores[i]-scores[i] for i in g) for g in groups])
        interval = np.quantile(sums[samples].sum(1)/group_counts[samples].sum(1), [.025, .975])
        lines.append(f'| Base width {w}, 20,000 | {sums.sum()/60:.9f} | [{interval[0]:.9f}, {interval[1]:.9f}] |')
    lines += ['',
        '| Width | Microbatch × accumulation | Effective batch | Saved CPU median seconds/image |', '|---:|---|---:|---:|']
    architecture = read(SCRIPTS / 'runs' / 'experiment_queue' / 'architecture_comparison.json')
    for w in [16, 32, 48, 64, 96, 128]:
        c = configs[f'hybrid{w}']
        runtime = next(r['cpu_seconds'] for r in architecture['candidates'] if r['run'] == f'hybrid{w}') if w <= 32 else read(SCRIPTS / 'runs' / f'hybrid{w}' / 'width_cpu' / 'metrics.json')['summary']['median_seconds']
        lines.append(f'| {w} | {c["batch"]} × {c["accumulate"]} | {c["batch"]*c["accumulate"]} | {runtime:.3f} |')
    lines += ['', 'CPU measurements use small validation subsets, four PyTorch threads, and different historical runs/load conditions; they are indicative, not a controlled hardware leaderboard. Memory probes selected microbatches with headroom while holding effective batch at 16. Width 128’s historical CPU median is about 49.7 seconds/image (about 16.6 minutes for 20 images before overhead); CUDA is used for the delivered PNGs. Check actual manifest timings for this run.', '',
        '## Training and fine-tuning recipe', '',
        'One step means one optimizer update after gradient accumulation, not one image and not an epoch. At effective batch 16, 20,000 steps sample 320,000 crops. Image selection is random with replacement, so this is not 941 complete deterministic passes over 340 images. For each absolute sample index, NumPy is seeded with 20260908 + index, preserving crop/augmentation choices across resumes when effective batch and recipe stay fixed.', '',
        'The first 1,000 updates use 128 × 128 crops; later updates use 256 × 256. The option `warmup_steps` controls crop size, **not learning-rate warmup**. Crops rotate by multiples of 90 degrees and randomly flip horizontally. The default mixture is 70% supplied noisy/clean pairs, 20% new synthetic corruptions, and 10% clean identity examples. Synthetic noise combines read noise U(0.003,0.07), shot coefficient U(0,0.025), sigma=sqrt(read²+shot*clean), a horizontal spatial scale, and optional sparse impulses; clipping/PNG quantization follows. This broad generator does not claim to reconstruct the organizer’s noise process.', '',
        'Hybrid features are cached at full-image resolution as uint8: noisy RGB, classical RGB, noise maps divided by 0.3, and clean RGB. Training restores the noise scale; synthetic branches recompute features for the sampled crop. Changing classical preprocessing requires an explicit cache rebuild. Training checks preprocessing and split hashes; each run archives configuration, source hashes, and code copies for its starting/resume segments.', '',
        'Base loss is MSE. AdamW uses initial lr 0.0002, weight decay 0.0001, gradient norm clipping 1.0, CUDA float16 autocast and GradScaler, with float32 loss arithmetic. The learning-rate rule is `lr0 * (0.05 + 0.95 * 0.5 * (1 + cos(pi * min(step,schedule_steps)/schedule_steps)))`; the base schedule spans 30,000 steps even when stopping at 20,000 (lr then 0.0000575). Extending duration must preserve that schedule for a comparable continuation.', '',
        'EMA decay is min(0.999, 1 - 1/(step+1)). Full validation uses EMA every 1,000 updates and at the requested endpoint. A strict score improvement replaces `best.pt`; `last.pt` is a resumable state including model, EMA, optimizer, scaler and step. Earlier intermediate best weights are overwritten unless separately preserved, so a score row does not imply that checkpoint remains recoverable. Logs are written about every 50 updates. One seed per configuration was run; neither deterministic settings nor these data establish multi-seed reproducibility.', '',
        'Width-32 SSIM fine-tunes initialize from the base model’s 20,000-step EMA, use a fresh optimizer at lr 0.00005, 2,000-step cosine schedule, no small-crop phase, and MSE + lambda*(1-SSIM), lambda in {0.05,0.1}. Local fine-tune step 1,000/2,000 means 21,000/22,000 total updates. Lambda 0.05 wins narrowly in measured validation. Do not assume its benefit transfers to width 128 without a new training/validation comparison.', '',
        'Earlier final training separately restarted width 32 on all 460 pairs for 20,000 MSE updates and 2,000 SSIM updates using the fixed recipe. Validation and best-checkpoint selection were disabled; it saves `final.pt`. Such a model has no remaining held-out score among those 460 images. Its `best_score=-1` status sentinel is not a quality score.', '',
        '## Overfitting: what the evidence does and does not say', '',
        'Overfitting is supported when fit on a fixed training evaluation set improves while independently measured validation quality deteriorates persistently. A noisy crop-training loss falling at the same time as a single validation dip is insufficient. Different crops, noise mixtures, EMA lag, and numerical variation can cause fluctuations. No matched full-image training-evaluation series is logged here, so no particular step can be certified as overfitting or as guaranteed free of it.', '',
        'Each measured step below has an evidence label: first check; new best (no measured validation decline); a deficit from the previous best (inconclusive); or three or more consecutive checks below the previous best (a persistent-deficit warning for investigation). This is a descriptive flag applied after training, not an early-stopping rule used by the historical queues. Repeatedly trying widths and steps on the same validation set also risks selection overfitting even if each curve improves.', '',
        'All six hybrid base widths end at their best measured validation score within their own completed duration. Thus there is no terminal validation deterioration indicating that the selected 20,000-step width-128 checkpoint overfits; there is also no evidence that steps beyond 20,000 would improve it. Steps not measured (e.g. 20,001) have no score or overfitting classification. Do not interpolate them into evidence.', '',
        'For further work, keep selection on this validation set, preserve groups before making a new independent holdout, log a fixed train-evaluation subset with identical inference/quantization, and compare multiple seeds and paired scene-group confidence intervals. Predeclare any patience and minimum-gain rule. Never use preliminary outputs, prior locked-test scores, or the final-round images to tune parameters. A new holdout must never have trained the model being assessed; the historical all-data model has already seen all 460 images.', '',
        '## Every recorded width, step, and analytical noise level', '',
        'Each Low/Medium/High cell is **PSNR dB / SSIM / composite score**. All is the full 60-image mean. Train loss is the mean of recorded loss-window averages from the preceding 1,000 updates; it is a changing-crop diagnostic, not directly comparable to full-image held-out PSNR. Fine-tune losses also include SSIM. CSVs retain full precision: [steps](validation_steps.csv), [levels](validation_levels.csv), [per-image results](validation_images.csv). Only actually measured checkpoints are listed.', '']
    for run in configs:
        lines += [f'### {run}', '', '| Step | All PSNR / SSIM / score | Low | Medium | High | Train loss | Evidence |', '|---:|---|---|---|---|---:|---|']
        for e in [x for x in entries if x['run'] == run]:
            fmt = lambda r: f'{r["psnr"]:.4f} / {r["ssim"]:.6f} / {r["composite_score"]:.9f}'
            loss = f'{e["logged_train_loss"]:.7f}' if e['logged_train_loss'] != '' else '—'
            lines.append(f'| {e["step"] or "—"} | {fmt(e)} | {fmt(e["level_metrics"]["Low"])} | {fmt(e["level_metrics"]["Medium"])} | {fmt(e["level_metrics"]["High"])} | {loss} | {e["overfitting_evidence"]} |')
        lines.append('')
    lines += ['## Delivered protection, robustness, and historical test', '',
        'The delivered threshold 0.03 blends noisy input with network output using squared clipped opponent-chroma noise confidence and a weak-texture luminance safeguard for grayscale/correlated noise. Threshold 0 disables blending. The historical wide-model validation curves use threshold 0, while their stress diagnostics explicitly use 0.03. The exported model fixes 0.03 and was therefore evaluated again on all 60 validation images. This prevents attributing the raw score to a different deployment configuration.', '',
        '| Delivered configuration | Count | PSNR dB | SSIM | Composite |', '|---|---:|---:|---:|---:|']
    for level in ['All'] + names:
        rows = [r for r in selected['rows'] if level == 'All' or levels[r['id']] == level]
        lines.append(f'| w128 / 20000 / threshold .03 / {level} | {len(rows)} | {statistics.mean(r["psnr"] for r in rows):.6f} | {statistics.mean(r["ssim"] for r in rows):.6f} | {statistics.mean(r["composite_score"] for r in rows):.9f} |')
    lines += ['', 'The following stress conditions use eight fixed central 256px validation crops, seed 914. They are synthetic diagnostic levels, separate from the analytical full-image noise bins. They have PSNR/SSIM/error scores but no official competition composite. The reported 120 dB on an unchanged clean crop is the implementation’s MSE floor, not a finite measured noise level.', '',
        '| Width | Stress condition | PSNR dB | SSIM | Input MSE | Output MSE |', '|---:|---|---:|---:|---:|---:|']
    for w in [48, 64, 96, 128]:
        robust = read(SCRIPTS / 'runs' / f'hybrid{w}' / 'width_robustness.json')
        for kind, m in robust['summary'].items():
            lines.append(f'| {w} | {kind} | {m["psnr"]:.6f} | {m["ssim"]:.6f} | {m["input_mse"]:.9f} | {m["output_mse"]:.9f} |')
    if runtime_path.exists():
        runtime = read(runtime_path)
        lines += ['', f'Final verification on 2026-09-13: all 20 outputs pass exact filename, 992×992 RGB PNG, SHA-256 and ZIP integrity checks. CPU image 021 took {runtime["cpu_seconds"]:.2f} seconds on its first pass under current machine conditions; two CPU passes were PNG-identical. CPU/CUDA maximum difference was {runtime["cpu_gpu_max_png_difference"]} uint8 level on that image. Regenerating delivered image 461 on CUDA was pixel-identical. These are explicitly one-image runtime checks, not a claim that every image was rerun on both devices.']
    locked = read(SCRIPTS / 'runs' / 'final_preparation' / 'locked_test' / 'metrics.json')
    lines += ['', 'Width 128 with threshold 0.03 passes the earlier diagnostic criteria: clean mean absolute error ≤0.003 and mild-Gaussian output MSE ≤1.1× input MSE. Grayscale/correlated noise remains harder than independent strong Gaussian noise. These eight crops cannot establish real-camera generalization.', '',
        f'Historical locked test: selected width-32 development model, threshold 0.02, 20,000 MSE + 2,000 SSIM updates; PSNR {locked["summary"]["psnr"]:.6f}, SSIM {locked["summary"]["ssim"]:.6f}, composite {locked["summary"]["composite_score"]:.9f} on 60 different images. **This score belongs only to that checkpoint**, not to the all-data width-32 model or the delivered width-128 model. Different split difficulty prevents ranking this score against the width-128 validation score.', '',
        '## Reproduce the outputs and continue development', '',
        'Run from the repository root with Python 3.12. The local interpreter is `scripts/.venv/Scripts/python.exe`. Runtime versions are pinned in [requirements.txt](../../requirements.txt); use [requirements-gpu.txt](../../requirements-gpu.txt) for the recorded CUDA build or [requirements-cpu.txt](../../requirements-cpu.txt) for CPU. The environment lock is retained. Install into your own environment; do not upload virtual environments.', '',
        '```powershell\npython -m pip install -r scripts/requirements-gpu.txt\n# Place the downloaded inference checkpoint at the path below.\npython scripts/denoise.py --noise_dir submissions/noisy --denoised_dir reproduced_images --checkpoint scripts/checkpoints/VISION_HUNTERS_w128_s20000.pt --device cuda --manifest reproduced_manifest.json\n# Substitute --device cpu when CUDA is unavailable.\n```', '',
        'Inference loads local EMA weights once, uses float32 512px tiles with 64px overlap and positive Hann blending, preserves dimensions, and writes sorted <id>.png names. No brightness/gamma/contrast adjustment is applied. CPU and CUDA may differ by a PNG quantization level; the manifest hashes describe the exact delivered CUDA files. Missing weights are an error, not a silent classical fallback.', '',
        '```powershell\n# Fresh development run, retaining the original split and effective batch.\npython scripts/prepare_cache.py\npython scripts/train.py --run new_w128 --width 128 --steps 20000 --schedule-steps 30000 --batch 1 --accumulate 16 --device cuda\n# Continue an existing run, preserving architecture, data mixture and schedule.\npython scripts/train.py --run new_w128 --width 128 --steps 25000 --schedule-steps 30000 --batch 1 --accumulate 16 --device cuda --resume scripts/runs/new_w128/last.pt\n# Separate SSIM experiment; this has NOT been run for width 128.\npython scripts/train.py --run new_w128_ssim005 --width 128 --steps 2000 --schedule-steps 2000 --warmup-steps 0 --lr 0.00005 --ssim-weight 0.05 --batch 1 --accumulate 16 --init scripts/runs/new_w128/best.pt --device cuda\n```', '',
        'Use a new run name for recipe changes. A different width cannot initialize directly from these weights because tensor shapes differ. `--resume` restores optimizer/scaler and sample position, while `--init` starts a new step counter/optimizer from EMA. An inference-only export lacks the optimizer and raw-model fields needed for exact resume: retain `best.pt`/`last.pt` locally for that purpose. Export weights only after selection and revalidate any deployment threshold. Rebuild this handoff with `python scripts/build_final_delivery.py` when its archived input runs and final-output files are available.', '',
        '## Provenance and publication', '',
        f'Checkpoint SHA-256: `{sha256(CHECKPOINT)}`. Exported weights size: {CHECKPOINT.stat().st_size:,} bytes. Source checkpoint and source-file hashes are in the export metadata; original measurement files are copied under [evidence](evidence) with [checksums](evidence_manifest.json). The ZIP manifest references the existing code base commit and separately records working-tree source hashes; it is not a claim that uncommitted handoff files were part of that old commit.', '',
        'The root README links this report; the earlier `scripts/reports/RESULTS.md` and prior PDF/technical guide describe earlier stages. Raw training runs, datasets, caches, environments and inference weights are excluded from ordinary Git. The final image list includes exact bytes and SHA-256, and the archive contains only 461.png–480.png. See [GITHUB_UPLOAD.md](../../../GITHUB_UPLOAD.md) for measured publication sizes, destination, and release-asset handling.', '']
    (REPORT / 'MODEL_GUIDE.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'Wrote {len(entries)} measured checkpoints, {len(level_rows)} level rows, {len(image_rows)} per-image rows.')


if __name__ == '__main__':
    main()
