# Adaptive denoising: Mora SP Cup 2026

This is an experimental solution under development, not a frozen competition submission.
All custom code is inside `scripts/`. The official baseline/evaluator are unchanged
from starter commit `6fe15af99b2dc5d808cf6004f7cb11df148b9f82`.

## Data and validation

The supplied data currently lives at `public/noisy`, `public/ground_truth`, and
`submissions/noisy`. Scripts also recognize the starter's canonical
`competition_data/public` layout. Original images are not moved or renamed.

`scripts/data/split.json` fixes 340 training, 60 validation, and 60 locked-test
images. The visual groups in `scene_groups.json` keep related views together.
The all-image audit is descriptive. Training noise statistics and model fitting
must only use the training partition. The 20 preliminary images are never used
to select settings or fit parameters.

## Environment

Python 3.12 is used locally. The project environment is `scripts/.venv`.
Install dependencies from `scripts/requirements.txt`. For GPU training, install
the CUDA build of PyTorch using the command in `scripts/requirements-gpu.txt`.
Inference loads local weights only and does not access the network.

Windows commands below assume the repository root. Replace `python` with
`scripts\.venv\Scripts\python.exe` to use the prepared environment.

## Development commands

```powershell
python scripts/audit.py
python scripts/benchmark.py --method baseline
python scripts/benchmark.py --method wavelet
python scripts/prepare_cache.py
python -m pytest scripts/tests -q
python scripts/train.py --run hybrid16 --steps 1000
python scripts/train.py --run hybrid16 --steps 10000 --resume scripts/runs/hybrid16/last.pt
python scripts/evaluate_checkpoint.py --checkpoint scripts/runs/hybrid16/best.pt --output scripts/runs/hybrid16_cpu_check --device cpu --limit 3
```

The audit refuses to change a split after experiment directories exist. A feature
cache manifest records preprocessing and split hashes. Training refuses stale
caches. Training uses scene-uniform deterministic crops, GPU mixed precision,
EMA, and full-image validation on PNG-quantized predictions. Progress, settings,
source hashes, metrics, and resumable checkpoints are stored inside each run.

## Denoising command

```powershell
python scripts/denoise.py --noise_dir submissions/noisy --denoised_dir scripts/outputs/preliminary --checkpoint scripts/runs/hybrid16/best.pt --device cpu --manifest scripts/outputs/preliminary_manifest.json
```

The required CLI flags are `--noise_dir` and `--denoised_dir`.
`--input_dir` and `--output_dir` are aliases. `<id>_noise.png` becomes `<id>.png`;
ordinary filenames keep their stem. PNG/JPEG RGB or grayscale inputs are
supported; an alpha channel is preserved separately. Output dimensions are
unchanged. Images are processed in a deterministic sorted order, with one local
model load, float32 inference, and overlapping tiles. No automatic brightness,
gamma, or contrast changes are applied.

CPU fallback runs the same neural model. Missing weights are an error rather
than a silent switch to another algorithm. To explicitly run the classical
method without weights, use `--method wavelet`. `--overwrite` is required to
replace existing outputs. Invalid inputs and filename collisions cause errors.

The default checkpoint location `scripts/checkpoints/selected.pt` is reserved
for the validated final selection and is not yet populated. No model is final
until the experiment and CPU checks are complete.

## Architecture and attribution

The classical front end estimates spatial RGB noise maps using robust diagonal
differences, optionally repairs strong isolated deviations, transforms into
orthonormal opponent colors, and applies locally adaptive wavelet shrinkage.
The hybrid model receives the original image, the classical estimate, and the
noise maps. It predicts a correction to the classical estimate while retaining
access to all original pixels. This anchor differs from the initial proposal's
noisy-image anchor, allowing the zero-initialized model to start from the tested
classical method. An RGB-only control starts from the noisy image.

The compact model uses the NAF block design from Chen et al., *Simple Baselines
for Image Restoration*, ECCV 2022: https://github.com/megvii-research/NAFNet .
The local implementation uses standard PyTorch operations and is trained from
scratch. No external checkpoints or datasets have been used.

## Submission freeze checklist (not yet performed)

- Final configuration selected from validation; locked test evaluated once.
- Report shorter than five pages, 12-point font, one-inch margins.
- Exactly 461.png through 480.png in the preliminary ZIP, reproduced by the code.
- CPU and offline installation rehearsed; checkpoint SHA-256 recorded.
- Organizer confirms the self-referential README/commit-SHA convention.
- Private repository access and final submission handled with the team.
- No changes to submitted code or weights after the deadline.
