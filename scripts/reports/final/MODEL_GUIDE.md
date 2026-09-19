# Final denoising results and model improvement guide

Prepared 2026-09-13 for NoMoreTokens. This report supersedes the early development scorecard. All values are measured local results, not competition leaderboard scores.

## What to submit

The highest measured base model is **hybrid128, width 128, step 20,000**, EMA weights: PSNR 30.949480 dB, SSIM 0.910601, composite 0.627013204 on the original 60 validation images. The delivered inference model uses the already stress-tested mild-protection threshold 0.03. Its separately measured full-validation score is **0.626890096**, PSNR **30.947021 dB**, SSIM **0.910539**.

Use [the 20 final PNGs](../../../final_output/images) or the image-only `final_output/NoMoreTokens.zip`. [Image list and checksums](../../../final_output/IMAGE_LIST.md). The checkpoint is `scripts/checkpoints/NoMoreTokens_w128_s20000.pt`; download it from the GitHub release if it is not present in a clone. This is the trained 340-image development model, not a new all-460 refit. No new long training run was needed for this handoff.

This is the strongest measured choice among the completed runs when quality is the priority and runtime permits it. It is not a guarantee of the best hidden-test result. Width 96 is a speed/size compromise; width 32 is the compact fallback. The earlier all-data width-32 output remains in `scripts/outputs/final_candidate`; its score cannot be compared as held-out validation because its training includes that partition. There is no all-data width-128 checkpoint or width-128 SSIM fine-tune in these experiments.

## Dataset, validation, and leakage boundaries

- Public data: 460 clean/noisy RGB pairs, IDs 001–460, each 992 × 992. Current paths: `public/ground_truth` and `public/noisy`; the code also recognizes `competition_data/public`.
- Development training: 340 image IDs. Their clean targets and noisy inputs may drive gradients and training statistics.
- Validation: 60 image IDs used repeatedly for architecture, step, and protection selection. Full images, never random validation crops, determine the selection score.
- Original locked test: 60 image IDs, evaluated once for the earlier selected width-32 model before all-data training. That assessment is historical; it is not an unopened test available for fresh width-128 model selection.
- Preliminary submission: 461–480, noisy images only, from `submissions/noisy`. No clean targets are supplied, so no honest PSNR/SSIM/competition score can be reported for these final outputs. They did not drive training or selection.
- Final-round IDs 481–500 are hidden according to the supplied challenge README; they are not present here.

The fixed split is [split.json](../../data/split.json), seed 20260908. Related views from visual scene review and the duplicate heuristic stay in one partition; the split allocator interleaves noisy-brightness and estimated-noise strata. No exact duplicate pairs were found in the recorded audit. Grouping reduces obvious leakage but cannot certify independence of every scene. Splitting occurs before crops. The actual loader samples **image IDs uniformly**, then random crops; despite its source comment, it does not sample scene groups uniformly.

**Validation IDs:** 021, 022, 023, 034, 047, 072, 075, 081, 083, 101, 114, 117, 118, 138, 139, 152, 156, 157, 168, 172, 173, 177, 183, 185, 197, 211, 217, 226, 236, 242, 246, 247, 249, 250, 276, 280, 281, 302, 317, 321, 326, 340, 342, 345, 371, 375, 379, 381, 384, 385, 391, 399, 413, 416, 422, 424, 429, 432, 434, 456.

**Original locked-test IDs:** 010, 019, 024, 030, 037, 041, 049, 059, 063, 064, 073, 080, 086, 100, 105, 107, 121, 126, 151, 163, 171, 176, 178, 181, 187, 203, 220, 231, 233, 243, 254, 255, 263, 264, 267, 271, 285, 289, 298, 316, 330, 335, 337, 352, 355, 356, 362, 367, 374, 383, 387, 389, 410, 425, 437, 444, 449, 450, 458, 460.

Training IDs are the remaining public IDs, enumerated in split.json. Caches may contain validation/test targets after the previous all-data run, but development TrainingCrops indexes only the 340 training IDs. Cache presence is not evidence of gradient use. The all-image audit is descriptive; fitted noise statistics and the diagnostic thresholds below use training IDs only. No external pretrained weights or datasets are recorded.

## Scores and the meaning of noise levels

The source of truth is the unmodified [official evaluator](../../../evaluation/evaluate.py). RGB is normalized to [0,1]. Predictions are clipped, rounded to uint8 PNG, then read back for evaluation. SSIM uses a 7 × 7 uniform window, sample covariance, channel_axis=-1, data_range=1, K1=0.01 and K2=0.03. For each image:

```text
delta_PSNR = PSNR(output, clean) - PSNR(noisy, clean)
delta_SSIM = SSIM(output, clean) - SSIM(noisy, clean)
score = 0.6 * clip(delta_PSNR / 15, 0, 1) + 0.4 * max(delta_SSIM, 0)
validation score = arithmetic mean of the 60 per-image scores
```

No authoritative per-image organizer severity labels were found in the provided metadata. Low/Medium/High below are explicitly **analytical noise bins**, not invented official levels. We take the per-image standard deviation of noisy-minus-clean residuals from the existing audit, calculate its 1/3 and 2/3 quantiles on the 340 training images, and apply those frozen boundaries to validation images. Ground truth is used only for this retrospective diagnostic; this is not a deployable severity classifier or routing rule.

Low: residual std < 0.095481850; Medium: 0.095481850 ≤ std < 0.120908491; High: std ≥ 0.120908491. Validation counts: {'Low': 20, 'Medium': 18, 'High': 22}. Units are normalized pixel intensities. Boundaries, memberships, and counts are exported in [level_definition.json](level_definition.json). These bins are not brightness levels or network stages.

## Architecture and why widths were chosen

The classical front end estimates local RGB noise maps, conservatively repairs isolated outliers, and uses opponent-color db2 wavelet shrinkage (up to three scales). The hybrid network receives nine channels: original RGB, classical RGB estimate, and three noise maps. A NAF-style encoder-decoder predicts a residual added to the classical estimate. Its zero-initialized output layer begins at that anchor. The RGB-only control instead uses three channels and anchors to the noisy input.

Width is the base feature-channel count, not image resolution. Encoder channels are w, 2w, 4w, 8w; the bottleneck has 16w channels. Encoder block counts are 1/1/2/4, middle 4, decoder one block per scale, with skip additions and pixel-shuffle upsampling. Four downsamples require padding to multiples of 16. Standard PyTorch implementation in [model.py](../../model.py), based on the NAF block design attributed there.

The first controlled comparison trained hybrid16, rgb16, hybrid32, and real-pairs-only hybrid16 to 4,000 updates. Width 32 won and was extended to 20,000. Widths 48 and 64 were later tested at exactly 4,000 updates against width 32’s 0.564605450; both improved and qualified for extension. The requested 96/128 queue trained both to 20,000 without a 4,000-step gate. Therefore width 16 has only 4,000-step evidence, and wider-versus-16 final scores mix capacity and duration. Compare equal-step rows for capacity claims. Parameter count and memory grow roughly quadratically in width.

| Run | Width | Best local step | Total updates | PSNR dB | SSIM | Score |
|---|---:|---:|---:|---:|---:|---:|
| hybrid128 | 128 | 20000 | 20000 | 30.949480 | 0.910601 | 0.627013204 |
| hybrid96 | 96 | 20000 | 20000 | 30.882558 | 0.910725 | 0.625354468 |
| hybrid64 | 64 | 20000 | 20000 | 30.733557 | 0.907315 | 0.618567422 |
| hybrid48 | 48 | 20000 | 20000 | 30.537013 | 0.904992 | 0.611368473 |
| hybrid32_ssim0.05 | 32 | 2000 | 22000 | 30.399974 | 0.906383 | 0.606984184 |
| hybrid32_ssim0.1 | 32 | 2000 | 22000 | 30.394534 | 0.906564 | 0.606817931 |
| hybrid32 | 32 | 20000 | 20000 | 30.354445 | 0.902484 | 0.604382665 |
| hybrid16 | 16 | 4000 | 4000 | 28.595903 | 0.843417 | 0.517370094 |
| hybrid16_real_only | 16 | 4000 | 4000 | 28.582153 | 0.842446 | 0.516431807 |
| rgb16 | 16 | 4000 | 4000 | 28.097404 | 0.812128 | 0.484914578 |
| wavelet_val_1 | — | — | — | 26.145290 | 0.771209 | 0.390462523 |
| baseline_val | — | — | — | 23.788519 | 0.626926 | 0.238256914 |

Scores above are base/raw validation scores unless a run name denotes SSIM fine-tuning; mild-protection results are separate below. Baseline and wavelet have no neural width or optimizer steps.

Paired uncertainty check: delivered protected width 128 versus saved base models on the same validation images. Resample the 41 scene groups (including singleton images) with replacement 10,000 times, seed 20260913; weight each bootstrap sample by its image count. These percentile intervals describe this dataset only and do not correct for repeated model selection.

| Comparator | Mean score gain | Paired group bootstrap 95% interval |
|---|---:|---|
| Base width 32, 20,000 | 0.022507431 | [0.017672123, 0.027782541] |
| Base width 64, 20,000 | 0.008322674 | [0.005453786, 0.011661348] |
| Base width 96, 20,000 | 0.001535628 | [-0.000868602, 0.003790811] |

| Width | Microbatch × accumulation | Effective batch | Saved CPU median seconds/image |
|---:|---|---:|---:|
| 16 | 8 × 2 | 16 | 2.424 |
| 32 | 8 × 2 | 16 | 5.836 |
| 48 | 8 × 2 | 16 | 10.632 |
| 64 | 4 × 4 | 16 | 19.855 |
| 96 | 2 × 8 | 16 | 29.780 |
| 128 | 1 × 16 | 16 | 49.730 |

CPU measurements use small validation subsets, four PyTorch threads, and different historical runs/load conditions; they are indicative, not a controlled hardware leaderboard. Memory probes selected microbatches with headroom while holding effective batch at 16. Width 128’s historical CPU median is about 49.7 seconds/image (about 16.6 minutes for 20 images before overhead); CUDA is used for the delivered PNGs. Check actual manifest timings for this run.

## Training and fine-tuning recipe

One step means one optimizer update after gradient accumulation, not one image and not an epoch. At effective batch 16, 20,000 steps sample 320,000 crops. Image selection is random with replacement, so this is not 941 complete deterministic passes over 340 images. For each absolute sample index, NumPy is seeded with 20260908 + index, preserving crop/augmentation choices across resumes when effective batch and recipe stay fixed.

The first 1,000 updates use 128 × 128 crops; later updates use 256 × 256. The option `warmup_steps` controls crop size, **not learning-rate warmup**. Crops rotate by multiples of 90 degrees and randomly flip horizontally. The default mixture is 70% supplied noisy/clean pairs, 20% new synthetic corruptions, and 10% clean identity examples. Synthetic noise combines read noise U(0.003,0.07), shot coefficient U(0,0.025), sigma=sqrt(read²+shot*clean), a horizontal spatial scale, and optional sparse impulses; clipping/PNG quantization follows. This broad generator does not claim to reconstruct the organizer’s noise process.

Hybrid features are cached at full-image resolution as uint8: noisy RGB, classical RGB, noise maps divided by 0.3, and clean RGB. Training restores the noise scale; synthetic branches recompute features for the sampled crop. Changing classical preprocessing requires an explicit cache rebuild. Training checks preprocessing and split hashes; each run archives configuration, source hashes, and code copies for its starting/resume segments.

Base loss is MSE. AdamW uses initial lr 0.0002, weight decay 0.0001, gradient norm clipping 1.0, CUDA float16 autocast and GradScaler, with float32 loss arithmetic. The learning-rate rule is `lr0 * (0.05 + 0.95 * 0.5 * (1 + cos(pi * min(step,schedule_steps)/schedule_steps)))`; the base schedule spans 30,000 steps even when stopping at 20,000 (lr then 0.0000575). Extending duration must preserve that schedule for a comparable continuation.

EMA decay is min(0.999, 1 - 1/(step+1)). Full validation uses EMA every 1,000 updates and at the requested endpoint. A strict score improvement replaces `best.pt`; `last.pt` is a resumable state including model, EMA, optimizer, scaler and step. Earlier intermediate best weights are overwritten unless separately preserved, so a score row does not imply that checkpoint remains recoverable. Logs are written about every 50 updates. One seed per configuration was run; neither deterministic settings nor these data establish multi-seed reproducibility.

Width-32 SSIM fine-tunes initialize from the base model’s 20,000-step EMA, use a fresh optimizer at lr 0.00005, 2,000-step cosine schedule, no small-crop phase, and MSE + lambda*(1-SSIM), lambda in {0.05,0.1}. Local fine-tune step 1,000/2,000 means 21,000/22,000 total updates. Lambda 0.05 wins narrowly in measured validation. Do not assume its benefit transfers to width 128 without a new training/validation comparison.

Earlier final training separately restarted width 32 on all 460 pairs for 20,000 MSE updates and 2,000 SSIM updates using the fixed recipe. Validation and best-checkpoint selection were disabled; it saves `final.pt`. Such a model has no remaining held-out score among those 460 images. Its `best_score=-1` status sentinel is not a quality score.

## Overfitting: what the evidence does and does not say

Overfitting is supported when fit on a fixed training evaluation set improves while independently measured validation quality deteriorates persistently. A noisy crop-training loss falling at the same time as a single validation dip is insufficient. Different crops, noise mixtures, EMA lag, and numerical variation can cause fluctuations. No matched full-image training-evaluation series is logged here, so no particular step can be certified as overfitting or as guaranteed free of it.

Each measured step below has an evidence label: first check; new best (no measured validation decline); a deficit from the previous best (inconclusive); or three or more consecutive checks below the previous best (a persistent-deficit warning for investigation). This is a descriptive flag applied after training, not an early-stopping rule used by the historical queues. Repeatedly trying widths and steps on the same validation set also risks selection overfitting even if each curve improves.

All six hybrid base widths end at their best measured validation score within their own completed duration. Thus there is no terminal validation deterioration indicating that the selected 20,000-step width-128 checkpoint overfits; there is also no evidence that steps beyond 20,000 would improve it. Steps not measured (e.g. 20,001) have no score or overfitting classification. Do not interpolate them into evidence.

For further work, keep selection on this validation set, preserve groups before making a new independent holdout, log a fixed train-evaluation subset with identical inference/quantization, and compare multiple seeds and paired scene-group confidence intervals. Predeclare any patience and minimum-gain rule. Never use preliminary outputs, prior locked-test scores, or the final-round images to tune parameters. A new holdout must never have trained the model being assessed; the historical all-data model has already seen all 460 images.

## Every recorded width, step, and analytical noise level

Each Low/Medium/High cell is **PSNR dB / SSIM / composite score**. All is the full 60-image mean. Train loss is the mean of recorded loss-window averages from the preceding 1,000 updates; it is a changing-crop diagnostic, not directly comparable to full-image held-out PSNR. Fine-tune losses also include SSIM. CSVs retain full precision: [steps](validation_steps.csv), [levels](validation_levels.csv), [per-image results](validation_images.csv). Only actually measured checkpoints are listed.

### baseline_val

| Step | All PSNR / SSIM / score | Low | Medium | High | Train loss | Evidence |
|---:|---|---|---|---|---:|---|
| — | 23.7885 / 0.626926 / 0.238256914 | 26.0470 / 0.732271 / 0.183213388 | 23.3304 / 0.538408 / 0.234964726 | 22.1102 / 0.603581 / 0.290990092 | — | First check |

### hybrid128

| Step | All PSNR / SSIM / score | Low | Medium | High | Train loss | Evidence |
|---:|---|---|---|---|---:|---|
| 1000 | 28.1531 / 0.840020 / 0.498301225 | 30.3300 / 0.897655 / 0.420689327 | 28.4320 / 0.823602 / 0.553843441 | 25.9460 / 0.801057 / 0.523413864 | 0.0013774 | First check |
| 2000 | 29.1144 / 0.875010 / 0.550518945 | 31.1708 / 0.918205 / 0.462540372 | 29.5720 / 0.868876 / 0.617115098 | 26.8705 / 0.840761 / 0.576011706 | 0.0010048 | New best; no measured decline |
| 3000 | 29.6921 / 0.890929 / 0.577957146 | 31.7006 / 0.928305 / 0.487772774 | 30.2047 / 0.885893 / 0.644830663 | 27.4468 / 0.861071 / 0.605228243 | 0.0009343 | New best; no measured decline |
| 4000 | 29.9408 / 0.895485 / 0.588969231 | 31.9214 / 0.931346 / 0.497818821 | 30.4908 / 0.890560 / 0.656228448 | 27.6904 / 0.866915 / 0.616802971 | 0.0009137 | New best; no measured decline |
| 5000 | 30.0893 / 0.897652 / 0.594626544 | 32.0553 / 0.933038 / 0.503852323 | 30.6507 / 0.893050 / 0.660910939 | 27.8427 / 0.869248 / 0.622915876 | 0.0008822 | New best; no measured decline |
| 6000 | 30.2070 / 0.899513 / 0.599108527 | 32.1653 / 0.934407 / 0.508802539 | 30.7872 / 0.895031 / 0.665118351 | 27.9519 / 0.871458 / 0.627196842 | 0.0008489 | New best; no measured decline |
| 7000 | 30.3241 / 0.900939 / 0.603418428 | 32.2670 / 0.935562 / 0.513329435 | 30.9264 / 0.896760 / 0.669581948 | 28.0650 / 0.872882 / 0.631183723 | 0.0008294 | New best; no measured decline |
| 8000 | 30.4228 / 0.902190 / 0.607893410 | 32.3578 / 0.936440 / 0.517312965 | 31.0555 / 0.898300 / 0.676210136 | 28.1460 / 0.874236 / 0.634343766 | 0.0008054 | New best; no measured decline |
| 9000 | 30.4801 / 0.903058 / 0.610321203 | 32.4258 / 0.937283 / 0.520370263 | 31.0977 / 0.899145 / 0.678410676 | 28.2062 / 0.875146 / 0.636385215 | 0.0007954 | New best; no measured decline |
| 10000 | 30.5551 / 0.904254 / 0.613107554 | 32.4913 / 0.937910 / 0.523243724 | 31.1851 / 0.900277 / 0.680961758 | 28.2793 / 0.876912 / 0.639284867 | 0.0007727 | New best; no measured decline |
| 11000 | 30.5968 / 0.905381 / 0.615050449 | 32.5525 / 0.938802 / 0.526045249 | 31.2159 / 0.901401 / 0.683117368 | 28.3124 / 0.878255 / 0.640273151 | 0.0007546 | New best; no measured decline |
| 12000 | 30.6381 / 0.906153 / 0.616944401 | 32.6064 / 0.939172 / 0.528352020 | 31.2532 / 0.902250 / 0.685028971 | 28.3455 / 0.879329 / 0.641777373 | 0.0007312 | New best; no measured decline |
| 13000 | 30.6582 / 0.906651 / 0.617629053 | 32.6285 / 0.939426 / 0.529337481 | 31.2583 / 0.902794 / 0.685103207 | 28.3759 / 0.880011 / 0.642687992 | 0.0007112 | New best; no measured decline |
| 14000 | 30.7133 / 0.907487 / 0.619805158 | 32.6704 / 0.939952 / 0.531223487 | 31.3564 / 0.903857 / 0.688660212 | 28.4078 / 0.880943 / 0.643997995 | 0.0007200 | New best; no measured decline |
| 15000 | 30.7835 / 0.908107 / 0.621450955 | 32.6959 / 0.940398 / 0.532420166 | 31.5106 / 0.904987 / 0.691849888 | 28.4501 / 0.881305 / 0.644788909 | 0.0007057 | New best; no measured decline |
| 16000 | 30.7999 / 0.908565 / 0.622095350 | 32.7386 / 0.940724 / 0.534259288 | 31.4933 / 0.905599 / 0.691313288 | 28.4700 / 0.881756 / 0.645313456 | 0.0006969 | New best; no measured decline |
| 17000 | 30.8458 / 0.909264 / 0.624002803 | 32.7830 / 0.941240 / 0.536241418 | 31.5432 / 0.906521 / 0.693473581 | 28.5142 / 0.882440 / 0.646946153 | 0.0006858 | New best; no measured decline |
| 18000 | 30.8614 / 0.909606 / 0.624562838 | 32.8040 / 0.941505 / 0.537186442 | 31.5733 / 0.906884 / 0.694273918 | 28.5131 / 0.882833 / 0.646959587 | 0.0006888 | New best; no measured decline |
| 19000 | 30.9071 / 0.910126 / 0.626277292 | 32.8344 / 0.941868 / 0.538549372 | 31.6362 / 0.907509 / 0.697056695 | 28.5584 / 0.883412 / 0.648119527 | 0.0006632 | New best; no measured decline |
| 20000 | 30.9495 / 0.910601 / 0.627013204 | 32.8540 / 0.942191 / 0.539463646 | 31.7275 / 0.907982 / 0.697661047 | 28.5815 / 0.884025 / 0.648800931 | 0.0006695 | New best; no measured decline |

### hybrid16

| Step | All PSNR / SSIM / score | Low | Medium | High | Train loss | Evidence |
|---:|---|---|---|---|---:|---|
| 1000 | 27.1468 / 0.799708 / 0.441923651 | 29.3937 / 0.873960 / 0.373757124 | 27.3230 / 0.772810 / 0.489166369 | 24.9601 / 0.754213 / 0.465240088 | 0.0017469 | First check |
| 2000 | 27.6653 / 0.814089 / 0.468414568 | 29.8745 / 0.883282 / 0.396719287 | 27.8570 / 0.789088 / 0.517039210 | 25.5001 / 0.771641 / 0.493808298 | 0.0014339 | New best; no measured decline |
| 3000 | 28.1705 / 0.828945 / 0.494564158 | 30.2639 / 0.891498 / 0.415582285 | 28.4548 / 0.807947 / 0.548493105 | 26.0347 / 0.789260 / 0.522242177 | 0.0012729 | New best; no measured decline |
| 4000 | 28.5959 / 0.843417 / 0.517370094 | 30.5861 / 0.898983 / 0.431462463 | 28.9724 / 0.827345 / 0.576959425 | 26.4786 / 0.806051 / 0.546713032 | 0.0012048 | New best; no measured decline |

### hybrid16_real_only

| Step | All PSNR / SSIM / score | Low | Medium | High | Train loss | Evidence |
|---:|---|---|---|---|---:|---|
| 1000 | 27.1421 / 0.800562 / 0.442074413 | 29.2214 / 0.874008 / 0.366883269 | 27.3872 / 0.774273 / 0.492321446 | 25.0512 / 0.755303 / 0.469318788 | 0.0021409 | First check |
| 2000 | 27.6701 / 0.814911 / 0.468937661 | 29.8031 / 0.884216 / 0.394237295 | 27.8976 / 0.790426 / 0.519198194 | 25.5450 / 0.771940 / 0.495724831 | 0.0017483 | New best; no measured decline |
| 3000 | 28.1783 / 0.829091 / 0.494934813 | 30.1953 / 0.892027 / 0.413050097 | 28.5101 / 0.808169 / 0.550794523 | 26.0731 / 0.788994 / 0.523672066 | 0.0015763 | New best; no measured decline |
| 4000 | 28.5822 / 0.842446 / 0.516431807 | 30.4899 / 0.898874 / 0.427570253 | 28.9958 / 0.825622 / 0.577203461 | 26.5094 / 0.804913 / 0.547492774 | 0.0014702 | New best; no measured decline |

### hybrid32

| Step | All PSNR / SSIM / score | Low | Medium | High | Train loss | Evidence |
|---:|---|---|---|---|---:|---|
| 1000 | 27.5435 / 0.813006 / 0.463107778 | 29.7735 / 0.882669 / 0.392435203 | 27.7245 / 0.787886 / 0.511259832 | 25.3679 / 0.770229 / 0.487958439 | 0.0015907 | First check |
| 2000 | 28.3240 / 0.839685 / 0.505000264 | 30.4291 / 0.897772 / 0.424696594 | 28.6401 / 0.822693 / 0.561806849 | 26.1516 / 0.800783 / 0.531525485 | 0.0011864 | New best; no measured decline |
| 3000 | 29.0156 / 0.865302 / 0.542686721 | 30.9819 / 0.911849 / 0.452440728 | 29.4797 / 0.856146 / 0.608025131 | 26.8482 / 0.830477 / 0.571269833 | 0.0010509 | New best; no measured decline |
| 4000 | 29.4452 / 0.879730 / 0.564605450 | 31.3475 / 0.920641 / 0.470580545 | 29.9860 / 0.873244 / 0.632650988 | 27.2733 / 0.847844 / 0.594409014 | 0.0010039 | New best; no measured decline |
| 5000 | 29.6957 / 0.886300 / 0.576065730 | 31.5619 / 0.925272 / 0.481009322 | 30.2521 / 0.880555 / 0.643593793 | 27.5440 / 0.855571 / 0.607230412 | 0.0009646 | New best; no measured decline |
| 6000 | 29.8422 / 0.889741 / 0.582371243 | 31.6940 / 0.927794 / 0.487304106 | 30.3967 / 0.884230 / 0.649327563 | 27.7052 / 0.859657 / 0.614013471 | 0.0009238 | New best; no measured decline |
| 7000 | 29.9317 / 0.891954 / 0.585910285 | 31.7810 / 0.929143 / 0.491322907 | 30.4902 / 0.886546 / 0.652060053 | 27.7934 / 0.862570 / 0.617776273 | 0.0009063 | New best; no measured decline |
| 8000 | 29.9873 / 0.893486 / 0.588162369 | 31.8472 / 0.930196 / 0.494393068 | 30.5392 / 0.887913 / 0.653937218 | 27.8449 / 0.864673 / 0.619591402 | 0.0008908 | New best; no measured decline |
| 9000 | 30.0300 / 0.894766 / 0.590349360 | 31.9026 / 0.930972 / 0.496920049 | 30.6119 / 0.889488 / 0.656491722 | 27.8516 / 0.866169 / 0.621168618 | 0.0008707 | New best; no measured decline |
| 10000 | 30.0884 / 0.895997 / 0.592594768 | 31.9643 / 0.931902 / 0.499760471 | 30.6500 / 0.890565 / 0.658301394 | 27.9236 / 0.867799 / 0.623229618 | 0.0008482 | New best; no measured decline |
| 11000 | 30.1279 / 0.896980 / 0.594537101 | 32.0124 / 0.932591 / 0.501958080 | 30.6927 / 0.891716 / 0.660708347 | 27.9525 / 0.868912 / 0.624559738 | 0.0008303 | New best; no measured decline |
| 12000 | 30.1892 / 0.897902 / 0.596834654 | 32.0735 / 0.933290 / 0.504681704 | 30.7439 / 0.892815 / 0.662736730 | 28.0224 / 0.869892 / 0.626690182 | 0.0008072 | New best; no measured decline |
| 13000 | 30.1982 / 0.898564 / 0.597770946 | 32.1141 / 0.933729 / 0.506481256 | 30.7627 / 0.893351 / 0.663351153 | 27.9947 / 0.870861 / 0.627105040 | 0.0007858 | New best; no measured decline |
| 14000 | 30.2379 / 0.899212 / 0.598771039 | 32.1567 / 0.934248 / 0.508394687 | 30.7727 / 0.893910 / 0.663625637 | 28.0560 / 0.871699 / 0.627868505 | 0.0007810 | New best; no measured decline |
| 15000 | 30.2804 / 0.899913 / 0.600709024 | 32.1889 / 0.934667 / 0.509848886 | 30.8228 / 0.894741 / 0.665914857 | 28.1016 / 0.872549 / 0.629958923 | 0.0007767 | New best; no measured decline |
| 16000 | 30.3044 / 0.900737 / 0.601870328 | 32.2142 / 0.935024 / 0.511005379 | 30.8387 / 0.895900 / 0.666533965 | 28.1310 / 0.873526 / 0.631568214 | 0.0007661 | New best; no measured decline |
| 17000 | 30.3041 / 0.901235 / 0.602323657 | 32.2349 / 0.935369 / 0.511971068 | 30.8547 / 0.896454 / 0.667071317 | 28.0982 / 0.874115 / 0.631487015 | 0.0007583 | New best; no measured decline |
| 18000 | 30.3387 / 0.901931 / 0.603963070 | 32.2636 / 0.935854 / 0.513309868 | 30.8867 / 0.897556 / 0.668971101 | 28.1406 / 0.874671 / 0.633186683 | 0.0007582 | New best; no measured decline |
| 19000 | 30.3508 / 0.902241 / 0.604286630 | 32.2697 / 0.936132 / 0.513664740 | 30.8905 / 0.897681 / 0.669206593 | 28.1649 / 0.875160 / 0.633553834 | 0.0007316 | New best; no measured decline |
| 20000 | 30.3544 / 0.902484 / 0.604382665 | 32.2697 / 0.936274 / 0.513721848 | 30.9080 / 0.897945 / 0.669344858 | 28.1604 / 0.875481 / 0.633650704 | 0.0007412 | New best; no measured decline |

### hybrid32_ssim0.05

| Step | All PSNR / SSIM / score | Low | Medium | High | Train loss | Evidence |
|---:|---|---|---|---|---:|---|
| 1000 | 30.3755 / 0.905703 / 0.605975827 | 32.2821 / 0.937711 / 0.514793409 | 30.9550 / 0.901257 / 0.672760183 | 28.1681 / 0.880242 / 0.634227189 | 0.0041471 | First check |
| 2000 | 30.4000 / 0.906383 / 0.606984184 | 32.3027 / 0.938089 / 0.515770295 | 30.9941 / 0.902165 / 0.674474067 | 28.1841 / 0.881010 / 0.634686906 | 0.0041414 | New best; no measured decline |

### hybrid32_ssim0.1

| Step | All PSNR / SSIM / score | Low | Medium | High | Train loss | Evidence |
|---:|---|---|---|---|---:|---|
| 1000 | 30.3752 / 0.905902 / 0.606004901 | 32.2742 / 0.937774 / 0.514501870 | 30.9724 / 0.901541 / 0.673566262 | 28.1602 / 0.880496 / 0.633911997 | 0.0075676 | First check |
| 2000 | 30.3945 / 0.906564 / 0.606817931 | 32.2944 / 0.938191 / 0.515479045 | 31.0019 / 0.902380 / 0.674939076 | 28.1705 / 0.881236 / 0.634117799 | 0.0075465 | New best; no measured decline |

### hybrid48

| Step | All PSNR / SSIM / score | Low | Medium | High | Train loss | Evidence |
|---:|---|---|---|---|---:|---|
| 1000 | 27.7815 / 0.821455 / 0.476007839 | 30.0365 / 0.887651 / 0.404947818 | 27.9703 / 0.798313 / 0.525259112 | 25.5769 / 0.780212 / 0.500311363 | 0.0015021 | First check |
| 2000 | 28.6640 / 0.853948 / 0.524304643 | 30.7392 / 0.905478 / 0.440183847 | 29.0466 / 0.842039 / 0.585803436 | 26.4643 / 0.816847 / 0.550460899 | 0.0010932 | New best; no measured decline |
| 3000 | 29.3267 / 0.877522 / 0.558949058 | 31.2971 / 0.919758 / 0.468214394 | 29.8353 / 0.871830 / 0.626355242 | 27.1193 / 0.843783 / 0.586284601 | 0.0009954 | New best; no measured decline |
| 4000 | 29.6800 / 0.887892 / 0.576238122 | 31.6142 / 0.926301 / 0.483514271 | 30.2315 / 0.883176 / 0.644035999 | 27.4705 / 0.856833 / 0.605061541 | 0.0009628 | New best; no measured decline |
| 5000 | 29.8637 / 0.892182 / 0.584397709 | 31.7814 / 0.929215 / 0.491366340 | 30.4255 / 0.887291 / 0.651677749 | 27.6607 / 0.862516 / 0.613924376 | 0.0009287 | New best; no measured decline |
| 6000 | 29.9760 / 0.894370 / 0.589183143 | 31.8940 / 0.930827 / 0.496516501 | 30.5379 / 0.889237 / 0.655959815 | 27.7725 / 0.865427 / 0.618790087 | 0.0008936 | New best; no measured decline |
| 7000 | 30.0631 / 0.895916 / 0.592412958 | 31.9813 / 0.931883 / 0.500432440 | 30.6473 / 0.890890 / 0.659521785 | 27.8414 / 0.867331 / 0.621124389 | 0.0008737 | New best; no measured decline |
| 8000 | 30.1147 / 0.896875 / 0.594362298 | 32.0491 / 0.932621 / 0.503439273 | 30.7227 / 0.892081 / 0.662498260 | 27.8587 / 0.868300 / 0.621271989 | 0.0008526 | New best; no measured decline |
| 9000 | 30.1703 / 0.897939 / 0.596607441 | 32.1157 / 0.933610 / 0.506498137 | 30.8081 / 0.893097 / 0.665086683 | 27.8799 / 0.869471 / 0.622496520 | 0.0008406 | New best; no measured decline |
| 10000 | 30.2349 / 0.899033 / 0.599244355 | 32.1630 / 0.934315 / 0.508670732 | 30.8694 / 0.894240 / 0.667977032 | 27.9631 / 0.870881 / 0.625348185 | 0.0008153 | New best; no measured decline |
| 11000 | 30.2898 / 0.899994 / 0.601755994 | 32.2064 / 0.935010 / 0.510684469 | 30.9535 / 0.895666 / 0.672238844 | 28.0044 / 0.871701 / 0.626880503 | 0.0007978 | New best; no measured decline |
| 12000 | 30.3202 / 0.900647 / 0.603216993 | 32.2649 / 0.935590 / 0.513258475 | 30.9472 / 0.896131 / 0.672460036 | 28.0392 / 0.872576 / 0.628344066 | 0.0007762 | New best; no measured decline |
| 13000 | 30.3359 / 0.901127 / 0.603710814 | 32.2913 / 0.935903 / 0.514440051 | 30.9438 / 0.896601 / 0.671820954 | 28.0610 / 0.873217 / 0.629139575 | 0.0007592 | New best; no measured decline |
| 14000 | 30.3879 / 0.901950 / 0.605801418 | 32.3116 / 0.936378 / 0.515441870 | 31.0416 / 0.897846 / 0.675400166 | 28.1041 / 0.874010 / 0.631002032 | 0.0007543 | New best; no measured decline |
| 15000 | 30.4186 / 0.902637 / 0.607033193 | 32.3415 / 0.936825 / 0.516817398 | 31.0760 / 0.898666 / 0.676281511 | 28.1326 / 0.874807 / 0.632389837 | 0.0007484 | New best; no measured decline |
| 16000 | 30.4589 / 0.903255 / 0.608368861 | 32.3723 / 0.937103 / 0.518159707 | 31.1433 / 0.899547 / 0.678390688 | 28.1595 / 0.875518 / 0.633086597 | 0.0007386 | New best; no measured decline |
| 17000 | 30.4925 / 0.903688 / 0.609375268 | 32.4048 / 0.937429 / 0.519590224 | 31.1586 / 0.900051 / 0.678576555 | 28.2091 / 0.875991 / 0.634378801 | 0.0007320 | New best; no measured decline |
| 18000 | 30.5070 / 0.904041 / 0.609728956 | 32.4051 / 0.937501 / 0.519630397 | 31.1859 / 0.900632 / 0.679390236 | 28.2260 / 0.876411 / 0.634641144 | 0.0007310 | New best; no measured decline |
| 19000 | 30.5296 / 0.904630 / 0.610848563 | 32.4173 / 0.937806 / 0.520240911 | 31.2194 / 0.901314 / 0.680988004 | 28.2490 / 0.877182 / 0.635832341 | 0.0007048 | New best; no measured decline |
| 20000 | 30.5370 / 0.904992 / 0.611368473 | 32.4296 / 0.938067 / 0.520837751 | 31.2517 / 0.901821 / 0.682541022 | 28.2317 / 0.877517 / 0.635437045 | 0.0007141 | New best; no measured decline |

### hybrid64

| Step | All PSNR / SSIM / score | Low | Medium | High | Train loss | Evidence |
|---:|---|---|---|---|---:|---|
| 1000 | 27.9414 / 0.829233 / 0.485514847 | 30.1664 / 0.892004 / 0.411884784 | 28.1636 / 0.808565 / 0.537093014 | 25.7367 / 0.789080 / 0.510250949 | 0.0014405 | First check |
| 2000 | 28.8513 / 0.864191 / 0.535798750 | 30.9026 / 0.911258 / 0.449032482 | 29.2934 / 0.855495 / 0.600740853 | 26.6247 / 0.828518 / 0.561542728 | 0.0010448 | New best; no measured decline |
| 3000 | 29.5061 / 0.884626 / 0.568604201 | 31.4753 / 0.924059 / 0.477060249 | 30.0413 / 0.879473 / 0.636830662 | 27.2781 / 0.852993 / 0.596004325 | 0.0009605 | New best; no measured decline |
| 4000 | 29.8359 / 0.892019 / 0.583645394 | 31.7676 / 0.928938 / 0.490703713 | 30.4116 / 0.887352 / 0.651850403 | 27.6089 / 0.862274 / 0.612333734 | 0.0009345 | New best; no measured decline |
| 5000 | 29.9964 / 0.895261 / 0.590486703 | 31.9073 / 0.931239 / 0.497213988 | 30.5797 / 0.890539 / 0.658273758 | 27.7821 / 0.866417 / 0.619817945 | 0.0009059 | New best; no measured decline |
| 6000 | 30.0865 / 0.896361 / 0.593732590 | 32.0057 / 0.932104 / 0.501495914 | 30.6660 / 0.891130 / 0.660158532 | 27.8677 / 0.868147 / 0.623235616 | 0.0008778 | New best; no measured decline |
| 7000 | 30.1941 / 0.897971 / 0.597744697 | 32.1024 / 0.933433 / 0.505893821 | 30.8176 / 0.893133 / 0.664955199 | 27.9493 / 0.869690 / 0.626255081 | 0.0008523 | New best; no measured decline |
| 8000 | 30.2740 / 0.899155 / 0.600846755 | 32.1885 / 0.934211 / 0.509650804 | 30.9242 / 0.894587 / 0.669604334 | 28.0016 / 0.871022 / 0.627495963 | 0.0008343 | New best; no measured decline |
| 9000 | 30.3221 / 0.899978 / 0.602790163 | 32.2547 / 0.934925 / 0.512582066 | 30.9861 / 0.895562 / 0.671956123 | 28.0219 / 0.871821 / 0.628207192 | 0.0008185 | New best; no measured decline |
| 10000 | 30.4046 / 0.900986 / 0.605909095 | 32.3253 / 0.935548 / 0.515658191 | 31.0904 / 0.896750 / 0.675724507 | 28.0974 / 0.873032 / 0.630833672 | 0.0007956 | New best; no measured decline |
| 11000 | 30.4322 / 0.901814 / 0.607416614 | 32.3814 / 0.936181 / 0.518155830 | 31.1075 / 0.897600 / 0.676917688 | 28.1076 / 0.874018 / 0.631698267 | 0.0007778 | New best; no measured decline |
| 12000 | 30.4729 / 0.902699 / 0.609035912 | 32.4217 / 0.936723 / 0.519982740 | 31.1524 / 0.898521 / 0.678819509 | 28.1454 / 0.875186 / 0.632897672 | 0.0007527 | New best; no measured decline |
| 13000 | 30.5077 / 0.903382 / 0.610640085 | 32.4471 / 0.937154 / 0.521172766 | 31.2094 / 0.899599 / 0.681239499 | 28.1706 / 0.875776 / 0.634210854 | 0.0007395 | New best; no measured decline |
| 14000 | 30.5423 / 0.904206 / 0.612165187 | 32.4786 / 0.937737 / 0.522665494 | 31.2415 / 0.900482 / 0.683183909 | 28.2098 / 0.876769 / 0.635422317 | 0.0007355 | New best; no measured decline |
| 15000 | 30.5738 / 0.904635 / 0.613335408 | 32.5074 / 0.938222 / 0.524011792 | 31.2811 / 0.901136 / 0.684598492 | 28.2372 / 0.876966 / 0.636232535 | 0.0007275 | New best; no measured decline |
| 16000 | 30.6007 / 0.905240 / 0.614286038 | 32.5311 / 0.938370 / 0.525019407 | 31.3138 / 0.902096 / 0.685304728 | 28.2622 / 0.877695 / 0.637331320 | 0.0007226 | New best; no measured decline |
| 17000 | 30.6479 / 0.906003 / 0.615903825 | 32.5507 / 0.938772 / 0.525961860 | 31.3981 / 0.903011 / 0.687738388 | 28.3044 / 0.878661 / 0.638895514 | 0.0007162 | New best; no measured decline |
| 18000 | 30.6976 / 0.906471 / 0.617157424 | 32.5713 / 0.939064 / 0.526902504 | 31.4704 / 0.903805 / 0.689617140 | 28.3620 / 0.879023 / 0.639922130 | 0.0007116 | New best; no measured decline |
| 19000 | 30.7190 / 0.906860 / 0.617813868 | 32.5837 / 0.939249 / 0.527474804 | 31.4613 / 0.904192 / 0.689294042 | 28.4164 / 0.879599 / 0.641456512 | 0.0006890 | New best; no measured decline |
| 20000 | 30.7336 / 0.907315 / 0.618567422 | 32.5976 / 0.939583 / 0.528161505 | 31.4694 / 0.904675 / 0.689838447 | 28.4370 / 0.880139 / 0.642441962 | 0.0006969 | New best; no measured decline |

### hybrid96

| Step | All PSNR / SSIM / score | Low | Medium | High | Train loss | Evidence |
|---:|---|---|---|---|---:|---|
| 1000 | 28.1573 / 0.839512 / 0.498263968 | 30.3454 / 0.897668 / 0.421311024 | 28.4223 / 0.822659 / 0.553080827 | 25.9512 / 0.800432 / 0.523371031 | 0.0013777 | First check |
| 2000 | 29.0462 / 0.873399 / 0.547235723 | 31.0998 / 0.917372 / 0.459365154 | 29.5105 / 0.867199 / 0.614107417 | 26.7993 / 0.838495 / 0.572404855 | 0.0010136 | New best; no measured decline |
| 3000 | 29.6599 / 0.890059 / 0.576592872 | 31.6586 / 0.927917 / 0.485938712 | 30.1908 / 0.885185 / 0.644326739 | 27.4084 / 0.859630 / 0.603587128 | 0.0009346 | New best; no measured decline |
| 4000 | 29.9471 / 0.895244 / 0.589179428 | 31.9041 / 0.931367 / 0.497136463 | 30.5021 / 0.890530 / 0.657215170 | 27.7140 / 0.866262 / 0.617189244 | 0.0009080 | New best; no measured decline |
| 5000 | 30.1037 / 0.897573 / 0.595484397 | 32.0283 / 0.932914 / 0.502724168 | 30.6909 / 0.893269 / 0.663146148 | 27.8737 / 0.868966 / 0.624452263 | 0.0008766 | New best; no measured decline |
| 6000 | 30.2277 / 0.899316 / 0.600212720 | 32.1368 / 0.934145 / 0.507557536 | 30.8500 / 0.895237 / 0.668164194 | 27.9830 / 0.870990 / 0.628848045 | 0.0008416 | New best; no measured decline |
| 7000 | 30.3246 / 0.900748 / 0.603692867 | 32.2371 / 0.935201 / 0.511990728 | 30.9720 / 0.896667 / 0.671649508 | 28.0561 / 0.872766 / 0.631457560 | 0.0008240 | New best; no measured decline |
| 8000 | 30.3987 / 0.901971 / 0.607080299 | 32.3363 / 0.936115 / 0.516322102 | 31.0275 / 0.897788 / 0.675087143 | 28.1228 / 0.874353 / 0.633945788 | 0.0008018 | New best; no measured decline |
| 9000 | 30.4741 / 0.902991 / 0.609596790 | 32.4077 / 0.936938 / 0.519507659 | 31.1067 / 0.898874 / 0.677658929 | 28.1986 / 0.875498 / 0.635808794 | 0.0007886 | New best; no measured decline |
| 10000 | 30.5506 / 0.903946 / 0.612350991 | 32.4773 / 0.937515 / 0.522523786 | 31.1808 / 0.899896 / 0.680362944 | 28.2835 / 0.876742 / 0.638365942 | 0.0007684 | New best; no measured decline |
| 11000 | 30.5928 / 0.905097 / 0.614145211 | 32.5385 / 0.938234 / 0.525260167 | 31.2000 / 0.900855 / 0.681639271 | 28.3272 / 0.878443 / 0.639727384 | 0.0007546 | New best; no measured decline |
| 12000 | 30.6154 / 0.905737 / 0.615158795 | 32.5653 / 0.938387 / 0.526393893 | 31.2035 / 0.901625 / 0.682972786 | 28.3615 / 0.879419 / 0.640369987 | 0.0007297 | New best; no measured decline |
| 13000 | 30.6393 / 0.906508 / 0.616043360 | 32.6157 / 0.939030 / 0.528666711 | 31.1915 / 0.902463 / 0.682557821 | 28.3907 / 0.880251 / 0.641055755 | 0.0007123 | New best; no measured decline |
| 14000 | 30.6708 / 0.907050 / 0.617417431 | 32.6509 / 0.939541 / 0.530277510 | 31.2554 / 0.903209 / 0.685348722 | 28.3925 / 0.880655 / 0.641055393 | 0.0007153 | New best; no measured decline |
| 15000 | 30.7252 / 0.907721 / 0.619482841 | 32.6831 / 0.940093 / 0.531785308 | 31.3433 / 0.904186 / 0.688422834 | 28.4394 / 0.881185 / 0.642802421 | 0.0007080 | New best; no measured decline |
| 16000 | 30.7525 / 0.908336 / 0.620409568 | 32.7195 / 0.940472 / 0.533395684 | 31.3545 / 0.904735 / 0.688447478 | 28.4718 / 0.882068 / 0.643845718 | 0.0006959 | New best; no measured decline |
| 17000 | 30.7869 / 0.909060 / 0.621731122 | 32.7612 / 0.941124 / 0.535322783 | 31.4084 / 0.905564 / 0.689996201 | 28.4835 / 0.882771 / 0.644430910 | 0.0006961 | New best; no measured decline |
| 18000 | 30.8323 / 0.909650 / 0.623433230 | 32.7971 / 0.941605 / 0.536951671 | 31.4749 / 0.906440 / 0.692277773 | 28.5203 / 0.883224 / 0.645725477 | 0.0006851 | New best; no measured decline |
| 19000 | 30.8606 / 0.910296 / 0.624870030 | 32.8130 / 0.941837 / 0.537680720 | 31.5164 / 0.907411 / 0.694643734 | 28.5490 / 0.883984 / 0.647045464 | 0.0006665 | New best; no measured decline |
| 20000 | 30.8826 / 0.910725 / 0.625354468 | 32.8286 / 0.942166 / 0.538436591 | 31.5668 / 0.907967 / 0.695637145 | 28.5536 / 0.884398 / 0.646866710 | 0.0006723 | New best; no measured decline |

### rgb16

| Step | All PSNR / SSIM / score | Low | Medium | High | Train loss | Evidence |
|---:|---|---|---|---|---:|---|
| 1000 | 25.7107 / 0.711460 / 0.349179769 | 28.2557 / 0.828619 / 0.310099619 | 25.7389 / 0.653462 / 0.378063601 | 23.3741 / 0.652405 / 0.361074954 | 0.0031011 | First check |
| 2000 | 27.1075 / 0.779951 / 0.432449133 | 29.1483 / 0.869944 / 0.362333967 | 27.3424 / 0.737035 / 0.475634840 | 25.0602 / 0.733251 / 0.460856433 | 0.0015461 | New best; no measured decline |
| 3000 | 27.7264 / 0.801413 / 0.465790097 | 29.7223 / 0.882537 / 0.390331860 | 27.9749 / 0.762742 / 0.511218205 | 25.7087 / 0.759304 / 0.497220041 | 0.0014345 | New best; no measured decline |
| 4000 | 28.0974 / 0.812128 / 0.484914578 | 30.1080 / 0.887244 / 0.407644016 | 28.3606 / 0.777412 / 0.532514723 | 26.0542 / 0.772244 / 0.516214968 | 0.0013234 | New best; no measured decline |

### wavelet_val_1

| Step | All PSNR / SSIM / score | Low | Medium | High | Train loss | Evidence |
|---:|---|---|---|---|---:|---|
| — | 26.1453 / 0.771209 / 0.390462523 | 28.5113 / 0.851245 / 0.329375388 | 26.1745 / 0.739260 / 0.429807899 | 23.9705 / 0.724589 / 0.413804610 | — | First check |

## Delivered protection, robustness, and historical test

The delivered threshold 0.03 blends noisy input with network output using squared clipped opponent-chroma noise confidence and a weak-texture luminance safeguard for grayscale/correlated noise. Threshold 0 disables blending. The historical wide-model validation curves use threshold 0, while their stress diagnostics explicitly use 0.03. The exported model fixes 0.03 and was therefore evaluated again on all 60 validation images. This prevents attributing the raw score to a different deployment configuration.

| Delivered configuration | Count | PSNR dB | SSIM | Composite |
|---|---:|---:|---:|---:|
| w128 / 20000 / threshold .03 / All | 60 | 30.947021 | 0.910539 | 0.626890096 |
| w128 / 20000 / threshold .03 / Low | 20 | 32.846656 | 0.942006 | 0.539094321 |
| w128 / 20000 / threshold .03 / Medium | 18 | 31.727536 | 0.907982 | 0.697661045 |
| w128 / 20000 / threshold .03 / High | 22 | 28.581476 | 0.884025 | 0.648800932 |

The following stress conditions use eight fixed central 256px validation crops, seed 914. They are synthetic diagnostic levels, separate from the analytical full-image noise bins. They have PSNR/SSIM/error scores but no official competition composite. The reported 120 dB on an unchanged clean crop is the implementation’s MSE floor, not a finite measured noise level.

| Width | Stress condition | PSNR dB | SSIM | Input MSE | Output MSE |
|---:|---|---:|---:|---:|---:|
| 48 | clean | 120.000000 | 1.000000 | 0.000000000 | 0.000000000 |
| 48 | correlated_color | 25.101360 | 0.763761 | 0.008840306 | 0.003167590 |
| 48 | gray_clean | 120.000000 | 1.000000 | 0.000000000 | 0.000000000 |
| 48 | gray_noise | 25.136219 | 0.762572 | 0.008873815 | 0.003133227 |
| 48 | mild_gaussian | 40.369159 | 0.984431 | 0.000099536 | 0.000091862 |
| 48 | signal_dependent | 31.793247 | 0.943566 | 0.006628115 | 0.000758023 |
| 48 | spatial_mixture | 32.439362 | 0.945062 | 0.005449469 | 0.000618429 |
| 48 | strong_gaussian | 30.533544 | 0.915440 | 0.008862070 | 0.000955838 |
| 64 | clean | 120.000000 | 1.000000 | 0.000000000 | 0.000000000 |
| 64 | correlated_color | 24.743349 | 0.741424 | 0.008840306 | 0.003400562 |
| 64 | gray_clean | 120.000000 | 1.000000 | 0.000000000 | 0.000000000 |
| 64 | gray_noise | 24.792399 | 0.744019 | 0.008873815 | 0.003363203 |
| 64 | mild_gaussian | 40.348107 | 0.984348 | 0.000099536 | 0.000092307 |
| 64 | signal_dependent | 31.918291 | 0.944466 | 0.006628115 | 0.000734218 |
| 64 | spatial_mixture | 32.551318 | 0.946181 | 0.005449469 | 0.000602540 |
| 64 | strong_gaussian | 30.573840 | 0.915696 | 0.008862070 | 0.000945081 |
| 96 | clean | 120.000000 | 1.000000 | 0.000000000 | 0.000000000 |
| 96 | correlated_color | 24.736686 | 0.737230 | 0.008840306 | 0.003408883 |
| 96 | gray_clean | 120.000000 | 1.000000 | 0.000000000 | 0.000000000 |
| 96 | gray_noise | 24.712501 | 0.736638 | 0.008873815 | 0.003424843 |
| 96 | mild_gaussian | 40.295771 | 0.984163 | 0.000099536 | 0.000093427 |
| 96 | signal_dependent | 32.058320 | 0.946466 | 0.006628115 | 0.000713472 |
| 96 | spatial_mixture | 32.737143 | 0.948386 | 0.005449469 | 0.000579755 |
| 96 | strong_gaussian | 30.725814 | 0.917619 | 0.008862070 | 0.000915262 |
| 128 | clean | 120.000000 | 1.000000 | 0.000000000 | 0.000000000 |
| 128 | correlated_color | 24.173432 | 0.710083 | 0.008840306 | 0.003857564 |
| 128 | gray_clean | 120.000000 | 1.000000 | 0.000000000 | 0.000000000 |
| 128 | gray_noise | 24.117583 | 0.711066 | 0.008873815 | 0.003900382 |
| 128 | mild_gaussian | 40.280944 | 0.984108 | 0.000099536 | 0.000093745 |
| 128 | signal_dependent | 32.110837 | 0.946898 | 0.006628115 | 0.000704449 |
| 128 | spatial_mixture | 32.828672 | 0.949091 | 0.005449469 | 0.000566937 |
| 128 | strong_gaussian | 30.775936 | 0.917785 | 0.008862070 | 0.000907011 |

Final verification on 2026-09-13: all 20 outputs pass exact filename, 992×992 RGB PNG, SHA-256 and ZIP integrity checks. CPU image 021 took 118.64 seconds on its first pass under current machine conditions; two CPU passes were PNG-identical. CPU/CUDA maximum difference was 1 uint8 level on that image. Regenerating delivered image 461 on CUDA was pixel-identical. These are explicitly one-image runtime checks, not a claim that every image was rerun on both devices.

Width 128 with threshold 0.03 passes the earlier diagnostic criteria: clean mean absolute error ≤0.003 and mild-Gaussian output MSE ≤1.1× input MSE. Grayscale/correlated noise remains harder than independent strong Gaussian noise. These eight crops cannot establish real-camera generalization.

Historical locked test: selected width-32 development model, threshold 0.02, 20,000 MSE + 2,000 SSIM updates; PSNR 31.362027, SSIM 0.916425, composite 0.647652094 on 60 different images. **This score belongs only to that checkpoint**, not to the all-data width-32 model or the delivered width-128 model. Different split difficulty prevents ranking this score against the width-128 validation score.

## Reproduce the outputs and continue development

Run from the repository root with Python 3.12. The local interpreter is `scripts/.venv/Scripts/python.exe`. Runtime versions are pinned in [requirements.txt](../../requirements.txt); use [requirements-gpu.txt](../../requirements-gpu.txt) for the recorded CUDA build or [requirements-cpu.txt](../../requirements-cpu.txt) for CPU. The environment lock is retained. Install into your own environment; do not upload virtual environments.

```powershell
python -m pip install -r scripts/requirements-gpu.txt
# Place the downloaded inference checkpoint at the path below.
python scripts/denoise.py --noise_dir submissions/noisy --denoised_dir reproduced_images --checkpoint scripts/checkpoints/NoMoreTokens_w128_s20000.pt --device cuda --manifest reproduced_manifest.json
# Substitute --device cpu when CUDA is unavailable.
```

Inference loads local EMA weights once, uses float32 512px tiles with 64px overlap and positive Hann blending, preserves dimensions, and writes sorted <id>.png names. No brightness/gamma/contrast adjustment is applied. CPU and CUDA may differ by a PNG quantization level; the manifest hashes describe the exact delivered CUDA files. Missing weights are an error, not a silent classical fallback.

```powershell
# Fresh development run, retaining the original split and effective batch.
python scripts/prepare_cache.py
python scripts/train.py --run new_w128 --width 128 --steps 20000 --schedule-steps 30000 --batch 1 --accumulate 16 --device cuda
# Continue an existing run, preserving architecture, data mixture and schedule.
python scripts/train.py --run new_w128 --width 128 --steps 25000 --schedule-steps 30000 --batch 1 --accumulate 16 --device cuda --resume scripts/runs/new_w128/last.pt
# Separate SSIM experiment; this has NOT been run for width 128.
python scripts/train.py --run new_w128_ssim005 --width 128 --steps 2000 --schedule-steps 2000 --warmup-steps 0 --lr 0.00005 --ssim-weight 0.05 --batch 1 --accumulate 16 --init scripts/runs/new_w128/best.pt --device cuda
```

Use a new run name for recipe changes. A different width cannot initialize directly from these weights because tensor shapes differ. `--resume` restores optimizer/scaler and sample position, while `--init` starts a new step counter/optimizer from EMA. An inference-only export lacks the optimizer and raw-model fields needed for exact resume: retain `best.pt`/`last.pt` locally for that purpose. Export weights only after selection and revalidate any deployment threshold. Rebuild this handoff with `python scripts/build_final_delivery.py` when its archived input runs and final-output files are available.

## Provenance and publication

Checkpoint SHA-256: `5bbd0b133f98603a72ab3b4679627dcbc9c2a2d1e3166e30d46130131f61287c`. Exported weights size: 734,541,793 bytes. Source checkpoint and source-file hashes are in the export metadata; original measurement files are copied under [evidence](evidence) with [checksums](evidence_manifest.json). The ZIP manifest references the existing code base commit and separately records working-tree source hashes; it is not a claim that uncommitted handoff files were part of that old commit.

The root README links this report; the earlier `scripts/reports/RESULTS.md` and prior PDF/technical guide describe earlier stages. Raw training runs, datasets, caches, environments and inference weights are excluded from ordinary Git. The final image list includes exact bytes and SHA-256, and the archive contains only 461.png–480.png. See [GITHUB_UPLOAD.md](../../../GITHUB_UPLOAD.md) for measured publication sizes, destination, and release-asset handling.
