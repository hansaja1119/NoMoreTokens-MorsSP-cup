# VISION_HUNTERS development scorecard

**Historical early scorecard.** For all completed widths and steps, current
selection, and the historical locked-test assessment, read the
[final model guide](final/MODEL_GUIDE.md).

All quality values below are from the same 60-image validation split. The locked test remains reserved.
Gain intervals resample 41 scene groups, using 10,000 paired bootstrap replicates.
They quantify uncertainty within this dataset, not the probability of winning.

| Run | Steps | PSNR | SSIM | Composite | Gain over baseline, 95% interval |
|---|---:|---:|---:|---:|---|
| baseline_val | - | 23.789 | 0.6269 | 0.23826 | [0.0000, 0.0000] |
| wavelet_val_1 | - | 26.145 | 0.7712 | 0.39046 | [0.1168, 0.1862] |
| hybrid16 | 1000 | 27.147 | 0.7997 | 0.44192 | [0.1643, 0.2411] |
| hybrid16 | 2000 | 27.665 | 0.8141 | 0.46841 | [0.1900, 0.2682] |
| hybrid16 | 3000 | 28.170 | 0.8289 | 0.49456 | [0.2148, 0.2963] |
| hybrid16 | 4000 | 28.596 | 0.8434 | 0.51737 | [0.2369, 0.3209] |
| rgb16 | 1000 | 25.711 | 0.7115 | 0.34918 | [0.0597, 0.1526] |
| rgb16 | 2000 | 27.108 | 0.7800 | 0.43245 | [0.1461, 0.2363] |
| rgb16 | 3000 | 27.726 | 0.8014 | 0.46579 | [0.1811, 0.2693] |
| rgb16 | 4000 | 28.097 | 0.8121 | 0.48491 | [0.2007, 0.2882] |

Runtime values in individual logs use different devices and concurrency. Do not compare their runtime columns as an isolated CPU benchmark.
The prototype processed all 20 preliminary images on CPU in 66.6 seconds while other development work ran.
The development ZIP and PDF are not a frozen submission. Longer comparisons and final training are still required.
