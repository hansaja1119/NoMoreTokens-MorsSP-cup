# Dataset audit

460 public pairs; completed in 27.6 seconds.

Split: 340 train / 60 validation / 60 locked test.
Automatic duplicate pairs: 0; conservative visual scene groups: 38.

| Measurement | Minimum | Median | Maximum |
|---|---:|---:|---:|
| brightness | 0.082255 | 0.358732 | 0.896720 |
| noisy_brightness | 0.096959 | 0.363438 | 0.874982 |
| noisy_sigma | 0.014535 | 0.059110 | 0.095933 |
| psnr | 15.445789 | 19.135556 | 30.169785 |
| residual_mean | -0.035083 | 0.004095 | 0.044466 |
| residual_std | 0.030781 | 0.108996 | 0.168740 |
| trimmed_std | 0.025655 | 0.073153 | 0.083831 |
| outlier_fraction | 0.000322 | 0.030097 | 0.148633 |
| residual_corr_h | 0.006497 | 0.033696 | 0.192496 |
| residual_corr_v | 0.006549 | 0.034704 | 0.191917 |
| row_bias_std | 0.004941 | 0.012161 | 0.024209 |
| col_bias_std | 0.004963 | 0.012255 | 0.025100 |
| noisy_black_fraction | 0.000000 | 0.022916 | 0.310235 |
| noisy_white_fraction | 0.000097 | 0.010125 | 0.105497 |

Noise-model parameters must use training_noise_statistics.json, not the all-image descriptive audit.
Locked-test metrics are not used for model selection.