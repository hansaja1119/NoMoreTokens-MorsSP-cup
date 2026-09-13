# GitHub publication and upload size

Destination: [abdul6996/MORA_SP_CUP](https://github.com/abdul6996/MORA_SP_CUP). The repository is public. The existing initial commit is preserved in the integrated history.

The user requested publication and explicitly selected inclusion of the selected model weights. Code, documentation, score evidence, final images, and their manifests belong in Git; the large selected weights and image-only ZIP belong in a GitHub release. Training datasets, feature caches, virtual environments, and intermediate checkpoints remain local.

| Upload component | Bytes before Git compression | Decimal MB |
|---|---:|---:|
| Current code, reports, evidence, and 20 final images (297 files) | 39,899,816 | 39.900 |
| Release: VISION_HUNTERS_w128_s20000.pt | 734,541,793 | 734.542 |
| Release: VISION_HUNTERS.zip | 30,804,119 | 30.804 |
| Release: submission_manifest.json | 2,561 | 0.003 |
| Release: inference_manifest.json | 5,994 | 0.006 |
| Release: VISION_HUNTERS_w128_s20000.json | 3,126 | 0.003 |
| Combined file payload | 805,257,409 | 805.257 |

Expected internet data for one successful publication is approximately **0.805 GB**, before accounting for Git compression, existing remote objects, protocol overhead, and history. Budget **0.9 GB** for this upload with modest overhead. This is a planning estimate, not a measurement of your internet provider’s meter; retries can increase it. Decimal MB = 1,000,000 bytes; GiB = 1,073,741,824 bytes. The 20 PNGs are included both in Git and in the release ZIP, so both transfers are counted.

The inference model is 734,541,793 bytes (734.542 MB; about 700.5 MiB). Its original training checkpoint is 2,938,122,309 bytes; the export removes the optimizer and raw-model duplicate while preserving the EMA used for inference. Do not upload all training checkpoints for this handoff.

GitHub blocks regular Git files above 100 MiB, so the weights are distributed as a release asset. [GitHub large-file documentation](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github). This plan does not use Git LFS. If LFS is chosen later, uploads count against the owner’s LFS storage; its bandwidth accounting concerns downloads. Your ISP still transfers upload bytes. [GitHub LFS billing documentation](https://docs.github.com/en/billing/concepts/product-billing/git-lfs).

[Release downloads](https://github.com/abdul6996/MORA_SP_CUP/releases/tag/denoising-final-2026-09-13) include the selected `.pt`, the image-only `VISION_HUNTERS.zip`, and provenance manifests. The publisher verifies each asset’s returned byte count and SHA-256 digest. The release tag identifies the final integrated code commit.

Reproduction checks: 17 project tests passed; final image names, dimensions, RGB/PNG mode, manifest hashes, and ZIP contents were checked; CPU repeat and CPU/CUDA consistency were checked on one validation image, and one final image was regenerated on CUDA. The Markdown guide contains full scope and limitations.

A small training-loader fix permits `--init` from the downloaded EMA-only inference export without trying to read an absent raw-model entry. An integration test initializes and trains one synthetic fixture step. Exact `--resume` still requires a full local training checkpoint. This fix changes future initialization, not the already trained weights or final inference behavior.
