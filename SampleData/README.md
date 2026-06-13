# SampleData — representative subset of Dataset4

This folder contains **321 image/mask pairs** (~87 MB) sampled from the project's
`Dataset4`, provided so the pipeline can be demonstrated without the full datasets.

- `Images/` — tampered RGB images.
- `Labels/` — corresponding binary ground-truth masks (white = Forged region).

**Why a subset?** The complete datasets used in this project are very large
(Dataset ≈ 22 GB, Dataset1 ≈ 2.3 GB, Dataset2 ≈ 31.5 GB, Dataset4 ≈ 5 GB) and exceed
GitHub's practical limits, so they are excluded via `.gitignore`. Trained model files
(`*.mat`, `*.pth`) larger than 100 MB are likewise excluded. To run on the full data,
point the scripts' `dataFolder` / `labelFolder` at your local dataset copies.
