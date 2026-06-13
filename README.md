# Semantic Segmentation Using FCN‑AlexNet — Image Forgery Localisation

Deep‑learning pipeline that reframes **digital image forgery detection as pixel‑level
semantic segmentation**: every pixel of a tampered image is labelled **Background**
(authentic) or **Forged**, localising the manipulated region. Built from the
**FCN‑AlexNet** construction (pre‑trained AlexNet → convolutionalised `fc6/fc7` →
transposed‑convolution decoder) and extended with an **encoder–decoder U‑Net** and
modern PyTorch backbones (ResNet‑50, MobileNet‑V2/V3, ConvNeXt, EfficientNet, DeepLabV3).

## Headline result (tuned U‑Net)

| Metric | Value |
|---|---|
| Forged‑class pixel **F1** | **0.731** (IoU 0.576) |
| **Global pixel accuracy** | 94.75 % |
| **ROC‑AUC** / **PR‑AUC** (Forged) | 0.936 / 0.700 |
| Train / Val accuracy | 99.05 % / 94.75 % |
| Train / Val loss | 0.046 / 0.153 |

## Repository layout

| Path | Contents |
|---|---|
| `*.m` | MATLAB pipelines — `fcnAlexNetExample.m` (FCN‑AlexNet), `forgery_detection_UNet_Segmentation_Final_*Tuned.m` (U‑Net), `forgery_models_compare_gpu_final_v2.m`, `GT_Reshape.m`/`reshape1.m` (mask synthesis), `apply_binary_mask_overlay.m`, `PlotBarGraph.m`, `gen_doc_figures.m` (result figures). |
| `*.py`, `*.pyx` | PyTorch backbones & multi‑model comparison (GPU‑oriented). |
| `Documentation/` | Full technical **report** (`Semantic_Segmentation_FCN_AlexNet_Report.md`) + figures (confusion matrix, ROC, PR/AUC) + its README. |
| `SampleData/` | 321 image/mask pairs (~87 MB) sampled from Dataset4 for demonstration. |
| `logs/` | Real execution logs (environment probe, U‑Net verification run, figure generation, tuned‑model training summary). |
| `CLAUDE.md` | Repository layout, run notes, path/robustness history. |

## Not included (size)

Full datasets (Dataset/Dataset1/Dataset2/Dataset4 ≈ 60 GB total) and trained model
blobs `> 100 MB` (`*.mat`, `*.pth`) are excluded via `.gitignore` — GitHub's per‑file
limit is 100 MB. Use `SampleData/`, or point the scripts at local dataset copies.

## Quick start

```matlab
% Train/evaluate the best model (needs Deep Learning + Computer Vision toolboxes)
forgery_detection_UNet_Segmentation_Final_Grayscale_Tuned
% Reproduce the report figures from a trained model
gen_doc_figures
```

See `Documentation/Semantic_Segmentation_FCN_AlexNet_Report.md` for the full write‑up
(Introduction, Literature Review, Methodology + block diagram, Experimental Work,
Results, Discussion, Conclusion, References).
