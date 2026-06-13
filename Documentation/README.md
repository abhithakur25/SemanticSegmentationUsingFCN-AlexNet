# Documentation — Semantic Segmentation Using FCN‑AlexNet (Image Forgery Localisation)

This folder is the **report wrapper** for the `SemanticSegmentationUsingFCN-AlexNet`
project. It contains the full technical write‑up and all generated result figures.

## Contents

| File | Description |
|---|---|
| `Semantic_Segmentation_FCN_AlexNet_Report.md` | The complete report: Introduction, Literature Review, Methodology + step‑by‑step code execution + block diagram, Experimental Work + comparison table, Results (figures), Discussion, Conclusion, References. |
| `README.md` | This file. |
| `figures/confusion_matrix.png` | Pixel‑level confusion matrix of the tuned U‑Net (from its real saved statistics). |
| `figures/roc_curve.png` | ROC curve + AUC for the Forged class (real inference, 40‑image validation subset). |
| `figures/auc_pr_curve.png` | Precision–Recall (AUC) curve for the Forged class. |
| `figures/roc_data.mat` | Raw curve data (FPR/TPR/Precision/Recall, AUC, AUC‑PR, confusion matrix). |

## How to read the report

Open `Semantic_Segmentation_FCN_AlexNet_Report.md` in any Markdown viewer. On GitHub
(or VS Code with Mermaid support) the **block diagram in §3.6 renders automatically**;
a plain‑text fallback of the diagram is printed directly beneath it.

## What the project does (one paragraph)

The project reframes **digital image‑forgery detection as pixel‑level semantic
segmentation**: it labels every pixel of a tampered image as **Background** (authentic)
or **Forged**, thereby *localising* the manipulated region. It starts from the
**FCN‑AlexNet** construction — a pre‑trained AlexNet whose fully connected layers are
converted to convolutions and followed by a transposed‑convolution up‑sampling decoder
— and adds an **encoder–decoder U‑Net** and several modern PyTorch backbones for
comparison.

## Headline result

| Metric (tuned U‑Net) | Value |
|---|---|
| Forged‑class pixel **F1** | **0.731** |
| Forged‑class **IoU** | 0.576 |
| **Global pixel accuracy** | 94.75 % |
| Final **training accuracy** | 99.05 % |
| Final **validation accuracy** | 94.75 % |
| Training / validation **loss** | 0.046 / 0.153 |

The **U‑Net (encoder–decoder with skip connections)** is the best model because it
maximises the **F1 / IoU of the minority Forged class** — the metrics that matter for
forensic localisation — rather than the misleading overall accuracy.

## Reproducing the figures

The figures are produced by `gen_doc_figures.m` (in the project root), which loads the
trained model `Final_Segmentation_Results_Tuned/netSeg_final.mat`, reads its real saved
confusion matrix, and runs inference on a validation subset of `Dataset4` to compute the
ROC and Precision–Recall curves:

```matlab
cd D:\claude\SemanticSegmentationUsingFCN-AlexNet1
gen_doc_figures      % writes Documentation/figures/*.png
```

Requirements: MATLAB with the Deep Learning and Computer Vision toolboxes. The host
used for this report is **CPU‑only**; a CUDA GPU substantially accelerates training and
inference.

## Source scripts referenced by the report

* `fcnAlexNetExample.m` — FCN‑AlexNet construction and training.
* `forgery_detection_UNet_Segmentation_Final_Grayscale_Tuned.m` — tuned U‑Net (best model).
* `forgery_detection_UNet_Segmentation_Final_Tuned.m` — U‑Net (adapted variant).
* `GT_Reshape.m` — ground‑truth mask synthesis (forged − authentic → binarise → morphology).
* `forgery_models_compare_gpu_final_v2.m` and the `*.py` scripts — multi‑backbone comparison.

See the project‑root `CLAUDE.md` for the full repository layout and run notes.
