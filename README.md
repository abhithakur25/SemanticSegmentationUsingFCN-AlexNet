# Semantic Segmentation Using FCN-AlexNet — Image Forgery Localisation

Deep-learning pipeline that reframes **digital image forgery detection as pixel-level
semantic segmentation**: every pixel of a tampered image is labelled **Background**
(authentic) or **Forged**, localising the manipulated region. Built from the
**FCN-AlexNet** construction (pre-trained AlexNet -> convolutionalised `fc6/fc7` ->
transposed-convolution decoder) and extended with an **encoder-decoder U-Net** and
modern PyTorch backbones (ResNet-50, MobileNet-V2/V3, ConvNeXt, EfficientNet, DeepLabV3).

## Headline result (tuned U-Net)

| Metric | Value |
|---|---|
| Forged-class pixel **F1** | **0.731** (IoU 0.576) |
| **Global pixel accuracy** | 94.75 % |
| **ROC-AUC** / **PR-AUC** (Forged) | 0.936 / 0.700 |
| Train / Val accuracy | 99.05 % / 94.75 % |
| Train / Val loss | 0.046 / 0.153 |

This was superseded by a transfer-learned DeepLabV3+ run on the full dataset —
see [Improved U-Net / DeepLabV3+ experiments](#improved-unet--deeplabv3-experiments)
below, **including the caveat that qualifies its higher score**.

## Repository layout

| Path | Contents |
|---|---|
| `*.m` | MATLAB pipelines — `fcnAlexNetExample.m` (FCN-AlexNet), `forgery_detection_UNet_Segmentation_Final_*Tuned.m` (original tuned U-Net), `forgery_detection_UNet_Segmentation_Improved.m` (rewritten, multi-variant U-Net/DeepLabV3+ pipeline — see below), `forgery_models_compare_gpu_final_v2.m`, `GT_Reshape.m`/`reshape1.m` (mask synthesis), `apply_binary_mask_overlay.m`, `PlotBarGraph.m`, `gen_doc_figures.m` (result figures). |
| `*.py`, `*.pyx` | PyTorch backbones & multi-model comparison (GPU-oriented). |
| `Documentation/` | Full technical **report** (`Semantic_Segmentation_FCN_AlexNet_Report.md`) + figures (confusion matrix, ROC, PR/AUC) + its README. `Documentation/article/` holds a separate, fully-verified **research article** (`.docx`) covering the improved experiments below, plus the Python/MATLAB scripts and raw measurement output that generated every number and figure in it. |
| `Improved_Segmentation_Results_{baseline,deeplab,transfer}/` | Saved metrics, figures, and sample predictions from the three variants of `forgery_detection_UNet_Segmentation_Improved.m` (trained network `.mat` files are gitignored — see [Not included](#not-included-size)). |
| `SampleData/` | 321 image/mask pairs (~87 MB) sampled from Dataset4 for demonstration. |
| `logs/` | Real execution logs (environment probe, U-Net verification run, figure generation, tuned-model training summary). |
| `CLAUDE.md` | Repository layout, run notes, path/robustness history. |

## Improved U-Net / DeepLabV3+ experiments

`forgery_detection_UNet_Segmentation_Improved.m` is a from-scratch rewrite of the
tuned U-Net script, aimed at two weaknesses the original showed (Forged-class
precision ≈ recall — i.e. boundary/localisation error, not class imbalance — and
a ~4-point train/val accuracy gap, i.e. overfitting):

* Ported to the MATLAB R2026a API (`unet()` + `trainnet()`, since `unetLayers`/
  `deeplabv3plusLayers` were removed).
* Geometric augmentation (reflection/rotation/translation/scale) applied
  identically to image and mask.
* Soft-Dice loss (optimises overlap directly, weights classes by area) as an
  alternative to cross-entropy.
* LR decay, early stopping, `OutputNetwork='best-validation-loss'`.
* A proper 70/15/15 train/val/**test** split — the test set is untouched until
  final evaluation (the original tuned script reported its headline metrics on
  the validation split it was monitored against).
* Consistent `>127` mask binarisation for both training and evaluation (the
  original mixed Otsu and `>127`, which disagreed on 27 % of boundary pixels).
* An optional DeepLabV3+ / ResNet-18 backbone in place of the from-scratch U-Net.

Run it as `forgery_detection_UNet_Segmentation_Improved`, or set
`VARIANT = 'baseline' | 'improved' | 'deeplab' | 'transfer'` first to pick a
preset (all four share the same seed and splits). Results land in
`Improved_Segmentation_Results_<VARIANT>/`.

| Variant | Data | Architecture / loss | Forged F1 | Forged IoU |
|---|---|---|---|---|
| `baseline` | `SampleData/` (321 pairs) | U-Net, cross-entropy, no augmentation — reproduces the original recipe as a control | 0.0002 | 0.0001 |
| `deeplab` | `SampleData/` (321 pairs) | DeepLabV3+/ResNet-18, soft-Dice, augmented | 0.136 | 0.073 |
| `transfer` | Full 47,824-pair dataset | DeepLabV3+/ResNet-18 (ImageNet-pretrained), Dice+CE, augmented, 10 epochs | **0.944** | **0.893** |

`baseline` and `deeplab` are small-data ablations on the 321-pair `SampleData/`
included in this repo (`baseline` collapses because 321 images is too little to
learn the Forged class from scratch with no augmentation — that is itself the
point of the comparison). `transfer` is the real result, trained on the full
Dataset4-derived corpus and evaluated on its own held-out 15 % test split.

**Caveat on the `transfer` score:** a duplication audit found the dataset is
video-derived — of 47,824 frames, only 14,457 have a distinct ground-truth
mask, and 90.4 % of test frames are bit-for-bit identical to a training frame's
mask (94.6 % near-duplicate by image hash). The 0.944 F1 therefore measures
within-corpus localisation against material very close to the training data,
not generalisation to unseen source footage. Re-partitioning by source clip
(rather than frame) is the natural next step. Full detail, methodology, and 20
literature references are in
[`Documentation/article/`](Documentation/article/Image_Forgery_Localisation_DeepLabV3plus_Article.docx).

## Not included (size)

Full datasets (Dataset/Dataset1/Dataset2/Dataset4 ~= 60 GB total) and trained model
blobs `> 100 MB` (`*.mat`, `*.pth`) are excluded via `.gitignore` — GitHub's per-file
limit is 100 MB. Use `SampleData/`, or point the scripts at local dataset copies.

---

## How to run — step by step

### Prerequisites

* **MATLAB R2020b+** with the **Deep Learning Toolbox** and **Computer Vision Toolbox**
  (Image Processing Toolbox recommended). For `fcnAlexNetExample.m` you also need the
  **AlexNet** support package (install via *Home -> Add-Ons -> Get Add-Ons ->
  "Deep Learning Toolbox Model for AlexNet Network"*).
* A **CUDA GPU** is optional but strongly recommended — training is many times faster
  than on CPU. The scripts use `'ExecutionEnvironment','auto'` and fall back to CPU.
* The PyTorch scripts (`*.py`) additionally need **Python 3.9+** with `torch`,
  `torchvision`, `scikit-learn`, `tqdm`, `Pillow`, `matplotlib`, and a CUDA GPU
  (the `*_gpu_only.py` scripts raise an error if no GPU is present).

### Step 1 — Clone the repository

```bash
git clone https://github.com/abhithakur25/SemanticSegmentationUsingFCN-AlexNet.git
cd SemanticSegmentationUsingFCN-AlexNet
```

### Step 2 — Get the data

* **Quick demo:** use the bundled **`SampleData/`** (321 image/mask pairs already in
  the repo — `SampleData/Images`, `SampleData/Labels`).
* **Full run:** place your own forgery dataset anywhere on disk, with an `Images/`
  folder (tampered RGB images) and a `Labels/` folder (binary masks, white = forged).
  A matching image and mask must share the same base filename.

### Step 3 — (Optional) Generate ground-truth masks

If you only have *forged* and *authentic* image pairs (no masks yet), edit the source
folders at the top of **`GT_Reshape.m`** and run it — it resizes the pair, subtracts
them, binarises and applies morphology to produce the `Forged`-region mask. Skip this
step if you already have masks (as in `SampleData/`).

### Step 4 — Point the scripts at your data  (IMPORTANT)

Every script has a **path block near the top**. Open the script you want to run and set
the folders to your machine. For the tuned U-Net
(`forgery_detection_UNet_Segmentation_Final_Grayscale_Tuned.m`):

```matlab
dataFolder    = 'C:\path\to\SemanticSegmentationUsingFCN-AlexNet\SampleData\Images';
labelFolder   = 'C:\path\to\SemanticSegmentationUsingFCN-AlexNet\SampleData\Labels';
resultsFolder = 'C:\path\to\SemanticSegmentationUsingFCN-AlexNet\Results_UNet';
```

(For `fcnAlexNetExample.m` set `imgDir` / `labelDir`; for the Python scripts set `DATA_ROOT`.)

### Step 5 — Train a model

In the MATLAB command window (from the repo folder):

```matlab
% Best model — encoder-decoder U-Net (recommended)
forgery_detection_UNet_Segmentation_Final_Grayscale_Tuned

% ...or the reference FCN-AlexNet model (needs the AlexNet add-on)
fcnAlexNetExample
```

Headless / batch mode (no GUI):
`matlab -batch "forgery_detection_UNet_Segmentation_Final_Grayscale_Tuned"`.
The U-Net trains for 12 epochs (Adam); FCN-AlexNet for 30 (SGDM). The trained network is
saved as `netSeg_final.mat` in your `resultsFolder`.

### Step 6 — Inspect results & metrics

After training, the `resultsFolder` contains:

* `PerClass_PixelMetrics.csv` — Precision / Recall / F1 / IoU per class.
* `Figures/` — confusion matrix, Precision/Recall/F1, train/val loss & accuracy.
* `BinarySamples/` and `OverlaySamples/` — predicted masks and overlays on the images.

### Step 7 — (Optional) Regenerate the report figures

To recreate the confusion-matrix, ROC and Precision-Recall/AUC figures from a trained
model, set the model/data paths inside **`gen_doc_figures.m`** and run:

```matlab
gen_doc_figures      % writes Documentation/figures/*.png
```

### Step 8 — Read the full report

Open **`Documentation/Semantic_Segmentation_FCN_AlexNet_Report.md`** for the complete
write-up: Introduction, Literature Review, Methodology + execution **block diagram**,
Experimental Work + comparison table, Results, Discussion, Conclusion and References.

### Optional — PyTorch backbone comparison (GPU)

```bash
python "ConvNeXt-Tiny.py"          # single-backbone classifier
python forgery_models_5version_compare_latest_gpu_only.py   # multi-model comparison
```

Edit `DATA_ROOT` at the top first; for the comparison script set the `RUN_*` flags to
`True` for the models you want, and raise the demo `EPOCHS`/`MAX_EPOCHS` values for a
real run.

---

See `Documentation/Semantic_Segmentation_FCN_AlexNet_Report.md` for the full technical
report, and `CLAUDE.md` for repository internals and run notes.
