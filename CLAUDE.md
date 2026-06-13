# CLAUDE.md — Semantic Segmentation Using FCN-AlexNet (Image Forgery Detection)

## What this project is
PhD thesis research (Dr. Abhishek Thakur) on **image forgery detection framed as pixel-level
semantic segmentation**: given a tampered image, segment the **forged region** vs. the
authentic/background region. It started from MathWorks' "Semantic Segmentation Using
FCN-AlexNet" example and was adapted to a **2-class** forgery problem, then extended with
modern CNN backbones in PyTorch.

Two task framings appear in the code:
- **Segmentation** (main goal): per-pixel `Forged` vs `Background/Authentic`.
- **Classification**: whole-image `forged` vs `authentic` (the standalone Python backbone
  scripts use `ImageFolder`, i.e. per-class subfolders).

## Languages / stack
- **MATLAB** (Deep Learning Toolbox + Computer Vision Toolbox; needs the `alexnet` support
  package). Used for FCN-AlexNet and U-Net segmentation, mask generation, plotting.
- **Python / PyTorch + torchvision** (most scripts are **GPU-only** — they `raise` if CUDA is
  absent). Used for the modern backbone comparison.

## Repository layout

### MATLAB — segmentation
- `fcnAlexNetExample.m` — original FCN-AlexNet pipeline: takes `alexnet`, converts fc6/fc7 to
  conv layers, adds a 32-stride `transposedConv2dLayer` upsampling head + crop/softmax/pixel
  classification. 2 classes (`Forged`=000, `Authentic`=250). Image size `[360 480]`. Note
  `MaxEpochs=1` (demo/quick-run value) and `'ExecutionEnvironment','multi-gpu'`.
- `fcnAlexNetExample.asv` — MATLAB autosave backup of the above (not a real source file).
- `forgery_detection_UNet_Segmentation_Final_Tuned.m` — U-Net (`unetLayers`, EncoderDepth 4,
  input `[352 480 3]`), Adam, 12 epochs. On-the-fly mask binarization (Otsu), saves 5 binary +
  5 overlay samples, computes pixel Precision/Recall/F1/IoU + confusion matrix.
- `forgery_detection_UNet_Segmentation_Final_Grayscale_Tuned.m` — same as above but grayscale
  figures (dashed/greyscale plots for print). This is the most polished/"final" segmentation
  script.

### MATLAB — combined model comparison
- `forgery_models_compare_gpu_final_v2.m` — single-file pipeline that trains/evaluates several
  models in one go: ResNet-50 & MobileNetV2 classification, plus FCN-AlexNet / U-Net-ResNet18 /
  DeepLabV3+-MobileNetV2 segmentation. Robust basename pairing of images↔labels. Toggle models
  via the `run.*` struct.
- `forgery_models_compare.m`, `plot_and_compare_all_models_fixed.m` — older/plotting variants.
- `Results/forgery_models_compare.m` — a copy.

### MATLAB — data prep & utilities
- `GT_Reshape.m` — **builds the ground-truth masks**: resizes forged/authentic image pairs to
  `[360 480]`, subtracts them, binarizes (`im2bw 0.1`), fills holes + dilates → forged-region
  mask. Writes into `Dataset2/{Images,ImagesReszed,Labels,LabelsReszed}`.
- `reshape1.m` — related reshape helper.
- `apply_binary_mask_overlay.m` — batch overlay: pairs RGB images with masks, writes mask /
  red-overlay / seg-only result folders.
- `PlotBarGraph.m` — textured grouped bar chart (400×300 px, in-plot vertical legend) from
  `PerClass_PixelMetrics.csv`. Publication figure generator.
- `GT_Reshape.m` and the `*Reszed` folders show the convention: `Images`/`Labels` = originals,
  `ImagesReszed`/`LabelsReszed` = `[360 480]` resized copies used for training.

### Python — modern backbones (GPU-only)
- `forgery_models_5version_compare_*.py` — three iterations of a multi-model comparison
  (`_fixed`, `_fixed_complete_gpu_only`, `_latest_gpu_only`). Latest models: EfficientNet-V2-S,
  ConvNeXt-Tiny, MobileNetV3-Large, MobileNetV3-Small patch classifier, DeepLabV3-MobileNetV3
  segmentation (with ResNet50/UNet fallbacks). Run flags `RUN_*` at top default to `False`.
- `ConvNeXt-Tiny.py`, `MobileNetV2.py`, `MobileNetV3-Large.py`, `ResNet-50 finetune.py` —
  standalone single-backbone fine-tuners (`ImageFolder` classification, ImageNet weights).
- `patch_mobilenetv3_small.py` — sliding-window patch classifier → forgery heatmap.
- `TinyCNN classifier.pyx` — small from-scratch CNN classifier (despite `.pyx`, it's plain
  Python, not Cython).

### Datasets (image ↔ mask pairs; `*Reszed` = `[360 480]` copies)
- `Dataset`  — Images + imagesReszed1 (no Labels folder).
- `Dataset1` — ~47k Images / ~47k Labels (largest; counts slightly mismatched: 46988 vs 46966).
- `Dataset2` — the one `GT_Reshape.m` / `fcnAlexNetExample.m` write to/read from.
- `Dataset3` — used by `forgery_models_compare_gpu_final_v2.m`.
- `Dataset4` — ~3986 Images / 3986 Labels; used by the U-Net scripts and most Python scripts.

### Outputs
- `Final_Segmentation_Results_Tuned/` & `Final_Segmentation_Results_Adapted/` — U-Net results:
  `BinarySamples/`, `OverlaySamples/`, `Figures/`, `PerClass_PixelMetrics.csv`,
  `PixelMetrics_final.mat`.
- `forgery_results/` — classification compare outputs (`*_class.mat`, `comparison_metrics_fixed.mat`,
  `figures/` with per-class precision/recall/F1 + confusion matrices).
- `Results/`, `outputFolder/`, `html/`, `resources/` — example/published output.
- `net.mat` (227 MB), `matlab.mat` (704 MB) — saved trained network / MATLAB workspace.
  `license.txt` is the MathWorks example BSD license.

## Path migration (done 2026-06-13)
All scripts originally hardcoded absolute paths to drives that don't exist here
(`F:\` / `E:\` / `D:\DRIVES\E Drive\PhD Work\Thesis Code in Process\Semantic Segmentation Using
FCN-AlexNet1\...`, and out-of-project `...\Abhishek_Deep\forgery_results[_v2]`). These have been
repointed to this machine's root `D:\claude\SemanticSegmentationUsingFCN-AlexNet1\`:
- **MATLAB:** `fcnAlexNetExample.m` (Dataset2 + Results), both
  `forgery_detection_UNet_Segmentation_Final_*Tuned.m` (Dataset4),
  `forgery_models_compare_gpu_final_v2.m` (Dataset3; results → project root),
  `forgery_models_compare.m`, `Results\forgery_models_compare.m`,
  `plot_and_compare_all_models_fixed.m` (Abhishek_Deep results → project root),
  `apply_binary_mask_overlay.m` (Dataset4), `GT_Reshape.m` (Dataset2 outputs), `PlotBarGraph.m`.
- **Python (all 9):** every `DATA_ROOT` → `D:/claude/SemanticSegmentationUsingFCN-AlexNet1/Dataset4`.
- **Robustness fixes also applied to `fcnAlexNetExample.m`:** `MaxEpochs` 1→30;
  `ExecutionEnvironment` `multi-gpu`→`auto`; live plot auto-disabled when headless
  (`usejava('desktop')` → `'none'`) + `Verbose` on; hardcoded test indices `2240:2250` replaced
  with `linspace`-spread sampling over the actual test set; test-loop `saveas` now uses captured
  figure handles and distinct `overlay_`/`compare_` filenames (was saving the wrong figure and
  overwriting).
- **NOT fixable — `GT_Reshape.m` lines 3-4:** its *input* datastores read
  `D:\DRIVES\F Drive\ImportPixelLabeledDatasetExample\Dataset\{F_All,L_All}` (forged/authentic
  source images used to synthesize masks). That source dataset is not present anywhere in this
  project, so those two lines were left as-is. `GT_Reshape.m` won't run until that data exists —
  but it's likely unneeded since the Dataset2/Dataset4 masks it produces already exist.

## ⚠️ Other gotchas before running anything
1. **Python scripts are GPU-only** — they raise `RuntimeError` if `torch.cuda.is_available()`
   is false. A CUDA NVIDIA GPU + matching PyTorch build is required.
2. **`MaxEpochs`/`EPOCHS` are still set to 1** (demo values) in several scripts (e.g. the
   `5version_compare` GPU-only scripts) — raise for real training. (`fcnAlexNetExample.m` already
   bumped to 30; the U-Net scripts use 12.)
3. **MATLAB needs the `alexnet` support package** for `fcnAlexNetExample.m`; `unetLayers`,
   `deeplabv3plusLayers`, etc. need the Computer Vision Toolbox.
4. Authorship banners say "ChatGPT"/"adapted by ChatGPT" — these scripts were generated/edited
   by an assistant previously; treat comments cautiously and verify against actual behavior.

## Metric conventions
Pixel-level (segmentation): per-class Precision, Recall, F1, IoU (Jaccard), global pixel
accuracy, confusion matrix. Classification: accuracy, per-class P/R/F1, confusion matrix,
ROC/AUC (one-vs-rest). The minority/positive class of interest is **Forged**.

## How to run (after fixing paths)
- MATLAB (headless): `matlab.exe -batch "forgery_detection_UNet_Segmentation_Final_Grayscale_Tuned"`
  (note: training-progress plots need a GUI; set `'Plots','none'` for headless).
- Python: `python "ConvNeXt-Tiny.py"` (needs CUDA + torch/torchvision/sklearn/tqdm/PIL/matplotlib).
