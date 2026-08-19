# Research article — Image forgery localisation with a transfer-learned DeepLabV3+

This folder holds the research article documenting the experiments stored in this
repository, together with the code that generates it and every figure it contains.

| File | What it is |
|---|---|
| `Image_Forgery_Localisation_DeepLabV3plus_Article.docx` | The article. Title page, abstract, introduction, 20-paper literature survey, experimental work, results, discussion, conclusion, future scope, APA references with DOIs. 15 figures, 7 tables. |
| `make_article.py` | Builds the `.docx` **and verifies it**. Before saving, it re-derives every headline number from the saved experiment artefacts and compares it against the value written into the document. 147 checks; the build aborts rather than emit an unverified figure. |
| `make_figures.py` | Generates all 15 figures as 300-dpi JPEG into `figures/`. |
| `figures/` | The figures, referenced by the article in order. |

## Rebuilding

```
python make_figures.py     # writes figures/*.jpg
python make_article.py     # verifies, then writes the .docx
```

`make_article.py` fails loudly if any number disagrees with its source artefact.

## Where the numbers come from

Every figure in the article traces to one of these:

| Source | Supplies |
|---|---|
| `Improved_Segmentation_Results_transfer/PixelMetrics_improved.mat` | test-split pixel confusion matrix and per-class metrics of the proposed model |
| `Improved_Segmentation_Results_transfer/PerClass_PixelMetrics.csv` | the same per-class metrics in text form |
| `Improved_Segmentation_Results_transfer/PerEpoch_Metrics.csv` | the ten-epoch training/validation history |
| `Improved_Segmentation_Results_{baseline,deeplab}/PerClass_PixelMetrics.csv` | the two small-data ablation runs |
| `logs/04_tuned_model_training_summary.log` | the tuned U-Net's metrics and confusion matrix |
| `logs/03_figure_generation.log` | the tuned U-Net's AUC(ROC) and AUC(PR) |

Two quantities were **not** recorded by the original runs and were measured post hoc
from the saved network, on the CPU-only host, by scripts kept in the session
scratchpad:

* **ROC / precision–recall curves and their areas** — a 10,000-bin score histogram
  accumulated over a fixed random subset of 1,200 of the 7,173 test images
  (207,360,000 scored pixels). Scores are resized to full ground-truth resolution
  before binning so the curves describe the same pixel population as the confusion
  matrix.
* **Duplication audit** — SHA-1 of every binarised mask and an 8x8 average hash of
  every image across the whole corpus, cross-referenced against the exact saved
  partition indices (`dup_audit.py`, `export_split.m`).
* **Training- and validation-split accuracy of the selected network** — 800 images
  from each split, evaluated with the same code path used for the test split, so
  that the generalisation gap is measured rather than inferred from training
  telemetry.

## Caveats stated in the article

These are recorded here as well because they bound what the results mean:

1. **Train/test leakage — the most important caveat.** The 70/15/15 split is drawn
   over frames, not source clips, and the corpus is video-derived. A direct audit of
   all 47,824 frames found only **14,457 distinct ground-truth masks**, and that
   **90.4 % of test frames carry a mask bit-for-bit identical to a training frame's**,
   while **94.6 %** have a near-duplicate image (64-bit average hash, Hamming ≤ 4) in
   the training split. The headline scores therefore measure within-corpus
   localisation against material very close to the training data — a real quantity,
   but not performance on unseen source footage. Re-partitioning by source clip is the
   first item of future work.
2. The tuned U-Net comparison row has **no held-out test split** — its figures come
   from the validation split it was monitored against.
3. **41.73 %** of forged pixels saturate the top score bin, so 42 % of the reported
   AUC-PR is a constant-precision extrapolation rather than a measured curve.
4. Half the corpus (the 23,912-frame `CI_` subset) is **not natural-colour imagery**;
   its mean channel values are (87.6, 23.1, 157.5) against near-neutral values
   elsewhere.
5. No cross-dataset evaluation (CASIA, Columbia, NIST16, Coverage, IMD2020) and no
   robustness study under recompression or noise were performed, so the comparison
   against published benchmark scores is indicative, not controlled.

## Authors

Abhishek Thakur and Hakam Singh, Department of Computer Science and Engineering,
Chitkara University, Himachal Pradesh, India. The co-author e-mail is left as a
bracketed placeholder in the title block and must be filled in before submission.

## `measurement/` — post-hoc measurement scripts and their raw output

Copies of the scripts that produced the quantities not recorded by the original
training runs, together with the exact outputs the article and the verifier read.
`make_article.py` reads these from the session scratchpad; the copies here exist so
the provenance chain is complete inside the repository.

| File | Produces |
|---|---|
| `extract_metrics.m` | `metrics_dump.txt` — confusion matrices, cfg and split sizes from every `PixelMetrics_improved.mat` |
| `roc_transfer.m` | the 10,000-bin score histogram over 1,200 test images |
| `fix_roc.m` | `roc_transfer_summary.txt`, `roc_transfer_curve.csv` — ROC/PR curves and areas |
| `train_acc.m` | `train_acc.txt` — accuracy on the training and validation splits under the test-split procedure |
| `export_split.m` | `split_idx.csv` — the exact saved partition indices |
| `dup_audit.py` | `dup_audit.txt` — the train/test duplication audit |

`fix_roc.m` exists because the first ROC pass integrated the curve with its operating
points in descending-FPR order and returned a negative area. It recomputes the curves
from the saved histograms without repeating inference.
