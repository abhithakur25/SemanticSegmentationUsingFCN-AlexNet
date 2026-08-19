"""
make_figures.py - build every figure used by the research article.

All figures are written as 300-dpi JPEG into Documentation/article/figures.
Colours follow the validated categorical palette (slots 1-4); because slots 3
and 4 fall below 3:1 contrast on a white surface, every bar carries a visible
direct label (the "relief" rule), which also keeps the figures legible when the
article is printed in greyscale.

    python make_figures.py
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Rectangle
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

SCRATCH = (r"C:\Users\USER\AppData\Local\Temp\claude"
           r"\F--Current-Work-SemanticSegmentationUsingFCN-AlexNet"
           r"\809c6f5e-39a4-4d1b-bc50-83811f54d99e\scratchpad")

# ----------------------------------------------------------------- palette
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK      = "#0b0b0b"
INK2     = "#52514e"
MUTED    = "#898781"
GRID     = "#e1e0d9"
AXIS     = "#c3c2b7"
SURFACE  = "#ffffff"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
BLUE_RAMP = LinearSegmentedColormap.from_list("brand_blue", SEQ)

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "lines.linewidth": 2.0,
})


def save(fig, name):
    path = os.path.join(FIG, name)
    fig.savefig(path, dpi=300, format="jpg", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print("wrote", os.path.relpath(path, ROOT))


def tidy(ax, ygrid=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    if ygrid:
        ax.set_axisbelow(True)
        ax.yaxis.grid(True)
        ax.xaxis.grid(False)


def bar_labels(ax, rects, fmt="{:.3f}", dy=0.008, size=7.0):
    for r in rects:
        h = r.get_height()
        ax.text(r.get_x() + r.get_width() / 2, h + dy, fmt.format(h),
                ha="center", va="bottom", fontsize=size, color=INK2)


# =====================================================================
#  Measured results (read from the artefacts written by the experiments)
# =====================================================================

def read_csv_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


PEREPOCH = read_csv_rows(os.path.join(
    ROOT, "Improved_Segmentation_Results_transfer", "PerEpoch_Metrics.csv"))


def col(rows, key, cast=float):
    return np.array([cast(r[key]) for r in rows])


EP        = col(PEREPOCH, "Epoch", int)
TRAINLOSS = col(PEREPOCH, "TrainLoss")
LR        = col(PEREPOCH, "LR")
FP_       = col(PEREPOCH, "Forged_P")
FR_       = col(PEREPOCH, "Forged_R")
FF1       = col(PEREPOCH, "Forged_F1")
FIOU      = col(PEREPOCH, "Forged_IoU")
BGF1      = col(PEREPOCH, "Bg_F1")
GACC      = col(PEREPOCH, "GlobalAcc")
MIOU      = col(PEREPOCH, "MeanIoU")
TRMIN     = col(PEREPOCH, "TrainMin")

# Pixel confusion matrices, exactly as stored in PixelMetrics_improved.mat
# rows = true (Background, Forged); cols = predicted (Background, Forged)
CM_TRANSFER = np.array([[1128511712, 6148140],
                        [5672571,   99161977]], dtype=float)
CM_BASELINE = np.array([[7858722, 843],
                        [434800,   35]], dtype=float)
CM_DEEPLAB  = np.array([[7504934, 354631],
                        [377222,   57613]], dtype=float)
# tuned U-Net (Dataset4, 12 epochs) from logs/04_tuned_model_training_summary.log
CM_TUNED    = np.array([[120684217, 3574122],
                        [3656118,   9807143]], dtype=float)

MODELS = [
    # label,                                     forged P, R, F1, IoU, global acc
    ("U-Net (baseline recipe)\n225 train · 48 test img",
     0.039863, 0.000080, 0.000161, 0.000080, 0.947477),
    ("DeepLabV3+/ResNet-18 (small data)\n225 train · 48 test img",
     0.139755, 0.132494, 0.136027, 0.072977, 0.911765),
    ("U-Net (tuned)\n3,189 train · 797 val img",
     0.732900, 0.728400, 0.730700, 0.575600, 0.947467),
    ("DeepLabV3+/ResNet-18 (proposed)\n33,477 train · 7,173 test img",
     0.941619, 0.945890, 0.943750, 0.893490, 0.990463),
]


# =====================================================================
#  Diagram primitives - boxes joined by real connectors
# =====================================================================

def box(ax, x, y, w, h, text, fc="#eef4fd", ec=S1, fontsize=8.2, weight="normal",
        radius=0.02, tc=INK):
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                       boxstyle=f"round,pad=0.004,rounding_size={radius}",
                       linewidth=1.3, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(p)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=tc, zorder=3, weight=weight, linespacing=1.35)
    return (x, y, w, h)


def diamond(ax, x, y, w, h, text, fc="#fdf1e9", ec=S2, fontsize=7.8):
    pts = [(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)]
    ax.add_patch(Polygon(pts, closed=True, linewidth=1.3,
                         edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=INK, zorder=3, linespacing=1.3)
    return (x, y, w, h)


def anchor(b, side):
    x, y, w, h = b
    return {"t": (x, y + h / 2), "b": (x, y - h / 2),
            "l": (x - w / 2, y), "r": (x + w / 2, y)}[side]


def connect(ax, b1, s1, b2, s2, label=None, color=INK2, style="-",
            rad=0.0, lw=1.2, labpos=0.5, fontsize=7.0, dx=0.0, dy=0.0):
    """Draw a labelled connector between two node anchors."""
    p1 = np.array(anchor(b1, s1), dtype=float)
    p2 = np.array(anchor(b2, s2), dtype=float)
    arrow = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=11,
                            linewidth=lw, color=color, linestyle=style,
                            connectionstyle=f"arc3,rad={rad}",
                            shrinkA=1.5, shrinkB=1.5, zorder=1)
    ax.add_patch(arrow)
    if label:
        m = p1 + (p2 - p1) * labpos
        ax.text(m[0] + dx, m[1] + dy, label, ha="center", va="center",
                fontsize=fontsize, color=INK2,
                bbox=dict(boxstyle="round,pad=0.18", fc=SURFACE, ec="none"),
                zorder=4)


def route(ax, pts, color=INK2, style="-", lw=1.2, label=None, lab_xy=None,
          fontsize=6.9, head=True):
    """Orthogonal (elbowed) connector through an explicit list of waypoints.

    Diagram edges are drawn as polylines rather than arcs so that they never
    cut diagonally across other nodes - the failure mode of curved connectors
    in a dense block diagram.
    """
    pts = np.asarray(pts, dtype=float)
    if len(pts) > 2:
        ax.plot(pts[:-1, 0], pts[:-1, 1], color=color, linestyle=style,
                linewidth=lw, zorder=1, solid_capstyle="round",
                solid_joinstyle="round")
    if head:
        ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>",
                                     mutation_scale=11, linewidth=lw,
                                     color=color, linestyle=style,
                                     shrinkA=0, shrinkB=1.2, zorder=1))
    else:
        ax.plot(pts[-2:, 0], pts[-2:, 1], color=color, linestyle=style,
                linewidth=lw, zorder=1)
    if label:
        lx, ly = lab_xy if lab_xy is not None else pts[len(pts) // 2]
        ax.text(lx, ly, label, ha="center", va="center", fontsize=fontsize,
                color=INK2, zorder=4,
                bbox=dict(boxstyle="round,pad=0.18", fc=SURFACE, ec="none"))


def hstep(b1, b2):
    """Right edge of b1 -> left edge of b2 (same row)."""
    return [anchor(b1, "r"), anchor(b2, "l")]


def vstep(b1, b2):
    """Bottom edge of b1 -> top edge of b2 (same column)."""
    return [anchor(b1, "b"), anchor(b2, "t")]


def elbow_down(b1, b2, side1="b", side2="t"):
    """b1 -> b2 via a single right-angle bend."""
    p1 = np.array(anchor(b1, side1), float)
    p2 = np.array(anchor(b2, side2), float)
    if side1 in ("b", "t"):
        return [p1, (p1[0], p2[1]), p2]
    return [p1, (p2[0], p1[1]), p2]


def blank_axes(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


# =====================================================================
#  Fig. 1 - block diagram of the proposed pipeline
# =====================================================================

def fig_block_diagram():
    """Four-stage serpentine block diagram, orthogonal connectors only."""
    fig, ax = blank_axes((7.9, 5.2))
    ax.set_xlim(-0.075, 1.0)
    W, H = 0.215, 0.105
    XS = [0.125, 0.375, 0.625, 0.875]
    YA, YB, YC, YD = 0.885, 0.655, 0.425, 0.170
    FS = 7.1

    GREY, GREYE = "#f4f4f2", MUTED
    BLUE, BLUEE = "#eef4fd", S1
    ORNG, ORNGE = "#fdf1e9", S2
    GRN,  GRNE  = "#eafaf3", S3

    # --- stage A: data preparation (left to right) ---
    a1 = box(ax, XS[0], YA, W, H, "47,824 tampered frames\n+ binary masks", BLUE, BLUEE, FS)
    a2 = box(ax, XS[1], YA, W, H, "Basename pairing\nimage \u2194 mask", GREY, GREYE, FS)
    a3 = box(ax, XS[2], YA, W, H, "Seeded 70/15/15 split\n33,477 / 7,174 / 7,173", GREY, GREYE, FS)
    a4 = box(ax, XS[3], YA, W, H, "Resize to 352\u00d7480\nImageNet z-score", GREY, GREYE, FS)

    # --- stage B: network (right to left) ---
    b4 = box(ax, XS[3], YB, W, H, "Geometric augmentation\nflip \u00b7 rotate \u00b7 shift \u00b7 scale", GRN, GRNE, FS)
    b3 = box(ax, XS[2], YB, W, H, "ResNet-18 encoder\nImageNet weights", ORNG, ORNGE, FS)
    b2 = box(ax, XS[1], YB, W, H, "ASPP\nrates 1 / 6 / 12 / 18", ORNG, ORNGE, FS)
    b1 = box(ax, XS[0], YB, W, H, "Decoder + skip\nbilinear \u00d74 upsample", ORNG, ORNGE, FS)

    # --- stage C: optimisation (left to right) ---
    c1 = box(ax, XS[0], YC, W, H, "Soft-max scores\n2 channels", ORNG, ORNGE, FS)
    c2 = box(ax, XS[1], YC, W, H, "Dice + cross-entropy\nAdam, lr 1\u00d710\u207b\u2074", GRN, GRNE, FS)
    c3 = box(ax, XS[2], YC, W, H, "Per-epoch checkpoint\nselect best Forged IoU", GRN, GRNE, FS)
    c4 = box(ax, XS[3], YC, W, H, "argmax \u2192 binary\nforged mask", GREY, GREYE, FS)

    # --- stage D: evaluation (right to left) ---
    d4 = box(ax, XS[3], YD, W, H, "Morphological\npost-processing", GREY, GREYE, FS)
    d3 = box(ax, XS[2], YD, W, H, "Full-resolution pixel\nconfusion matrix", GREY, GREYE, FS)
    d2 = box(ax, XS[1], YD, W, H, "Precision / Recall / F1\nIoU \u00b7 sensitivity \u00b7 specificity", BLUE, BLUEE, 6.7)
    d1 = box(ax, XS[0], YD, W, H, "ROC and PR curves\nAUC", BLUE, BLUEE, FS)

    for p, q in ((a1, a2), (a2, a3), (a3, a4)):
        route(ax, hstep(p, q))
    route(ax, vstep(a4, b4), label="train split", lab_xy=(XS[3] + 0.075, (YA + YB) / 2))
    for p, q in ((b4, b3), (b3, b2), (b2, b1)):
        route(ax, [anchor(p, "l"), anchor(q, "r")])
    route(ax, vstep(b1, c1))
    for p, q in ((c1, c2), (c2, c3), (c3, c4)):
        route(ax, hstep(p, q))
    route(ax, vstep(c4, d4), label="best epoch", lab_xy=(XS[3] + 0.072, (YC + YD) / 2))
    for p, q in ((d4, d3), (d3, d2), (d2, d1)):
        route(ax, [anchor(p, "l"), anchor(q, "r")])

    # back-propagation: c2 top -> up the gutter -> b3 bottom
    ygut = (YB + YC) / 2
    route(ax, [anchor(c2, "t"), (XS[1], ygut), (XS[2], ygut), anchor(b3, "b")],
          color=S3, style="--", label="back-propagation",
          lab_xy=((XS[1] + XS[2]) / 2, ygut + 0.028))

    for y, lab in ((YA, "A · Data preparation"), (YB, "B · Forward path"),
                   (YC, "C · Optimisation"), (YD, "D · Evaluation")):
        ax.text(-0.040, y, lab, ha="center", va="center", rotation=90,
                fontsize=7.0, color=MUTED, weight="bold")

    save(fig, "fig01_block_diagram.jpg")


# =====================================================================
#  Fig. 2 - flowchart
# =====================================================================

def fig_flowchart():
    """Single main column, side branches to the right, loop-back down the left
    gutter. Every connector is orthogonal, so no edge crosses a node."""
    fig, ax = blank_axes((7.2, 9.6))
    XC, XR, XL = 0.46, 0.855, 0.055      # main column, right branch, return lane
    W, H = 0.40, 0.056
    WR, HR = 0.235, 0.050
    FS = 7.3

    y = dict(start=0.978, pair=0.912, dpair=0.838, split=0.762,
             net=0.688, init=0.614, dlr=0.538, train=0.462, save=0.388,
             diou=0.312, best=0.236, dstop=0.160, final=0.084, end=0.020)

    n_start = box(ax, XC, y["start"], 0.20, 0.042, "Start", "#eef4fd", S1, 8.0, weight="bold", radius=0.03)
    n_pair  = box(ax, XC, y["pair"], W, H, "Read image and mask folders;\npair files by lower-case basename", fontsize=FS)
    d_pair  = diamond(ax, XC, y["dpair"], 0.38, 0.062, "Any pairs found ?", fontsize=FS)
    n_abort = box(ax, XR, y["dpair"], WR, HR, "Abort with error", "#fdecec", "#d03b3b", FS)
    n_split = box(ax, XC, y["split"], W, H, "rng(42); randperm(N); split 70/15/15;\nbuild datastores (352\u00d7480, mask >127)", fontsize=FS)
    n_net   = box(ax, XC, y["net"], W, H, "Instantiate DeepLabV3+ / ResNet-18;\nrestore ImageNet mean and std", fontsize=FS)
    n_init  = box(ax, XC, y["init"], W, H, "epoch \u2190 1;  lr \u2190 1\u00d710\u207b\u2074;\nbest \u2190 \u2212\u221e;  stale \u2190 0", "#eafaf3", S3, FS)
    d_lr    = diamond(ax, XC, y["dlr"], 0.38, 0.062, "mod(epoch \u2212 1, 5) = 0 ?", fontsize=FS)
    n_lr    = box(ax, XR, y["dlr"], WR, HR, "lr \u2190 0.3 \u00d7 lr", "#eafaf3", S3, FS)
    n_train = box(ax, XC, y["train"], W, H, "Train one epoch: augmentation +\nDice + cross-entropy loss (Adam)", fontsize=FS)
    n_save  = box(ax, XC, y["save"], W, H, "Save net_epoch_kk.mat;\nevaluate 1,000 validation images", fontsize=FS)
    d_iou   = diamond(ax, XC, y["diou"], 0.38, 0.064, "Forged IoU > best ?", fontsize=FS)
    n_stale = box(ax, XR, y["diou"], WR, HR, "stale \u2190 stale + 1", "#fdf1e9", S2, FS)
    n_best  = box(ax, XC, y["best"], W, H, "best \u2190 Forged IoU;  stale \u2190 0;\nsave as netSeg_improved.mat", "#eafaf3", S3, FS)
    d_stop  = diamond(ax, XC, y["dstop"], 0.40, 0.066, "stale \u2265 4   or   epoch = 10 ?", fontsize=FS)
    n_final = box(ax, XC, y["final"], W, H, "Restore best epoch; tune minArea on\nvalidation; evaluate 7,173 test images", "#eef4fd", S1, FS)
    n_end   = box(ax, XC, y["end"], 0.20, 0.042, "Stop", "#eef4fd", S1, 8.0, weight="bold", radius=0.03)

    # main spine
    for p, q in ((n_start, n_pair), (n_pair, d_pair), (n_split, n_net),
                 (n_net, n_init), (n_init, d_lr), (n_train, n_save), (n_save, d_iou),
                 (n_best, d_stop), (n_final, n_end)):
        route(ax, vstep(p, q))

    route(ax, vstep(d_pair, n_split), label="yes", lab_xy=(XC + 0.035, (y["dpair"] - 0.031 + y["split"] + 0.028) / 2))
    route(ax, hstep(d_pair, n_abort), label="no", lab_xy=((XC + 0.19 + XR - 0.117) / 2, y["dpair"] + 0.016))
    route(ax, vstep(d_lr, n_train), label="no", lab_xy=(XC + 0.032, (y["dlr"] - 0.031 + y["train"] + 0.028) / 2))
    route(ax, hstep(d_lr, n_lr), label="yes", lab_xy=((XC + 0.19 + XR - 0.117) / 2, y["dlr"] + 0.016))
    route(ax, vstep(d_iou, n_best), label="yes", lab_xy=(XC + 0.035, (y["diou"] - 0.032 + y["best"] + 0.028) / 2))
    route(ax, hstep(d_iou, n_stale), label="no", lab_xy=((XC + 0.19 + XR - 0.117) / 2, y["diou"] + 0.016))
    route(ax, vstep(d_stop, n_final), label="yes", lab_xy=(XC + 0.035, (y["dstop"] - 0.033 + y["final"] + 0.028) / 2))

    # lr branch rejoins the spine just above the training block
    yj = (y["dlr"] - 0.031 + y["train"] + 0.028) / 2
    route(ax, [anchor(n_lr, "b"), (XR, yj), (XC, yj)], color=S3, head=False)

    # stale branch rejoins just above the stop test
    yk = (y["best"] - 0.028 + y["dstop"] + 0.033) / 2
    route(ax, [anchor(n_stale, "b"), (XR, yk), (XC, yk)], color=S2, head=False)

    # loop-back: stop test -> left gutter -> back into the lr test
    route(ax, [anchor(d_stop, "l"), (XL, y["dstop"]), (XL, y["dlr"]), anchor(d_lr, "l")],
          color=S1, style="--", label="no \u2014 epoch \u2190 epoch + 1",
          lab_xy=(XL + 0.095, (y["dstop"] + y["dlr"]) / 2))

    save(fig, "fig02_flowchart.jpg")


# =====================================================================
#  Fig. 3 - network architecture
# =====================================================================

def fig_architecture():
    """Encoder row, ASPP bus, decoder row - all connectors orthogonal."""
    fig, ax = blank_axes((7.8, 5.6))
    FS = 7.2

    def band(y, text):
        ax.add_patch(Rectangle((0.0, y - 0.021), 1.0, 0.042, facecolor="#f4f4f2",
                               edgecolor="none", zorder=0))
        ax.text(0.012, y, text, ha="left", va="center", fontsize=8.0,
                color=INK, weight="bold", zorder=1)

    # ---------------- encoder ----------------
    band(0.955, "Encoder \u2014 ResNet-18 backbone, ImageNet-pretrained")
    YE = 0.845
    XE = [0.095, 0.265, 0.435, 0.605, 0.775, 0.930]
    enc_spec = [
        ("Input\n352\u00d7480\u00d73", "z-score", "#eef4fd", S1, 0.125),
        ("conv1 7\u00d77 /2\nmaxpool /2", "88\u00d7120", "#fdf1e9", S2, 0.125),
        ("res2x\n64 ch", "88\u00d7120  (/4)", "#fdf1e9", S2, 0.125),
        ("res3x\n128 ch", "44\u00d760  (/8)", "#fdf1e9", S2, 0.125),
        ("res4x\n256 ch", "22\u00d730  (/16)", "#fdf1e9", S2, 0.125),
        ("res5x dilated\n512 ch", "22\u00d730  (/16)", "#fdf1e9", S2, 0.125),
    ]
    enc = []
    for x, (t, sub, fc, ec, w) in zip(XE, enc_spec):
        b = box(ax, x, YE, w, 0.080, t, fc, ec, FS)
        ax.text(x, YE - 0.062, sub, ha="center", va="center", fontsize=6.5, color=MUTED)
        enc.append(b)
    for p, q in zip(enc[:-1], enc[1:]):
        route(ax, hstep(p, q))

    # ---------------- ASPP ----------------
    band(0.700, "Atrous Spatial Pyramid Pooling (output stride 16)")
    YB_IN, YA = 0.640, 0.560
    aspp_x = [0.140, 0.310, 0.480, 0.650, 0.845]
    aspp_lab = ["1\u00d71 conv", "3\u00d73 conv\nrate 6", "3\u00d73 conv\nrate 12",
                "3\u00d73 conv\nrate 18", "image-level\npooling"]
    # distribution bus - dropped clear of the res5x size caption
    XBUS = XE[-1] + 0.048
    route(ax, [(XE[-1], YE - 0.040), (XE[-1], YE - 0.040), (XBUS, YE - 0.040),
               (XBUS, YB_IN)], head=False)
    ax.plot([aspp_x[0], XBUS], [YB_IN, YB_IN], color=INK2, linewidth=1.2, zorder=1)
    aspp = []
    for x, lab in zip(aspp_x, aspp_lab):
        route(ax, [(x, YB_IN), (x, YA + 0.040)])
        aspp.append(box(ax, x, YA, 0.145, 0.080, lab, "#eafaf3", S3, FS))

    # collection bus
    YB_OUT = 0.470
    for b_ in aspp:
        route(ax, [anchor(b_, "b"), (b_[0], YB_OUT)], head=False)
    ax.plot([aspp_x[0], aspp_x[-1]], [YB_OUT, YB_OUT], color=INK2, linewidth=1.2, zorder=1)
    cat = box(ax, 0.480, 0.400, 0.34, 0.058, "concatenate \u2192 1\u00d71 conv, 256 ch",
              "#eafaf3", S3, FS)
    route(ax, [(0.480, YB_OUT), anchor(cat, "t")])

    # ---------------- decoder ----------------
    band(0.300, "Decoder")
    YD = 0.185
    dec = [
        box(ax, 0.135, YD, 0.185, 0.080, "bilinear \u00d74\n\u2192 88\u00d7120", "#eef4fd", S1, FS),
        box(ax, 0.375, YD, 0.195, 0.080, "concatenate low-level\nres2x \u2192 1\u00d71, 48 ch", "#eef4fd", S1, FS),
        box(ax, 0.615, YD, 0.185, 0.080, "3\u00d73 conv \u00d72\n256 ch", "#eef4fd", S1, FS),
        box(ax, 0.865, YD, 0.215, 0.080, "1\u00d71 conv \u2192 2 ch\nbilinear \u00d74 + soft-max", "#eef4fd", S1, FS),
    ]
    route(ax, [anchor(cat, "l"), (0.135, 0.400), anchor(dec[0], "t")])
    for p, q in zip(dec[:-1], dec[1:]):
        route(ax, hstep(p, q))

    # skip connection: res2x -> down into the gap above the ASPP band -> left
    # gutter -> along beneath the decoder row -> decoder concatenation block
    XG, YSKIP = 0.016, YD - 0.098
    route(ax, [anchor(enc[2], "b"), (XE[2], 0.742), (XG, 0.742), (XG, YSKIP),
               (dec[1][0], YSKIP), anchor(dec[1], "b")],
          color=S1, style="--", label="low-level skip connection",
          lab_xy=(0.195, YSKIP + 0.024))

    ax.text(0.5, 0.048, "Output: per-pixel Background / Forged probability map at 352\u00d7480, "
                        "resized to ground-truth resolution before scoring",
            ha="center", fontsize=7.3, color=MUTED, style="italic")
    save(fig, "fig03_architecture.jpg")


# =====================================================================
#  Fig. 4 - class imbalance
# =====================================================================

def fig_class_balance():
    tot = CM_TRANSFER.sum()
    bg = CM_TRANSFER[0].sum() / tot * 100
    fg = CM_TRANSFER[1].sum() / tot * 100

    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    r = ax.bar(["Background\n(authentic)", "Forged"], [bg, fg],
               color=[S1, S2], width=0.5)
    bar_labels(ax, r, fmt="{:.2f}%", dy=1.2, size=8.5)
    ax.set_ylabel("Share of evaluated pixels (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Pixel-class prevalence over the 7,173-image test split\n"
                 "(1,239,494,400 pixels)")
    tidy(ax)
    save(fig, "fig04_class_balance.jpg")


# =====================================================================
#  Fig. 5 - per-epoch training loss and validation accuracy
# =====================================================================

def fig_training_curves():
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))

    ax = axes[0]
    ax.plot(EP, TRAINLOSS, marker="o", markersize=6, color=S1, label="Training loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dice + cross-entropy loss")
    ax.set_title("Training loss per epoch")
    ax.set_xticks(EP)
    ax.axvline(6, color=MUTED, linewidth=0.9, linestyle=":")
    ax.text(6.1, TRAINLOSS.max() * 0.92, "lr 1e-4 \u2192 3e-5",
            fontsize=7, color=MUTED)
    tidy(ax)

    ax = axes[1]
    ax.plot(EP, GACC * 100, marker="o", markersize=6, color=S1,
            label="Global pixel accuracy")
    ax.plot(EP, FF1 * 100, marker="s", markersize=6, color=S2, linestyle="--",
            label="Forged-class F1")
    ax.plot(EP, FIOU * 100, marker="^", markersize=6, color=S3, linestyle="-.",
            label="Forged-class IoU")
    for x, y in zip(EP[[0, -1]], (GACC * 100)[[0, -1]]):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=7, color=INK2)
    for x, y in zip(EP[[0, -1]], (FIOU * 100)[[0, -1]]):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                    xytext=(0, -12), ha="center", fontsize=7, color=INK2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation metric (%)")
    ax.set_title("Validation accuracy, F1 and IoU per epoch")
    ax.set_xticks(EP)
    ax.set_ylim(65, 102)
    ax.legend(loc="lower right", fontsize=7.5)
    tidy(ax)

    save(fig, "fig05_training_curves.jpg")


# =====================================================================
#  Fig. 6 - grouped bar chart of per-epoch validation metrics
# =====================================================================

def fig_epoch_bars():
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    x = np.arange(len(EP), dtype=float)
    w = 0.20
    series = [("Precision", FP_, S1), ("Recall", FR_, S2),
              ("F1", FF1, S3), ("IoU", FIOU, S4)]
    for k, (lab, vals, c) in enumerate(series):
        r = ax.bar(x + (k - 1.5) * w, vals, w * 0.92, label=lab, color=c)
        if k in (0, 3):
            bar_labels(ax, r, fmt="{:.2f}", dy=0.012, size=5.6)
    ax.set_xticks(x)
    ax.set_xticklabels([str(e) for e in EP])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Forged-class metric")
    # full 0-1 scale: truncating a bar axis exaggerates the epoch-to-epoch deltas
    ax.set_ylim(0, 1.05)
    ax.set_title("Forged-class validation metrics after every training epoch")
    ax.legend(ncol=4, loc="lower center", fontsize=8,
              bbox_to_anchor=(0.5, -0.30))
    tidy(ax)
    save(fig, "fig06_epoch_bars.jpg")


# =====================================================================
#  Fig. 7 - training vs validation loss / accuracy comparison
# =====================================================================

def read_kv(path):
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if "=" in line:
                k, v = line.split("=", 1)
                try:
                    out[k.strip()] = float(v.strip())
                except ValueError:
                    out[k.strip()] = v.strip()
    return out


TRAINACC = read_kv(os.path.join(SCRATCH, "train_acc.txt"))


def fig_train_val_bars():
    """Only measured quantities are plotted.

    Panel (a) uses the per-epoch training loss recorded during the run and the
    validation error (1 - global accuracy) measured after each epoch; the run
    monitored validation with accuracy rather than loss, so the error rate is
    the like-for-like validation quantity.

    Panel (b) uses accuracy measured post-hoc with the selected network on a
    fixed random subset of its own training split and of its validation split
    (train_acc.m), alongside the tuned U-Net figures recorded in
    logs/04_tuned_model_training_summary.log.
    """
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4),
                             gridspec_kw={"wspace": 0.30})

    ax = axes[0]
    x = np.arange(len(EP), dtype=float)
    r1 = ax.bar(x - 0.19, TRAINLOSS, 0.35, label="Training loss", color=S1)
    r2 = ax.bar(x + 0.19, 1 - GACC, 0.35, label="Validation error (1 − accuracy)", color=S2)
    bar_labels(ax, r1, fmt="{:.3f}", dy=0.004, size=5.6)
    bar_labels(ax, r2, fmt="{:.3f}", dy=0.004, size=5.6)
    ax.set_xticks(x)
    ax.set_xticklabels([str(e) for e in EP])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss / error")
    ax.set_ylim(0, float(max(TRAINLOSS.max(), (1 - GACC).max())) * 1.42)
    ax.set_title("(a)  Training loss and validation error", fontsize=9)
    ax.legend(fontsize=7.2, loc="upper right")
    tidy(ax)

    ax = axes[1]
    if not TRAINACC:
        raise SystemExit("train_acc.txt missing - run train_acc.m first")
    labels = ["Training split\n(800 images)", "Validation split\n(800 images)",
              "Test split\n(7,173 images)"]
    acc = [TRAINACC["train_globalAcc"] * 100.0,
           TRAINACC["val_globalAcc"] * 100.0,
           0.990463 * 100.0]
    f1 = [TRAINACC["train_forged_F1"] * 100.0,
          TRAINACC["val_forged_F1"] * 100.0,
          0.943750 * 100.0]
    x = np.arange(3, dtype=float)
    r1 = ax.bar(x - 0.17, acc, 0.32, label="Global pixel accuracy", color=S1)
    r2 = ax.bar(x + 0.17, f1, 0.32, label="Forged-class F1", color=S2)
    bar_labels(ax, r1, fmt="{:.2f}", dy=0.35, size=7.0)
    bar_labels(ax, r2, fmt="{:.2f}", dy=0.35, size=7.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.4)
    ax.set_ylabel("Metric (%)")
    ax.set_ylim(88, 102)
    ax.set_title("(b)  Proposed model on all three splits\n"
                 "train − test accuracy gap: %+.2f pp" % (acc[0] - acc[2]),
                 fontsize=9)
    ax.legend(fontsize=7.2, loc="lower center", ncol=2,
              bbox_to_anchor=(0.5, -0.30))
    tidy(ax)

    save(fig, "fig07_train_val_bars.jpg")


# =====================================================================
#  Fig. 8 - per-class precision / recall / F1 / IoU of the proposed model
# =====================================================================

def fig_per_class():
    bg = (0.994999, 0.994582, 0.994790, 0.989634)
    fg = (0.941619, 0.945890, 0.943750, 0.893490)
    names = ["Precision", "Recall", "F1-score", "IoU"]
    x = np.arange(4, dtype=float)

    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    r1 = ax.bar(x - 0.19, bg, 0.35, label="Background (authentic)", color=S1)
    r2 = ax.bar(x + 0.19, fg, 0.35, label="Forged", color=S2)
    bar_labels(ax, r1, fmt="{:.4f}", dy=0.010, size=7.0)
    bar_labels(ax, r2, fmt="{:.4f}", dy=0.010, size=7.0)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Pixel-level metric")
    ax.set_ylim(0, 1.12)
    ax.set_title("Per-class pixel metrics of the proposed model\n"
                 "(held-out test split, 7,173 images)")
    ax.legend(fontsize=8, ncol=2, loc="lower center")
    tidy(ax)
    save(fig, "fig08_per_class_metrics.jpg")


# =====================================================================
#  Fig. 9 / 10 - ROC and Precision-Recall curves
# =====================================================================

def _roc_rows():
    path = os.path.join(SCRATCH, "roc_transfer_curve.csv")
    if not os.path.isfile(path):
        return None
    return read_csv_rows(path)


def _roc_summary():
    path = os.path.join(SCRATCH, "roc_transfer_summary.txt")
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if "=" in line:
                k, v = line.split("=", 1)
                try:
                    out[k.strip()] = float(v.strip())
                except ValueError:
                    out[k.strip()] = v.strip()
    return out


def fig_roc_pr():
    rows = _roc_rows()
    if rows is None:
        print("!! ROC data not available yet - skipping fig09/fig10")
        return
    s = _roc_summary()
    fpr = col(rows, "FPR")
    tpr = col(rows, "TPR")
    pre = col(rows, "Precision")
    rec = col(rows, "Recall")
    auc = s.get("AUC", float("nan"))
    aucpr = s.get("AUC_PR", float("nan"))

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot(fpr, tpr, color=S1, label=f"Proposed model (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle=":", linewidth=1.2,
            label="Chance (AUC = 0.5000)")
    ax.set_xlabel("False positive rate  (1 \u2212 specificity)")
    ax.set_ylabel("True positive rate  (sensitivity)")
    ax.set_title("ROC curve \u2014 Forged class")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.005)
    ax.legend(loc="lower right", fontsize=8)
    tidy(ax)
    ax.xaxis.grid(True)
    save(fig, "fig09_roc_curve.jpg")

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    prev = s.get("prevalence", CM_TRANSFER[1].sum() / CM_TRANSFER.sum())
    ax.plot(rec, pre, color=S2, label=f"Proposed model (AUC-PR = {aucpr:.4f})")
    ax.axhline(prev, color=MUTED, linestyle=":", linewidth=1.2,
               label=f"Chance (prevalence = {prev:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision\u2013Recall curve \u2014 Forged class")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.005)
    ax.legend(loc="lower left", fontsize=8)
    tidy(ax)
    ax.xaxis.grid(True)
    save(fig, "fig10_pr_curve.jpg")


# =====================================================================
#  Fig. 11 - confusion matrix
# =====================================================================

def fig_confusion():
    cm = CM_TRANSFER
    rown = cm.sum(axis=1, keepdims=True)
    norm = cm / rown

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.5))
    classes = ["Background", "Forged"]

    for ax, mat, title, fmt in (
            (axes[0], cm, "Pixel counts", None),
            (axes[1], norm * 100, "Row-normalised (%)", "{:.3f}%")):
        im = ax.imshow(norm, cmap=BLUE_RAMP, vmin=0, vmax=1)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Predicted\nBackground", "Predicted\nForged"])
        ax.set_yticklabels(["True\nBackground", "True\nForged"])
        ax.set_title(title)
        for i in range(2):
            for j in range(2):
                v = mat[i, j]
                txt = f"{v:,.0f}" if fmt is None else fmt.format(v)
                tag = [["TN", "FP"], ["FN", "TP"]][i][j]
                ax.text(j, i - 0.12, tag, ha="center", va="center",
                        fontsize=8, weight="bold",
                        color="#ffffff" if norm[i, j] > 0.55 else INK)
                ax.text(j, i + 0.14, txt, ha="center", va="center",
                        fontsize=8.5,
                        color="#ffffff" if norm[i, j] > 0.55 else INK)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)

    fig.suptitle("Pixel-level confusion matrix of the proposed model "
                 "(7,173 test images, 1,239,494,400 pixels)",
                 fontsize=9.5, weight="bold", color=INK, y=1.02)
    save(fig, "fig11_confusion_matrix.jpg")


# =====================================================================
#  Fig. 12 - model comparison
# =====================================================================

def fig_model_comparison():
    labels = [m[0] for m in MODELS]
    P  = np.array([m[1] for m in MODELS])
    R  = np.array([m[2] for m in MODELS])
    F1 = np.array([m[3] for m in MODELS])
    IU = np.array([m[4] for m in MODELS])

    x = np.arange(len(labels), dtype=float)
    w = 0.20
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    for k, (lab, vals, c) in enumerate([("Precision", P, S1), ("Recall", R, S2),
                                        ("F1-score", F1, S3), ("IoU", IU, S4)]):
        r = ax.bar(x + (k - 1.5) * w, vals, w * 0.92, label=lab, color=c)
        bar_labels(ax, r, fmt="{:.3f}", dy=0.012, size=5.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.6)
    ax.set_ylabel("Forged-class pixel metric")
    ax.set_ylim(0, 1.12)
    ax.set_title("Ablation across the four trained configurations. The tuned U-Net has "
                 "no held-out test split;\nits figures are measured on the validation "
                 "split it was monitored against.", fontsize=8.6)
    ax.legend(ncol=4, fontsize=8, loc="upper left")
    tidy(ax)
    save(fig, "fig12_model_comparison.jpg")


# =====================================================================
#  Fig. 13 - qualitative results
# =====================================================================

def fig_qualitative():
    import matplotlib.image as mpimg
    ovdir = os.path.join(ROOT, "Improved_Segmentation_Results_transfer", "OverlaySamples")
    bindir = os.path.join(ROOT, "Improved_Segmentation_Results_transfer", "BinarySamples")
    names = sorted(os.listdir(bindir))[:5]
    bases = [n.replace("Binary_", "").replace(".png", "") for n in names]

    fig, axes = plt.subplots(2, len(bases), figsize=(7.6, 3.4))
    for k, b in enumerate(bases):
        for row, (d, pref, lab) in enumerate((
                (ovdir, "Overlay_", "Predicted region\nmasked on image"),
                (bindir, "Binary_", "Predicted binary\nforgery mask"))):
            p = os.path.join(d, f"{pref}{b}.png")
            ax = axes[row, k]
            ax.imshow(mpimg.imread(p), cmap="gray")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(AXIS)
            if row == 0:
                ax.set_title(b, fontsize=7.5, color=INK2, weight="normal")
            if k == 0:
                ax.set_ylabel(lab, fontsize=7, color=INK2)
    fig.suptitle("Qualitative localisation results on held-out test images",
                 fontsize=9.5, weight="bold", color=INK, y=1.00)
    fig.tight_layout()
    save(fig, "fig13_qualitative.jpg")


# =====================================================================
#  Fig. 14 - comparison against the literature
# =====================================================================

# Each entry: (label, value, metric kind, dataset). The metric kind is printed on
# the axis because these are NOT the same quantity - three are pixel-level
# localisation F1 and two are image-level classification accuracy.
LIT_COMPARE = [
    ("Wu et al.\n(2025)\nAGU\u00b2-Net",              0.5560, "pixel F1", "CASIA"),
    ("Zeng et al.\n(2022)\nAttDAU-Net",           0.7736, "pixel F1", "CASIA1"),
    ("Proposed\n(this work)\nDeepLabV3+/R-18",    0.9438, "pixel F1", "47,824-frame corpus"),
    ("Pourkashani et al.\n(2021)\nCNN + k-means", 0.9698, "image F1", "MICC-F2000"),
    ("Kumar et al.\n(2022)\nVI-NET",              0.9800, "image F1", "CoMoFoD"),
    ("Qazi et al.\n(2022)\nResNet50v2",           0.9930, "image acc.", "CASIA v2"),
]


def fig_literature_comparison():
    labels = [r[0] for r in LIT_COMPARE]
    vals = np.array([r[1] for r in LIT_COMPARE])
    kinds = [r[2] for r in LIT_COMPARE]
    # colour by what the number actually measures, not by rank
    cmap = {"pixel F1": S1, "image F1": S3, "image acc.": S4}
    colors = [cmap[k] for k in kinds]
    edges = ["none" if "Proposed" not in l else INK for l in labels]

    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    r = ax.bar(labels, vals, 0.55, color=colors, edgecolor=edges, linewidth=1.6)
    bar_labels(ax, r, fmt="{:.4f}", dy=0.012, size=7.5)
    for k, (rect, kind) in enumerate(zip(r, kinds)):
        ax.text(rect.get_x() + rect.get_width() / 2, 0.028, kind, ha="center",
                va="bottom", fontsize=6.6, color="#ffffff", rotation=0)
    ax.set_ylabel("Reported score")
    ax.set_ylim(0, 1.12)
    ax.set_title("Proposed model against five closely related published methods\n"
                 "Bars measure different quantities (pixel-level localisation vs "
                 "image-level classification) on different test sets")
    tidy(ax)
    ax.tick_params(axis="x", labelsize=6.9)
    save(fig, "fig14_literature_comparison.jpg")


# =====================================================================
#  Fig. 15 - sensitivity / specificity summary
# =====================================================================

def fig_sensitivity_specificity():
    cm = CM_TRANSFER
    TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    sens = TP / (TP + FN)
    spec = TN / (TN + FP)
    ppv  = TP / (TP + FP)
    npv  = TN / (TN + FN)
    acc  = (TP + TN) / cm.sum()
    mcc  = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    bacc = 0.5 * (sens + spec)

    names = ["Sensitivity\n(TPR)", "Specificity\n(TNR)", "Precision\n(PPV)",
             "NPV", "Balanced\naccuracy", "Global\naccuracy", "MCC"]
    vals = [sens, spec, ppv, npv, bacc, acc, mcc]

    fig, ax = plt.subplots(figsize=(7.0, 3.1))
    r = ax.bar(names, vals, 0.55, color=[S1, S1, S2, S2, S3, S3, S4])
    bar_labels(ax, r, fmt="{:.4f}", dy=0.010, size=7.4)
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("Value")
    ax.set_title("Confusion-matrix-derived rates for the proposed model")
    tidy(ax)
    ax.tick_params(axis="x", labelsize=7.4)
    save(fig, "fig15_rates.jpg")

    print("\nDerived rates: sens %.6f  spec %.6f  ppv %.6f  npv %.6f  "
          "bacc %.6f  acc %.6f  mcc %.6f"
          % (sens, spec, ppv, npv, bacc, acc, mcc))
    print("FPR %.6f  FNR %.6f" % (FP / (FP + TN), FN / (FN + TP)))


# =====================================================================

if __name__ == "__main__":
    fig_block_diagram()
    fig_flowchart()
    fig_architecture()
    fig_class_balance()
    fig_training_curves()
    fig_epoch_bars()
    fig_train_val_bars()
    fig_per_class()
    fig_confusion()
    fig_model_comparison()
    fig_qualitative()
    fig_literature_comparison()
    fig_sensitivity_specificity()
    fig_roc_pr()
    print("\nAll figures written to", FIG)
