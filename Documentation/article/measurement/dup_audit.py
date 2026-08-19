"""
dup_audit.py - quantify duplication between the training and test splits.

The article's headline numbers come from a random frame-level 70/15/15 split of a
corpus built from video frames. If near-duplicate frames straddle the split, the
test score is optimistic. This measures that directly:

  * exact duplicate ground-truth masks (SHA-1 of the binarised mask)
  * near-duplicate images (8x8 average hash, Hamming distance <= 4)

and reports how much of the test split has a duplicate partner in the training
split under each criterion.
"""
import os
import hashlib
import numpy as np
from PIL import Image

IMG = r"F:\Current_Work\Semantic Segmentation Using FCN-AlexNet1\Dataset\Images"
LBL = r"F:\Current_Work\Semantic Segmentation Using FCN-AlexNet1\Dataset\Lables"
OUT = os.path.dirname(os.path.abspath(__file__))


def list_files(d, exts):
    out = []
    for e in exts:
        out += [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(e)]
    return out


def strip(p):
    return os.path.splitext(os.path.basename(p))[0].lower()


# Reproduce the MATLAB pairing order exactly: extensions in the listed order,
# dir() returns names sorted by the OS, and ismember preserves the image order.
imgs = list_files(IMG, ['.jpg', '.jpeg', '.png', '.tif', '.tiff'])
masks = list_files(LBL, ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'])
mask_by_base = {strip(m): m for m in masks}
pairs = [(i, mask_by_base[strip(i)]) for i in imgs if strip(i) in mask_by_base]
print("paired", len(pairs))

# The split cannot be reproduced from Python (MATLAB's Mersenne stream differs),
# so the split membership is read back from the saved index vectors, exported by
# export_split.m. If that export is absent, fall back to reporting corpus-wide
# duplication only.
split_file = os.path.join(OUT, "split_idx.csv")
tr = te = None
if os.path.isfile(split_file):
    import csv
    rows = list(csv.DictReader(open(split_file)))

    def col(name):
        out = set()
        for r in rows:
            v = (r.get(name) or "").strip()
            if v and v.lower() != "nan":
                out.add(int(float(v)))
        return out

    tr = col("trIdx")
    te = col("teIdx")
    print("split loaded: train %d, test %d" % (len(tr), len(te)))
else:
    print("!! split_idx.csv absent - reporting corpus-wide duplication only")

mask_hash = []
img_hash = []
for k, (ip, mp) in enumerate(pairs, start=1):
    m = Image.open(mp).convert("L")
    a = (np.asarray(m) > 127)
    mask_hash.append(hashlib.sha1(np.packbits(a).tobytes()).hexdigest())

    g = np.asarray(Image.open(ip).convert("L").resize((8, 8), Image.BILINEAR),
                   dtype=float)
    bits = (g > g.mean()).flatten()
    img_hash.append(int("".join("1" if b else "0" for b in bits), 2))

    if k % 4000 == 0:
        print("  hashed %d/%d" % (k, len(pairs)))

mask_hash = np.array(mask_hash)
img_hash = np.array(img_hash, dtype=np.uint64)

# ---- corpus-wide exact mask duplication ----
uniq, counts = np.unique(mask_hash, return_counts=True)
print("\nexact-duplicate masks: %d distinct masks over %d frames"
      % (len(uniq), len(mask_hash)))
print("frames whose mask is shared with at least one other frame: %d (%.2f %%)"
      % (int(counts[counts > 1].sum()),
         100.0 * counts[counts > 1].sum() / len(mask_hash)))

lines = ["distinct_masks = %d" % len(uniq),
         "total_frames = %d" % len(mask_hash),
         "frames_with_shared_mask = %d" % int(counts[counts > 1].sum())]

if tr is not None:
    tr_i = np.array(sorted(tr)) - 1
    te_i = np.array(sorted(te)) - 1
    tr_masks = set(mask_hash[tr_i])
    hit = np.array([h in tr_masks for h in mask_hash[te_i]])
    print("\ntest frames whose EXACT mask also occurs in the training split: "
          "%d of %d (%.2f %%)" % (hit.sum(), len(te_i), 100.0 * hit.mean()))
    lines.append("test_frames_mask_in_train = %d" % int(hit.sum()))
    lines.append("test_frames_total = %d" % len(te_i))
    lines.append("test_frames_mask_in_train_pct = %.4f" % (100.0 * hit.mean()))

    # near-duplicate images by average hash
    trh = img_hash[tr_i]
    teh = img_hash[te_i]
    near = 0
    CH = 512
    for s in range(0, len(teh), CH):
        blk = teh[s:s + CH][:, None]
        d = np.bitwise_xor(blk, trh[None, :])
        # popcount
        c = np.zeros_like(d)
        v = d.copy()
        while v.any():
            c += (v & np.uint64(1))
            v >>= np.uint64(1)
        near += int((c.min(axis=1) <= 4).sum())
    print("test frames with a NEAR-duplicate image (aHash Hamming <= 4) in the "
          "training split: %d of %d (%.2f %%)" % (near, len(teh),
                                                  100.0 * near / len(teh)))
    lines.append("test_frames_near_dup_image_in_train = %d" % near)
    lines.append("test_frames_near_dup_pct = %.4f" % (100.0 * near / len(teh)))

open(os.path.join(OUT, "dup_audit.txt"), "w").write("\n".join(lines) + "\n")
print("\nwrote dup_audit.txt")
