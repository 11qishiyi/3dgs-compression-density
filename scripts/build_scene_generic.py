"""Build clean and q20-hybrid training sets for a COLMAP benchmark scene without
resizing: the released T&T / DeepBlending images are already at training resolution.

Usage: python build_scene_generic.py <scene_dir> <out_root> <tag>

Writes <out_root>/<tag>_clean and <out_root>/<tag>_q20_hyb, keeping the COLMAP
sparse folder and (when present) poses_bounds.npy.
"""
import os
import shutil
import sys

from PIL import Image

SRC, DST_ROOT, TAG = sys.argv[1], sys.argv[2], sys.argv[3]
Q_CLEAN, Q_LOW = 98, 20

names = sorted(f for f in os.listdir(os.path.join(SRC, "images"))
               if f.lower().endswith((".jpg", ".jpeg", ".png")))
probe = Image.open(os.path.join(SRC, "images", names[0]))
print(f"{TAG}: {len(names)} images at {probe.width}x{probe.height} (no resize)")


def make(tag, quality, hybrid=False):
    out = os.path.join(DST_ROOT, tag)
    img_dir = os.path.join(out, "images")
    os.makedirs(img_dir, exist_ok=True)
    for sub in ("sparse", "poses_bounds.npy"):
        s, d = os.path.join(SRC, sub), os.path.join(out, sub)
        if os.path.isdir(s):
            if os.path.exists(d):
                shutil.rmtree(d)
            shutil.copytree(s, d)
        elif os.path.isfile(s):
            shutil.copy2(s, d)
    for i, f in enumerate(names):
        im = Image.open(os.path.join(SRC, "images", f)).convert("RGB")
        q = Q_CLEAN if (hybrid and i % 8 == 0) else quality
        im.save(os.path.join(img_dir, f), "JPEG", quality=q,
                subsampling=0 if q == Q_CLEAN else 2, optimize=False)
    tot = sum(os.path.getsize(os.path.join(img_dir, x)) for x in os.listdir(img_dir))
    print(f"  {tag}: {len(names)} imgs, {tot / 1e6:.1f}MB, q={quality}, hybrid={hybrid}")


make(f"{TAG}_clean", Q_CLEAN)
make(f"{TAG}_q20_hyb", Q_LOW, hybrid=True)
print("done")
