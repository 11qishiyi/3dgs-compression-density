"""通用 r4res 构建：原图 ÷4 → clean(q98) 与 q20 hybrid（test 划分保留原图）。

用法: python build_scene.py <scene>
产出: data/mip360/<scene>_r4res, data/mip360/<scene>_r4res_q20_hyb
"""
import os
import shutil
import sys

from PIL import Image

SCENE = sys.argv[1]
SRC = os.path.join(os.path.dirname(__file__), "data", "mip360", SCENE)
DST_ROOT = os.path.join(os.path.dirname(__file__), "data", "mip360")
Q_CLEAN, Q_LOW = 98, 20

names = sorted(f for f in os.listdir(os.path.join(SRC, "images"))
               if f.lower().endswith((".jpg", ".jpeg", ".png")))
probe = Image.open(os.path.join(SRC, "images", names[0]))
W, H = probe.width // 4, probe.height // 4
print(f"scene={SCENE} images={len(names)} original={probe.width}x{probe.height} -> r4res={W}x{H}")


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
        im = Image.open(os.path.join(SRC, "images", f)).convert("RGB").resize((W, H), Image.LANCZOS)
        q = Q_CLEAN if (hybrid and i % 8 == 0) else quality
        im.save(os.path.join(img_dir, f), "JPEG", quality=q,
                subsampling=0 if q == Q_CLEAN else 2, optimize=False)
    tot = sum(os.path.getsize(os.path.join(img_dir, x)) for x in os.listdir(img_dir))
    print(f"{tag}: {len(names)} imgs, {tot/1e6:.1f}MB, q={quality}, hybrid={hybrid}")


make(f"{SCENE}_r4res", Q_CLEAN)
make(f"{SCENE}_r4res_q20_hyb", Q_LOW, hybrid=True)
print("done")
