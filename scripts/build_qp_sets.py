"""构建 q50 / q80 的 r4res 数据集（含 hybrid：test 换回干净图）"""
import os
import shutil

from PIL import Image

SRC = r"D:\3dgs\proj-D6\cluster\data\mip360\bonsai"
DST_ROOT = r"D:\3dgs\proj-D6\cluster\data\mip360"
W, H = 779, 519
Q_CLEAN = 98

names = sorted(f for f in os.listdir(os.path.join(SRC, "images"))
               if f.lower().endswith((".jpg", ".jpeg", ".png")))


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
    print(f"{tag}: {len(names)} imgs {tot/1e6:.1f}MB q={quality} hybrid={hybrid}")


for q in (50, 80):
    make(f"bonsai_r4res_q{q}", q)
    make(f"bonsai_r4res_q{q}_hyb", q, hybrid=True)
print("done")
