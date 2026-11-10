"""构建某场景的 q20 4:4:4 候选档（不同 QP），并报有效码率，用于 iso-bpp 对照。"""
import os
import shutil
import sys

from PIL import Image

SCENE, = (sys.argv[1],)
QPS = [int(x) for x in sys.argv[2:]] or [24, 26, 28]
ROOT = r"D:\3dgs\proj-D6\cluster\data\mip360"
SRC = os.path.join(ROOT, SCENE)
W, H = 779, 519
TARGET = 0.441  # q20 4:2:0 的 bpp

names = sorted(f for f in os.listdir(os.path.join(SRC, "images"))
               if f.lower().endswith((".jpg", ".jpeg", ".png")))
train = [f for i, f in enumerate(names) if i % 8 != 0]
print(f"{SCENE}: {len(names)} imgs, train={len(train)}, target bpp={TARGET}")

for qp in QPS:
    tag = f"{SCENE}_r4res_q444_q{qp}_hyb"
    dst = os.path.join(ROOT, tag)
    os.makedirs(os.path.join(dst, "images"), exist_ok=True)
    for sub in ("sparse", "poses_bounds.npy"):
        s, d = os.path.join(SRC, sub), os.path.join(dst, sub)
        if os.path.isdir(s):
            if os.path.exists(d):
                shutil.rmtree(d)
            shutil.copytree(s, d)
        elif os.path.isfile(s):
            shutil.copy2(s, d)
    for i, f in enumerate(names):
        im = Image.open(os.path.join(SRC, "images", f)).convert("RGB").resize((W, H), Image.LANCZOS)
        q = 98 if i % 8 == 0 else qp
        im.save(os.path.join(dst, "images", f), "JPEG", quality=q, subsampling=0, optimize=False)
    tot = sum(os.path.getsize(os.path.join(dst, "images", f)) for f in train)
    bpp = tot * 8 / (W * H * len(train))
    print(f"  q{qp}: bpp={bpp:.3f} (Δ vs target {bpp-TARGET:+.3f}), size={tot/1e6:.1f}MB")
