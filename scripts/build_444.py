"""构建 bonsai q20 但 4:4:4（无色度下采样）的 hybrid 数据集。

用途：隔离"色度下采样"与"块量化/DCT 量化"两个因素。
  - q20_hyb（现有）      : 4:2:0 + q20 量化
  - q20_444_hyb（本脚本）: 4:4:4 + q20 量化（DCT-only，无下采样/无模糊）
  - blur_hyb（现有）     : 仅高斯模糊，无量化
"""
import os
import shutil

from PIL import Image

SRC = r"D:\3dgs\proj-D6\cluster\data\mip360\bonsai"
DST = r"D:\3dgs\proj-D6\cluster\data\mip360\bonsai_r4res_q20_444_hyb"
W, H = 779, 519
Q_CLEAN, Q_LOW = 98, 20

names = sorted(f for f in os.listdir(os.path.join(SRC, "images"))
               if f.lower().endswith((".jpg", ".jpeg", ".png")))
os.makedirs(os.path.join(DST, "images"), exist_ok=True)
for sub in ("sparse", "poses_bounds.npy"):
    s, d = os.path.join(SRC, sub), os.path.join(DST, sub)
    if os.path.isdir(s):
        if os.path.exists(d):
            shutil.rmtree(d)
        shutil.copytree(s, d)
    elif os.path.isfile(s):
        shutil.copy2(s, d)

for i, f in enumerate(names):
    im = Image.open(os.path.join(SRC, "images", f)).convert("RGB").resize((W, H), Image.LANCZOS)
    q = Q_CLEAN if i % 8 == 0 else Q_LOW
    im.save(os.path.join(DST, "images", f), "JPEG", quality=q, subsampling=0, optimize=False)

tot = sum(os.path.getsize(os.path.join(DST, "images", x)) for x in os.listdir(os.path.join(DST, "images")))
print(f"q20_444_hyb: {len(names)} imgs, {tot/1e6:.1f}MB (4:4:4, hybrid)")
