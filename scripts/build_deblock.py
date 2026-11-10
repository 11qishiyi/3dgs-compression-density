"""构建去块基线数据集：降采样 -> JPEG q20 -> 解码 -> NLM 去噪 -> 保存。

hybrid 口径：test 划分（idx%8==0）用干净图，其余用去噪图。
"""
import io
import os
import shutil

import cv2
import numpy as np
from PIL import Image

SRC = r"D:\3dgs\proj-D6\cluster\data\mip360\bonsai"
DST = r"D:\3dgs\proj-D6\cluster\data\mip360\bonsai_r4res_deblock_hyb"
W, H = 779, 519

names = sorted(f for f in os.listdir(os.path.join(SRC, "images"))
               if f.lower().endswith((".jpg", ".jpeg", ".png")))
img_dir = os.path.join(DST, "images")
os.makedirs(img_dir, exist_ok=True)

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
    if i % 8 == 0:                      # test：干净
        im.save(os.path.join(img_dir, f), "JPEG", quality=98, subsampling=0)
        continue
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=20, subsampling=2)
    buf.seek(0)
    dec = np.array(Image.open(buf).convert("RGB"))
    den = cv2.fastNlMeansDenoisingColored(dec, None, 6, 6, 7, 21)
    Image.fromarray(den).save(os.path.join(img_dir, f), "JPEG", quality=95, subsampling=0)

tot = sum(os.path.getsize(os.path.join(img_dir, x)) for x in os.listdir(img_dir))
print(f"deblock hybrid: {len(names)} imgs, {tot/1e6:.1f}MB -> {DST}")
