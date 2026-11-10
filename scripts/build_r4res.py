"""
构建 r4res 口径数据集（监督者第 4 条 B）

真实链路：原图 → LANCZOS 降到训练分辨率(779x519) → JPEG 压缩 → 送进 3DGS。
因此训练用 -r 1（像素量等于 r=4 的快路径），块效应才不会被二次降采样抹掉。

产出三个目录：
  bonsai_r4res         : 降采样后 q98（近似无损，"干净"）
  bonsai_r4res_q20     : 降采样后 q20（4:2:0）
  bonsai_r4res_q20_hyb : 同上，但 LLFF test 划分（idx%8==0）换回 clean 图
"""
import os
import shutil
import sys

from PIL import Image

SRC = r"D:\3dgs\proj-D6\cluster\data\mip360\bonsai"
DST_ROOT = r"D:\3dgs\proj-D6\cluster\data\mip360"
W, H = 779, 519
Q_CLEAN, Q_LOW = 98, 20

names = sorted(f for f in os.listdir(os.path.join(SRC, "images"))
               if f.lower().endswith((".jpg", ".jpeg", ".png")))
print(f"source images: {len(names)}")


def make(tag, quality, hybrid=False):
    out = os.path.join(DST_ROOT, tag)
    img_dir = os.path.join(out, "images")
    os.makedirs(img_dir, exist_ok=True)

    for sub in ("sparse", "poses_bounds.npy"):
        s = os.path.join(SRC, sub)
        d = os.path.join(out, sub)
        if os.path.isdir(s):
            if os.path.exists(d):
                shutil.rmtree(d)
            shutil.copytree(s, d)
        elif os.path.isfile(s):
            shutil.copy2(s, d)

    for i, f in enumerate(names):
        im = Image.open(os.path.join(SRC, "images", f)).convert("RGB")
        im = im.resize((W, H), Image.LANCZOS)
        # hybrid: test 划分用 clean 质量，train 用低质量
        q = Q_CLEAN if (hybrid and i % 8 == 0) else quality
        im.save(os.path.join(img_dir, f), "JPEG", quality=q, subsampling=0 if q == Q_CLEAN else 2,
                optimize=False)

    tot = sum(os.path.getsize(os.path.join(img_dir, x)) for x in os.listdir(img_dir))
    print(f"{tag}: {len(names)} imgs, {tot/1e6:.1f}MB, q={quality}, hybrid={hybrid}")


make("bonsai_r4res", Q_CLEAN)
make("bonsai_r4res_q20", Q_LOW)
make("bonsai_r4res_q20_hyb", Q_LOW, hybrid=True)
print("done")
