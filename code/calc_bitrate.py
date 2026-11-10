"""计算两场景 q20 与 clean 的有效码率（bits/pixel，仅训练图，排除 LLFF test）。"""
import os

from PIL import Image

ROOT = r"D:\3dgs\proj-D6\cluster\data\mip360"
CASES = [
    ("bonsai", "bonsai_r4res"),
    ("bonsai", "bonsai_r4res_q20_hyb"),
    ("bicycle", "bicycle_r4res"),
    ("bicycle", "bicycle_r4res_q20_hyb"),
]

for scene, d in CASES:
    img_dir = os.path.join(ROOT, d, "images")
    names = sorted(f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    train = [f for i, f in enumerate(names) if i % 8 != 0]      # hybrid 的 test 划分
    im = Image.open(os.path.join(img_dir, train[0]))
    W, H = im.size
    total_bytes = sum(os.path.getsize(os.path.join(img_dir, f)) for f in train)
    bpp = total_bytes * 8 / (W * H * len(train))
    print(f"{d:26s} {W}x{H} train={len(train):3d} "
          f"avg={total_bytes/len(train)/1024:6.1f}KB  bpp={bpp:.3f}")
