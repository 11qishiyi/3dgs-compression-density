"""构建机制分解数据集：blur（只保留高频损失）与 noise（只保留块噪声）。

匹配准则（相对 q20）：
- blur：搜索高斯模糊 sigma，使 Laplacian 绝对值均值（高频能量）最接近 q20；
- noise：块相关噪声的像素标准差 = 实测 (q20 - clean) 的 RMS。

两者的 test 划分（idx%8==0）都用干净图（hybrid 口径）。
"""
import io
import os
import shutil

import cv2
import numpy as np
from PIL import Image

ROOT = r"D:\3dgs\proj-D6\cluster\data\mip360"
CLEAN = os.path.join(ROOT, "bonsai_r4res")
Q20 = os.path.join(ROOT, "bonsai_r4res_q20")
W, H = 779, 519

names = sorted(f for f in os.listdir(os.path.join(CLEAN, "images"))
               if f.lower().endswith((".jpg", ".jpeg", ".png")))


def load(d, f):
    return np.asarray(Image.open(os.path.join(d, "images", f)).convert("RGB"), dtype=np.float32)


def hf(a):
    g = cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return float(np.abs(cv2.Laplacian(g, cv2.CV_32F)).mean())


sample = names[::60][:5]
hf_clean = np.mean([hf(load(CLEAN, f)) for f in sample])
hf_q20 = np.mean([hf(load(Q20, f)) for f in sample])
print(f"HF: clean={hf_clean:.3f}  q20={hf_q20:.3f}")

best = None
for s in np.arange(0.4, 3.01, 0.1):
    vals = []
    for f in sample:
        a = load(CLEAN, f)
        b = cv2.GaussianBlur(a, (0, 0), float(s))
        vals.append(hf(b))
    d = abs(np.mean(vals) - hf_q20)
    if best is None or d < best[1]:
        best = (float(s), d)
sigma_blur = best[0]
print(f"blur sigma={sigma_blur:.2f} (HF gap={best[1]:.3f})")

res = [load(Q20, f) - load(CLEAN, f) for f in sample]
sigma_noise = float(np.sqrt(np.mean([np.mean(r ** 2) for r in res])))
print(f"noise sigma={sigma_noise:.3f} (q20-clean RMS)")

D8 = None


def block_noise(shape, sigma):
    nb_h, nb_w = shape[0] // 8, shape[1] // 8
    coef = np.random.randn(nb_h, nb_w, 8, 8) * (sigma * (shape[0] // 8) * 0)
    # 简化：直接用 8x8 块均值滤波后的白噪声
    z = np.random.randn(*shape).astype(np.float32)
    k = np.ones((8, 8), np.float32) / 64.0
    z = cv2.filter2D(z, -1, k)
    s = z.std() + 1e-9
    return z * (sigma / s)


def build(tag, kind):
    out = os.path.join(ROOT, tag)
    img_dir = os.path.join(out, "images")
    os.makedirs(img_dir, exist_ok=True)
    for sub in ("sparse", "poses_bounds.npy"):
        s, d = os.path.join(CLEAN, sub), os.path.join(out, sub)
        if os.path.isdir(s):
            if os.path.exists(d):
                shutil.rmtree(d)
            shutil.copytree(s, d)
        elif os.path.isfile(s):
            shutil.copy2(s, d)

    for i, f in enumerate(names):
        a = load(CLEAN, f)
        if i % 8 == 0:                             # test：干净
            img = a
        elif kind == "blur":
            img = cv2.GaussianBlur(a, (0, 0), sigma_blur)
        else:
            img = np.clip(a + block_noise((H, W, 3), sigma_noise), 0, 255)
        Image.fromarray(img.astype(np.uint8)).save(
            os.path.join(img_dir, f), "JPEG", quality=98, subsampling=0)

    tot = sum(os.path.getsize(os.path.join(img_dir, x)) for x in os.listdir(img_dir))
    print(f"{tag}: {len(names)} imgs {tot/1e6:.1f}MB")


build("bonsai_r4res_blur_hyb", "blur")
build("bonsai_r4res_noise_hyb", "noise")
print("done")
