"""
密度错配分析（对应监督者第 2 条）
================================
对训练好的高斯模型，做三类统计：
 A. 按局部纹理能量分桶（平坦/中/纹理）：高斯数、平均不透明度、平均尺度
 B. 高斯中心到 JPEG 8x8 块边界的距离直方图（全分辨率像素坐标系）
 C. 浮渣/针状比例：各向异性(scale_max/scale_min) 与低不透明度

用法: python3 analyze_density.py <model_dir> <image_dir> [n_cams]
"""
import json
import os
import sys

import numpy as np
from PIL import Image

model_dir = sys.argv[1]
img_dir = sys.argv[2]
n_cams = int(sys.argv[3]) if len(sys.argv) > 3 else 6

# ---------- 读 ply ----------
ply = os.path.join(model_dir, "point_cloud", "iteration_%s" % (
    sorted(os.listdir(os.path.join(model_dir, "point_cloud")))[-1].split("_")[-1]),
    "point_cloud.ply")

props = []
with open(ply, "rb") as f:
    n = 0
    dtype = {}
    while True:
        line = f.readline().decode("ascii", errors="replace").strip()
        if line.startswith("element vertex"):
            n = int(line.split()[-1])
        if line.startswith("property"):
            parts = line.split()
            dtype[parts[-1]] = parts[1]
        if line == "end_header":
            break
    order = list(dtype.keys())
    types = {"float": "f4", "double": "f8", "uchar": "u1", "int": "i4"}
    dt = np.dtype([(k, types.get(dtype[k], "f4")) for k in order])
    data = np.frombuffer(f.read(n * dt.itemsize), dtype=dt, count=n)

xyz = np.stack([data["x"], data["y"], data["z"]], 1).astype(np.float64)
opacity = 1.0 / (1.0 + np.exp(-data["opacity"].astype(np.float64)))
scales = np.exp(np.stack([data["scale_0"], data["scale_1"], data["scale_2"]], 1).astype(np.float64))

print(f"gaussians={n}")

# ---------- 读相机 ----------
cams = json.load(open(os.path.join(model_dir, "cameras.json")))
cams = cams[:n_cams]

tex_vals, block_dist = [], []
in_view = np.zeros(len(xyz), dtype=bool)
per_cam_counts = []

for c in cams:
    R = np.array(c["rotation"], dtype=np.float64)          # world2cam rotation
    t = np.array(c["position"], dtype=np.float64)
    fx, fy = float(c["fx"]), float(c["fy"])
    W, H = int(c["width"]), int(c["height"])

    pc = (xyz - t[None]) @ R.T
    z = pc[:, 2]
    front = z > 0.2
    u = fx * pc[:, 0] / np.where(front, z, 1) + W / 2
    v = fy * pc[:, 1] / np.where(front, z, 1) + H / 2
    ok = front & (u >= 0) & (u < W) & (v >= 0) & (v < H)

    img_path = os.path.join(img_dir, c["img_name"])
    if not os.path.exists(img_path):
        continue
    img = np.asarray(Image.open(img_path).convert("L"), dtype=np.float32)
    if img.shape[1] != W or img.shape[0] != H:
        from PIL import Image as I
        img = np.asarray(I.fromarray(img.astype(np.uint8)).resize((W, H)), dtype=np.float32)

    idx = np.where(ok)[0]
    in_view[idx] = True
    per_cam_counts.append(len(idx))

    # 局部纹理能量（8x8 局部标准差）与块边界距离
    for i in idx[:: max(1, len(idx) // 4000)]:
        uu, vv = int(u[i]), int(v[i])
        patch = img[max(0, vv - 4):vv + 4, max(0, uu - 4):uu + 4]
        if patch.size:
            tex_vals.append(float(patch.std()))
            du = min(uu % 8, 8 - uu % 8)
            dv = min(vv % 8, 8 - vv % 8)
            block_dist.append(float(min(du, dv)))

tex_vals = np.array(tex_vals)
block_dist = np.array(block_dist)
vis = np.where(in_view)[0]

q1, q2 = np.quantile(tex_vals, [1 / 3, 2 / 3]) if len(tex_vals) else (0, 0)
buckets = {"flat": tex_vals <= q1, "mid": (tex_vals > q1) & (tex_vals <= q2), "texture": tex_vals > q2}

aniso = scales.max(1) / np.maximum(scales.min(1), 1e-9)
floater = float(np.mean((aniso > 10) & (opacity < 0.1))) if len(opacity) else 0.0
needle = float(np.mean(aniso > 20)) if len(opacity) else 0.0

# 纹理分桶的比例（在可见高斯上抽样得到的比例，按抽样计数归一）
tot = sum(b.sum() for b in buckets.values()) or 1
bucket_share = {k: float(v.sum() / tot) for k, v in buckets.items()}

# 块边界距离直方图（越小=越贴边）
hist = np.histogram(block_dist, bins=[0, 1, 2, 3, 4, 5])[0].tolist() if len(block_dist) else [0] * 5
edge_ratio = float(np.mean(block_dist <= 1)) if len(block_dist) else 0.0

out = dict(model=os.path.basename(model_dir), gaussians=int(n),
           visible=float(len(vis) / n), bucket_share=bucket_share,
           block_hist=hist, edge_ratio=edge_ratio,
           floater_ratio=floater, needle_ratio=needle,
           mean_opacity=float(opacity.mean()),
           mean_scale=float(scales.mean()))
print(json.dumps(out, indent=1))
