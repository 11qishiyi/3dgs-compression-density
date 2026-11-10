"""实验 1：空间密度（解析投影）。

对每个模型（clean/q20/q20_444）：
  1) 把基元中心解析投影到训练视角（cameras.json 的 position/rotation/fx/fy）
  2) 16×16 块统计基元数
  3) 解析投影覆盖 Σπr²/(HW)，r 由 Σ2D = J Σ3D Jᵀ 的最大特征值开方 ×3
  4) 与干净训练图的块级梯度/Laplacian 能量对齐，算相关与高/低能量块的丢失率
输出 JSON 到 results_spatial/<tag>.json
"""
import glob
import json
import os
import sys

import numpy as np
from PIL import Image


def read_ply(path):
    props, types = [], []
    with open(path, "rb") as f:
        n = 0
        while True:
            line = f.readline().decode("ascii", errors="replace").strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            if line.startswith("property"):
                p = line.split()
                props.append(p[-1])
                types.append(p[1])
            if line == "end_header":
                break
        dt = np.dtype([(k, {"float": "f4", "double": "f8", "uchar": "u1", "int": "i4"}[t])
                       for k, t in zip(props, types)])
        data = np.frombuffer(f.read(n * dt.itemsize), dtype=dt, count=n)
    xyz = np.stack([data["x"], data["y"], data["z"]], 1).astype(np.float64)
    scales = np.exp(np.stack([data["scale_0"], data["scale_1"], data["scale_2"]], 1).astype(np.float64))
    rot = np.stack([data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"]], 1).astype(np.float64)
    opacity = 1.0 / (1.0 + np.exp(-data["opacity"].astype(np.float64)))
    return xyz, scales, rot, opacity


def quat_to_R(q):
    """向量化：q 为 N×4 (w,x,y,z) → N×3×3。"""
    q = np.atleast_2d(q)
    n = np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
    w, x, y, z = (q / n).T
    R = np.empty((q.shape[0], 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def block_energy(gray, bs=16):
    h, w = gray.shape
    nh, nw = h // bs, w // bs
    gy, gx = np.gradient(gray)
    mag = np.sqrt(gy ** 2 + gx ** 2)[: nh * bs, : nw * bs]
    return mag.reshape(nh, bs, nw, bs).mean(axis=(1, 3))


def main(tag, model_dir, clean_img_dir, n_cams=8, bs=16, out_dir="/public/home/qq18676607705/results_spatial"):
    it_dir = sorted(glob.glob(os.path.join(model_dir, "point_cloud", "iteration_*")))[-1]
    ply = os.path.join(it_dir, "point_cloud.ply")
    xyz, scales, rot, opacity = read_ply(ply)
    cams = json.load(open(os.path.join(model_dir, "cameras.json")))

    counts, cover, energies, n_blocks_total = [], [], [], 0
    per_cam = []
    for c in cams[:n_cams]:
        p = os.path.join(clean_img_dir, c["img_name"])
        if not os.path.exists(p):
            continue
        im = Image.open(p).convert("L")
        wi, hi = im.size
        gray = np.asarray(im, dtype=np.float32) / 255.0
        W0, H0 = float(c["width"]), float(c["height"])
        sx, sy = wi / W0, hi / H0
        fx, fy = float(c["fx"]) * sx, float(c["fy"]) * sy
        cx, cy = wi / 2.0, hi / 2.0

        R = np.array(c["rotation"], dtype=np.float64)
        t = np.array(c["position"], dtype=np.float64)
        pc = (xyz - t[None]) @ R.T
        z = pc[:, 2]
        front = z > 0.2
        u = fx * pc[:, 0] / np.where(front, z, 1) + cx
        v = fy * pc[:, 1] / np.where(front, z, 1) + cy
        inside = front & (u >= 0) & (u < wi) & (v >= 0) & (v < hi) & (opacity > 0.01)
        idx = np.where(inside)[0]
        if idx.size == 0:
            continue

        # 16x16 块计数
        bu = np.clip((u[idx] // bs).astype(int), 0, wi // bs - 1)
        bv = np.clip((v[idx] // bs).astype(int), 0, hi // bs - 1)
        nh, nw = hi // bs, wi // bs
        cnt = np.zeros((nh, nw), dtype=np.float64)
        np.add.at(cnt, (bv, bu), 1.0)

        # 解析覆盖：Σ2D = J Σ3D Jᵀ
        Rq = quat_to_R(rot[idx])
        S = np.zeros((idx.size, 3, 3))
        S[:, 0, 0], S[:, 1, 1], S[:, 2, 2] = scales[idx].T
        Sig3 = Rq @ S @ S @ np.transpose(Rq, (0, 2, 1))
        zz = z[idx]
        J = np.zeros((idx.size, 2, 3))
        J[:, 0, 0] = fx / zz
        J[:, 0, 2] = -fx * pc[idx, 0] / zz ** 2
        J[:, 1, 1] = fy / zz
        J[:, 1, 2] = -fy * pc[idx, 1] / zz ** 2
        Sig2 = J @ Sig3 @ np.transpose(J, (0, 2, 1))
        lam = np.linalg.eigvalsh(Sig2)[:, -1]
        r = 3.0 * np.sqrt(np.clip(lam, 0, None))
        cover.append(float((np.pi * r ** 2).sum() / (wi * hi)))

        e = block_energy(gray, bs)
        e = e[:nh, :nw]
        energies.append(e.reshape(-1))
        counts.append(cnt.reshape(-1))
        per_cam.append(int(idx.size))
        n_blocks_total += cnt.size

    cnt = np.concatenate(counts)
    en = np.concatenate(energies)
    q = np.quantile(en, [0.25, 0.75])
    low = en <= q[0]
    high = en >= q[1]
    res = {
        "tag": tag,
        "n_gauss": int(xyz.shape[0]),
        "cams": len(per_cam),
        "visible_per_cam": per_cam,
        "coverage_per_pixel": float(np.mean(cover)) if cover else None,
        "blocks": int(cnt.size),
        "count_mean": float(cnt.mean()),
        "corr_count_energy": float(np.corrcoef(cnt, en)[0, 1]),
        "mean_count_lowE": float(cnt[low].mean()),
        "mean_count_highE": float(cnt[high].mean()),
        "energy_mean": float(en.mean()),
    }
    os.makedirs(out_dir, exist_ok=True)
    json.dump(res, open(os.path.join(out_dir, tag + ".json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
