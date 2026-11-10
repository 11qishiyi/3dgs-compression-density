"""渲染先验去量化：POCS 量化格投影（C2）与维纳频域混合（C3）。

约定
----
* 图像张量为 3×H×W、取值 [0,1]（与 3DGS 的 original_image 一致）。
* JPEG 量化表直接从数据集文件的 PIL `quantization` 属性读取（zigzag 反排），
  不使用"标准表 + 质量猜"。
* 在 YCbCr（JPEG 全范围 BT.601）域做 DCT 域运算：
  C = D · block · Dᵀ，D 为 8×8 正交 DCT-II 矩阵。
* 4:2:0 色度下采样**未**复现：色度按其量化表在全分辨率上直接处理（近似），
  报告中必须注明。

两种目标
--------
pocs_project(R, y, qtabs)
    对每个 8×8 块、每个系数：由 y 的系数得量化格 [ (q-0.5)Q, (q+0.5)Q )，
    把 R 的系数裁剪进该格（DCT 域正交 ⇒ 是格内离 R 最近的 L2 投影）。

wiener_mix(R, y, qtabs)
    逐频点可靠性 w = σ_x² / (σ_x² + Q²/12)，
    σ_x² 由 R 该频点在所有块上的方差估计，Q²/12 为均匀量化噪声方差。
    x̂ = w·y + (1−w)·R（低频信码流、高频信渲染）。
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_ZZ = [0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5,
       12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
       35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
       58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63]

_DCT_CACHE = {}


def dct_matrix(device):
    key = str(device)
    if key not in _DCT_CACHE:
        n = 8
        k = np.arange(n)
        C = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            C[i, :] = np.cos(np.pi * (2 * k + 1) * i / (2 * n))
        C[0, :] *= np.sqrt(1.0 / n)
        C[1:, :] *= np.sqrt(2.0 / n)
        _DCT_CACHE[key] = torch.tensor(C, dtype=torch.float32, device=device)
    return _DCT_CACHE[key]


def load_qtables(path):
    """返回 {0: Y, 1: Cb, 2: Cr} 的 8×8 float32 张量（JPEG 顺序 → 空间顺序）。"""
    im = Image.open(path)
    q = im.quantization
    tabs = {}
    for k, v in q.items():
        # natural (row-major) order, not the file's zigzag scan (fixed 2026-11-09)
        tabs[k] = np.asarray(v, dtype=np.float32).reshape(8, 8)
    return tabs


def rgb_to_ycbcr(x):
    r, g, b = x[0:1], x[1:2], x[2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return torch.cat([y, cb, cr], 0) * 255.0


def ycbcr_to_rgb(x):
    x = x / 255.0
    y, cb, cr = x[0:1], x[1:2] - 0.5, x[2:3] - 0.5
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return torch.clamp(torch.cat([r, g, b], 0), 0.0, 1.0)


def _to_blocks(x):
    C, H, W = x.shape
    ph, pw = (-H) % 8, (-W) % 8
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
    b = x.unfold(1, 8, 8).unfold(2, 8, 8)          # C × nh × nw × 8 × 8
    return b, (H, W, ph, pw)


def _from_blocks(b, meta):
    H, W, ph, pw = meta
    C, nh, nw = b.shape[0], b.shape[1], b.shape[2]
    x = b.permute(0, 1, 3, 2, 4).reshape(C, nh * 8, nw * 8)
    return x[:, :H, :W]


def _chan_q(qtabs, c, device):
    key = {0: 0, 1: 1, 2: 2}[c]
    if key not in qtabs:            # 灰度 JPEG 只有一张表
        key = min(qtabs.keys())
    return torch.as_tensor(qtabs[key], dtype=torch.float32, device=device)


def pocs_project(R, y, qtabs):
    """把渲染 R 投影到 y 的 JPEG 量化格内（DCT 域裁剪）。"""
    device = R.device
    D = dct_matrix(device)
    Ry, Yy = rgb_to_ycbcr(R.detach()), rgb_to_ycbcr(y.detach())
    outs = []
    for c in range(3):
        Q = _chan_q(qtabs, c, device)
        rb, meta = _to_blocks(Ry[c:c + 1])
        yb, _ = _to_blocks(Yy[c:c + 1])
        cr_ = D @ rb @ D.t()
        cy = D @ yb @ D.t()
        q = torch.round(cy / Q)
        lo, hi = (q - 0.5) * Q, (q + 0.5) * Q
        rec = D.t() @ torch.clamp(cr_, lo, hi) @ D
        outs.append(_from_blocks(rec, meta))
    return ycbcr_to_rgb(torch.cat(outs, 0))


def wiener_mix(R, y, qtabs):
    """逐频点维纳混合：低频信码流、高频信渲染。"""
    device = R.device
    D = dct_matrix(device)
    Ry, Yy = rgb_to_ycbcr(R.detach()), rgb_to_ycbcr(y.detach())
    outs = []
    for c in range(3):
        Q = _chan_q(qtabs, c, device)
        rb, meta = _to_blocks(Ry[c:c + 1])
        yb, _ = _to_blocks(Yy[c:c + 1])
        cr_ = D @ rb @ D.t()
        cy = D @ yb @ D.t()
        var_r = cr_.var(dim=(1, 2), keepdim=True)          # 每频点跨块方差
        noise = (Q ** 2) / 12.0                            # 均匀量化噪声
        w = var_r / (var_r + noise + 1e-8)
        rec = D.t() @ (w * cy + (1.0 - w) * cr_) @ D
        outs.append(_from_blocks(rec, meta))
    return ycbcr_to_rgb(torch.cat(outs, 0))
