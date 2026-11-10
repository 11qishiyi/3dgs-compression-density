"""JSGS-lite：按每图 JPEG 量化表加权的 DCT 域损失。

L = mean_{blocks,uv} w_uv * |DCT(render − target)|²
w_uv = 1 / (1 + Q_uv / Q_ref)，Q_ref 取该图量化表均值。
逐图从 PIL quantization 读取（jpge 顺序 → 自然顺序）。
"""
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_ZZ = [0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5,
       12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
       35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
       58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63]

_DCT = {}


def dct_mat(device):
    k = str(device)
    if k not in _DCT:
        n = 8
        idx = np.arange(n)
        C = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            C[i] = np.cos(np.pi * (2 * idx + 1) * i / (2 * n))
        C[0] *= np.sqrt(1.0 / n)
        C[1:] *= np.sqrt(2.0 / n)
        _DCT[k] = torch.tensor(C, dtype=torch.float32, device=device)
    return _DCT[k]


def weight_maps(paths, device):
    """返回 {stem: 3×8×8 权重}；Y/Cb/Cr 各自归一。"""
    out = {}
    for p in paths:
        im = Image.open(p)
        q = im.quantization
        ws = []
        for ch in range(3):
            key = ch if ch in q else min(q.keys())
            # PIL returns the table already in natural (row-major) order; the
            # file's zigzag scan is undone by the decoder. Assuming zigzag here
            # scrambles the per-frequency steps (fixed 2026-11-09).
            Q = np.asarray(q[key], dtype=np.float32).reshape(8, 8)
            qref = float(Q.mean()) + 1e-6
            ws.append(1.0 / (1.0 + Q / qref))
        out[os.path.splitext(os.path.basename(p))[0]] = torch.tensor(
            np.stack(ws), dtype=torch.float32, device=device)
    return out


def rgb_to_ycbcr(x):
    r, g, b = x[0:1], x[1:2], x[2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b
    return torch.cat([y, cb, cr], 0)


def blocks(x):
    C, H, W = x.shape
    ph, pw = (-H) % 8, (-W) % 8
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
    b = x.unfold(1, 8, 8).unfold(2, 8, 8)      # C×nh×nw×8×8
    return b, (H, W)


def dct_loss(render, target, w):
    """render/target: 3×H×W in [0,1]；w: 3×8×8 权重。"""
    r = rgb_to_ycbcr(render)
    t = rgb_to_ycbcr(target)
    diff = r - t
    b, _ = blocks(diff)
    D = dct_mat(render.device)
    coef = D @ b @ D.t()                        # C×nh×nw×8×8
    ww = w.view(3, 1, 1, 8, 8)
    scale = float(os.environ.get("DCT_SCALE", "1"))
    return scale * ((coef ** 2) * ww).mean()

def dct_l1(render, target, w):
    """Weighted L1 in the DCT domain, normalized to the pixel-L1 scale.

    render/target: 3xHxW in [0,1]; w: 3x8x8 weights read from the JPEG tables.
    """
    r = rgb_to_ycbcr(render)
    t = rgb_to_ycbcr(target)
    diff = r - t
    b, _ = blocks(diff)
    D = dct_mat(render.device)
    coef = D @ b @ D.t()
    ww = w.view(3, 1, 1, 8, 8)
    return ((coef.abs() * ww).mean()) / (ww.mean() + 1e-8)


def qtab_maps(paths, device, unit=1.0 / 255.0):
    """Return {stem: 3x8x8 JPEG quantization steps in [0,1] pixel units}."""
    out = {}
    for p in paths:
        im = Image.open(p)
        q = im.quantization
        qs = []
        for ch in range(3):
            key = ch if ch in q else min(q.keys())
            arr = np.asarray(q[key], dtype=np.float32).reshape(8, 8)
            qs.append(arr * unit)
        out[os.path.splitext(os.path.basename(p))[0]] = torch.tensor(
            np.stack(qs), dtype=torch.float32, device=device)
    return out


def dct_hinge(render, target, qtab, relative=True):
    """JSGS-style coefficient-interval constraint on the luminance channel.

    JPEG pins every DCT coefficient of an observed block to a cell of half-width
    q/2 around the decoded value. A rendered coefficient inside that cell cannot
    be distinguished from the observation, so it is not penalized; one that falls
    outside is pulled back to the boundary. With ``relative`` the excess is
    divided by the local half-width, so a coefficient is judged against its own
    quantizer uncertainty; the penalty is the squared normalized excess averaged
    over blocks and coefficients.
    """
    r = rgb_to_ycbcr(render)[0:1]
    t = rgb_to_ycbcr(target)[0:1]
    b_r, _ = blocks(r)
    b_t, _ = blocks(t)
    D = dct_mat(render.device)
    cr = D @ b_r @ D.t()
    ct = D @ b_t @ D.t()
    half = (0.5 * qtab[0]).view(1, 1, 8, 8)
    excess = (cr - ct).abs() - half
    if relative:
        excess = excess / half.clamp_min(1e-6)
    viol = torch.clamp(excess, min=0.0)
    return (viol ** 2).mean()


def frame_weights(paths, train_names=None, lo=0.33, hi=3.0):
    """CompSplat-style per-frame severity weights read from the codec tables.

    Severity is the mean luminance quantization step of a frame: the larger the
    steps, the less reliable its high-frequency evidence, so the smaller its
    weight. Weights are normalized over the training split (test views are never
    used by the loss and must not shift the normalization), then clipped so that
    one frame cannot dominate the batch.
    """
    qs = {}
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        if train_names is not None and stem not in train_names:
            continue
        im = Image.open(p)
        q = im.quantization
        key = 0 if 0 in q else min(q.keys())
        arr = np.asarray(q[key], dtype=np.float32)
        Q = np.zeros(64, dtype=np.float32)
        Q[_ZZ] = arr
        qs[stem] = float(Q.mean())
    mean_q = sum(qs.values()) / len(qs)
    raw = {k: mean_q / max(v, 1e-6) for k, v in qs.items()}
    norm = sum(raw.values()) / len(raw)
    return {k: min(hi, max(lo, v / norm)) for k, v in raw.items()}
