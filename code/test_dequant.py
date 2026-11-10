"""本地验证 dequant 的数学正确性（CPU）。"""
import io
import numpy as np
import torch
from PIL import Image

import dequant

torch.manual_seed(0)
H, W = 96, 128
x = (torch.rand(3, H, W) * 0.6 + 0.2)
x[:, 20:60, 30:90] = torch.rand(3, 1, 1) * 0.5 + 0.25   # 结构块

def jpeg_roundtrip(img, quality):
    arr = (img.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "JPEG", quality=quality, subsampling=0)
    buf.seek(0)
    out = Image.open(buf)
    q = out.quantization
    y = torch.tensor(np.asarray(out.convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)
    return y, q

for quality in (20, 50, 80):
    y, q = jpeg_roundtrip(x, quality)
    tabs = {}
    zz = dequant._ZZ
    for k, v in q.items():
        a = np.asarray(v, dtype=np.float32)
        o = np.zeros(64, dtype=np.float32)
        o[zz] = a
        tabs[k] = o.reshape(8, 8)

    # 1) 幂等性：把 y 自己投影，应还原 y（格内点不动）
    y2 = dequant.pocs_project(y, y, tabs)
    err_idem = (y2 - y).abs().max().item()

    # 2) 用干净 x 做"渲染"，投影后应比 y 更接近 x
    yp = dequant.pocs_project(x, y, tabs)
    err_y = (y - x).abs().mean().item()
    err_p = (yp - x).abs().mean().item()

    # 3) 维纳：w=1 时退化为 y
    yw = dequant.wiener_mix(y, y, tabs)
    err_w_idem = (yw - y).abs().max().item()
    xw = dequant.wiener_mix(x, y, tabs)
    err_w = (xw - x).abs().mean().item()

    print(f"q{quality}: idem_max={err_idem:.2e}  "
          f"MAE(y)={err_y:.4f} MAE(pocs)={err_p:.4f}  "
          f"wiener(y)={err_w_idem:.2e}  MAE(wiener)={err_w:.4f}")
