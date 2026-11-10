"""生成 train_dct.py：在 train_seed.py 基础上加入 DCT 加权损失（DCT_W=1 时启用）。"""
G = r"D:\3dgs\proj-D6\code\gaussian-splatting"
src = open(f"{G}\\train_seed.py", encoding="utf-8").read()

# 1) 顶部导入
src = src.replace("from c3 import c3_mask", "from c3 import c3_mask\nimport dct_loss as dctlib", 1)

# 2) 权重缓存与初始化
helper = '''

_WCACHE = {}


def _weights_for(stem, img_dir):
    if stem in _WCACHE:
        return _WCACHE[stem]
    if not _WCACHE:
        import os as _os
        paths = [_os.path.join(img_dir, f) for f in sorted(_os.listdir(img_dir))
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        _WCACHE.update(dctlib.weight_maps(paths, "cuda"))
    return _WCACHE[stem]

'''
src = src.replace("def training(dataset, opt, pipe", helper + "def training(dataset, opt, pipe", 1)

# 3) 损失块替换
old = """        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)"""
new = """        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if os.environ.get("DCT_W", "0") == "1":
            _stem = os.path.splitext(viewpoint_cam.image_name)[0]
            _w = _weights_for(_stem, os.environ["DCT_IMG_DIR"])
            loss = dctlib.dct_loss(image, gt_image, _w)
        else:
            if FUSED_SSIM_AVAILABLE:
                ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                ssim_value = ssim(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)"""
assert old in src, "loss block not found"
src = src.replace(old, new, 1)

open(f"{G}\\train_dct.py", "w", encoding="utf-8").write(src)
print("train_dct.py written", len(src))
