"""批量评测：PSNR / SSIM / LPIPS（3DGS metrics + lpips），输出 CSV 行。

用法: python eval_metrics.py <tag> <dataset> <iteration>
"""
import sys

import torch

from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams
from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.image_utils import psnr
from utils.loss_utils import ssim

try:
    from lpips import LPIPS
    LPIPS_OK = True
except Exception:
    LPIPS_OK = False


def main():
    tag, dataset_path, it = sys.argv[1], sys.argv[2], int(sys.argv[3])
    parser = ArgumentParser()
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    args = parser.parse_args([
        "-s", dataset_path,
        "-m", f"/public/home/qq18676607705/output/{tag}",
        "--eval", "--data_device", "cpu",
    ])
    dataset = lp.extract(args)
    pipe = pp.extract(args)

    gaussians = GaussianModel(dataset.sh_degree, "default")
    scene = Scene(dataset, gaussians, load_iteration=it, shuffle=False)
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")

    lpips_fn = LPIPS(net="vgg", verbose=False).cuda() if LPIPS_OK else None

    cams = scene.getTestCameras()
    p = s = l = 0.0
    n = 0
    with torch.no_grad():
        for cam in cams:
            img = torch.clamp(render(cam, gaussians, pipe, bg)["render"], 0, 1)
            gt = torch.clamp(cam.original_image.cuda(), 0, 1)
            p += float(psnr(img, gt).mean())
            s += float(ssim(img, gt))
            if lpips_fn is not None:
                l += float(lpips_fn(img.unsqueeze(0) * 2 - 1, gt.unsqueeze(0) * 2 - 1))
            n += 1
    print(f"{tag},{p/n:.4f},{s/n:.4f},{(l/n if n else float('nan')):.4f},{n}")


if __name__ == "__main__":
    main()
