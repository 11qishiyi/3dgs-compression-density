"""Evaluate the F1 run matrix: test-view PSNR / SSIM and primitive count.

Reads $F1_BASE/f1_eval.tsv (tag<TAB>scene<TAB>iterations, one per line) and
prints CSV rows: tag,psnr,ssim,n_test,gaussians,iterations
"""
import gc
import os
import re
import sys

import torch
from argparse import ArgumentParser

from arguments import ModelParams, PipelineParams
from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.image_utils import psnr
from utils.loss_utils import ssim

BASE = os.environ.get("F1_BASE", "/root/f1")
OUT = os.path.join(BASE, "output")
DATA = os.path.join(BASE, "data", "mip360")


def count_ply(tag, it):
    p = os.path.join(OUT, tag, "point_cloud", f"iteration_{it}", "point_cloud.ply")
    try:
        with open(p, "rb") as fh:
            head = fh.read(4000)
        m = re.search(rb"element vertex (\d+)", head)
        return int(m.group(1)) if m else -1
    except Exception:
        return -1


def main():
    parser = ArgumentParser()
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    rows = [l.split() for l in open(os.path.join(BASE, "f1_eval.tsv")) if l.strip()]
    print("tag,psnr,ssim,n_test,gaussians,iterations", flush=True)
    for tag, scene, it in rows:
        it = int(it)
        args = parser.parse_args([
            "-s", os.path.join(DATA, scene), "-m", os.path.join(OUT, tag),
            "--eval", "--data_device", "cpu",
        ])
        dataset = lp.extract(args)
        pipe = pp.extract(args)
        gaussians = GaussianModel(dataset.sh_degree, "default")
        sc = Scene(dataset, gaussians, load_iteration=it, shuffle=False)
        bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                          dtype=torch.float32, device="cuda")
        p = s = 0.0
        n = 0
        with torch.no_grad():
            for cam in sc.getTestCameras():
                img = torch.clamp(render(cam, gaussians, pipe, bg)["render"], 0, 1)
                gt = torch.clamp(cam.original_image.cuda(), 0, 1)
                p += float(psnr(img, gt).mean())
                s += float(ssim(img, gt))
                n += 1
        g = count_ply(tag, it)
        print(f"{tag},{p/n:.4f},{s/n:.4f},{n},{g},{it}", flush=True)
        del gaussians, sc
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
