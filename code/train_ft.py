"""Fine-tune 版训练：冻结致密化，只换训练目标（C1/C2/C3），用于信号级去量化 screening。

与 train.py 的差异（其余保持逐行一致）：
1. 用 --start_ply 从已有 point_cloud.ply 初始化（仓库只保存 ply，没有 .pth）；
2. 不做 densify/prune（由 --densify_until_iter 0 保证），也不触发 opacity reset；
3. 目标图可由 --ft_mode 选择：
     c1 = 量化图 y（对照，等于普通续训）
     c2 = POCS 量化格投影 x̂ = Proj(R)
     c3 = 维纳频域混合 x̂ = w·y + (1−w)·R
   c4（oracle）不在这里实现：直接用 c1 跑干净数据集即可。
4. 学习率整体调小（命令行传入），四个 run 完全一致。
"""
import os
import random
import sys

import numpy as np
from argparse import ArgumentParser
from random import randint

import torch
from tqdm import tqdm

import dequant
import dct_loss as dctlib
from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except Exception:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except Exception:
    SPARSE_ADAM_AVAILABLE = False


CODEC_MODE = os.environ.get("CODEC_MODE", "")
CODEC_LAMBDA = float(os.environ.get("CODEC_LAMBDA", "1.0"))
_CODEC_IMG_DIR = os.environ.get("DCT_IMG_DIR", "")
_CODEC_QT = {}


def _codec_qtab(stem):
    if stem in _CODEC_QT:
        return _CODEC_QT[stem]
    if not _CODEC_QT:
        paths = [os.path.join(_CODEC_IMG_DIR, f) for f in sorted(os.listdir(_CODEC_IMG_DIR))
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        _CODEC_QT.update(dctlib.qtab_maps(paths, "cuda"))
    return _CODEC_QT[stem]


QTABS = None


def report(iteration, testing_iterations, scene, pipe, background, train_test_exp):
    if iteration not in testing_iterations:
        return
    torch.cuda.empty_cache()
    for name, cams in (("test", scene.getTestCameras()), ("train", scene.getTrainCameras()[:5])):
        if not cams:
            continue
        l1_t = psnr_t = 0.0
        for cam in cams:
            img = torch.clamp(render(cam, scene.gaussians, pipe, background,
                                     use_trained_exp=train_test_exp,
                                     separate_sh=SPARSE_ADAM_AVAILABLE)["render"], 0.0, 1.0)
            gt = torch.clamp(cam.original_image.to("cuda"), 0.0, 1.0)
            l1_t += l1_loss(img, gt).mean().double()
            psnr_t += psnr(img, gt).mean().double()
        print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(
            iteration, name, l1_t / len(cams), psnr_t / len(cams)), flush=True)
    torch.cuda.empty_cache()


def training(dataset, opt, pipe, testing_iterations, saving_iterations, start_ply, ft_mode, img_dir):
    global QTABS, _CODEC_QT
    os.makedirs(dataset.model_path, exist_ok=True)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)

    if start_ply:
        gaussians.load_ply(start_ply)
        print("LOADED_PLY {} gaussians={}".format(start_ply, gaussians.get_xyz.shape[0]), flush=True)
    gaussians.training_setup(opt)

    if ft_mode in ("c2", "c3"):
        imgs = sorted(os.path.join(img_dir, f) for f in os.listdir(img_dir)
                      if f.lower().endswith((".jpg", ".jpeg", ".png")))
        # hybrid 数据集中 test 图是干净的(q98)，train 图才是 q20：必须逐图取表
        QMAP = {}
        for p in imgs:
            QMAP[os.path.splitext(os.path.basename(p))[0]] = dequant.load_qtables(p)
        _k0 = os.path.splitext(os.path.basename(imgs[0]))[0]
        _k1 = os.path.splitext(os.path.basename(imgs[1]))[0]
        print("QTAB n={} sample[{}].Y0={} sample[{}].Y0={}".format(
            len(QMAP), _k0, QMAP[_k0][0].tolist()[0], _k1, QMAP[_k1][0].tolist()[0]), flush=True)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    viewpoint_stack = scene.getTrainCameras().copy()
    ema = 0.0
    progress_bar = tqdm(range(1, opt.iterations + 1), desc="FT progress")

    for iteration in range(1, opt.iterations + 1):
        gaussians.update_learning_rate(iteration)

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        pkg = render(viewpoint_cam, gaussians, pipe, bg,
                     use_trained_exp=dataset.train_test_exp,
                     separate_sh=SPARSE_ADAM_AVAILABLE)
        image = pkg["render"]
        if viewpoint_cam.alpha_mask is not None:
            image = image * viewpoint_cam.alpha_mask.cuda()

        gt_image = viewpoint_cam.original_image.cuda()
        if ft_mode in ("c2", "c3"):
            stem = os.path.splitext(viewpoint_cam.image_name)[0]
            _q = QMAP.get(stem)
            if _q is None:
                raise KeyError("no qtable for camera image: %s" % viewpoint_cam.image_name)
            gt_image = (dequant.pocs_project(image, gt_image, _q) if ft_mode == "c2"
                        else dequant.wiener_mix(image, gt_image, _q))

        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        if CODEC_MODE == "hinge":
            loss = loss + CODEC_LAMBDA * dctlib.dct_hinge(
                image, gt_image, _codec_qtab(os.path.splitext(viewpoint_cam.image_name)[0]))
        loss.backward()

        with torch.no_grad():
            ema = 0.4 * loss.item() + 0.6 * ema
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema:.7f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            report(iteration, testing_iterations, scene, pipe, background, dataset.train_test_exp)

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration), flush=True)
                scene.save(iteration)

            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none=True)
                if opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE:
                    gaussians.optimizer.step(pkg["radii"] > 0, pkg["radii"].shape[0])
                else:
                    gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    parser = ArgumentParser(description="Fine-tune with dequantized targets")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[600])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--disable_viewer", action="store_true", default=True)
    parser.add_argument("--start_ply", type=str, default=None)
    parser.add_argument("--ft_mode", type=str, default="c1", choices=["c1", "c2", "c3"])
    parser.add_argument("--img_dir", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.disable_viewer = True

    print("Fine-tune " + args.model_path, flush=True)
    _seed = int(os.environ.get('GSSEED', '0'))
    torch.manual_seed(_seed); random.seed(_seed); np.random.seed(_seed)
    safe_state(args.quiet)
    print('GSSEED', _seed, flush=True)
    training(lp.extract(args), op.extract(args), pp.extract(args),
             args.test_iterations, args.save_iterations, args.start_ply,
             args.ft_mode, args.img_dir)
    print("\nFine-tune complete.", flush=True)
