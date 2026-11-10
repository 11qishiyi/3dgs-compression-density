# Reproducing the manuscript

Repository: <https://github.com/11qishiyi/3dgs-compression-density>

This guide assumes a Linux machine with one NVIDIA GPU (the paper used an A100-40GB and an
RTX 4090/3090-class card for some runs) and a working CUDA toolchain.

## 1. Upstream code

```bash
git clone https://github.com/graphdeco-inria/gaussian-splatting.git third_party/gaussian-splatting
cd third_party/gaussian-splatting
git checkout <commit-used>          # record the exact commit in your fork; the paper used the
                                    # public main branch as of 2026-08
pip install submodules/diff-gaussian-rasterization submodules/simple-knn
cd ../..
```

The paper builds on the official implementation without changing the rasterizer. The files that
were modified or added are listed in `patches/README.md`; the main training entry point is
`code/train_codec2.py` (derived from `train.py`).

## 2. Scenes

Download the three collections (links in the top-level README) and arrange them as:

```
data/mip360/<scene>/images/  data/mip360/<scene>/sparse/0/
data/bench/<scene>/images/   data/bench/<scene>/sparse/0/
```

Build the quality variants (clean, q80, q50, q20; 4:2:0 and 4:4:4):

```bash
python scripts/build_r4res.py          # clean + q20, 4:2:0 and 4:4:4, resized 4x
python scripts/build_444.py            # 4:4:4 counterfactual at matched QP
python scripts/build_444_qp.py         # QP sweep for the 4:4:4 arm
python scripts/build_blur_noise.py     # Gaussian-blur control
python scripts/build_deblock.py        # deblocking pre-filter
```

## 3. Training

Uniform-quality runs (the main grid; see `scripts/s3_matrix.tsv` for the exact tag list):

```bash
python code/train_codec2.py -s data/mip360/counter_r4res -m output/counter_clean_7k \
       --iterations 7000 --seed 0
python code/train_codec2.py -s data/mip360/counter_r4res_q20_hyb -m output/counter_q20_hyb_7k \
       --iterations 7000 --seed 0
```

Fine-tuning arms (post-hoc remedies, frozen topology):

```bash
python code/train_ft.py  ...   # 600-iteration fine-tuning from the 3k checkpoint
python code/train_ft4.py ...   # threshold/objective variants of the same protocol
```

The full run matrix, seeds and iteration budgets are in `data/runs.csv`; the exact command lines
are recorded in the `notes` column of that ledger.

## 4. Evaluation

```bash
python code/eval_metrics.py -m output/counter_q20_hyb_7k -s data/mip360/counter_r4res \
       --iteration 7000 --split test
```

This reports PSNR, SSIM and LPIPS on the held-out original views and the primitive count.

## 5. Figures and numbers

```bash
python code/make_figs_v8.py            # Figs. 1-11 (reads the aggregate JSON in data/)
python code/make_fig12_3view.py        # Fig. 12 (needs renders_3v/fig12_3v/{clean,q20}/*.png)
python code/audit_tvc_numbers.py       # re-derives every number in the manuscript
```

`code/audit_tvc_numbers.py` reads `data/runs.csv` and `data/metrics.csv` and checks the tables,
the abstract numbers, the decomposition, the elasticity fits and the seed statistics. It is the
script used to verify the submitted manuscript (266 checks).

## 6. Notes

- Seeds: configurations share camera order and random draws (common random numbers), so seed
  effects are common-mode rather than confounded with the compression variable.
- All runs use the same 200-image training split per scene (240 for counter, 194 for bicycle,
  251 for truck, 225 for playroom); test views are always the original uncompressed images.
- The figures are generated from the aggregate JSON in `data/`, not hand-edited; the manuscript's
  image-integrity audit reproduced all 12 figures bit-for-bit from these scripts.

## 7. Hardware and time

Reference timings on one A100-40GB: counter 7k ≈ 2.6 min, counter 30k ≈ 11 min, bicycle 7k ≈ 5 min,
truck 7k ≈ 3.5 min, playroom 7k ≈ 5 min. The complete grid in `data/runs.csv` took about 3.5 GPU
hours on a mix of A100/4090/3090 cards.
