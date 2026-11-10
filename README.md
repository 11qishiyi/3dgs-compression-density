# Density Is a Symptom, Not a Lever — code and data release

Companion repository for the manuscript

> **Density Is a Symptom, Not a Lever: A Controlled Study of Lossy Input Compression in 3D Gaussian Splatting**
> Tianji Su, Business School, Ningbo University, China

The paper measures what lossy input compression does to 3D Gaussian Splatting (3DGS) under a
single-variable protocol: only the JPEG quality factor at training resolution is varied, a
dirty-train/clean-test protocol is used, and the analysis covers seven Mip-NeRF 360 scenes plus
two benchmark scenes (Tanks and Temples, Deep Blending) over up to four training budgets.

Key findings the released data support: compression *reduces* the primitive count in every scene;
the fidelity gap grows by about 0.45 dB per doubling of the training budget; the loss decomposes
almost additively into block quantization and chroma subsampling; and density moved through the
densification threshold buys far less fidelity per percent than the same density moved by changing
the input.

## What is in this repository

```
data/        run ledger, per-run metrics, bitrate ledger, aggregate tables used in the paper
code/        training/evaluation/figure scripts written for this study
scripts/     scene builders, cluster runners, render/eval helpers
figures/     the 12 manuscript figures as 300 dpi PNG
patches/     description of the modifications made to the official 3DGS training code
docs/        step-by-step reproduction guide
```

`data/runs.csv` contains every training run (196 rows, including failed and blocked ones) with
scene, quality level, iterations, seed, status and the path of the log; `data/metrics.csv` contains
the per-run evaluation metrics; `data/bpp.csv` the bitrate ledger. The aggregate tables used by
Tables 3–9 of the manuscript are in `data/f1-agg.json` and `data/s3-agg.json`.

## Quick start

```bash
git clone https://github.com/11qishiyi/3dgs-compression-density.git
cd 3dgs-compression-density
pip install -r requirements.txt
# follow docs/REPRODUCE.md: clone the official 3DGS code, build the scenes, train, evaluate
```

The official 3DGS implementation is **not** redistributed here. `docs/REPRODUCE.md` explains which
upstream commit was used, which files were modified (also summarised in `patches/README.md`), and
how to apply the modifications. Our modified training entry point is included as
`code/train_codec2.py` for transparency; it is a derivative of the official `train.py` and remains
subject to the Inria/MPII Gaussian-Splatting license (non-commercial research use), whose text is
reproduced in `third_party/LICENSE-Inria.md`.

## Reproducing the numbers

1. Build the scene variants with `scripts/build_r4res.py` (clean and q20 at 4:4:4 and 4:2:0),
   `scripts/build_444.py`, `scripts/build_444_qp.py`, `scripts/build_blur_noise.py` and
   `scripts/build_deblock.py`.
2. Train with `code/train_codec2.py` (uniform quality) or the fine-tuning entries
   `code/train_ft.py` / `code/train_ft4.py` for the post-hoc routes.
3. Evaluate with `code/eval_metrics.py` (PSNR/SSIM/LPIPS on held-out original views).
4. Regenerate the figures with `code/make_figs_v8.py` (Figs. 1–11) and
   `code/make_fig12_3view.py` (Fig. 12, requires the six renders in `renders_3v/fig12_3v/`).
5. Run `code/audit_tvc_numbers.py` to re-derive every number in the manuscript from
   `data/runs.csv` and `data/metrics.csv`.

## Data

The scene images are **not** redistributed. They are publicly available from the original authors:

- Mip-NeRF 360 — <https://jonbarron.info/mipnerf360/>
- Tanks and Temples — <https://www.tanksandtemples.org/>
- Deep Blending — <http://visual.cs.ucl.ac.uk/pubs/deepblending/>

The compressed training sets are deterministic functions of the originals and the released
`scripts/build_*.py` code; no images are needed to check the numbers in the paper, because the run
ledger and metrics are included.

## License

- Our code (`code/`, `scripts/`) is released under the MIT license (`LICENSE`).
- Our data (`data/`, `figures/`) is released under CC BY 4.0.
- `code/train_codec2.py` and any file derived from the official 3DGS code remain under the
  Inria/MPII Gaussian-Splatting license (non-commercial research use); see
  `third_party/LICENSE-Inria.md`.

## Repository

<https://github.com/11qishiyi/3dgs-compression-density>

## Citation

If you use this code or data, please cite the manuscript (see `CITATION.cff`). A Zenodo DOI for
this release will be added after acceptance.
