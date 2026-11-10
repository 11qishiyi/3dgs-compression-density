# Modifications to the official 3DGS implementation

The paper uses the public `graphdeco-inria/gaussian-splatting` code (Inria/MPII license) as the
training and rendering backbone. The rasterizer (`diff-gaussian-rasterization`) and `simple-knn`
are used unmodified. The following files carry the study-specific changes; all of them are
included in `code/` for inspection, and `code/train_codec2.py` is the main entry point.

## `train_codec2.py` (derived from `train.py`)

1. **Pre-built scene loading.** Instead of running COLMAP, the loader reads a scene directory that
   already contains `images/` (at training resolution), `sparse/0/` and `poses_bounds.npy`. This is
   what makes the single-variable design possible: clean and compressed variants of the same scene
   share the identical camera set.
2. **Loss options.** Pixel L1 (default), pixel L2, uniform DCT-L2, quantization-weighted DCT-L2 and
   the quantization-cell (hinge/interval) objective at two weights. The DCT operators live in
   `dct_loss.py`.
3. **Dequantization modes.** Projection onto convex sets (POCS) cell projection and the
   Wiener-style mixing `w = σ²/(σ² + Q²/12)`; both in `dequant.py` (with unit tests in
   `test_dequant.py`, not shipped here to keep the release small).
4. **Densification instrumentation.** At every densification event the trainer records the median,
   90th and 99th percentiles of the view-space positional gradient norm over visible Gaussians, the
   fraction above the threshold, and the primitive count before and after the event. These are the
   numbers behind Section 4.2 and Fig. 8.
5. **Seed control.** A `--seed` flag sets Python/NumPy/PyTorch seeds and the camera-order shuffle, so
   that configurations sharing a seed share random draws (common random numbers).
6. **Evaluation hook.** At the final iteration the trainer evaluates on the held-out original views
   and writes PSNR/SSIM/LPIPS and the primitive count to the run log.

## `train_ft.py`, `train_ft4.py`

Fine-tuning from the 3k checkpoint with densification and pruning disabled (topology frozen at
812,673 primitives), 600 iterations, reduced learning rates. `train_ft.py` compares target
constructions (compressed, clean, POCS, Wiener); `train_ft4.py` implements the threshold/objective
variants of the same protocol.

## `train_seed.py`

Seed-only variant used for the multi-seed top-ups (same schedule, different seed), so the ledger
rows for seeds 1–3 are directly comparable with the seed-0 rows.

## `eval_metrics.py`

Renders the held-out test views from a trained model and reports PSNR, SSIM and LPIPS against the
original uncompressed images, plus the primitive count.

## `calc_bitrate.py`, `bpp_train.py`, `calc_bpp_all.py`

Bitrate ledger: compute bits per pixel over the training split for each quality level, including
the 4:4:4 counterfactual and the mixed-quality arm.

## `make_figs_v8.py`, `make_fig12_3view.py`, `make_figs_export.py`

Figure generation. Every figure is produced from the aggregate JSON in `data/` (or from the six
renders in `renders_3v/fig12_3v/` for Fig. 12); no manual editing is involved.

## `audit_tvc_numbers.py`

Independent re-derivation of every number in the manuscript from `data/runs.csv` and
`data/metrics.csv` (Tables 1–11, abstract numbers, decomposition, elasticity, seed statistics).
