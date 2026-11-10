"""Rebuild Fig. 12 as a three-view plate once render_3views.sh / fetch_3views.py
have put the renders in proj-D6/renders_3v/fig12_3v/.

Layout: 3 columns = three held-out views (test indices 00000, 00009, 00018);
rows = clean-trained 30k, q20-trained 30k, |difference| x8.

Output (overwrites the single-view versions):
    proj-D6/figs_v8/fig12_qualitative.pdf
    proj-D6/figs_v8/fig12_qualitative.png        (300 dpi)
Run proj-D6/make_figs_export.py afterwards for the EPS / 600 dpi / TIFF masters.

The script refuses to overwrite anything unless all six renders exist.
Check the printed caption block against the manuscript text.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})

SRC = r"D:\3dgs\proj-D6\renders_3v\fig12_3v"
OUT = r"D:\3dgs\proj-D6\figs_v8"
VIEWS = ["00000", "00009", "00018"]
mm_to_inch = 1 / 25.4
W2 = 174 * mm_to_inch
BLACK = "#222222"


def load(kind, view):
    p = os.path.join(SRC, kind, view + ".png")
    if not os.path.exists(p):
        sys.exit(f"missing render: {p}\nRun render_3views.sh + fetch_3views.py first.")
    return np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255


def main():
    clean = [load("clean", v) for v in VIEWS]
    q20 = [load("q20", v) for v in VIEWS]

    hh = min(min(a.shape[0] for a in clean), min(a.shape[0] for a in q20))
    ww = min(min(a.shape[1] for a in clean), min(a.shape[1] for a in q20))
    clean = [a[:hh, :ww] for a in clean]
    q20 = [a[:hh, :ww] for a in q20]
    diff = [np.clip(np.abs(a - b).mean(axis=2) * 8, 0, 1) for a, b in zip(clean, q20)]

    fig, axes = plt.subplots(3, 3, figsize=(W2, 118 * mm_to_inch),
                             gridspec_kw={"wspace": 0.025, "hspace": 0.025})
    rows = [("clean-trained", clean), ("q20-trained", q20), (r"$|$difference$|$ $\times$8", diff)]
    for r, (row_label, imgs) in enumerate(rows):
        for c in range(3):
            ax = axes[r, c]
            ax.imshow(imgs[c], cmap="magma" if r == 2 else None,
                      vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if c == 0:
                ax.set_ylabel(row_label, fontsize=7, labelpad=4)

    # panel letters, row-major, matching the caption order (a)-(i)
    letters = "abcdefghi"
    for r in range(3):
        for c in range(3):
            axes[r, c].text(0.015, 0.975, f"({letters[3 * r + c]})", transform=axes[r, c].transAxes,
                            fontsize=7, fontweight="bold", color=BLACK,
                            ha="left", va="top",
                            bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none", alpha=0.85))

    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(OUT, "fig12_qualitative." + ext)
        fig.savefig(p, bbox_inches="tight", dpi=300)
        print("wrote", p)

    print("\ncaption draft (replace the Fig. 12 caption in the manuscript):")
    print("**Fig. 12** Qualitative comparison on three held-out views of bonsai at 30k iterations: "
          "renderings of the models trained on clean inputs (a-c) and on q20 inputs (d-f), and the "
          "absolute differences between them, amplified eight times (g-i). Compression removes "
          "high-frequency detail across viewpoints rather than adding structured artifacts, and the "
          "error spreads over textured regions instead of concentrating on 8x8 block boundaries")


if __name__ == "__main__":
    main()
