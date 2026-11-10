# Academic Figure Skill Asset Confirmation (verified against assets/figures/)
# (a) protocol schematic           → assets/figures/other (schematic)       → param inherit
# (b) line response vs quality     → assets/figures/LineTrend/plot_trend.py → param inherit (Class A: lw 1.1-1.2, ms 3.4-4.0; Class C: no grid)
# (c) forest / paired CI           → assets/figures/BarComparison           → param inherit (dot + CI bar, reference rule)
# (d) budget law + fit             → assets/figures/LineTrend/plot_trend.py → param inherit (band α=0.15, direct Δ labels)
# (e) cross-scene horizontal bars  → assets/figures/BarAblation/plot_bar_ablation.py → param inherit (bar height 0.62, value labels)
# (f) mechanism 3-panel            → assets/figures/basic-plots (scatter/stem) → param inherit
# (g) quantile dot plot            → assets/figures/BarDistribution          → param inherit (dot instead of bar: log axis, VI-4)
# (h) spatial 3-panel              → assets/figures/BarComparison            → param inherit (direct labels, no legend)
# (i) decomposition waterfall      → assets/figures/BarAblation              → param inherit (floating bars, in-bar values)
# (j) remedies dot plot            → assets/figures/BarComparison            → param inherit (dot plot avoids zero-baseline distortion)
# (k) elasticity lollipop (log x)  → assets/figures/BarDistribution          → param inherit
# (l) qualitative image plate      → assets/figures/other (image plate)      → param inherit (raster ≥300 dpi)
# RULE: no production script matches 3DGS measurement data structures, so every panel is "param inherit":
#       the drawing functions copy Class A (colour hex, lw, alpha, marker size), Class B (font sizes at
#       7-8 pt base, 84/174 mm widths) and Class C (no grid, no top/right spines, frameon=False,
#       direct value labels) from the named assets.
#
# Journal override: target is The Visual Computer (Springer), whose artwork guidance specifies 84 mm
#   (double-column text area) and 174 mm (single-column text area) figure widths instead of the CNS
#   89/183 mm of references/journal-specs.md. Heights are fixed at creation time.

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

# ------------------------- Academic Figure Skill Typography Baseline (verbatim) -------------------------
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 8,
    "figure.titlesize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
})

# ------------------------- Academic Figure Skill Color Palette (verbatim) -------------------------
CATEGORICAL = ["#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666"]
CATEGORICAL_EXTENDED = [
    "#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666",
    "#4393C3", "#D6604D", "#5AAE61", "#B35806", "#9970AB", "#999999",
]
DIVERGING = ["#2166AC", "#F7F7F7", "#B2182B"]
SEQUENTIAL = ["#F7FBFF", "#6BAED6", "#08306B"]
ACCENT_RED = "#B2182B"
GREY = "#999999"
BLACK = "#222222"

# ------------------------- Academic Figure Skill Export Baseline (verbatim) -------------------------
mpl.rcParams.update({
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})


def save_cns_figure(fig, filename):
    """Standard Academic Figure Skill export: vector PDF + 300dpi PNG preview."""
    fig.savefig(f"{filename}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{filename}.png", bbox_inches="tight", dpi=300)


# ------------------------- semantic roles (Class A) -------------------------
C_CLEAN = CATEGORICAL[0]   # blue  : clean / control / reference
C_Q20 = CATEGORICAL[1]     # red   : compressed input (the effect under study)
C_444 = CATEGORICAL[3]     # orange: 4:4:4 counterfactual / secondary condition
C_GOOD = CATEGORICAL[2]    # green : headroom / clean-target control
C_PROXY = CATEGORICAL[4]   # purple: scalar proxies (blur / noise)
C_OTHER = GREY             # grey  : reference rules

mm_to_inch = 1 / 25.4
W1 = 84 * mm_to_inch    # TVC single-column text area (Springer: 84 mm)
W2 = 174 * mm_to_inch   # TVC double-column text area (Springer: 174 mm)
W2C = 168 * mm_to_inch  # page-safe canvas: tight cropping adds the axis-label margins,
                        # so a 168 mm canvas renders at <=174 mm, the TVC limit

OUT = r"D:\3dgs\proj-D6\figs_v8"
os.makedirs(OUT, exist_ok=True)


def panel_label(ax, s, x=-0.14, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8, fontweight="bold",
            ha="left", va="bottom", color=BLACK)


def save(fig, name, tight=True):
    if tight:
        fig.tight_layout(pad=0.4)
    save_cns_figure(fig, os.path.join(OUT, name))
    plt.close(fig)
    print("saved", name)


# ============================ Fig. 1  measurement protocol ============================
fig, ax = plt.subplots(figsize=(W2C, 36 * mm_to_inch))
ax.axis("off")
steps = [
    ("original image\n3118 x 2078", "#F2F2F2", BLACK),
    ("resize x1/4\n779 x 519", "#F2F2F2", BLACK),
    ("JPEG q20\n8x8 DCT, 4:2:0", "#FBE3D6", C_Q20),
    ("3DGS training\n(-r 1)", "#DCE9F4", C_CLEAN),
    ("fidelity on\nheld-out originals", "#F2F2F2", BLACK),
]
x, w, h = 0.015, 0.168, 0.44
for i, (txt, face, edge) in enumerate(steps):
    ax.add_patch(FancyBboxPatch((x, 0.32), w, h, boxstyle="round,pad=0.008",
                                facecolor=face, edgecolor=edge, lw=0.8))
    ax.text(x + w / 2, 0.32 + h / 2, txt, ha="center", va="center", fontsize=7, color=BLACK)
    if i < len(steps) - 1:
        ax.add_patch(FancyArrowPatch((x + w, 0.32 + h / 2), (x + w + 0.033, 0.32 + h / 2),
                                     arrowstyle="-|>", mutation_scale=7, lw=0.7, color=BLACK))
    x += w + 0.033
ax.text(0.5, 0.10,
        "dirty-train / clean-test: only the bitstream changes; poses, initialisation, "
        "densification rule and budget are held fixed",
        ha="center", va="center", fontsize=7, color=BLACK)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
save(fig, "fig01_protocol")

# ============================ Fig. 2  response to the quality factor at 3k ============================
qp = ["clean", "q80", "q50", "q20"]
psnr3 = [28.848, 28.660, 28.444, 27.997]
prim3 = np.array([914349, 883809, 850501, 810585]) / 1e3
fig, axes = plt.subplots(2, 1, figsize=(W1, 66 * mm_to_inch), sharex=True,
                         gridspec_kw={"hspace": 0.18})
axes[0].plot(range(4), psnr3, "-o", color=C_CLEAN, ms=3.6, lw=1.1, mfc="white", mew=1.0)
axes[0].set_ylabel("PSNR (dB)")
axes[0].set_ylim(27.85, 28.95)
for xi, v in zip(range(4), psnr3):
    axes[0].annotate(f"{v:.3f}", (xi, v), xytext=(0, 5), textcoords="offset points",
                     ha="center", fontsize=6.9, color=C_CLEAN)
axes[1].plot(range(4), prim3, "--s", color=C_Q20, ms=3.4, lw=1.1, mfc="white", mew=1.0)
axes[1].set_ylabel("primitives (k)")
axes[1].set_ylim(795, 935)
for xi, v in zip(range(4), prim3):
    axes[1].annotate(f"{v:.0f}", (xi, v), xytext=(0, 5), textcoords="offset points",
                     ha="center", fontsize=6.9, color=C_Q20)
axes[1].set_xticks(range(4)); axes[1].set_xticklabels(qp)
axes[1].set_xlim(-0.35, 3.35)
axes[1].set_xlabel("JPEG quality factor (clean = q98)")
panel_label(axes[0], "(a)", x=-0.22, y=1.02)
panel_label(axes[1], "(b)", x=-0.22, y=1.02)
save(fig, "fig02_qp_response")

# ============================ Fig. 3  paired fidelity change with 95% CI ============================
labels3 = ["q80", "q50", "q20"]
means3 = [-0.163, -0.379, -0.852]
ci3 = [0.063, 0.108, 0.087]
fig, ax = plt.subplots(figsize=(W1, 40 * mm_to_inch))
y = np.arange(3)[::-1]
for yi, m, c in zip(y, means3, ci3):
    ax.plot([m - c, m + c], [yi, yi], color=C_Q20, lw=1.2, solid_capstyle="butt")
    ax.plot([m], [yi], "o", color=C_Q20, ms=4.2, mec="white", mew=0.6, zorder=3)
    ax.annotate(f"{m:+.3f}", (m + c, yi), xytext=(5, 0), textcoords="offset points",
                ha="left", va="center", fontsize=7.0, color=C_Q20)
ax.axvline(0, color=C_OTHER, lw=0.6, ls=(0, (4, 2)))
ax.set_yticks(y); ax.set_yticklabels(labels3)
ax.set_xlabel("paired PSNR change vs clean (dB), 95% CI")
ax.set_xlim(-1.02, 0.02)
ax.set_ylim(-0.65, 2.55)
ax.text(0.015, 2.50, "no change", fontsize=6, color=C_OTHER, ha="right", va="top")
save(fig, "fig03_paired_ci")

# ============================ Fig. 4  budget-dependent degradation ============================
its = np.array([3000, 7000, 15000, 30000])
clean = np.array([28.848, 30.251, 31.624, 32.771])
q20 = np.array([27.997, 29.022, 29.850, 30.450])
fig, ax = plt.subplots(figsize=(W1, 58 * mm_to_inch))
xx = its / 1000
ax.fill_between(xx, q20, clean, color=C_Q20, alpha=0.15, lw=0)
ax.plot(xx, clean, "-o", color=C_CLEAN, ms=3.6, lw=1.2, mfc="white", mew=1.0, label="clean")
ax.plot(xx, q20, "--s", color=C_Q20, ms=3.4, lw=1.2, mfc="white", mew=1.0, label="q20")
ax.set_xscale("log", base=2)
ax.set_xticks(xx); ax.set_xticklabels(["3k", "7k", "15k", "30k"]); ax.minorticks_off()
ax.set_xlim(2.1, 42)
ax.set_xlabel("training iterations")
ax.set_ylabel("PSNR on original test views (dB)")
ax.set_ylim(27.35, 33.5)
ax.legend(loc="lower right", handlelength=1.6)
for xi, c, q in zip(xx, clean, q20):
    ax.annotate(f"{q - c:+.2f} dB", (xi * 1.10, (c + q) / 2), fontsize=7.0, color=C_Q20,
                va="center", ha="left",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
ax.text(2.35, 32.9, "clean$-$q20 gap grows by 0.45 dB\nper doubling of the budget ($R^2$ = 0.984)",
        fontsize=6.9, color=C_Q20, va="top", ha="left")
save(fig, "fig04_budget")

# ============================ Fig. 5  cross-scene behaviour ============================
scenes = ["bonsai 3k", "bonsai 7k", "bonsai 15k", "bonsai 30k",
          "counter 3k", "counter 7k", "counter 30k", "bicycle 7k",
          "garden 3k", "kitchen 3k", "room 3k", "stump 3k",
          "truck 3k", "truck 7k", "playroom 3k", "playroom 7k"]
dpsnr = [-0.852, -1.229, -1.774, -2.321, -0.514, -0.808, -1.430, -0.125,
         -0.462, -1.560, -0.608, -0.109, -0.110, -0.224, -0.024, -0.477]
ddens = [-11.3, -11.6, -12.3, -10.8, -13.0, -19.6, -20.2, -8.6,
         -13.5, -18.4, -15.0, -1.0, -13.9, -20.3, -9.8, -15.5]
order = np.argsort(dpsnr)[::-1]  # largest fidelity cost at the top
scenes_s = [scenes[i] for i in order]
dpsnr_s = np.array(dpsnr)[order]
ddens_s = np.array(ddens)[order]
fig, axes = plt.subplots(1, 2, figsize=(W2, 62 * mm_to_inch), gridspec_kw={"wspace": 0.30})
xpos = np.arange(len(scenes_s))
axes[0].barh(xpos, dpsnr_s, color=C_Q20, height=0.62, lw=0)
axes[0].set_yticks(xpos); axes[0].set_yticklabels(scenes_s, fontsize=6)
axes[0].tick_params(axis="y", length=2)
axes[0].axvline(0, color=BLACK, lw=0.6)
axes[0].set_xlabel("PSNR change, q20 $-$ clean (dB)")
axes[0].set_xlim(-2.95, 0.28)
axes[0].set_ylim(-0.7, len(scenes_s) - 0.3)
for yi, v in zip(xpos, dpsnr_s):
    axes[0].annotate(f"{v:.2f}", (v - 0.06, yi), ha="right", va="center", fontsize=6.8, color=C_Q20)
axes[1].barh(xpos, ddens_s, color=C_444, height=0.62, lw=0)
axes[1].set_yticks(xpos); axes[1].set_yticklabels([])
axes[1].tick_params(axis="y", length=0)
axes[1].axvline(0, color=BLACK, lw=0.6)
axes[1].set_xlabel("primitive-count change (%)")
axes[1].set_xlim(-25.5, 2.5)
axes[1].set_ylim(-0.7, len(scenes_s) - 0.3)
for yi, v in zip(xpos, ddens_s):
    axes[1].annotate(f"{v:.1f}", (v - 0.6, yi), ha="right", va="center", fontsize=6.8, color="#B35806")
panel_label(axes[0], "(a)", x=-0.40)
panel_label(axes[1], "(b)", x=-0.06)
save(fig, "fig05_cross_scene")

# ============================ Fig. 6  mechanism chain ============================
names = ["clean", "q80", "q50", "q20", "q20 4:4:4", "blur", "noise"]
hf = [1.000, 1.003, 0.917, 0.805, 0.805, 0.812, 1.062]
dens = [1.000, 0.967, 0.930, 0.887, 0.888, 0.968, 0.928]
frac = [0.0698, 0.0629, 0.0624, 0.0681]
dens_f = [1.000, 0.887, 0.888, 0.968]
frac_labels = ["clean", "q20", "q20 4:4:4", "blur"]
fig, axes = plt.subplots(1, 3, figsize=(W2, 50 * mm_to_inch), gridspec_kw={"wspace": 0.40})
col = [GREY, C_CLEAN, C_CLEAN, C_Q20, C_444, C_444, C_PROXY]
for n, hv, dv, c in zip(names, hf, dens, col):
    axes[0].plot(hv, dv, "o", ms=4.2, color=c, mec="white", mew=0.5, zorder=3)
ann = [("clean", (0, 4), "center"), ("q80", (5, 3), "left"), ("q50", (5, -9), "left"),
       ("q20", (-3, -10), "right"), ("q20 4:4:4", (5, 3), "left"),
       ("blur", (-5, 5), "right"), ("noise", (-5, 5), "right")]
for (lab, off, ha), hv, dv in zip(ann, hf, dens):
    axes[0].annotate(lab, (hv, dv), xytext=off, textcoords="offset points", fontsize=6.8, ha=ha)
axes[0].set_xlabel("high-frequency energy ratio")
axes[0].set_ylabel("relative primitive count")
axes[0].set_xlim(0.74, 1.12); axes[0].set_ylim(0.855, 1.025)
axes[0].text(0.755, 1.012, "$R^2$ = 0.28", fontsize=7.0, color=C_OTHER)

xfr = np.arange(4)
for xi, v, c in zip(xfr, frac, [GREY, C_Q20, C_444, C_444]):
    axes[1].plot([xi, xi], [0, v], color=c, lw=1.4, solid_capstyle="butt")
    axes[1].plot([xi], [v], "o", color=c, ms=4.0, mec="white", mew=0.5, zorder=3)
    axes[1].annotate(f"{v:.4f}", (xi, v), xytext=(0, 4), textcoords="offset points",
                     ha="center", fontsize=6.8, color=c)
axes[1].set_xticks(xfr); axes[1].set_xticklabels(frac_labels, fontsize=6)
axes[1].set_ylabel(r"fraction of gradients above $\tau$")
axes[1].set_ylim(0, 0.086)
axes[1].set_xlabel("training input")

fx = np.array(frac) / frac[0]
axes[2].plot([0.86, 1.01], [0.86, 1.01], ls=(0, (4, 2)), lw=0.6, color=C_OTHER)
for xi, yi, c in zip(fx, dens_f, [GREY, C_Q20, C_444, C_444]):
    axes[2].plot(xi, yi, "o", ms=4.2, color=c, mec="white", mew=0.5, zorder=3)
axes[2].set_xlabel(r"relative fraction above $\tau$")
axes[2].set_ylabel("relative primitive count")
axes[2].set_xlim(0.855, 1.015); axes[2].set_ylim(0.855, 1.015)
axes[2].text(0.862, 1.000, "1:1", fontsize=7.0, color=C_OTHER)
for ax, lab, off in [(axes[0], "(a)", -0.30), (axes[1], "(b)", -0.30), (axes[2], "(c)", -0.30)]:
    panel_label(ax, lab, x=off)
save(fig, "fig06_mechanism")

# ============================ Fig. 7  gradient quantiles ============================
conds = ["clean", "q20", "q20 4:4:4"]
p50 = [6.59e-6, 8.06e-6, 8.02e-6]
p90 = [1.44e-4, 1.35e-4, 1.35e-4]
p99 = [6.50e-4, 5.80e-4, 5.79e-4]
fig, ax = plt.subplots(figsize=(W1, 50 * mm_to_inch))
xc = np.arange(3)
series = [(p50, "p50", "o", C_CLEAN, -0.14), (p90, "p90", "s", C_444, 0.0), (p99, "p99", "^", C_Q20, 0.14)]
for vals, lab, mk, c, dx in series:
    ax.plot(xc + dx, vals, mk + "-", color=c, ms=3.8, lw=0.9, label=lab, mfc="white", mew=0.9)
ax.set_yscale("log")
ax.set_xticks(xc); ax.set_xticklabels(conds)
ax.set_xlabel("training input")
ax.set_ylabel("screen-space gradient norm")
ax.set_xlim(-0.55, 2.55)
ax.set_ylim(3e-6, 1.6e-3)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, handlelength=1.2,
          columnspacing=1.2, fontsize=7)
ax.annotate("median rises", (1 - 0.14, 8.06e-6), xytext=(-2, 14), textcoords="offset points",
            fontsize=6.8, color=C_CLEAN, ha="center")
ax.annotate("tail trimmed", (2.14, 5.79e-4), xytext=(0, 14), textcoords="offset points",
            fontsize=6.8, color=C_Q20, ha="center")
fig.subplots_adjust(left=0.26, right=0.96, top=0.97, bottom=0.30)
save(fig, "fig07_quantiles", tight=False)

# ============================ Fig. 8  spatial control ============================
edge = [-0.30, -2.53, -1.37, -1.07]
floater = [2.14, 1.86, 1.56, 1.29]
needle = [3.52, 3.00, 2.72, 2.25]
cov = [131.15, 120.77, 124.64]
qp4 = ["clean", "q80", "q50", "q20"]
fig, axes = plt.subplots(1, 3, figsize=(W2, 48 * mm_to_inch), gridspec_kw={"wspace": 0.48})
xe = np.arange(4)
axes[0].bar(xe, edge, color=C_CLEAN, width=0.6, lw=0)
axes[0].axhline(0, color=BLACK, lw=0.6)
axes[0].set_xticks(xe); axes[0].set_xticklabels(qp4, fontsize=7.0)
axes[0].set_ylabel("edge ratio $-$ random (pp)")
axes[0].set_ylim(-3.05, 0.55)
for xi, v in zip(xe, edge):
    axes[0].annotate(f"{v:.2f}", (xi, v), xytext=(0, -9 if v < -2 else 4), textcoords="offset points",
                     ha="center", fontsize=6.8, color=BLACK)
w = 0.34
axes[1].bar(xe - w / 2 - 0.01, floater, w, color=GREY, lw=0)
axes[1].bar(xe + w / 2 + 0.01, needle, w, color="#555555", lw=0)
axes[1].set_xticks(xe); axes[1].set_xticklabels(qp4, fontsize=7.0)
axes[1].set_ylabel("primitives affected (%)")
axes[1].set_ylim(0, 4.75)
axes[1].annotate("floater", (-w / 2 - 0.01, floater[0]), xytext=(0, 3), textcoords="offset points",
                 ha="center", va="bottom", fontsize=6.9, color="#777777")
axes[1].annotate("needle", (w / 2 + 0.01, needle[0]), xytext=(0, 3), textcoords="offset points",
                 ha="center", va="bottom", fontsize=6.9, color="#333333")
axes[2].bar(np.arange(3), cov, color=[C_CLEAN, C_Q20, C_444], width=0.55, lw=0)
axes[2].set_xticks(np.arange(3)); axes[2].set_xticklabels(["clean", "q20", "q20 4:4:4"], fontsize=6)
axes[2].set_ylabel(r"analytic coverage $\Sigma\pi r^2/(HW)$")
axes[2].set_ylim(0, 162)
for i, v in enumerate(cov):
    axes[2].annotate(f"{v:.1f}", (i, v), xytext=(0, 3), textcoords="offset points",
                     ha="center", fontsize=6.8, color=BLACK)
for ax, lab, off in [(axes[0], "(a)", -0.34), (axes[1], "(b)", -0.30), (axes[2], "(c)", -0.32)]:
    panel_label(ax, lab, x=off)
save(fig, "fig08_spatial")

# ============================ Fig. 9  additive decomposition ============================
clean0, quant, chroma = 28.726, 0.624, 0.195
pred = clean0 - quant - chroma
measured = 27.908
fig, ax = plt.subplots(figsize=(W1, 58 * mm_to_inch))
ax.bar(0, clean0 - 27.4, bottom=27.4, width=0.58, color=C_CLEAN, lw=0)
ax.bar(1, quant, bottom=clean0 - quant, width=0.58, color="#4393C3", lw=0)
ax.bar(2, chroma, bottom=pred, width=0.58, color=C_444, lw=0)
ax.bar(3, pred - 27.4, bottom=27.4, width=0.58, color="#BBBBBB", lw=0)
ax.plot([3], [measured], "D", ms=4.4, color=C_Q20, zorder=5, mec="white", mew=0.6)
ax.annotate(f"measured {measured:.3f}", (3, measured), xytext=(12, -16), textcoords="offset points",
            fontsize=6.9, color=C_Q20, ha="left")
ax.annotate(f"{clean0:.3f}", (0, clean0), xytext=(0, 4), textcoords="offset points",
            ha="center", fontsize=6.9, color=C_CLEAN)
ax.annotate(f"$-$ {quant:.3f}", (1, clean0 - quant / 2), ha="center", va="center",
            fontsize=6.9, color="white", fontweight="bold")
ax.annotate(f"$-$ {chroma:.3f}", (2, pred + chroma / 2), ha="center", va="center",
            fontsize=6.9, color="white", fontweight="bold")
ax.annotate(f"{pred:.3f}", (3, pred), xytext=(0, 5), textcoords="offset points",
            ha="center", fontsize=6.9, color=BLACK)
ax.axhline(measured, color=C_Q20, lw=0.6, ls=(0, (4, 2)))
ax.set_xticks(range(4))
ax.set_xticklabels(["clean\n(q98, 4:2:0)", "block\nquantisation", "chroma\nsubsampling",
                    "predicted\n(sum)"], fontsize=6.9)
ax.set_ylabel("PSNR on original test views (dB)")
ax.set_ylim(27.4, 28.98)
ax.set_xlim(-0.55, 3.95)
ax.set_xlabel("codec factors, added one at a time")
save(fig, "fig09_decomposition")

# ============================ Fig. 10  remediation routes ============================
routes = ["deblock", "statistical\ngating", "threshold\ntuning", "render prior\n(POCS)",
          "render prior\n(Wiener)", "loss\ndomain"]
vals = [27.700, 27.559, 27.824, 28.427, 28.639, 26.944]
fig, ax = plt.subplots(figsize=(W2C, 56 * mm_to_inch))
xr = np.arange(len(routes))
for xi, v in zip(xr, vals):
    c = C_444 if v < 27.908 else C_GOOD
    ax.plot([xi, xi], [27.908, v], color=c, lw=0.9, ls=(0, (2, 2)), zorder=2)
    ax.plot([xi], [v], "o", ms=4.4, color=c, mec="white", mew=0.6, zorder=3)
    ax.annotate(f"{v:.3f}", (xi + 0.13, v), ha="left", va="center", fontsize=6.9, color=c)
ax.axhline(27.908, color=BLACK, lw=0.6, ls=(0, (1, 1)))
ax.axhline(28.564, color=C_CLEAN, lw=0.9)
ax.axhline(28.931, color=C_GOOD, lw=0.9)
ax.text(-0.52, 27.918, "3k baseline", fontsize=6.9, va="bottom", ha="left", color=BLACK)
ax.text(-0.52, 28.566, "C1: continuation on the compressed target", fontsize=6.9, va="bottom",
        ha="left", color=C_CLEAN)
ax.text(-0.52, 28.930, "C4: clean target accessible", fontsize=6.9, va="bottom", ha="left", color=C_GOOD)
ax.set_xticks(xr); ax.set_xticklabels(routes, fontsize=6.9)
ax.set_ylabel("PSNR on original test views (dB)")
ax.set_xlim(-0.55, len(routes) - 0.30)
ax.set_ylim(26.72, 29.12)
ax.set_xlabel("post-hoc route (all consume the same compressed observations)")
save(fig, "fig10_routes")

# ============================ Fig. 11  density elasticity ============================
within = [0.033, 0.051, 0.194, 0.075]
within_lab = [r"$\tau$ 2e-4 $\rightarrow$ 5e-4", r"$\tau$ 5e-4 $\rightarrow$ 1e-3",
              r"$\tau$ 1e-3 $\rightarrow$ 2e-3", r"full $\tau$ range"]
across = [0.75, 1.06, 1.44, 2.14]
across_lab = ["3k", "7k", "15k", "30k"]
fig, ax = plt.subplots(figsize=(W1, 62 * mm_to_inch))
ypos, labels = [], []
yy = 0
for lab, v in zip(within_lab, within):
    ax.plot([0.02, v], [yy, yy], color=C_CLEAN, lw=1.0)
    ax.plot([v], [yy], "o", ms=3.8, color=C_CLEAN, mec="white", mew=0.5, zorder=3)
    ax.annotate(f"{v:.3f}", (v, yy), xytext=(4, 0), textcoords="offset points",
                fontsize=6.8, color=C_CLEAN, va="center")
    ypos.append(yy); labels.append(lab); yy += 1
yy += 1.1
for lab, v in zip(across_lab, across):
    ax.plot([0.02, v], [yy, yy], color=C_Q20, lw=1.0)
    ax.plot([v], [yy], "o", ms=3.8, color=C_Q20, mec="white", mew=0.5, zorder=3)
    ax.annotate(f"{v:.2f}", (v, yy), xytext=(4, 0), textcoords="offset points",
                fontsize=6.8, color=C_Q20, va="center")
    ypos.append(yy); labels.append("q20 vs clean, " + lab); yy += 1
ax.set_yticks(ypos); ax.set_yticklabels(labels, fontsize=6)
ax.set_xscale("log")
ax.set_xlim(0.02, 8.0)
ax.set_xticks([0.03, 0.1, 0.3, 1, 3])
ax.set_xticklabels(["0.03", "0.1", "0.3", "1", "3"])
ax.minorticks_off()
ax.set_xlabel("elasticity of PSNR to density (dB per 10% primitives)")
ax.set_ylim(-0.9, yy - 0.2)
ax.axhspan(-0.7, len(within_lab) - 0.35, color=C_CLEAN, alpha=0.06, lw=0)
ax.axhspan(len(within_lab) + 0.55, yy - 0.75, color=C_Q20, alpha=0.06, lw=0)
ax.text(0.0215, -0.75, "same input, density moved by $\\tau$", fontsize=6.9,
        color=C_CLEAN, va="bottom", ha="left")
ax.text(0.0215, len(within_lab) + 0.62, "different inputs, q20 vs clean", fontsize=6.9,
        color=C_Q20, va="bottom", ha="left")
save(fig, "fig11_elasticity")

# ============================ Fig. 12  qualitative comparison ============================
A = np.asarray(Image.open(r"D:\3dgs\proj-D6\renders\00000.png").convert("RGB")).astype(np.float32) / 255
B = np.asarray(Image.open(r"D:\3dgs\proj-D6\renders\q20\00000.png").convert("RGB")).astype(np.float32) / 255
hh = min(A.shape[0], B.shape[0]); ww = min(A.shape[1], B.shape[1])
A, B = A[:hh, :ww], B[:hh, :ww]
D = np.abs(A - B).mean(axis=2)
fig, axes = plt.subplots(2, 3, figsize=(W2, 96 * mm_to_inch), gridspec_kw={"wspace": 0.03, "hspace": 0.10})
labels_top = ["clean-trained model", "q20-trained model", r"$|$difference$|$ $\times$8"]
imgs = [A, B, np.clip(D * 8, 0, 1)]
for j, (lab, im) in enumerate(zip(labels_top, imgs)):
    ax = axes[0, j]
    ax.imshow(im, cmap="magma" if j == 2 else None, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel(lab, fontsize=7, labelpad=3)
y0 = int(hh * 0.30); y1 = y0 + int(hh * 0.26)
x0 = int(ww * 0.34); x1 = x0 + int(ww * 0.26)
for j, im in enumerate(imgs):
    ax = axes[1, j]
    ax.imshow(im[y0:y1, x0:x1], cmap="magma" if j == 2 else None, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(BLACK); s.set_linewidth(0.7); s.set_visible(True)
    ax.set_xlabel("zoom", fontsize=7, labelpad=3)
for j, lab in enumerate(["(a)", "(b)", "(c)"]):
    panel_label(axes[0, j], lab, x=-0.05, y=1.02)
for j, lab in enumerate(["(d)", "(e)", "(f)"]):
    panel_label(axes[1, j], lab, x=-0.05, y=1.02)
save(fig, "fig12_qualitative")

print("OUT:", OUT)



