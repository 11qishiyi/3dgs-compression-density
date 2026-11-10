#!/usr/bin/env python3
"""TVC 数字审计：把稿件表格数字与 runs.csv / metrics.csv / 本地数据集逐项对账。

用法: python audit_tvc_numbers.py [manuscript.md]
输出: PASS/FAIL 列表 + 关键重算值。
"""
import csv
import glob
import math
import os
import re
import statistics as st
import sys

RUNS = r"D:\3dgs\proj-D6\runs.csv"
METRICS = r"D:\3dgs\proj-D6\metrics.csv"
DATA = r"D:\3dgs\proj-D6\cluster\data\mip360"
MS = sys.argv[1] if len(sys.argv) > 1 else r"D:\3dgs\proj-D6\TVC-en-v2.md"

rows = list(csv.DictReader(open(RUNS, encoding="utf-8")))
by = {r["run_id"]: r for r in rows}


def f(x):
    try:
        return float(x)
    except Exception:
        return None


def psnr(tag):
    return f(by[tag]["psnr_test"])


def gauss(tag):
    return f(by[tag]["gaussians"])


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def sd(vals):
    vals = [v for v in vals if v is not None]
    return st.pstdev(vals) if len(vals) > 1 else 0.0


checks = []


def chk(name, got, want, tol):
    ok = got is not None and want is not None and abs(got - want) <= tol
    checks.append((ok, name, got, want))


# ---------- Table 4: quality means ----------
groups = {
    ("bonsai", 3000, "clean"): ["fast3k_clean", "clean_3k_s1", "clean_3k_s2", "clean_3k_s3"],
    ("bonsai", 3000, "q80"): ["fast3k_q80", "q80_hyb_3k_s1", "q80_hyb_3k_s2"],
    ("bonsai", 3000, "q50"): ["fast3k_q50", "q50_hyb_3k_s1", "q50_hyb_3k_s2"],
    ("bonsai", 3000, "q20"): ["fast3k_q20", "q20_hyb_3k_s1", "q20_hyb_3k_s2", "q20_hyb_3k_s3"],
    ("bonsai", 7000, "clean"): ["clean_7k_t251", "clean_7k_s1", "bonsai_clean_7k_s2"],
    ("bonsai", 7000, "q20"): ["q20_hyb_7k_t251", "q20_hyb_7k_s1", "bonsai_q20_7k_s2"],
    ("bonsai", 15000, "clean"): ["clean_15k", "clean_15k_s1", "bonsai_clean_15k_s2"],
    ("bonsai", 15000, "q20"): ["q20_15k", "q20_hyb_15k_s1", "bonsai_q20_15k_s2"],
    ("bonsai", 30000, "clean"): ["bonsai_clean_30k", "bonsai_clean_30k_s1", "bonsai_clean_30k_s2"],
    ("bonsai", 30000, "q20"): ["q20_hyb_30k", "bonsai_q20_30k_s1", "bonsai_q20_30k_s2"],
    ("counter", 3000, "clean"): ["counter_clean_3k", "counter_clean_3k_s1", "counter_clean_3k_s2"],
    ("counter", 3000, "q20"): ["counter_q20_hyb_3k", "counter_q20_3k_s1", "counter_q20_3k_s2"],
    ("counter", 7000, "clean"): ["counter_clean_7k", "counter_clean_7k_s1", "counter_clean_7k_s2"],
    ("counter", 7000, "q20"): ["counter_q20_hyb_7k", "counter_q20_hyb_7k_s1", "counter_q20_hyb_7k_s2"],
    ("counter", 30000, "clean"): ["counter_clean_30k", "counter_clean_30k_s1", "counter_clean_30k_s2"],
    ("counter", 30000, "q20"): ["counter_q20_hyb_30k", "counter_q20_hyb_30k_s1", "counter_q20_hyb_30k_s2"],
    ("bicycle", 7000, "clean"): ["bicycle_clean_7k", "bicycle_clean_7k_s1", "bicycle_clean_7k_s2"],
    ("bicycle", 7000, "q20"): ["bicycle_q20_hyb_7k", "bicycle_q20_hyb_7k_s1", "bicycle_q20_hyb_7k_s2"],
    ("garden", 3000, "clean"): ["garden_clean_3k", "garden_clean_3k_s1", "garden_clean_3k_s2"],
    ("garden", 3000, "q20"): ["garden_q20_hyb_3k", "garden_q20_3k_s1", "garden_q20_3k_s2"],
    ("truck", 3000, "clean"): ["truck_clean_3k_s1", "truck_clean_3k_s2", "truck_clean_3k_s3"],
    ("truck", 3000, "q20"): ["truck_q20_3k_s1", "truck_q20_3k_s2", "truck_q20_3k_s3"],
    ("truck", 7000, "clean"): ["truck_clean_7k_s1", "truck_clean_7k_s2", "truck_clean_7k_s3"],
    ("truck", 7000, "q20"): ["truck_q20_7k_s1", "truck_q20_7k_s2", "truck_q20_7k_s3"],
    ("playroom", 3000, "clean"): ["playroom_clean_3k_s1", "playroom_clean_3k_s2", "playroom_clean_3k_s3"],
    ("playroom", 3000, "q20"): ["playroom_q20_3k_s1", "playroom_q20_3k_s2", "playroom_q20_3k_s3"],
    ("playroom", 7000, "clean"): ["playroom_clean_7k_s1", "playroom_clean_7k_s2", "playroom_clean_7k_s3"],
    ("playroom", 7000, "q20"): ["playroom_q20_7k_s1", "playroom_q20_7k_s2", "playroom_q20_7k_s3"],
    ("kitchen", 3000, "clean"): ["kitchen_clean_3k", "kitchen_clean_3k_s1", "kitchen_clean_3k_s2"],
    ("kitchen", 3000, "q20"): ["kitchen_q20_hyb_3k", "kitchen_q20_3k_s1", "kitchen_q20_3k_s2"],
    ("room", 3000, "clean"): ["room_clean_3k", "room_clean_3k_s1", "room_clean_3k_s2"],
    ("room", 3000, "q20"): ["room_q20_hyb_3k", "room_q20_3k_s1", "room_q20_3k_s2"],
    ("stump", 3000, "clean"): ["stump_clean_3k", "stump_clean_3k_s1", "stump_clean_3k_s2"],
    ("stump", 3000, "q20"): ["stump_q20_hyb_3k", "stump_q20_3k_s1", "stump_q20_3k_s2"],
}
means = {}
gmean = {}
for k, ids in groups.items():
    means[k] = mean([psnr(i) for i in ids if i in by])
    gmean[k] = mean([gauss(i) for i in ids if i in by])

# published table values (from manuscript Table 4)
t4 = {
    ("bonsai", 3000): (28.848, 27.997, -0.852, -11.3),
    ("bonsai", 7000): (30.251, 29.022, -1.229, -11.6),
    ("bonsai", 15000): (31.624, 29.850, -1.774, -12.3),
    ("bonsai", 30000): (32.771, 30.450, -2.321, -10.8),
    ("counter", 3000): (26.248, 25.734, -0.514, -13.0),
    ("counter", 7000): (27.542, 26.734, -0.808, -19.6),
    ("counter", 30000): (29.358, 27.929, -1.430, -20.2),
    ("bicycle", 7000): (23.733, 23.608, -0.125, -8.6),
    ("kitchen", 3000): (27.768, 26.208, -1.560, -18.4),
    ("garden", 3000): (24.670, 24.208, -0.462, -13.5),
    ("truck", 3000): (22.823, 22.713, -0.110, -13.9),
    ("truck", 7000): (23.913, 23.689, -0.224, -20.3),
    ("playroom", 3000): (27.944, 27.920, -0.024, -9.8),
    ("playroom", 7000): (29.686, 29.208, -0.477, -15.5),
    ("room", 3000): (28.895, 28.287, -0.608, -15.0),
    ("stump", 3000): (23.843, 23.734, -0.109, -1.0),
}
for (sc, b), (c, q, d, dg) in t4.items():
    chk(f"T4 {sc}{b//1000}k clean", means.get((sc, b, "clean")), c, 0.002)
    chk(f"T4 {sc}{b//1000}k q20", means.get((sc, b, "q20")), q, 0.002)
    if means.get((sc, b, "clean")) and means.get((sc, b, "q20")):
        chk(f"T4 {sc}{b//1000}k delta", means[(sc, b, "q20")] - means[(sc, b, "clean")], d, 0.003)
    if dg is not None and gmean.get((sc, b, "clean")) and gmean.get((sc, b, "q20")):
        got = (gmean[(sc, b, "q20")] - gmean[(sc, b, "clean")]) / gmean[(sc, b, "clean")] * 100
        chk(f"T4 {sc}{b//1000}k density%", got, dg, 0.15)

# ---------- Table 3: four QP at 3k ----------
t3 = {"clean": (28.848, 28.7263, 0.9145, 0.2072), "q80": (28.660, 28.5847, 0.9085, 0.2177),
      "q50": (28.444, 28.3281, 0.9006, 0.2327), "q20": (27.997, 27.9081, 0.8805, 0.2696)}
m3 = {r["run_id"]: r for r in csv.DictReader(open(METRICS, encoding="utf-8"))}
seed0 = {"clean": "clean_3k", "q80": "q80_hyb_3k", "q50": "q50_hyb_3k", "q20": "q20_hyb_3k"}
for qp, (p, p0, ssim, lpips) in t3.items():
    chk(f"T3 {qp} psnr-seed0", f(m3[seed0[qp]]["PSNR"]), p0, 0.002)
    chk(f"T3 {qp} ssim", f(m3[seed0[qp]]["SSIM"]), ssim, 0.0002)
    chk(f"T3 {qp} lpips", f(m3[seed0[qp]]["LPIPS"]), lpips, 0.0002)
    chk(f"T3 {qp} mean-PSNR", means[(("bonsai"), 3000, qp)] if ((("bonsai"), 3000, qp)) in means else None, p, 0.003)
# paired deltas (per-seed pairing, not difference of means)
pairs = {
    "q80": [("fast3k_clean", "fast3k_q80"), ("clean_3k_s1", "q80_hyb_3k_s1"),
            ("clean_3k_s2", "q80_hyb_3k_s2")],
    "q50": [("fast3k_clean", "fast3k_q50"), ("clean_3k_s1", "q50_hyb_3k_s1"),
            ("clean_3k_s2", "q50_hyb_3k_s2")],
    "q20": [("fast3k_clean", "fast3k_q20"), ("clean_3k_s1", "q20_hyb_3k_s1"),
            ("clean_3k_s2", "q20_hyb_3k_s2"), ("clean_3k_s3", "q20_hyb_3k_s3")],
}
want_paired = {"q80": -0.163, "q50": -0.379, "q20": -0.852}
for qp, ps in pairs.items():
    deltas = [psnr(b) - psnr(a) for a, b in ps]
    chk(f"T3 paired {qp}", mean(deltas), want_paired[qp], 0.003)

# ---------- Table 2: bpp / ratio from local datasets ----------
def bpp(ds, w, h):
    imgs = sorted(glob.glob(os.path.join(DATA, ds, "images", "*")))
    if not imgs:
        return None, None, None
    train = [p for i, p in enumerate(imgs) if i % 8 != 0]
    tb = sum(os.path.getsize(p) for p in train)
    return tb * 8 / (w * h * len(train)), len(train), len(imgs) - len(train)


t2 = {
    "bonsai": ("bonsai_r4res", "bonsai_r4res_q20_hyb", 779, 519, 0.441, 11.4),
    "counter": ("counter_r4res", "counter_r4res_q20_hyb", 778, 519, 0.483, 12.2),
    "garden": ("garden_r4res", "garden_r4res_q20_hyb", 1296, 840, 0.642, 13.4),
    "bicycle": ("bicycle_r4res", "bicycle_r4res_q20_hyb", 1236, 821, 0.656, 10.4),
    "kitchen": ("kitchen_r4res", "kitchen_r4res_q20_hyb", 778, 519, 0.570, 12.7),
    "stump": ("stump_r4res", "stump_r4res_q20_hyb", 1244, 825, 0.569, 11.1),
    "room": ("room_r4res", "room_r4res_q20_hyb", 778, 519, None, None),
}
for sc, (cd, qd, w, h, want_bpp, want_ratio) in t2.items():
    cb = bpp(cd, w, h)[0] if os.path.isdir(os.path.join(DATA, cd)) else None
    qb, nt, nte = bpp(qd, w, h) if os.path.isdir(os.path.join(DATA, qd)) else (None, None, None)
    if want_bpp is not None:
        chk(f"T2 {sc} q20 bpp", qb, want_bpp, 0.004)
        if cb and qb:
            chk(f"T2 {sc} ratio", cb / qb, want_ratio, 0.15)

# ---------- budget law fit ----------
its = [3000, 7000, 15000, 30000]
gaps = [means[("bonsai", b, "clean")] - means[("bonsai", b, "q20")] for b in its]
lx = [math.log2(b / 1000) for b in its]
n = len(lx); mx = sum(lx) / n; my = sum(gaps) / n
slope = sum((x - mx) * (y - my) for x, y in zip(lx, gaps)) / sum((x - mx) ** 2 for x in lx)
inter = my - slope * mx
pred = [inter + slope * x for x in lx]
r2 = 1 - sum((y - p) ** 2 for y, p in zip(gaps, pred)) / sum((y - my) ** 2 for y in gaps)
chk("fit intercept", inter, 0.074, 0.02)
chk("fit slope", slope, 0.445, 0.01)
chk("fit R2", r2, 0.984, 0.005)

# ---------- Table 5 decomposition ----------
chk("T5 quant", psnr("q20_444_3k_s0") - psnr("fast3k_clean"), -0.624, 0.002)
chk("T5 chroma", psnr("fast3k_q20") - psnr("q20_444_3k_s0"), -0.195, 0.002)
chk("T5 seed1 quant", psnr("q20_444_3k_s1") - psnr("clean_3k_s1"), -0.674, 0.002)
chk("T5 seed1 chroma", psnr("q20_hyb_3k_s1") - psnr("q20_444_3k_s1"), -0.204, 0.002)
chk("T5 counter quant", psnr("counter_q20_444_3k") - psnr("counter_clean_3k"), -0.364, 0.002)
chk("T5 counter chroma", psnr("counter_q20_hyb_3k") - psnr("counter_q20_444_3k"), -0.200, 0.002)

# ---------- Table 7 / 8: re-estimated series with corrected quantization tables ----------
chk("T7 deblock", psnr("fast3k_deblock"), 27.700, 0.002)
chk("T7 gate", psnr("fast3k_c3_q20"), 27.559, 0.002)
chk("T7 oracle best", max(psnr("oracle_tau_1e-4_3k"), psnr("oracle_tau_5e-4_3k"),
                          psnr("oracle_tau_1e-3_3k"), psnr("oracle_tau_2e-3_3k")), 27.824, 0.002)
chk("T7 POCS", psnr("f1_ft_c2"), 28.427, 0.003)
chk("T7 Wiener", psnr("f1_ft_c3"), 28.639, 0.003)
chk("T7 C1", psnr("f1_ft_c1"), 28.564, 0.003)
chk("T7 C4", psnr("f1_ft_c4"), 28.931, 0.003)
chk("T7 free DCT-L2 uniform", psnr("f1_dct_uniform_3k"), 26.944, 0.003)
chk("T7 free DCT-L2 weighted", psnr("f1_dct_weighted_3k"), 26.939, 0.003)
chk("T8 L2", psnr("f1_ft4_l2"), 28.625, 0.003)
chk("T8 DCTu", psnr("f1_ft4_dct_u"), 28.631, 0.003)
chk("T8 DCTw", psnr("f1_ft4_dct_w"), 28.649, 0.003)
chk("T8 interval 0.003", psnr("f1_ft_h003"), 28.572, 0.003)
chk("T8 interval 0.01", psnr("f1_ft_h01"), 28.587, 0.003)
# prose numbers of the re-estimated series
chk("prose continuation gain", psnr("f1_ft_c1") - psnr("f1_c0"), 0.670, 0.003)
chk("prose total recovery", psnr("f1_ft_c4") - psnr("f1_c0"), 1.037, 0.003)
chk("prose headroom", psnr("f1_ft_c4") - psnr("f1_ft_c1"), 0.367, 0.003)
chk("prose POCS delta", psnr("f1_ft_c2") - psnr("f1_ft_c1"), -0.137, 0.003)
chk("prose Wiener delta", psnr("f1_ft_c3") - psnr("f1_ft_c1"), 0.075, 0.003)
chk("prose best fixed-topology", psnr("f1_ft4_dct_w") - psnr("f1_ft_c1"), 0.085, 0.003)
# lambda probe percentages quoted in Section 5.6
gbase = gauss("q20_hyb_3k_s1")
for _tag, _want in (("f1_a_hinge003_3k_s1", 4), ("f1_a_hinge01_3k_s1", 11),
                    ("f1_a_hinge03_3k_s1", 34), ("f1_a_hinge30_3k_s1", 378)):
    chk(f"prose probe {_tag} density%", round(100 * (gauss(_tag) / gbase - 1)), _want, 1)
chk("prose lambda-1 psnr", psnr("f1_a_hinge100_3k_s1"), 25.404, 0.01)
chk("prose dequant % of headroom",
    100 * max(psnr("f1_ft_c2") - psnr("f1_ft_c1"), psnr("f1_ft_c3") - psnr("f1_ft_c1"))
    / (psnr("f1_ft_c4") - psnr("f1_ft_c1")),
    21, 1.0)

# ---------- Table 8 elasticity ----------
def elast(d_psnr, d_pct):
    return abs(d_psnr) / (abs(d_pct) / 10)


def g(tag):
    return gauss(tag)


tau_runs = [("oracle_tau_5e-4_3k", 5e-4), ("oracle_tau_1e-3_3k", 1e-3),
            ("oracle_tau_2e-3_3k", 2e-3)]
base = ("fast3k_q20", psnr("fast3k_q20"), gauss("fast3k_q20"))
seq = [base] + [(t, psnr(t), gauss(t)) for t, _ in tau_runs]
for i in range(1, len(seq)):
    dp = seq[i][1] - seq[i - 1][1]
    dg = (seq[i][2] - seq[i - 1][2]) / seq[i - 1][2] * 100
    chk(f"T8 within step{i} density%", dg, [-58.9, -27.6, -11.2][i - 1], 0.3)
    chk(f"T8 within step{i} dPSNR", dp, [-0.197, -0.140, -0.217][i - 1], 0.002)
    chk(f"T8 within step{i} elast", elast(dp, dg), [0.033, 0.051, 0.194][i - 1], 0.002)
# across inputs
for b, want in [(3000, 0.75), (7000, 1.06), (15000, 1.44), (30000, 2.14)]:
    c = means[("bonsai", b, "clean")]; q = means[("bonsai", b, "q20")]
    dgg = (gmean[("bonsai", b, "q20")] - gmean[("bonsai", b, "clean")]) / gmean[("bonsai", b, "clean")] * 100
    chk(f"T8 cross {b}", elast(q - c, dgg), want, 0.02)

# ---------- parse manuscript tables and compare ----------
txt = open(MS, encoding="utf-8").read()
body = txt[:txt.rfind("## References")]


def num(s):
    s = s.replace("−", "-").replace("\\", "").strip()
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group()) if m else None


# Table 2 rows
t2_rows = re.findall(r"^\| Mip-NeRF 360 \| (\w+) \| [^|]+ \| [^|]+ \| ([\d.]+) \| ([\d.]+) \| ([\d.]+)×", body, re.M)
for sc, clean_s, bpp_s, ratio_s in t2_rows:
    if sc in t2:
        cd, qd, w, h, want_bpp, want_ratio = t2[sc]
        qb = bpp(qd, w, h)[0] if os.path.isdir(os.path.join(DATA, qd)) else None
        cb = bpp(cd, w, h)[0] if os.path.isdir(os.path.join(DATA, cd)) else None
        if cb is not None:
            chk(f"MS T2 {sc} clean bpp", float(clean_s), cb, 0.02)
        if qb is not None:
            chk(f"MS T2 {sc} bpp", float(bpp_s), qb, 0.004)
        if cb and qb:
            chk(f"MS T2 {sc} ratio", float(ratio_s), cb / qb, 0.15)

# Table 4 rows
t4_rows = re.findall(r"^\| (\w+) (\d+)k \| ([\d.]+)(?: ± [\d.]+)? \| ([\d.]+)(?: ± [\d.]+)? \| (−?[\d.]+) \| (−?[\d.]+)% \| (\d+) \|", body, re.M)
if not t4_rows:
    t4_rows = re.findall(r"^\| (\w+) (\d+)k \| ([\d.]+)[^|]*\| ([\d.]+) \| (−?[\d.]+) \| (−?[\d.]+)% \|", body, re.M)
for sc, b_s, c, q, d, dg, _n in t4_rows:
    b = int(b_s) * 1000
    key = (sc, b)
    if key not in t4:
        continue
    c, q, d, dg = float(c), float(q), num(d), num(dg)
    chk(f"MS T4 {sc}{b_s}k clean", means.get((sc, b, "clean")), c, 0.002)
    chk(f"MS T4 {sc}{b_s}k q20", means.get((sc, b, "q20")), q, 0.002)
    chk(f"MS T4 {sc}{b_s}k delta", (means.get((sc, b, "q20")) or 0) - (means.get((sc, b, "clean")) or 0), d, 0.004)
    if gmean.get((sc, b, "clean")) and gmean.get((sc, b, "q20")):
        got = (gmean[(sc, b, "q20")] - gmean[(sc, b, "clean")]) / gmean[(sc, b, "clean")] * 100
        chk(f"MS T4 {sc}{b_s}k density%", got, dg, 0.15)

# Table 8 cross-input rows
t8_rows = re.findall(r"^\| across inputs: (\d+)k \| (−?[\d.]+)% \| (−?[\d.]+) \| ([\d.]+) \|", body, re.M)
for b_s, dg, d, e in t8_rows:
    b = int(b_s) * 1000
    dg, d, e = num(dg), num(d), float(e)
    if ("bonsai", b, "clean") in means:
        c = means[("bonsai", b, "clean")]; q = means[("bonsai", b, "q20")]
        chk(f"MS T8 {b_s}k dPSNR", q - c, d, 0.004)
        got_dg = (gmean[("bonsai", b, "q20")] - gmean[("bonsai", b, "clean")]) / gmean[("bonsai", b, "clean")] * 100
        chk(f"MS T8 {b_s}k density%", got_dg, dg, 0.15)
        chk(f"MS T8 {b_s}k elast", abs(d) / (abs(got_dg) / 10), e, 0.02)

# ---------- Table 9: codec-aware comparison ----------
A_SCENE, B_SCENE = "bonsai_r4res_q20_hyb", "bonsai_mix_hyb"


def fm(tags):
    return mean([psnr(t) for t in tags if t in by])


def fg(tags):
    return mean([gauss(t) for t in tags if t in by])


t9_spec = {
    ("q20 baseline", "uniform"): (["q20_hyb_3k_s1", "q20_hyb_3k_s2"],
                                  ["q20_hyb_7k_s1", "bonsai_q20_7k_s2"]),
    ("interval, λ = 0.003", "uniform"): (["f1_a_hinge003_3k_s1", "f1_a_hinge003_3k_s2"],
                                         ["f1_a_hinge003_7k_s1", "f1_a_hinge003_7k_s2"]),
    ("interval, λ = 0.01", "uniform"): (["f1_a_hinge01_3k_s1", "f1_a_hinge01_3k_s2"],
                                        ["f1_a_hinge01_7k_s1"]),
    ("mixed baseline", "mixed"): (["f1_b_base_3k_s1", "f1_b_base_3k_s2"],
                                  ["f1_b_base_7k_s1", "f1_b_base_7k_s2"]),
    ("interval, λ = 0.003", "mixed"): (["f1_b_hinge_3k_s1", "f1_b_hinge_3k_s2"],
                                       ["f1_b_hinge_7k_s1", "f1_b_hinge_7k_s2"]),
    ("frame-weighted", "mixed"): (["f1_b_frame_3k_s1", "f1_b_frame_3k_s2"],
                                  ["f1_b_frame_7k_s1", "f1_b_frame_7k_s2"]),
}
ms = open(MS, encoding="utf-8").read()
mbody = ms[:ms.rfind("## References")]
rows9 = re.findall(r"^\| ([^|]+?) \| (uniform q20|q20/q50) \| ([\d.]+) \| ([\d]+) \| ([\d.]+) \| ([\d]+) \|", mbody, re.M)
for arm, view, p3, g3, p7, g7 in rows9:
    arm = arm.strip()
    key = (arm, "mixed" if view == "q20/q50" else "uniform")
    if key not in t9_spec:
        continue
    tags3, tags7 = t9_spec[key]
    chk(f"MS T9 {arm}/{view} 3k psnr", fm(tags3), float(p3), 0.02)
    chk(f"MS T9 {arm}/{view} 3k g", round(fg(tags3) / 1000), float(g3), 1.0)
    chk(f"MS T9 {arm}/{view} 7k psnr", fm(tags7), float(p7), 0.02)
    chk(f"MS T9 {arm}/{view} 7k g", round(fg(tags7) / 1000), float(g7), 1.0)
# Table 8 delta column
t8b = re.findall(r"^\| \((?:b|c|d|e|f)\) \| [^|]+ \| ([\d.]+) \| ([−+-][\d.]+) \|", mbody, re.M)
for val, delta in t8b:
    chk(f"MS T8 delta {val}", float(val) - psnr("f1_ft_c1"), float(delta.replace("−", "-")), 0.003)

# scene count claims
for pat in [r"across (\w+) Mip-NeRF 360 scenes", r"Across (\w+) Mip-NeRF 360 scenes",
            r"coverage is (\w+) Mip-NeRF 360 scenes"]:
    for m in re.finditer(pat, body):
        print(f"NOTE scene-count phrase: '{m.group(0)}'")

# ---------- report ----------
bad = [c for c in checks if not c[0]]
for ok, name, got, want in checks:
    if not ok:
        print(f"FAIL {name}: got={got} want={want}")
print(f"\n{len(checks) - len(bad)}/{len(checks)} checks passed")
print("recomputed: slope=%.3f intercept=%.3f R2=%.4f" % (slope, inter, r2))
print("gaps:", [round(x, 3) for x in gaps])

