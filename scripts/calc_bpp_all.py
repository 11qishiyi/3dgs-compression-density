"""全场景全档位 effective bpp 台账 → bpp.csv"""
import csv
import os

from PIL import Image

ROOT = r"D:\3dgs\proj-D6\cluster\data\mip360"
OUT = r"D:\3dgs\proj-D6\bpp.csv"

rows = []
for d in sorted(os.listdir(ROOT)):
    img_dir = os.path.join(ROOT, d, "images")
    if not os.path.isdir(img_dir):
        continue
    names = sorted(f for f in os.listdir(img_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    train = [f for i, f in enumerate(names) if i % 8 != 0]
    if not train:
        continue
    im = Image.open(os.path.join(img_dir, train[0]))
    w, h = im.size
    tot = sum(os.path.getsize(os.path.join(img_dir, f)) for f in train)
    bpp = tot * 8 / (w * h * len(train))
    scene = d.split("_r4res")[0]
    rows.append([d, scene, len(names), len(train), f"{w}x{h}",
                 round(tot / 1e6, 1), round(bpp, 3)])

with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["dataset", "scene", "images", "train", "size", "train_MB", "bpp"])
    w.writerows(rows)

for r in rows:
    print(f"{r[0]:34s}{r[4]:>10s}{r[6]:>8.3f}")
print("wrote", OUT)
