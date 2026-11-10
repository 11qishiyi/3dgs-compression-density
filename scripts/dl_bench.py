"""Download one Tanks&Temples / DeepBlending scene from the HF mirror.

Usage: python dl_bench.py <truck|train|playroom|drjohnson>

Writes data/bench/<scene>/{images,sparse} next to this script.
"""
import json
import os
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

REPOS = {
    "truck": "nvs-bench/tanksandtemples",
    "train": "nvs-bench/tanksandtemples",
    "playroom": "nvs-bench/deepblending",
    "drjohnson": "nvs-bench/deepblending",
}
API = "https://hf-mirror.com/api/datasets/{repo}/tree/main?recursive=true"
BASE = "https://hf-mirror.com/datasets/{repo}/resolve/main/"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "bench")

scene = sys.argv[1]
repo = REPOS[scene]
os.makedirs(OUT, exist_ok=True)

with urllib.request.urlopen(API.format(repo=repo), timeout=90) as r:
    tree = json.load(r)

rels = []
for f in tree:
    p = f.get("path", "")
    if f.get("type") != "file" or not p.startswith(scene + "/"):
        continue
    if "/images/" in p or "/sparse/" in p:
        rels.append(p)
print(f"{scene}: {len(rels)} files to fetch")
missing = []


def fetch(rel):
    dst = os.path.join(OUT, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return
    url = BASE.format(repo=repo) + urllib.parse.quote(rel)
    last = None
    for _ in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(dst, "wb") as f:
                f.write(r.read())
            return
        except Exception as e:  # noqa: BLE001
            last = e
    missing.append((rel, str(last)))


with ThreadPoolExecutor(max_workers=8) as ex:
    list(ex.map(fetch, rels))

n_img = sum(1 for p in rels if "/images/" in p)
n_sp = sum(1 for p in rels if "/sparse/" in p)
print(f"done: images={n_img} sparse={n_sp} missing={len(missing)}")
for m in missing[:5]:
    print("MISSING", m)
