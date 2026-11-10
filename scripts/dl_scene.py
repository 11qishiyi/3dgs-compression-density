"""从 HF 镜像并行下载 Mip-NeRF 360 场景（含 COLMAP sparse）。

用法:
  python dl_scene.py <scene> [limit]      # limit=0 表示全量
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

REPO = "YuanhaoXD/Mip-NeRF360"
BASE = f"https://hf-mirror.com/datasets/{REPO}/resolve/main/"
API = f"https://hf-mirror.com/api/datasets/{REPO}"
OUT = os.path.join(os.path.dirname(__file__), "data", "mip360")

scene = sys.argv[1]
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0


def get_tree():
    with urllib.request.urlopen(API, timeout=40) as r:
        return [s["rfilename"] for s in json.load(r)["siblings"]]


files = get_tree()
sel = [f for f in files
       if f.startswith(scene + "/")
       and ("/sparse/" in f or f.endswith("poses_bounds.npy")
            or (f.endswith((".JPG", ".jpg", ".png")) and "/images/" in f)
            or f.endswith("transforms.json"))]
sel.sort()
if limit:
    imgs = [f for f in sel if "/images/" in f][:limit]
    others = [f for f in sel if "/images/" not in f]
    sel = imgs + others
print(f"scene={scene} files={len(sel)}")

done = {"n": 0, "bytes": 0}
t0 = time.time()


def fetch(rel):
    dst = os.path.join(OUT, rel.replace("/", os.sep))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return os.path.getsize(dst), 0
    url = BASE + urllib.parse.quote(rel)
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=90) as r:
                data = r.read()
            tmp = dst + ".part"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, dst)
            return len(data), len(data)
        except Exception as e:
            if attempt == 2:
                print("FAIL", rel, repr(e)[:120])
                return 0, 0
            time.sleep(1)
    return 0, 0


with ThreadPoolExecutor(max_workers=12) as ex:
    for size, new in ex.map(fetch, sel):
        done["n"] += 1
        done["bytes"] += new
        if done["n"] % 20 == 0:
            el = time.time() - t0
            print(f"  {done['n']}/{len(sel)}  {done['bytes']/1e6:.1f}MB  "
                  f"{done['bytes']/el/1024:.0f} KB/s  elapsed={el:.0f}s", flush=True)

el = time.time() - t0
print(f"DONE files={done['n']} downloaded={done['bytes']/1e6:.1f}MB "
      f"avg={done['bytes']/el/1024:.0f} KB/s in {el:.0f}s")
