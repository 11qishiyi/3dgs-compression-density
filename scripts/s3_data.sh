#!/bin/bash
# Data-only stage: download and build the four scenes (setup.sh stopped in the
# simple-knn directory before the download stage).
set -u
BASE=/root/s3
cd "$BASE" || exit 1
export PYTHONPATH=$BASE/code/gaussian-splatting:$BASE/code/diff-gaussian-rasterization:$BASE/code/simple-knn
mkdir -p data/mip360 data/bench logs

nimg() { ls "$1" 2>/dev/null | wc -l; }

for sc in counter bicycle; do
  if [ "$(nimg "$BASE/data/mip360/$sc/images")" -ge 100 ]; then
    echo "$sc already downloaded ($(nimg "$BASE/data/mip360/$sc/images") imgs)"
  else
    echo "== download $sc =="
    python3 dl_scene.py "$sc" 2>&1 | tail -3
  fi
  if [ "$(nimg "$BASE/data/mip360/${sc}_r4res/images")" -ge 100 ]; then
    echo "$sc_r4res exists"
  else
    echo "== build $sc =="
    python3 build_scene.py "$sc" 2>&1 | tail -3
  fi
done

for sc in truck playroom; do
  if [ "$(nimg "$BASE/data/bench/$sc/images")" -ge 50 ]; then
    echo "$sc already downloaded ($(nimg "$BASE/data/bench/$sc/images") imgs)"
  else
    echo "== download $sc =="
    python3 dl_bench.py "$sc" 2>&1 | tail -3
  fi
  if [ "$(nimg "$BASE/data/bench/${sc}_clean/images")" -ge 50 ]; then
    echo "${sc}_clean exists"
  else
    echo "== build $sc =="
    python3 build_scene_generic.py "$BASE/data/bench/$sc" "$BASE/data/bench" "$sc" 2>&1 | tail -3
  fi
done

for d in counter_r4res counter_r4res_q20_hyb bicycle_r4res bicycle_r4res_q20_hyb \
         truck_clean truck_q20_hyb playroom_clean playroom_q20_hyb; do
  echo "$d images=$(nimg "$BASE/data/mip360/$d/images")"
done
echo "S3_DATA_DONE"
