#!/bin/bash
# D9 seed top-up: setup on a fresh node.
# Builds the code extensions and the four remaining scenes (counter, bicycle,
# truck, playroom) in clean + q20-hybrid form.
set -u
BASE=${S3_BASE:-/root/s3}
cd "$BASE" || exit 1
mkdir -p output logs data/mip360 data/bench

echo "== unpack =="
tar -xzf code.tgz || { echo NO_CODE; exit 1; }
ls code

echo "== build rasterizer =="
cd "$BASE/code/diff-gaussian-rasterization" || exit 1
python3 setup.py build_ext --inplace > build.log 2>&1
echo "dgr_exit=$? $(ls diff_gaussian_rasterization/_C*.so 2>/dev/null | head -1)"

echo "== build simple-knn =="
cd "$BASE/code/simple-knn" || exit 1
python3 setup.py build_ext --inplace > build.log 2>&1
echo "sknn_exit=$? $(ls simple_knn/_C*.so 2>/dev/null | head -1)"

export PYTHONPATH=$BASE/code/gaussian-splatting:$BASE/code/diff-gaussian-rasterization:$BASE/code/simple-knn
python3 -c "import torch, diff_gaussian_rasterization, simple_knn._C; print('IMPORT_OK')" 2>&1 | tail -1

echo "== Mip-NeRF 360: counter, bicycle =="
for sc in counter bicycle; do
  if [ -d "$BASE/data/mip360/$sc/images" ]; then echo "$sc present"; else
    python3 dl_scene.py "$sc" 2>&1 | tail -2
  fi
  [ -d "$BASE/data/mip360/${sc}_r4res/images" ] || python3 build_scene.py "$sc" 2>&1 | tail -3
done

echo "== benchmark: truck, playroom =="
for sc in truck playroom; do
  if [ -d "$BASE/data/bench/$sc/images" ]; then echo "$sc present"; else
    python3 dl_bench.py "$sc" 2>&1 | tail -2
  fi
  [ -d "$BASE/data/bench/${sc}_clean/images" ] || python3 build_scene_generic.py \
      "$BASE/data/bench/$sc" "$BASE/data/bench" "$sc" 2>&1 | tail -3
done

echo "== expose bench scenes under data/mip360 for the shared eval script =="
for d in truck_clean truck_q20_hyb playroom_clean playroom_q20_hyb; do
  ln -sfn "$BASE/data/bench/$d" "$BASE/data/mip360/$d"
done
for d in counter_r4res counter_r4res_q20_hyb bicycle_r4res bicycle_r4res_q20_hyb \
         truck_clean truck_q20_hyb playroom_clean playroom_q20_hyb; do
  echo "$d images=$(ls "$BASE/data/mip360/$d/images" 2>/dev/null | wc -l)"
done
echo "S3_SETUP_DONE"
