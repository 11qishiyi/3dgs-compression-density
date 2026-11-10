#!/bin/bash
# F1 setup on the rented node: unpack code, build the CUDA extensions, fetch and
# build the bonsai scenes (clean, uniform q20, mixed q20/q50).
# Usage: bash f1_setup.sh 2>&1 | tee setup.log
set -u
BASE=${F1_BASE:-/root/f1}
cd "$BASE" || exit 1
mkdir -p output logs data/mip360

echo "== unpack =="
tar -xzf code.tgz || { echo "NO_CODE"; exit 1; }
ls code

echo "== python/torch =="
python3 -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))"

echo "== build rasterizer =="
cd "$BASE/code/diff-gaussian-rasterization" || exit 1
python3 setup.py build_ext --inplace > build.log 2>&1
echo "dgr_exit=$? $(ls diff_gaussian_rasterization/_C*.so 2>/dev/null | head -1)"
tail -3 build.log

echo "== build simple-knn =="
cd "$BASE/code/simple-knn" || exit 1
python3 setup.py build_ext --inplace > build.log 2>&1
echo "sknn_exit=$? $(ls simple_knn/_C*.so 2>/dev/null | head -1)"
tail -3 build.log

echo "== import check =="
export PYTHONPATH=$BASE/code/gaussian-splatting:$BASE/code/diff-gaussian-rasterization:$BASE/code/simple-knn
cd "$BASE" || exit 1
python3 -c "import torch, diff_gaussian_rasterization as d, simple_knn._C as sk; print('IMPORT_OK')" 2>&1 | tail -3

echo "== data: download bonsai =="
if [ -d "$BASE/data/mip360/bonsai/images" ] && [ "$(ls "$BASE/data/mip360/bonsai/images" | wc -l)" -ge 280 ]; then
  echo "bonsai already present"
else
  python3 dl_scene.py bonsai 2>&1 | tail -3
fi
ls "$BASE/data/mip360/bonsai/images" | wc -l

echo "== build scenes =="
[ -d "$BASE/data/mip360/bonsai_r4res/images" ] || python3 build_scene.py bonsai 2>&1 | tail -3
[ -d "$BASE/data/mip360/bonsai_mix_hyb/images" ] || python3 build_scene_mix.py bonsai 2>&1 | tail -2
for d in bonsai_r4res bonsai_r4res_q20_hyb bonsai_mix_hyb; do
  n=$(ls "$BASE/data/mip360/$d/images" 2>/dev/null | wc -l)
  sz=$(du -sh "$BASE/data/mip360/$d" 2>/dev/null | cut -f1)
  echo "$d images=$n size=$sz"
done

echo "SETUP_DONE"
