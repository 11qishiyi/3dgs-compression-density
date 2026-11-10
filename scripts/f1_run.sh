#!/bin/bash
# F1 run matrix (see f1_matrix.tsv). Sequential on one GPU, resumable: a run
# whose log contains the completion marker is skipped.
#
# columns: tag  scene  iters  seed  mode  lambda  kind
#   kind=train : train_codec2.py with CODEC_MODE=mode (hinge/frame/dct/none)
#   kind=ft    : train_ft.py on the f1_c0 3k checkpoint with frozen topology,
#                mode = c1|c2|c3 (target construction) or h (c1 + interval term)
#
# Usage: setsid nohup bash f1_run.sh > logs/f1_run.log 2>&1 < /dev/null &
set -u
BASE=${F1_BASE:-/root/f1}
G=$BASE/code/gaussian-splatting
D=$BASE/data/mip360
O=$BASE/output
export PYTHONPATH=$G:$BASE/code/diff-gaussian-rasterization:$BASE/code/simple-knn
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
mkdir -p "$O" "$BASE/logs"
cd "$G" || exit 1
PLY=$O/f1_c0/point_cloud/iteration_3000/point_cloud.ply
LR="--position_lr_init 1e-5 --position_lr_final 1e-6 --feature_lr 2.5e-4 --opacity_lr 2.5e-3 --scaling_lr 5e-4 --rotation_lr 1e-4"

run_train() {  # tag scene iters seed mode lam
  local tag=$1 scene=$2 it=$3 seed=$4 mode=$5 lam=$6
  [ "$mode" = "none" ] && mode=""
  local log=$O/$tag.log
  if [ -f "$log" ] && grep -q "Training complete" "$log"; then echo "SKIP $tag"; return; fi
  echo "RUN $tag scene=$scene it=$it seed=$seed mode=${mode:-none} lambda=$lam $(date +%H:%M:%S)"
  CODEC_MODE="$mode" CODEC_LAMBDA="$lam" DCT_IMG_DIR="$D/$scene/images" GSSEED="$seed" \
    python3 -u train_codec2.py -s "$D/$scene" -m "$O/$tag" \
      --iterations "$it" -r 1 --eval --disable_viewer --data_device cpu \
      --test_iterations "$it" > "$log" 2>&1
  echo "EXIT $tag train $? $(date +%H:%M:%S)"
}

run_ft() {  # tag scene mode lam
  local tag=$1 scene=$2 mode=$3 lam=$4
  local log=$O/$tag.log
  if [ -f "$log" ] && grep -q "Fine-tune complete" "$log"; then echo "SKIP $tag"; return; fi
  if [ ! -f "$PLY" ]; then echo "MISSING_PLY $PLY"; return; fi
  local ft=$mode cm=""
  if [ "$mode" = "h" ]; then ft="c1"; cm="hinge"; fi
  echo "RUN $tag ft=$ft codec=${cm:-none} lambda=$lam $(date +%H:%M:%S)"
  CODEC_MODE="$cm" CODEC_LAMBDA="$lam" DCT_IMG_DIR="$D/$scene/images" GSSEED=0 \
    python3 -u train_ft.py -s "$D/$scene" -m "$O/$tag" \
      --iterations 600 --test_iterations 600 --eval --disable_viewer --data_device cpu \
      --densify_until_iter 0 --start_ply "$PLY" --ft_mode "$ft" $LR > "$log" 2>&1
  echo "EXIT $tag ft $? $(date +%H:%M:%S)"
}

while read -r tag scene it seed mode lam kind; do
  [ -z "${tag:-}" ] && continue
  case "$tag" in \#*) continue;; esac
  if [ "${kind:-train}" = "ft" ]; then run_ft "$tag" "$scene" "$mode" "$lam"; else run_train "$tag" "$scene" "$it" "$seed" "$mode" "$lam"; fi
done < "$BASE/f1_matrix.tsv"

echo "F1_ALL_DONE $(date +%H:%M:%S)"
