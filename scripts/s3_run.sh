#!/bin/bash
# D9 seed top-up runs (20 runs): counter 7k/30k +2 seeds, bicycle 7k +2 seeds,
# truck and playroom +1 seed at 3k/7k.
# Usage: setsid nohup bash s3_run.sh > logs/s3_run.log 2>&1 < /dev/null &
set -u
BASE=${S3_BASE:-/root/s3}
G=$BASE/code/gaussian-splatting
D=$BASE/data/mip360
O=$BASE/output
export PYTHONPATH=$G:$BASE/code/diff-gaussian-rasterization:$BASE/code/simple-knn
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
mkdir -p "$O" "$BASE/logs"
cd "$G" || exit 1

run() {  # tag scene iters seed
  local tag=$1 scene=$2 it=$3 seed=$4
  local log=$O/$tag.log
  if [ -f "$log" ] && grep -q "Training complete" "$log"; then echo "SKIP $tag"; return; fi
  echo "RUN $tag scene=$scene it=$it seed=$seed $(date +%H:%M:%S)"
  GSSEED="$seed" python3 -u train_codec2.py -s "$D/$scene" -m "$O/$tag" \
    --iterations "$it" -r 1 --eval --disable_viewer --data_device cpu \
    --test_iterations "$it" > "$log" 2>&1
  echo "EXIT $tag $? $(date +%H:%M:%S)"
}

while read -r tag scene it seed; do
  [ -z "${tag:-}" ] && continue
  case "$tag" in \#*) continue;; esac
  run "$tag" "$scene" "$it" "$seed"
done < "${1:-$BASE/s3_matrix.tsv}"

echo "S3_ALL_DONE $(date +%H:%M:%S)"
