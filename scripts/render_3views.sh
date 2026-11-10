#!/bin/bash
# Render three held-out views of the two bonsai 30k models (clean vs q20) so that
# Fig. 12 can show more than one viewpoint. Mirrors the render lines of post2.sh.
#
# Run on the DCU instance (the shared /public/home tree must still hold
# output/bonsai_clean_30k and output/q20_hyb_30k):
#   bash /public/home/qq18676607705/render_3views.sh
# then, locally:
#   python proj-D6/cluster/fetch_3views.py      (downloads + unpacks renders_3v/)
#   python proj-D6/make_fig12_3view.py          (rebuilds figs_v8/fig12_qualitative.*)
#
# Test-view indices are 0, 9 and 18 of the LLFF holdout (idx % 8 == 0), i.e. three
# viewpoints spread over the capture, not three neighbouring frames.
set -u
BASE=/public/home/qq18676607705
G=$BASE/code/gaussian-splatting
O=$BASE/output
D=$BASE/data/r4res
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MIOPEN_FIND_MODE=FAST PYTHONUNBUFFERED=1
source $BASE/dgr_env.sh
export PYTHONPATH=$G:$BASE/code/diff-gaussian-rasterization:$BASE/code/simple-knn
cd "$G" || exit 1

for M in bonsai_clean_30k q20_hyb_30k; do
  if [ ! -d "$O/$M/point_cloud/iteration_30000" ]; then
    echo "MISSING checkpoint: $O/$M/point_cloud/iteration_30000" >&2
    exit 2
  fi
done

python3 render.py -s $D/bonsai_r4res -m $O/bonsai_clean_30k --iteration 30000 --skip_train --eval > $BASE/render_clean_3v.log 2>&1
python3 render.py -s $D/bonsai_r4res -m $O/q20_hyb_30k   --iteration 30000 --skip_train --eval > $BASE/render_q20_3v.log 2>&1

PKG=$BASE/fig12_3v
rm -rf "$PKG"
mkdir -p "$PKG/clean" "$PKG/q20" "$PKG/gt"
: > "$PKG/missing.log"
for i in 00000 00009 00018; do
  cp "$O/bonsai_clean_30k/test/ours_30000/renders/$i.png" "$PKG/clean/" 2>>"$PKG/missing.log"
  cp "$O/q20_hyb_30k/test/ours_30000/renders/$i.png"     "$PKG/q20/"   2>>"$PKG/missing.log"
  cp "$O/bonsai_clean_30k/test/ours_30000/gt/$i.png"     "$PKG/gt/"    2>>"$PKG/missing.log"
done

echo "--- manifest ---"
ls -l "$PKG/clean" "$PKG/q20"
echo "--- missing (empty = all good) ---"
cat "$PKG/missing.log"

# A single archive keeps the download one file.
tar czf $BASE/fig12_3v.tar.gz -C $BASE fig12_3v
echo "FIG12_3V_DONE $BASE/fig12_3v.tar.gz"
