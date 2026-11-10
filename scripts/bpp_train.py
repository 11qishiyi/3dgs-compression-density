import os, sys

base = "/public/home/qq18676607705/data/r4res"
for tag, W, H in [("room_r4res", 778, 518), ("room_r4res_q20_hyb", 778, 518),
                  ("stump_r4res", 1244, 825), ("stump_r4res_q20_hyb", 1244, 825)]:
    d = os.path.join(base, tag, "images")
    if not os.path.isdir(d):
        print(tag, "MISSING")
        continue
    names = sorted(f for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    train = [n for i, n in enumerate(names) if i % 8 != 0]
    tb = sum(os.path.getsize(os.path.join(d, n)) for n in train)
    print("%s files=%d train=%d test=%d trainMB=%.2f bpp=%.3f"
          % (tag, len(names), len(train), len(names) - len(train), tb / 1e6,
             tb * 8 / (W * H * len(train))))
