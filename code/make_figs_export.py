"""Springer/TVC artwork export: vector EPS masters + 600 dpi raster for halftone/combination art.

Reads the matplotlib outputs in figs_v8/ and writes:
  - <name>.eps        vector (pdftops -eps), for Springer's preferred vector format
  - <name>_600.png   600 dpi raster preview
  - fig12_qualitative.tif   600 dpi TIFF (combination art: raster plate + lettering)

Usage: python make_figs_export.py
"""
import glob
import os
import subprocess

from PIL import Image

SRC = r"D:\3dgs\proj-D6\figs_v8"
PDFTOPS = r"D:\texlive\2026\bin\windows\pdftops.exe"
PDFTOCAIRO = r"D:\texlive\2026\bin\windows\pdftocairo.exe"
DPI = 600


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("FAILED:", " ".join(cmd), r.stderr[:300])
    return r.returncode == 0


def main():
    pdfs = sorted(glob.glob(os.path.join(SRC, "fig*.pdf")))
    print(f"found {len(pdfs)} vector masters")
    for pdf in pdfs:
        base = os.path.splitext(pdf)[0]
        eps = base + ".eps"
        # 1) EPS vector master (fonts embedded by pdf.fonttype=42 in the PDF)
        run([PDFTOPS, "-eps", "-level3", pdf, eps])

        # 2) 600 dpi PNG preview
        png600 = base + "_600.png"
        tmp_base = base + "_600_tmp"          # pdftocairo appends .png itself
        run([PDFTOCAIRO, "-png", "-r", str(DPI), "-singlefile", pdf, tmp_base])
        tmp = tmp_base + ".png"
        if os.path.exists(tmp):
            Image.open(tmp).convert("RGB").save(png600, dpi=(DPI, DPI))
            os.remove(tmp)

        # 3) combination art (raster plate + lettering) -> TIFF 600 dpi
        if "fig12" in os.path.basename(pdf):
            tif = base + ".tif"
            im = Image.open(png600).convert("RGB")
            im.save(tif, format="TIFF", compression="tiff_lzw", dpi=(DPI, DPI))
            print("TIFF:", os.path.basename(tif), im.size)

    # 4) confirm every figure has all three deliverables
    for pdf in pdfs:
        base = os.path.splitext(pdf)[0]
        have = [os.path.exists(base + ext) for ext in (".pdf", ".eps", "_600.png")]
        print(f"{os.path.basename(base):26s} pdf={have[0]} eps={have[1]} png600={have[2]}")


if __name__ == "__main__":
    main()
