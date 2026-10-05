"""Build the README figures and docs/results/demo_metrics.json from a demo set (maintainer tool, PIL + NumPy only).

  python scripts/make_demo_figures.py --clean DIR --lq DIR --restored DIR --out docs

<clean>/<name>.png are the 512x512 source images; <lq>/<name>_<degradation>.png the degraded inputs given to the node;
<restored>/<name>_<degradation>.png its outputs (same file names as the inputs).
Writes, for every source image, a grid (rows: degradations; columns: clean source | degraded input | restored)
with cells downscaled to 256 px for display, one 1:1 detail crop figure, and per-image PSNR against the clean source
(full 512x512, RGB, 8-bit values, no crop or shift; PSNR is not a perceptual measure and generative restoration
can lower it). Every result is written; nothing is filtered.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

DEGRADATIONS = ["noise", "blur", "jpeg"]
COLUMNS = ["clean source (synthetic)", "degraded input", "PixRestore-S output"]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else round(float(10 * np.log10(255.0 ** 2 / mse)), 3)


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def grid(cells: list[list[Image.Image]], col_labels, row_labels, cell=256, bar=34, side=90) -> Image.Image:
    rows, cols = len(cells), len(cells[0])
    canvas = Image.new("RGB", (side + cols * cell, bar + rows * cell), (24, 24, 24))
    d = ImageDraw.Draw(canvas)
    f = font(16)
    for c, label in enumerate(col_labels):
        d.text((side + c * cell + 8, 9), label, fill=(235, 235, 235), font=f)
    for r, label in enumerate(row_labels):
        d.text((8, bar + r * cell + cell // 2 - 8), label, fill=(235, 235, 235), font=f)
        for c in range(cols):
            canvas.paste(cells[r][c].resize((cell, cell), Image.Resampling.LANCZOS), (side + c * cell, bar + r * cell))
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", type=Path, required=True)
    ap.add_argument("--lq", type=Path, required=True)
    ap.add_argument("--restored", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--crop-name", default="p1_s7_l4")
    ap.add_argument("--crop-box", default="150,250,350,450", help="x0,y0,x1,y1 of the 1:1 detail crop")
    a = ap.parse_args()
    (a.out / "images").mkdir(parents=True, exist_ok=True)
    (a.out / "results").mkdir(parents=True, exist_ok=True)
    names = sorted(p.stem for p in a.clean.glob("*.png"))
    rows = []
    for name in names:
        clean = Image.open(a.clean / f"{name}.png").convert("RGB")
        cells = []
        for deg in DEGRADATIONS:
            lq_p, out_p = a.lq / f"{name}_{deg}.png", a.restored / f"{name}_{deg}.png"
            lq, out = Image.open(lq_p).convert("RGB"), Image.open(out_p).convert("RGB")
            c, l_, o = (np.asarray(x) for x in (clean, lq, out))
            row = {"name": name, "degradation": deg, "psnr_input_db": psnr(l_, c), "psnr_output_db": psnr(o, c),
                   "clean_sha256": sha256(a.clean / f"{name}.png"), "input_sha256": sha256(lq_p), "output_sha256": sha256(out_p)}
            row["psnr_change_db"] = round(row["psnr_output_db"] - row["psnr_input_db"], 3)
            rows.append(row)
            cells.append([clean, lq, out])
        grid(cells, COLUMNS, DEGRADATIONS).save(a.out / "images" / f"grid_{name}.png", optimize=True)
    x0, y0, x1, y1 = (int(v) for v in a.crop_box.split(","))
    clean = Image.open(a.clean / f"{a.crop_name}.png").convert("RGB").crop((x0, y0, x1, y1))
    cells = [[clean, Image.open(a.lq / f"{a.crop_name}_{d}.png").convert("RGB").crop((x0, y0, x1, y1)),
              Image.open(a.restored / f"{a.crop_name}_{d}.png").convert("RGB").crop((x0, y0, x1, y1))] for d in DEGRADATIONS]
    grid(cells, COLUMNS, DEGRADATIONS, cell=x1 - x0).save(a.out / "images" / f"crop_{a.crop_name}_1to1.png", optimize=True)
    summary = {d: {"n": int(sum(r["degradation"] == d for r in rows)),
                   "psnr_up": int(sum(r["degradation"] == d and r["psnr_change_db"] > 0 for r in rows)),
                   "mean_change_db": round(float(np.mean([r["psnr_change_db"] for r in rows if r["degradation"] == d])), 3)}
               for d in DEGRADATIONS}
    (a.out / "results" / "demo_metrics.json").write_text(json.dumps(
        {"metric": "PSNR in dB against the clean source; full 512x512, RGB, 8-bit, no crop or shift",
         "note": "12 results = every image x every degradation; no selection. Not a benchmark: 4 synthetic images.",
         "summary": summary, "rows": rows}, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
