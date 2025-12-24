#!/usr/bin/env python3
"""B0 スイープ結果を重ねて壁 AMR の効果を比べる可視化スクリプト"""

import sys
import csv
import re
from pathlib import Path
import matplotlib.pyplot as plt


def read_csv(path: Path):
    """CSV を読み込み、辞書のリストで返す"""
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    header_idx = None
    for i, ln in enumerate(lines):
        if ln.startswith("d0,"):
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError(f"CSV header not found in {path}")
    reader = csv.DictReader(lines[header_idx:])
    for r in reader:
        rows.append(r)
    return rows


def to_float(x: str) -> float:
    """数値化できない値は NaN にする簡易パーサ"""
    try:
        return float(x)
    except Exception:
        return float("nan")


def main():
    """コマンドライン引数を解釈してグラフを描画する"""
    if len(sys.argv) < 4:
        print("Usage: python scripts/plot_b0_amr_overlay.py <results_dir> <shape_tag> <out.png>")
        print("  shape_tag: prefix used in CSV filenames after 'b0_' (e.g. yourmodel)")
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    shape_tag = sys.argv[2]
    out_path = Path(sys.argv[3])

    # run_b0_sweeps.ps1 の命名規則に合わせる
    # 例: b0_yourmodel_amr2_thr0p5.csv
    pat = re.compile(rf"^b0_{re.escape(shape_tag)}_amr(\\d+)_thr([0-9p]+)\\.csv$")

    items = []
    for p in sorted(results_dir.glob(f"b0_{shape_tag}_amr*_thr*.csv")):
        m = pat.match(p.name)
        if not m:
            continue
        amr = int(m.group(1))
        thr = float(m.group(2).replace("p", "."))
        rows = read_csv(p)
        d0 = [int(r["d0"]) for r in rows]
        home = [to_float(r["mean_absdiff_home"]) for r in rows]
        hyb = [to_float(r["mean_absdiff_hybrid"]) for r in rows]
        items.append((amr, thr, d0, home, hyb, p.name))

    if not items:
        raise RuntimeError(f"No CSVs found for shape '{shape_tag}' in {results_dir}")

    # HOME vs Legacy の誤差曲線を描画（d0 でほぼフラットだが AMR ごとに差が出る）
    plt.figure()
    for amr, thr, d0, home, hyb, name in sorted(items, key=lambda t: (t[1], t[0])):
        plt.plot(d0, home, marker="o", label=f"HOME (amr={amr}, thr={thr})")
    plt.yscale("log")
    plt.xlabel("d0")
    plt.ylabel("Mean abs diff of near-wall |u| vs Legacy")
    plt.title(f"{shape_tag}: HOME vs Legacy (wall AMR sweep)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    print("Saved:", out_path)

    # HYBRID(B0) vs Legacy の誤差曲線を描画
    out2 = out_path.with_name(out_path.stem + "_hybrid.png")
    plt.figure()
    for amr, thr, d0, home, hyb, name in sorted(items, key=lambda t: (t[1], t[0])):
        plt.plot(d0, hyb, marker="o", label=f"Hybrid(B0) (amr={amr}, thr={thr})")
    plt.yscale("log")
    plt.xlabel("d0")
    plt.ylabel("Mean abs diff of near-wall |u| vs Legacy")
    plt.title(f"{shape_tag}: Hybrid(B0) vs Legacy (wall AMR sweep)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(fontsize=8)
    plt.savefig(out2, dpi=180)
    print("Saved:", out2)


if __name__ == "__main__":
    main()
