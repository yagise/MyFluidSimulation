#!/usr/bin/env python3
"""summary: plot_b0_sweep.py の処理内容をまとめたスクリプト

plot_b0_sweep.py

Plot B0 sweep CSV produced by:
 fluidsim_compare.exe --hybrid-sweep-max M --hybrid-sweep-steps S ...

Usage:
 python scripts/plot_b0_sweep.py results_b0/b0_teardrop_amr2.csv out.png

Notes:
- The sweep output includes a log line like "[hybrid] ...", then a CSV header.
 This script skips non-CSV lines automatically."""
import sys
import re
import csv
from pathlib import Path

import matplotlib.pyplot as plt

def read_csv(path: Path):
    """summary: 入力を読み取る
param path: 入力パラメータ
return: 戻り値"""
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        # keep only the CSV header + numeric lines
        lines = [ln.strip() for ln in f if ln.strip()]
    # find header line (starts with "d0,")
    header_idx = None
    for i, ln in enumerate(lines):
        if ln.startswith("d0,"):
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError("CSV header not found (expected a line starting with 'd0,').")
    csv_lines = lines[header_idx:]
    reader = csv.DictReader(csv_lines)
    for r in reader:
        rows.append(r)
    return rows

def to_float(x):
    """summary: to_float の処理を行う
param x: 入力パラメータ
return: 戻り値"""
    try:
        return float(x)
    except Exception:
        return float("nan")

def main():
    """summary: スクリプトのエントリポイントを実行する
param: なし
return: 戻り値"""
    if len(sys.argv) < 3:
        print("Usage: python scripts/plot_b0_sweep.py <in.csv> <out.png>")
        sys.exit(1)

    in_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    rows = read_csv(in_path)

    d0 = [int(r["d0"]) for r in rows]
    mean_absdiff_home = [to_float(r["mean_absdiff_home"]) for r in rows]
    mean_absdiff_hyb  = [to_float(r["mean_absdiff_hybrid"]) for r in rows]
    legacy_ratio      = [to_float(r["legacy_ratio"]) for r in rows]
    ms_legacy         = [to_float(r["ms_per_step_legacy"]) for r in rows]
    ms_home           = [to_float(r["ms_per_step_home"]) for r in rows]
    ms_hyb            = [to_float(r["ms_per_step_hybrid"]) for r in rows]

    plt.figure()
    plt.plot(d0, mean_absdiff_home, marker="o", label="HOME vs Legacy (near-wall |u|)")
    plt.plot(d0, mean_absdiff_hyb,  marker="o", label="Hybrid(B0) vs Legacy (near-wall |u|)")
    plt.yscale("log")
    plt.xlabel("d0 (legacy band thickness)")
    plt.ylabel("Mean absolute difference of near-wall |u|")
    plt.title(in_path.name)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    print("Saved:", out_path)

    # Optional second plot: cost proxy (legacy ratio and ms/step)
    out2 = out_path.with_name(out_path.stem + "_cost.png")
    plt.figure()
    plt.plot(d0, legacy_ratio, marker="o", label="Legacy ratio (cells)")
    plt.plot(d0, ms_hyb, marker="o", label="Hybrid ms/step")
    # repeat baselines as flat lines (use first value)
    if ms_legacy:
        plt.plot(d0, [ms_legacy[0]]*len(d0), label="Legacy ms/step")
    if ms_home:
        plt.plot(d0, [ms_home[0]]*len(d0), label="HOME ms/step")
    plt.xlabel("d0")
    plt.title(in_path.name + " (cost)")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.savefig(out2, dpi=180)
    print("Saved:", out2)

if __name__ == "__main__":
    main()
