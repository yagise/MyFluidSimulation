"""TGV 計算結果の CSV を読み込み、エネルギーや L2 誤差を描画する"""

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def load(csv_path):
    """CSV を読み込み、時刻と各種メトリクスの配列を返す"""
    times = []
    e_exact = []
    e_legacy = []
    e_home = []
    l2_legacy = []
    l2_home = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row["time"]))
            e_exact.append(float(row["energy_exact"]))
            e_legacy.append(float(row["energy_legacy"]))
            e_home.append(float(row["energy_home"]))
            l2_legacy.append(float(row["l2_legacy"]))
            l2_home.append(float(row["l2_home"]))
    return times, e_exact, e_legacy, e_home, l2_legacy, l2_home


def main():
    """コマンドライン引数を解釈してグラフを描画する"""
    if len(sys.argv) < 2:
        print("usage: python scripts/plot_tgv.py <tgv_results.csv> [out.png]")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else csv_path.with_suffix(".png")
    times, e_exact, e_legacy, e_home, l2_legacy, l2_home = load(csv_path)

    plt.figure(figsize=(8.0, 6.0))

    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(times, e_exact, label="analytic", color="k", linewidth=2.0)
    ax1.plot(times, e_legacy, label="legacy", linestyle="--", color="tab:orange")
    ax1.plot(times, e_home, label="home", linestyle="-.", color="tab:blue")
    ax1.set_ylabel("Kinetic energy")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend()

    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(times, l2_legacy, label="legacy L2", linestyle="--", color="tab:orange")
    ax2.plot(times, l2_home, label="home L2", linestyle="-.", color="tab:blue")
    ax2.set_yscale("log")
    ax2.set_xlabel("Time [lattice units]")
    ax2.set_ylabel("Velocity L2 error")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()

