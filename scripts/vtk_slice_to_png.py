"""vtk_slice_to_png.py の処理内容をまとめたスクリプト"""

import argparse
import glob
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_structured_scalar(path: Path):
    """入力を読み取る"""
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    dims = None
    data_start = None
    for i, line in enumerate(lines):
        if line.startswith("DIMENSIONS"):
            parts = line.strip().split()
            dims = tuple(int(v) for v in parts[1:4])
        if line.startswith("LOOKUP_TABLE"):
            data_start = i + 1
            break
    if dims is None or data_start is None:
        raise RuntimeError(f"Failed to parse DIMENSIONS or data start in {path}")
    vals = []
    for line in lines[data_start:]:
        vals.extend(float(x) for x in line.strip().split())
    arr = np.asarray(vals, dtype=np.float32)
    if arr.size != dims[0] * dims[1] * dims[2]:
        raise RuntimeError(f"Data size mismatch in {path}, got {arr.size}, expected {dims}")
    return arr.reshape((dims[2], dims[1], dims[0]))  # z, y, x


def save_frame(z_slice_legacy, z_slice_home, step, out_dir):
    """ファイル等へ書き出す"""
    diff = z_slice_home - z_slice_legacy
    vmax = max(z_slice_legacy.max(), z_slice_home.max())
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    im0 = axes[0].imshow(z_slice_legacy, origin="lower", cmap="viridis", vmin=0, vmax=vmax)
    axes[0].set_title(f"Legacy speed (step {step})")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(z_slice_home, origin="lower", cmap="viridis", vmin=0, vmax=vmax)
    axes[1].set_title("HOME speed")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    vmax_diff = np.max(np.abs(diff)) + 1e-12
    im2 = axes[2].imshow(diff, origin="lower", cmap="coolwarm", vmin=-vmax_diff, vmax=vmax_diff)
    axes[2].set_title("HOME - Legacy")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    out_path = out_dir / f"frame_{step:06d}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    """スクリプトのエントリポイントを実行する"""
    ap = argparse.ArgumentParser(description="Create mid-plane slice PNGs (and optional GIF) from VTK outputs.")
    ap.add_argument("vtk_dir", help="Directory containing speed_legacy_*.vtk and speed_home_*.vtk")
    ap.add_argument("out_dir", help="Output directory for PNG frames (and optional GIF)")
    ap.add_argument("--gif", help="If set, write an animated GIF to this path (requires imageio)", default=None)
    args = ap.parse_args()

    vtk_dir = Path(args.vtk_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect matching pairs by step number
    pattern = re.compile(r"speed_legacy_(\d+)\.vtk$")
    legacy_files = {}
    for p in glob.glob(str(vtk_dir / "speed_legacy_*.vtk")):
        m = pattern.search(os.path.basename(p))
        if m:
            legacy_files[int(m.group(1))] = Path(p)

    steps = sorted(legacy_files.keys())
    if not steps:
        raise SystemExit("No speed_legacy_*.vtk files found.")

    frame_paths = []
    for step in steps:
        legacy_path = legacy_files[step]
        home_path = vtk_dir / f"speed_home_{step:06d}.vtk"
        if not home_path.exists():
            print(f"[warn] missing HOME file for step {step}, skipping")
            continue

        legacy = read_structured_scalar(legacy_path)
        home = read_structured_scalar(home_path)
        # take middle z plane
        z = legacy.shape[0] // 2
        frame_paths.append(save_frame(legacy[z], home[z], step, out_dir))
        print(f"[frame] step {step} -> {frame_paths[-1]}")

    if args.gif and frame_paths:
        try:
            import imageio.v2 as imageio  # type: ignore
        except ImportError:
            print("[warn] imageio not installed; skipping GIF creation")
        else:
            images = [imageio.imread(p) for p in frame_paths]
            imageio.mimsave(args.gif, images, duration=0.3)
            print(f"[gif] saved {args.gif}")


if __name__ == "__main__":
    main()
