"""summary: generate_tgv_2d.py の処理内容をまとめたスクリプト

generate_tgv_2d.py

2D テイラー・グリーン渦 (Taylor-Green vortex) の初期場を Python で生成するスクリプト。

このリポジトリの fluidsim_tgv (src/bench/main_taylor_green.cpp) と同じ式を使い、
格子点(セル中心)における (rho, u, v) を計算して VTK(LEGACY ASCII) に書き出します。

論文用メモ:
 - 2Dで渦を発生させるコードがあったはずという要望に対応し、
 解析解に基づく初期条件生成をスクリプトとして残します。
 - 3D ソルバで可視化したい場合は、Nz=1 として使うか、
 出力された VTK を任意のツール(ParaView など)で読み込んでください。

実行例:
 python scripts/generate_tgv_2d.py --nx 128 --ny 128 --u0 0.06 --out out_vtk

出力:
 out_vtk/tgv2d_rho.vtk
 out_vtk/tgv2d_vel.vtk (vector field)"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Field2D:
    nx: int
    ny: int
    rho: List[float]
    u: List[float]
    v: List[float]


def analytic_tgv_cell_center(ix: int, iy: int, nx: int, ny: int, u0: float, cs2: float = 1.0 / 3.0) -> Tuple[float, float, float]:
    """summary: analytic_tgv_cell_center の処理を行う
param ix: 入力パラメータ
param iy: 入力パラメータ
param nx: 入力パラメータ
param ny: 入力パラメータ
param u0: 入力パラメータ
param cs2: 入力パラメータ
return: 戻り値

セル中心での Taylor-Green 渦の解析解 (t=0) を返す。

 C++ 側と合わせて、波数は
 kx_wave = 2*pi / Nx, ky_wave = 2*pi / Ny
 とし、tx=(ix+0.5)*kx_wave, ty=(iy+0.5)*ky_wave で評価する。

 Returns:
 rho, u, v"""
    two_pi = 2.0 * math.pi
    kx_wave = two_pi / float(nx)
    ky_wave = two_pi / float(ny)
    tx = (float(ix) + 0.5) * kx_wave
    ty = (float(iy) + 0.5) * ky_wave

    sin_x = math.sin(tx)
    cos_x = math.cos(tx)
    sin_y = math.sin(ty)
    cos_y = math.cos(ty)

    u = u0 * sin_x * cos_y
    v = -u0 * cos_x * sin_y
    # t=0 の密度場（C++ 実装と同じ形）
    rho = 1.0 + (u0 * u0 / (4.0 * cs2)) * (math.cos(2.0 * tx) + math.cos(2.0 * ty))
    return rho, u, v


def build_tgv2d(nx: int, ny: int, u0: float) -> Field2D:
    """summary: データを生成する
param nx: 入力パラメータ
param ny: 入力パラメータ
param u0: 入力パラメータ
return: 戻り値

2D TGV の (rho,u,v) フィールドを生成する。"""
    rho: List[float] = [0.0] * (nx * ny)
    u: List[float] = [0.0] * (nx * ny)
    v: List[float] = [0.0] * (nx * ny)

    for iy in range(ny):
        for ix in range(nx):
            r, ux, uy = analytic_tgv_cell_center(ix, iy, nx, ny, u0)
            idx = ix + nx * iy
            rho[idx] = float(r)
            u[idx] = float(ux)
            v[idx] = float(uy)

    return Field2D(nx=nx, ny=ny, rho=rho, u=u, v=v)


def write_vtk_structured_points_scalar(path: str, name: str, nx: int, ny: int, nz: int, data: List[float]) -> None:
    """summary: ファイル等へ書き出す
param path: 入力パラメータ
param name: 入力パラメータ
param nx: 入力パラメータ
param ny: 入力パラメータ
param nz: 入力パラメータ
param data: 入力パラメータ
return: なし

VTK(LEGACY ASCII) の STRUCTURED_POINTS としてスカラー場を書き出す。"""
    assert len(data) == nx * ny * nz
    with open(path, "w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"{name}\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {nx} {ny} {nz}\n")
        f.write("ORIGIN 0 0 0\n")
        f.write("SPACING 1 1 1\n")
        f.write(f"POINT_DATA {nx * ny * nz}\n")
        f.write(f"SCALARS {name} float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for v in data:
            f.write(f"{v}\n")


def write_vtk_structured_points_vector(path: str, name: str, nx: int, ny: int, nz: int, data_xyz: List[Tuple[float, float, float]]) -> None:
    """summary: ファイル等へ書き出す
param path: 入力パラメータ
param name: 入力パラメータ
param nx: 入力パラメータ
param ny: 入力パラメータ
param nz: 入力パラメータ
param data_xyz: 入力パラメータ
return: なし

VTK(LEGACY ASCII) の STRUCTURED_POINTS としてベクトル場を書き出す。"""
    assert len(data_xyz) == nx * ny * nz
    with open(path, "w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"{name}\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {nx} {ny} {nz}\n")
        f.write("ORIGIN 0 0 0\n")
        f.write("SPACING 1 1 1\n")
        f.write(f"POINT_DATA {nx * ny * nz}\n")
        f.write(f"VECTORS {name} float\n")
        for (x, y, z) in data_xyz:
            f.write(f"{x} {y} {z}\n")


def main() -> int:
    """summary: スクリプトのエントリポイントを実行する
param: なし
return: 終了コード"""
    parser = argparse.ArgumentParser(description="Generate 2D Taylor-Green vortex fields (t=0) and dump VTK.")
    parser.add_argument("--nx", type=int, default=128, help="grid size in x")
    parser.add_argument("--ny", type=int, default=128, help="grid size in y")
    parser.add_argument("--u0", type=float, default=0.06, help="velocity amplitude")
    parser.add_argument("--out", type=str, default="out_vtk", help="output directory")
    parser.add_argument("--nz", type=int, default=1, help="optional z size for extrusion (default 1)")
    args = parser.parse_args()

    nx, ny, nz = int(args.nx), int(args.ny), int(args.nz)
    if nx <= 0 or ny <= 0 or nz <= 0:
        raise SystemExit("nx, ny, nz must be positive")

    os.makedirs(args.out, exist_ok=True)
    field2d = build_tgv2d(nx, ny, float(args.u0))

    # 3D 形式に合わせて nz 回複製する（ParaView などで扱いやすい）
    rho3 = []
    vel3 = []
    for iz in range(nz):
        for i in range(nx * ny):
            rho3.append(field2d.rho[i])
            vel3.append((field2d.u[i], field2d.v[i], 0.0))

    rho_path = os.path.join(args.out, "tgv2d_rho.vtk")
    vel_path = os.path.join(args.out, "tgv2d_vel.vtk")
    write_vtk_structured_points_scalar(rho_path, "rho", nx, ny, nz, rho3)
    write_vtk_structured_points_vector(vel_path, "vel", nx, ny, nz, vel3)
    print(f"[ok] wrote: {rho_path}")
    print(f"[ok] wrote: {vel_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
