# 実験メモ（再現用のコマンド集）

---

## 1) 同一モデルを 3 手法で回して比較する（推奨）

同じ格子/外力/障害物で

- `fluidsim_legacy`
- `fluidsim_home`
- `fluidsim_hybrid`

を走らせます。

```bash
MODEL=path/to/model.stl

./build/fluidsim_legacy \
  --nx 200 --ny 150 --nz 120 \
  --tau 0.58 --force 3e-6 0 0 \
  --stl $MODEL \
  --steps 2000 --vtk-dir out_vtk/legacy --vtk-every 200

./build/fluidsim_home \
  --nx 200 --ny 150 --nz 120 \
  --tau 0.58 --force 3e-6 0 0 \
  --stl $MODEL \
  --steps 2000 --vtk-dir out_vtk/home --vtk-every 200

./build/fluidsim_hybrid \
  --nx 200 --ny 150 --nz 120 \
  --tau 0.58 --force 3e-6 0 0 \
  --stl $MODEL \
  --hybrid-band 2 \
  --steps 2000 --vtk-dir out_vtk/hybrid --vtk-every 200
```

### VTK を GIF にする

`fluidsim_tgv` の VTK (speed_legacy_*, speed_home_*) を想定したスクリプトですが、
同様の形式なら流用できます。

```bash
python scripts/vtk_slice_to_png.py out_vtk/tgv_vtk out_vtk/frames --gif out_vtk/midplane.gif
```

---

## 2) Hybrid(B0) の band(d0) を sweep する

GUI 実行ファイル `fluidsim_compare` には、B0 の band 厚 `d0` を sweep して
統計量を CSV で出す headless モードがあります。

例（任意の STL、AMR=2、d0=0..6 を 800 step ずつ）:

```bash
./build/fluidsim_compare \
  --stl path/to/model.stl \
  --amr-factor 2 --amr-threshold 0.5 \
  --hybrid-sweep-max 6 --hybrid-sweep-steps 800 \
  > results.csv
```

プロット:

```bash
python scripts/plot_b0_sweep.py results.csv out.png
```

---

## 3) Taylor–Green vortex (TGV) ベンチ

`fluidsim_tgv` は 2D の Taylor–Green vortex を 3D に押し出した初期条件で走らせ、
エネルギー減衰と誤差を CSV に出します。

```bash
# Windows(例): build/Release/...
./build/fluidsim_tgv --steps 400 --sample-every 20 --u0 0.06 --tau 0.58

# プロット
python scripts/plot_tgv.py out_vtk/tgv_results.csv out_vtk/tgv_plot.png
```

---

## 4) 2D 渦（初期場だけ欲しい場合）

シミュレーションを回すのではなく、まず「2D の渦初期条件」を VTK に吐きたい場合:

```bash
python scripts/generate_tgv_2d.py --nx 128 --ny 128 --u0 0.06 --out out_vtk
```

ParaView で `tgv2d_rho.vtk`, `tgv2d_vel.vtk` を読み込んで確認できます。

---

## 注意（再現性のために）

- 障害物は **自動投入されません**。必ず `--stl` を明示してください。
- 比較するときは、少なくとも以下を揃えるのが安全です。
  - `(Nx,Ny,Nz)`, `tau`, `force`, `steps`, `substeps`
  - voxelize パラメータ (`--voxel-scale`, `--voxel-translate`, `--amr-factor`, `--amr-threshold`)
