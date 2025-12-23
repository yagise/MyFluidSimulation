# 使い方（コマンドライン引数）

このリポジトリには複数の実行ファイルがあります。

- `fluidsim_compare` : GUI で可視化しながら比較（OpenGL 必須）
- `fluidsim_legacy`  : 従来法(BGK)のみ headless
- `fluidsim_home`    : HOME 法のみ headless
- `fluidsim_hybrid`  : Hybrid(B0)のみ headless
- `fluidsim_tgv`     : Taylor–Green vortex ベンチ
- `fluidsim_bench`   : 計測/実験用ベンチ（用途はソース参照）

> 注意: `fluidsim_legacy/home/hybrid` は「論文に載せやすい」ことを優先しているため、
> 引数は必要最低限のみ実装しています（GUI 専用の回転/移動などは入っていません）。

---

## 1) 論文用 headless 実行（共通）

対象:

- `fluidsim_legacy`
- `fluidsim_home`
- `fluidsim_hybrid`

### よく使う例

```bash
# 2000 step 実行し、200 step ごとに VTK を出力
./build/fluidsim_legacy --steps 2000 --vtk-dir out_vtk/legacy --vtk-every 200 --stl model.stl
./build/fluidsim_home   --steps 2000 --vtk-dir out_vtk/home   --vtk-every 200 --stl model.stl
./build/fluidsim_hybrid --steps 2000 --hybrid-band 2 --vtk-dir out_vtk/hybrid --vtk-every 200 --stl model.stl
```

### 引数一覧

#### ドメイン設定

- `--nx <int>` / `--ny <int>` / `--nz <int>`
  - 格子数（既定: 200×150×120）
- `--tau <float>`
  - 緩和時間（既定: 0.58）
- `--force <fx> <fy> <fz>`
  - 外力（既定: `fx=3e-6, fy=0, fz=0`）
- `--force-x <fx>` / `--force-y <fy>` / `--force-z <fz>`

#### 障害物（STL/手続き生成）

> デフォルトは **障害物なし**です。必要なら必ず明示してください。

- `--stl <path>`
  - STL を読み込んで障害物にする
- `--fan` / `--teardrop`
  - 手続き生成モデルを使う（明示したときだけ）

voxelize 関連（必要なら調整）:

- `--voxel-scale <float>`
  - 単位立方体に収めるためのスケール（既定: 0.5）
- `--voxel-translate <tx> <ty> <tz>`
  - モデルの平行移動（既定: 0.5 0.5 0.5）
- `--amr`
  - 境界の voxelize を 2倍 supersampling（`--amr-factor 2` 相当）
- `--amr-factor <int>`
  - supersampling 倍率（既定: 1）
- `--amr-threshold <0..1>`
  - coarse cell を solid と判定する占有率（既定: 0.5）

#### 実行と出力

- `--steps <int>`
  - 実行ステップ数（既定: 200）
- `--substeps <int>`
  - 1 step あたりの内部 step 回数（既定: 1）
- `--vtk-dir <dir>`
  - VTK 出力先ディレクトリ（指定しないと出力しない）
- `--vtk-every <int>`
  - 何 step ごとに VTK を出すか（0 で無効）

#### 初期条件（必要な場合のみ）

- `--init none|gauss|slab`
  - 初期条件の種類（既定: `none`）
- `--amp <float>`
  - ρ の摂動振幅（既定: 1e-3）
- `--sigma <float>`
  - gauss の幅（既定: 0.08）
- `--center <x> <y> <z>`
  - gauss 中心（既定: 0.6 0.5 0.5）
- `--axis x|y|z`
  - slab の軸（既定: x）
- `--slab-center <float>`
  - slab 中心（既定: 0.5）
- `--width <float>`
  - slab 幅（既定: 0.2）

#### Hybrid(B0) のみ

- `--hybrid-band <int>` / `--hybrid-d0 <int>`
  - 壁から距離 `dist_to_solid <= d0` の流体セルを Legacy 扱いにする band 厚
  - 既定: 2

---

## 2) GUI 比較 (`fluidsim_compare`)

基本:

```bash
./build/fluidsim_compare --stl path/to/model.stl
```

手法の切り替え:

```bash
./build/fluidsim_compare --method legacy
./build/fluidsim_compare --method home
./build/fluidsim_compare --method hybrid
./build/fluidsim_compare --method compare
```

主な引数（抜粋）:

- `--stl <path>` / `--fan` / `--teardrop`
- `--amr` / `--amr-factor <n>` / `--amr-threshold <0..1>`
- `--init none|gauss|slab` + `--amp` 等（初期条件）
- `--vtk-dir <dir>` + `--vtk-every <n>`
  - GUI 実行でも VTK を出せます
- `--vtk-solver legacy|home`
  - どちらのソルバで VTK を出すか
- `--hybrid-band <int>`
- `--hybrid-sweep-max <M>` / `--hybrid-sweep-steps <S>`
  - B0 sweep を headless で実行して CSV を stdout に出して終了

キー操作:

- `1`: Pressure (rho)
- `2`: Speed (|u|)
- `3`: Error (rho_HOME - rho_Legacy)
- `O`: 障害物メッシュ表示の ON/OFF

---

## 3) VTK の可視化

- VTK は legacy ASCII です。
- ParaView 等で `rho_*.vtk`, `speed_*.vtk`, `vel_*.vtk` を読み込んでください。

---

## 4) 2D 渦の生成（Python）

```bash
python scripts/generate_tgv_2d.py --nx 128 --ny 128 --u0 0.06 --out out_vtk
```

出力:

- `tgv2d_rho.vtk`
- `tgv2d_vel.vtk`