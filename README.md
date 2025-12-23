# FluidSimExtended

CUDA 上で動く 3D LBM(D3Q19) の実験用コードです。
論文で扱いやすいように、次の **3つの手法を main ファイルごとに切り替え**ています。
- **従来法(Legacy / BGK)**  … `fluidsim_legacy`
- **HOME 法** … `fluidsim_home`
- **従来法 + HOME 法(Hybrid(B0))** … `fluidsim_hybrid`

> メモ: このリポジトリの HOME は「モーメント保存 (moment-encoded)」を導入した版です。
> - 1セルあたり `rho,u(3),S(6)` の **10変数バッファ** を保持し、`f_i` は保持しません。
> - streaming に必要な `f_i` は毎ステップ `(rho,u,S)` から **2次までの regularized 形式**で再構成します。
> - あらためて HOME-LBM の全要素（D3Q27、3次Hermite再構成、central-moment 衝突など）は本コードには含めていません。

加えて、OpenGL で可視化しながら比較できる GUI 実行ファイル `fluidsim_compare`、
ベンチマーク用の `fluidsim_tgv` / `fluidsim_bench` も含みます。

> **重要**
> - デフォルトでは **障害物モデル(fan / teardrop) 自動投入**は行いません。
>   障害物を入れる場合は `--stl` / `--fan` / `--teardrop` を **必ず明示**してください。
> - 回転体や移動壁などの「比較条件を増やす機能」は入れていません。
---

## 目次

- [依存関係](#依存関係)
- [ビルド](#ビルド)
- [実行例](#実行例)
- [出力(VTK)](#出力vtk)
- [Python スクリプト](#python-スクリプト)
- [ディレクトリ構成](#ディレクトリ構成)
- [論文での利用](#論文での利用)

詳細な説明は `docs/` 以下も参照してください。
- `docs/BUILD.md`  : 依存関係・ビルドの詳説 (vcpkg 推奨)
- `docs/USAGE.md`  : コマンドライン引数一覧
- `docs/EXPERIMENTS.md` : B0 sweep / TGV などの実験メモ

---

## 依存関係
必須

- NVIDIA GPU + CUDA Toolkit
- CMake >= 3.20
- C++17 コンパイラ

GUI(`fluidsim_compare`) で必須

- GLFW
- GLAD
- GLM

このリポジトリでは `find_package(glfw3)`, `find_package(glad CONFIG)`, `find_package(glm CONFIG)` を使うため、
**vcpkg を用いた依存導入が最も確実**です（後述）。

---

## ビルド
### 推奨: vcpkg(manifest) を使う
このリポジトリには `vcpkg.json` を同梱してあるので、vcpkg の manifest モードが使えます。
```bash
# 例: Linux
export VCPKG_ROOT=/path/to/vcpkg

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake

cmake --build build -j
```

GUI(OpenGL) が不要で headless 実行だけ使いたい場合は、CMake に次を追加してください。
```bash
-DFLUIDSIM_BUILD_GUI=OFF
```

Windows(Visual Studio) の例は `docs/BUILD.md` を参照してください。
### ヒント: CUDA アーキテクチャ

環境によっては `CMAKE_CUDA_ARCHITECTURES` を指定すると安定します。
```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=native
```

---

## 実行例
### 1) GUI で比較 (`fluidsim_compare`)

```bash
./build/fluidsim_compare --stl path/to/model.stl

# 手続き生成モデルは明示したときだけ使われます。
./build/fluidsim_compare --teardrop
./build/fluidsim_compare --fan
```

キー操作

- `1` : 圧力（密度 rho）
- `2` : 速度の大きさ |u|
- `3` : 誤差（rho_HOME - rho_Legacy）
- `O` : 障害物表示の ON/OFF
- `G` : Legacy 表示
- `H` : HOME 表示
- `B` : Hybrid(B0) 表示
- `C` : Compare 表示

### 2) 論文用 headless 実行
GUI を使わず、各手法を独立に回して VTK を出力します。
```bash
# 例: 2000 step 回して 200 step ごとに VTK 出力
./build/fluidsim_legacy --steps 2000 --vtk-dir out_vtk/legacy --vtk-every 200 --stl path/to/model.stl
./build/fluidsim_home   --steps 2000 --vtk-dir out_vtk/home   --vtk-every 200 --stl path/to/model.stl
./build/fluidsim_hybrid --steps 2000 --hybrid-band 2 --vtk-dir out_vtk/hybrid --vtk-every 200 --stl path/to/model.stl
```

障害物を使わない場合（デフォルト）は `--stl` を外します。
### 3) 2D 渦 (Taylor–Green vortex) 初期場生成

2D で渦を発生させる初期場を Python で生成します（VTK 出力）。
```bash
python scripts/generate_tgv_2d.py --nx 128 --ny 128 --u0 0.06 --out out_vtk
```

---

## 出力(VTK)

headless 実行の VTK 出力は **VTK legacy ASCII** です。

- `rho_legacy_000200.vtk` : 圧力
- `speed_home_000200.vtk` : |u|
- `vel_hybrid_000200.vtk` : 速度ベクトル

ParaView 等で読み込んで可視化してください。
---

## Python スクリプト

`scripts/` に簡単な補助スクリプトがあります。
- `generate_tgv_2d.py` : 2D Taylor–Green 渦の初期場(VTK)生成（標準ライブラリのみ）
- `plot_tgv.py` : `fluidsim_tgv` の CSV をプロット（要 matplotlib）
- `vtk_slice_to_png.py` : VTK の中間断面を PNG/GIF に変換（要 numpy, matplotlib, imageio。任意で ffmpeg）
- `plot_b0_sweep.py`, `plot_b0_amr_overlay.py` : B0 sweep の可視化

Python 依存は `requirements.txt` を参照してください。
---

## ディレクトリ構成

```
.
├─ src/
│  ├─ lbm/        # LBM 本体 (Legacy / HOME / Hybrid)
│  ├─ apps/       # 論文向け headless main
│  ├─ bench/      # ベンチマーク (TGV など)
│  ├─ geom/       # STL ローダ / voxelize
│  ├─ fields/     # 初期条件・場の演算
│  └─ render/     # GUI 用 OpenGL 可視化
├─ scripts/       # Python / PowerShell 補助
└─ docs/          # ビルド・使い方・実験手順
```

---

## 論文での利用

- 研究で利用する場合は `CITATION.cff` を（必要に応じて編集して）利用してください。
- ライセンスは `LICENSE` を参照し、公開時に必要事項を確定してください。
