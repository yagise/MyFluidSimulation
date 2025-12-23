# ビルド手順

このプロジェクトは **CUDA + CMake** を使ってビルドします。

GUI ビューア (`fluidsim_compare`) をビルドする場合は **OpenGL 関連の依存 (GLFW / GLAD)** が追加で必要です。
一方、論文用の headless 実行 (`fluidsim_legacy/home/hybrid`) だけを使う場合は、`-DFLUIDSIM_BUILD_GUI=OFF` で GUI を無効化できます。

---

## 必要環境

必須:

- NVIDIA GPU (CUDA 対応)
- CUDA Toolkit (nvcc)
- CMake >= 3.20
- C++17 対応コンパイラ

依存ライブラリ:

- **GLM**（ヘッダオンリー。voxelize などでも使用）
- **GLFW / GLAD**（GUI をビルドする場合のみ）

このリポジトリは `find_package(glfw3)`, `find_package(glad CONFIG)`, `find_package(glm CONFIG)` を使うため、
**vcpkg** による導入が最も簡単です。

---

## 推奨: vcpkg(manifest) で依存を導入

このリポジトリには `vcpkg.json` を同梱してあります。

### 1) vcpkg を用意する

```bash
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.sh
```

Windows の場合は `bootstrap-vcpkg.bat` を実行してください。

### 2) 環境変数を設定する

```bash
export VCPKG_ROOT=/path/to/vcpkg
```

Windows PowerShell 例:

```powershell
$env:VCPKG_ROOT = "C:\path\to\vcpkg"
```

---

## Linux (single-config) のビルド例

### GUI あり（既定）

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake

cmake --build build -j
```

### GUI なし（headless のみ）

HPC/サーバ等で OpenGL を入れたくない場合は、GUI を OFF にします。

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DFLUIDSIM_BUILD_GUI=OFF \
  -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake

cmake --build build -j
```

---

## Windows (Visual Studio) のビルド例

Visual Studio のジェネレータは multi-config なので、実行ファイルは通常 `build/Release/` にできます。

```powershell
cmake -S . -B build `
  -DCMAKE_TOOLCHAIN_FILE=$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake

cmake --build build --config Release -j
```

GUI なしにしたい場合:

```powershell
cmake -S . -B build `
  -DFLUIDSIM_BUILD_GUI=OFF `
  -DCMAKE_TOOLCHAIN_FILE=$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake

cmake --build build --config Release -j
```

---

## CMakePresets を使う（任意）

`CMakePresets.json` を同梱してあります。
`VCPKG_ROOT` を環境変数で設定した上で、次のようにできます。

```bash
cmake --preset release-vcpkg
cmake --build --preset build-release
```

---

## よくあるトラブル

### `Could not find glfw3` / `Could not find glad`

- `FLUIDSIM_BUILD_GUI=ON` のままだと GUI 依存が必須です。
  headless のみでよければ `-DFLUIDSIM_BUILD_GUI=OFF` を指定してください。
- vcpkg を使う場合は `-DCMAKE_TOOLCHAIN_FILE=.../vcpkg.cmake` を付けてください。

### CUDA のアーキテクチャ

環境によっては `CMAKE_CUDA_ARCHITECTURES` を指定すると安定します。

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=native \
  -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
```

`native` が使えない CMake 版本の場合は数値(例: `86`)を指定してください。
