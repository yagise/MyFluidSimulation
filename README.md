# FluidSimExtended

CUDA 荳翫〒蜍輔￥ 3D LBM(D3Q19) 縺ｮ螳滄ｨ鍋畑繧ｳ繝ｼ繝峨〒縺吶・
隲匁枚縺ｧ謇ｱ縺・ｄ縺吶＞繧医≧縺ｫ縲∵ｬ｡縺ｮ **3縺､縺ｮ謇区ｳ輔ｒ main 繝輔ぃ繧､繝ｫ縺斐→縺ｫ蛻・屬**縺励※縺・∪縺吶・
- **蠕捺擂豕・(Legacy / BGK)**  窶ｦ `fluidsim_legacy`
- **HOME 豕・* 窶ｦ `fluidsim_home`
- **蠕捺擂豕・+ HOME 豕・(Hybrid(B0))** 窶ｦ `fluidsim_hybrid`

> 繝｡繝｢: 縺薙・繝ｪ繝昴ず繝医Μ蜀・・ HOME 縺ｯ縲後Δ繝ｼ繝｡繝ｳ繝井ｿ晏ｭ假ｼ・oment-encoded・峨阪ｒ蟆主・縺励◆迚医〒縺吶・> - 1繧ｻ繝ｫ縺ゅ◆繧・`rho,u(3),S(6)` 縺ｮ **10螟画焚 ﾃ・繝舌ャ繝輔ぃ** 繧剃ｿ晄戟縺励∝・蟶・`f_i` 縺ｯ菫晄戟縺励∪縺帙ｓ縲・> - streaming 縺ｫ蠢・ｦ√↑ `f_i` 縺ｯ豈弱せ繝・ャ繝・`(rho,u,S)` 縺九ｉ **2谺｡縺ｾ縺ｧ縺ｮ regularized 蠖｢蠑・*縺ｧ蜀肴ｧ区・縺励∪縺吶・> - 縺・ｏ繧・ｋ HOME-LBM 縺ｮ蜈ｨ隕∫ｴ・井ｾ・ D3Q27, 3谺｡Hermite蜀肴ｧ区・, central-moment 陦晉ｪ√↑縺ｩ・峨・譛ｬ繧ｳ繝ｼ繝峨↓縺ｯ蜷ｫ繧√※縺・∪縺帙ｓ縲・
蜉縺医※縲＾penGL 縺ｧ蜿ｯ隕門喧縺励↑縺後ｉ豈碑ｼ・〒縺阪ｋ GUI 螳溯｡後ヵ繧｡繧､繝ｫ `fluidsim_compare`縲・繝吶Φ繝√・繝ｼ繧ｯ逕ｨ縺ｮ `fluidsim_tgv` / `fluidsim_bench` 繧ょ性縺ｿ縺ｾ縺吶・
> **驥崎ｦ・*
> - 繝・ヰ繝・げ逶ｮ逧・・ **髫懷ｮｳ迚ｩ繝｢繝・Ν(fan / teardrop) 閾ｪ蜍墓兜蜈･**縺ｯ陦後＞縺ｾ縺帙ｓ縲・>   髫懷ｮｳ迚ｩ繧貞・繧後ｋ蝣ｴ蜷医・ `--stl` / `--fan` / `--teardrop` 繧・**蠢・★譏守､ｺ**縺励※縺上□縺輔＞縲・> - 蝗櫁ｻ｢菴薙・遘ｻ蜍募｣√↑縺ｩ縺ｮ縲梧ｯ碑ｼ・擅莉ｶ繧貞｢励ｄ縺呎ｩ溯・縲阪・蜈･繧後※縺・∪縺帙ｓ縲・
---

## 逶ｮ谺｡

- [萓晏ｭ倬未菫・(#萓晏ｭ倬未菫・
- [繝薙Ν繝云(#繝薙Ν繝・
- [螳溯｡御ｾ犠(#螳溯｡御ｾ・
- [蜃ｺ蜉・VTK)](#蜃ｺ蜉孥tk)
- [Python 繧ｹ繧ｯ繝ｪ繝励ヨ](#python-繧ｹ繧ｯ繝ｪ繝励ヨ)
- [繝・ぅ繝ｬ繧ｯ繝医Μ讒区・](#繝・ぅ繝ｬ繧ｯ繝医Μ讒区・)
- [隲匁枚縺ｧ縺ｮ蛻ｩ逕ｨ](#隲匁枚縺ｧ縺ｮ蛻ｩ逕ｨ)

隧ｳ邏ｰ縺ｪ隱ｬ譏弱・ `docs/` 莉･荳九ｂ蜿ら・縺励※縺上□縺輔＞縲・
- `docs/BUILD.md`  : 萓晏ｭ倬未菫ゅ・繝薙Ν繝峨・隧ｳ隱ｬ (vcpkg 謗ｨ螂ｨ)
- `docs/USAGE.md`  : 繧ｳ繝槭Φ繝峨Λ繧､繝ｳ蠑墓焚荳隕ｧ
- `docs/EXPERIMENTS.md` : B0 sweep / TGV 縺ｪ縺ｩ縺ｮ螳滄ｨ薙Γ繝｢

---

## 萓晏ｭ倬未菫・
蠢・・

- NVIDIA GPU + CUDA Toolkit
- CMake >= 3.20
- C++17 繧ｳ繝ｳ繝代う繝ｩ

GUI(`fluidsim_compare`) 縺ｧ蠢・ｦ・

- GLFW
- GLAD
- GLM

縺薙・繝ｪ繝昴ず繝医Μ縺ｧ縺ｯ `find_package(glfw3)`, `find_package(glad CONFIG)`, `find_package(glm CONFIG)` 繧剃ｽｿ縺・◆繧√・**vcpkg 繧堤畑縺・◆萓晏ｭ伜ｰ主・縺梧怙繧ら｢ｺ螳・*縺ｧ縺呻ｼ亥ｾ瑚ｿｰ・峨・
---

## 繝薙Ν繝・
### 謗ｨ螂ｨ: vcpkg(manifest) 繧剃ｽｿ縺・
縺薙・繝ｪ繝昴ず繝医Μ縺ｫ縺ｯ `vcpkg.json` 繧貞酔譴ｱ縺励※縺ゅｋ縺ｮ縺ｧ縲」cpkg 縺ｮ manifest 繝｢繝ｼ繝峨′菴ｿ縺医∪縺吶・
```bash
# 萓・ Linux
export VCPKG_ROOT=/path/to/vcpkg

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake

cmake --build build -j
```

GUI(OpenGL) 縺御ｸ崎ｦ√〒 headless 螳溯｡後□縺台ｽｿ縺・◆縺・ｴ蜷医・縲，Make 縺ｫ谺｡繧定ｿｽ蜉縺励※縺上□縺輔＞縲・
```bash
-DFLUIDSIM_BUILD_GUI=OFF
```

Windows(Visual Studio) 縺ｮ萓九・ `docs/BUILD.md` 繧貞盾辣ｧ縺励※縺上□縺輔＞縲・
### 繝偵Φ繝・ CUDA 繧｢繝ｼ繧ｭ繝・け繝√Ε

迺ｰ蠅・↓繧医▲縺ｦ縺ｯ `CMAKE_CUDA_ARCHITECTURES` 繧呈欠螳壹☆繧九→螳牙ｮ壹＠縺ｾ縺吶・
```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=native
```

---

## 螳溯｡御ｾ・
### 1) GUI 縺ｧ豈碑ｼ・(`fluidsim_compare`)

```bash
./build/fluidsim_compare --stl path/to/model.stl

# 謇狗ｶ壹″逕滓・繝｢繝・Ν・域・遉ｺ縺励◆縺ｨ縺阪□縺台ｽｿ繧上ｌ縺ｾ縺呻ｼ・./build/fluidsim_compare --teardrop
./build/fluidsim_compare --fan
```

繧ｭ繝ｼ謫堺ｽ・

<<<<<<< HEAD
- `1` : 圧力（密度 rho）
- `2` : 速度の大きさ |u|
- `3` : 誤差（rho_HOME - rho_Legacy）
- `O` : 障害物表示の ON/OFF
- `G` : Legacy 表示
- `H` : HOME 表示
- `B` : Hybrid(B0) 表示
- `C` : Compare 表示

### 2) GUIなしでvtk出力

GUI を使わず、各手法を独立に回して VTK を出力します。
=======
- `1` : 蝨ｧ蜉幢ｼ亥ｯ・ｺｦ rho・・- `2` : 騾溷ｺｦ縺ｮ螟ｧ縺阪＆ |u|
- `3` : 隱､蟾ｮ・・ho_HOME - rho_Legacy・・- `O` : 髫懷ｮｳ迚ｩ陦ｨ遉ｺ縺ｮ ON/OFF
- `G` : Legacy 陦ｨ遉ｺ
- `H` : HOME 陦ｨ遉ｺ
- `B` : Hybrid(B0) 陦ｨ遉ｺ
- `C` : Compare 陦ｨ遉ｺ
>>>>>>> 648ee65 (feat:バグ修正)

### 2) 隲匁枚逕ｨ・・eadless 螳溯｡鯉ｼ・
GUI 繧剃ｽｿ繧上★縲∝推謇区ｳ輔ｒ迢ｬ遶九↓蝗槭＠縺ｦ VTK 繧貞・蜉帙＠縺ｾ縺吶・
```bash
# 萓・ 2000 step 蝗槭＠縺ｦ 200 step 縺斐→縺ｫ VTK 蜃ｺ蜉・./build/fluidsim_legacy --steps 2000 --vtk-dir out_vtk/legacy --vtk-every 200 --stl path/to/model.stl
./build/fluidsim_home   --steps 2000 --vtk-dir out_vtk/home   --vtk-every 200 --stl path/to/model.stl
./build/fluidsim_hybrid --steps 2000 --hybrid-band 2 --vtk-dir out_vtk/hybrid --vtk-every 200 --stl path/to/model.stl
```

髫懷ｮｳ迚ｩ繧剃ｽｿ繧上↑縺・ｴ蜷茨ｼ医ョ繝輔か繝ｫ繝茨ｼ峨・ `--stl` 繧貞､悶＠縺ｾ縺吶・
### 3) 2D 貂ｦ・・aylor窶敵reen vortex・峨・逕滓・

縲・D縺ｧ貂ｦ繧堤匱逕溘＆縺帙ｋ縲榊・譛溷ｴ繧・Python 縺ｧ逕滓・縺励∪縺呻ｼ・TK蜃ｺ蜉幢ｼ峨・
```bash
python scripts/generate_tgv_2d.py --nx 128 --ny 128 --u0 0.06 --out out_vtk
```

---

## 蜃ｺ蜉・VTK)

<<<<<<< HEAD
VTK 出力は **VTK legacy ASCII** です。
=======
headless 螳溯｡後・ VTK 蜃ｺ蜉帙・ **VTK legacy ASCII** 縺ｧ縺吶・
萓・
>>>>>>> 648ee65 (feat:バグ修正)

- `rho_legacy_000200.vtk` : 蟇・ｺｦ
- `speed_home_000200.vtk` : |u|
<<<<<<< HEAD
- `vel_hybrid_000200.vtk` : 速度ベクトル

=======
- `vel_hybrid_000200.vtk` : 騾溷ｺｦ繝吶け繝医Ν

ParaView 遲峨〒隱ｭ縺ｿ霎ｼ繧薙〒蜿ｯ隕門喧縺励※縺上□縺輔＞縲・
>>>>>>> 648ee65 (feat:バグ修正)
---

## Python 繧ｹ繧ｯ繝ｪ繝励ヨ

`scripts/` 縺ｫ邁｡蜊倥↑陬懷勧繧ｹ繧ｯ繝ｪ繝励ヨ縺後≠繧翫∪縺吶・
- `generate_tgv_2d.py` : 2D Taylor窶敵reen 貂ｦ縺ｮ蛻晄悄蝣ｴ(VTK)逕滓・・域ｨ呎ｺ悶Λ繧､繝悶Λ繝ｪ縺ｮ縺ｿ・・- `plot_tgv.py` : `fluidsim_tgv` 縺ｮ CSV 繧偵・繝ｭ繝・ヨ・郁ｦ・ matplotlib・・- `vtk_slice_to_png.py` : VTK 縺ｮ荳ｭ髢捺妙髱｢繧・PNG/GIF 縺ｫ螟画鋤・郁ｦ・ numpy, matplotlib, imageio(莉ｻ諢・・・- `plot_b0_sweep.py`, `plot_b0_amr_overlay.py` : B0 sweep 縺ｮ蜿ｯ隕門喧

Python 萓晏ｭ倥・ `requirements.txt` 繧貞盾辣ｧ縺励※縺上□縺輔＞縲・
---

## 繝・ぅ繝ｬ繧ｯ繝医Μ讒区・

```
.
<<<<<<< HEAD
├─ src/
│  ├─ lbm/        # LBM 本体 (Legacy / HOME / Hybrid)
│  ├─ apps/       # headless
│  ├─ bench/      # ベンチマーク (TGV 等)
│  ├─ geom/       # STL ロード / voxelize
│  ├─ fields/     # 初期条件・場の演算
│  └─ render/     # GUI 用 OpenGL 可視化
├─ scripts/       # Python / PowerShell 補助
└─ docs/          # ビルド・使い方・実験手順
```

=======
笏懌楳 src/
笏・ 笏懌楳 lbm/        # LBM 譛ｬ菴・(Legacy / HOME / Hybrid)
笏・ 笏懌楳 apps/       # 隲匁枚蜷代￠ headless main
笏・ 笏懌楳 bench/      # 繝吶Φ繝√・繝ｼ繧ｯ (TGV 遲・
笏・ 笏懌楳 geom/       # STL 繝ｭ繝ｼ繝・/ voxelize
笏・ 笏懌楳 fields/     # 蛻晄悄譚｡莉ｶ繝ｻ蝣ｴ縺ｮ貍皮ｮ・笏・ 笏披楳 render/     # GUI 逕ｨ OpenGL 蜿ｯ隕門喧
笏懌楳 scripts/       # Python / PowerShell 陬懷勧
笏披楳 docs/          # 繝薙Ν繝峨・菴ｿ縺・婿繝ｻ螳滄ｨ捺焔鬆・```

---

## 隲匁枚縺ｧ縺ｮ蛻ｩ逕ｨ

- 遐皮ｩｶ縺ｧ蛻ｩ逕ｨ縺吶ｋ蝣ｴ蜷医・ `CITATION.cff` 繧抵ｼ亥ｿ・ｦ√↓蠢懊§縺ｦ邱ｨ髮・＠縺ｦ・牙茜逕ｨ縺励※縺上□縺輔＞縲・- 繝ｩ繧､繧ｻ繝ｳ繧ｹ縺ｯ `LICENSE` 繧貞盾辣ｧ・亥・髢区凾縺ｫ蠢・★蜀・ｮｹ繧堤｢ｺ螳壹＠縺ｦ縺上□縺輔＞・峨・
>>>>>>> 648ee65 (feat:バグ修正)
