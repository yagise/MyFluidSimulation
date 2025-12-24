// sim_common.hpp
//
// 共通ユーティリティです。
//
// 目的:
// - 各手法の main を分離しつつ、引数処理・障害物(voxelize)・VTK 出力などを共通化する
// - デバッグ用の自動モデル挿入（手続き生成モデルを勝手に入れる等）は行わない
// - 任意の STL モデルで実験できるようにする
//
// 注意:
// - ここは実験・比較を回すためのユーティリティであり、数値手法そのもの
// (衝突・ストリーミング) は各 LBM 実装側にあります。
//

#pragma once

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "geom/stl_loader.hpp"
#include "geom/voxelize.hpp"
#include "lbm/lbm_common.hpp"
// 出力データを書き出す
extern "C" void write_scalar_vtk(const char* filename, const float* d_field, int Nx, int Ny, int Nz);
// 出力データを書き出す
extern "C" void write_vector_vtk(const char* filename,
                                 const float* d_fx,
                                 const float* d_fy,
                                 const float* d_fz,
                                 int Nx, int Ny, int Nz);
extern "C" void make_rho_gauss(float* d_rho, int Nx, int Ny, int Nz,
                               float cx, float cy, float cz, float sigma, float amp);
extern "C" void make_rho_slab(float* d_rho, int Nx, int Ny, int Nz,
                              int axis, float amp, float center, float width);
extern "C" void reinit_equilibrium_from_macro(const float* d_rho,
                                              const float* d_ux,
                                              const float* d_uy,
                                              const float* d_uz,
                                              float*       d_f,
                                              int          N);

//
// 実験に必要な設定
//

// / 実験用のドメイン設定（格子数・緩和時間・外力）
struct DomainConfig {
    int Nx = 200;
    int Ny = 150;
    int Nz = 120;
    float tau = 0.58f;
    float fx = 0.0f;
    float fy = 0.0f;
    float fz = 0.0f;
};

// / 障害物（STL or 手続き生成）に関する設定
struct ObstacleConfig {
    // 入力モデル
    std::string stlPath;         ///< --stl で指定された STL

    // voxelize
    float uniformScale = 0.5f;   ///< 単位立方体[0,1]^3 に収めるためのスケール
    float translateX = 0.5f;     ///< 位置合わせ（中心）
    float translateY = 0.5f;
    float translateZ = 0.5f;
    int amrFactor = 1;           ///< supersampling
    float amrThreshold = 0.5f;   ///< coarse cell を solid と判定する占有率
};

// / 実行ステップや VTK 出力設定
struct RunConfig {
    int steps = 200;
    int substeps = 1;
    std::string vtkDir;  ///< 空なら無効
    int vtkEvery = 0;    ///< 0 なら無効

    // 初期条件（必要な人だけ）
    // - none : rho=1, u=0
    // - gauss : rho=1+gaussian
    // - slab : rho=1+slab
    std::string initMode = "none";
    float amp = 1e-3f;
    float sigma = 0.08f;
    float cx = 0.6f, cy = 0.5f, cz = 0.5f;
    int slabAxis = 0;
    float slabCenter = 0.5f;
    float slabWidth = 0.2f;
};

// / B0（従来法+HOME）の band 設定
struct HybridConfig {
    int bandD0 = 2;  ///< dist_to_solid <= d0 のセルを Legacy として扱う
};

// / voxelize 後に得られる障害物データ
struct ObstacleData {
    bool hasObstacle = false;
    TriangleMesh mesh;
    std::vector<unsigned char> solidMask;  ///< size = Nx*Ny*Nz
};

//
// 引数処理
inline bool is_option(const char* s) {
    return s && s[0] == '-';
}
// 引数や入力設定を解析する
inline void parse_common_args(int argc, char** argv,
                              DomainConfig& dom,
                              ObstacleConfig& obs,
                              RunConfig& run,
                              HybridConfig* hyb /*=nullptr*/)
{
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](int remain) { return (i + remain) < argc; };

        // --- ドメイン ---
        if (a == "--nx" && need(1)) dom.Nx = std::stoi(argv[++i]);
        else if (a == "--ny" && need(1)) dom.Ny = std::stoi(argv[++i]);
        else if (a == "--nz" && need(1)) dom.Nz = std::stoi(argv[++i]);
        else if (a == "--tau" && need(1)) dom.tau = std::stof(argv[++i]);
        else if (a == "--force" && need(3)) {
            dom.fx = std::stof(argv[++i]);
            dom.fy = std::stof(argv[++i]);
            dom.fz = std::stof(argv[++i]);
        } else if (a == "--force-x" && need(1)) dom.fx = std::stof(argv[++i]);
        else if (a == "--force-y" && need(1)) dom.fy = std::stof(argv[++i]);
        else if (a == "--force-z" && need(1)) dom.fz = std::stof(argv[++i]);

        // --- 障害物 ---
        else if (a == "--stl" && need(1)) obs.stlPath = argv[++i];
        else if (a == "--voxel-scale" && need(1)) obs.uniformScale = std::stof(argv[++i]);
        else if (a == "--voxel-translate" && need(3)) {
            obs.translateX = std::stof(argv[++i]);
            obs.translateY = std::stof(argv[++i]);
            obs.translateZ = std::stof(argv[++i]);
        }
        else if ((a == "--amr")) {
            obs.amrFactor = std::max(obs.amrFactor, 2);
        }
        else if ((a == "--amr-factor" || a == "--amr-refine") && need(1)) {
            obs.amrFactor = std::max(1, std::stoi(argv[++i]));
        }
        else if (a == "--amr-threshold" && need(1)) {
            obs.amrThreshold = std::clamp(std::stof(argv[++i]), 0.0f, 1.0f);
        }

        // --- 実行 ---
        else if (a == "--steps" && need(1)) run.steps = std::max(0, std::stoi(argv[++i]));
        else if (a == "--substeps" && need(1)) run.substeps = std::max(1, std::stoi(argv[++i]));
        else if (a == "--vtk-dir" && need(1)) run.vtkDir = argv[++i];
        else if ((a == "--vtk-every" || a == "--vtk-interval") && need(1)) run.vtkEvery = std::max(0, std::stoi(argv[++i]));

        // --- 初期条件 ---
        else if (a == "--init" && need(1)) run.initMode = argv[++i];
        else if (a == "--amp" && need(1)) run.amp = std::stof(argv[++i]);
        else if (a == "--sigma" && need(1)) run.sigma = std::stof(argv[++i]);
        else if (a == "--center" && need(3)) {
            run.cx = std::stof(argv[++i]);
            run.cy = std::stof(argv[++i]);
            run.cz = std::stof(argv[++i]);
        }
        else if (a == "--axis" && need(1)) {
            char ax = argv[++i][0];
            run.slabAxis = (ax == 'y') ? 1 : (ax == 'z') ? 2 : 0;
        }
        else if (a == "--width" && need(1)) run.slabWidth = std::stof(argv[++i]);
        else if (a == "--slab-center" && need(1)) run.slabCenter = std::stof(argv[++i]);

        // --- Hybrid (B0) ---
        else if ((a == "--hybrid-band" || a == "--hybrid-d0") && need(1)) {
            // Hybrid 実行のときだけ有効。従来法/HOME の main では hyb==nullptr のまま。
            const int v = std::max(0, std::stoi(argv[++i]));
            if (hyb) hyb->bandD0 = v;
        }
    }

    // 最低限のクリップ
    dom.Nx = std::max(dom.Nx, 8);
    dom.Ny = std::max(dom.Ny, 8);
    dom.Nz = std::max(dom.Nz, 1);
    dom.tau = std::max(dom.tau, 0.51f);
    run.substeps = std::max(run.substeps, 1);
}

//
// 障害物ロード / voxelize
//
// 入力データを読み込む
inline std::optional<TriangleMesh> load_obstacle_mesh(const ObstacleConfig& obs) {
    if (!obs.stlPath.empty()) {
        TriangleMesh m;
        if (!load_stl(obs.stlPath, m)) {
            std::fprintf(stderr, "[error] STL の読み込みに失敗しました: %s\n", obs.stlPath.c_str());
            return std::nullopt;
        }
        return m;
    }

    // 何も指定されていない場合は障害物なし
    return std::nullopt;
}
inline ObstacleData build_obstacle(const DomainConfig& dom, const ObstacleConfig& obs) {
    const int N = dom.Nx * dom.Ny * dom.Nz;

    ObstacleData out;
    out.solidMask.assign((size_t)N, 0);

    auto meshOpt = load_obstacle_mesh(obs);
    if (!meshOpt.has_value()) {
        // 障害物なし
        out.hasObstacle = false;
        return out;
    }

    out.hasObstacle = true;
    out.mesh = std::move(*meshOpt);

    VoxelParams vp;
    vp.Nx = dom.Nx;
    vp.Ny = dom.Ny;
    vp.Nz = dom.Nz;
    vp.uniformScale = obs.uniformScale;
    vp.translate = glm::vec3(obs.translateX, obs.translateY, obs.translateZ);
    vp.refine = std::max(1, obs.amrFactor);
    vp.coverageThreshold = std::clamp(obs.amrThreshold, 0.0f, 1.0f);

    out.solidMask = voxelize_mesh_to_mask(out.mesh, vp);
    return out;
}

//
// 初期条件
inline float* build_initial_rho_device(const DomainConfig& dom, const RunConfig& run) {
    const int N = dom.Nx * dom.Ny * dom.Nz;
    float* d_rho = nullptr;
    cudaMalloc(&d_rho, sizeof(float) * (size_t)N);

    // 初期条件: none は rho=1
    if (run.initMode == "gauss") {
        make_rho_gauss(d_rho, dom.Nx, dom.Ny, dom.Nz,
                       run.cx, run.cy, run.cz, run.sigma, run.amp);
    } else if (run.initMode == "slab") {
        make_rho_slab(d_rho, dom.Nx, dom.Ny, dom.Nz,
                      run.slabAxis, run.amp, run.slabCenter, run.slabWidth);
    } else {
        // まず 1 で埋める
        std::vector<float> h((size_t)N, 1.0f);
        cudaMemcpy(d_rho, h.data(), sizeof(float) * (size_t)N, cudaMemcpyHostToDevice);
    }
    return d_rho;
}

//
// VTK 出力
inline std::string vtk_path(const std::filesystem::path& dir,
                            const std::string& stem,
                            int step)
{
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%s_%06d.vtk", stem.c_str(), step);
    return (dir / buf).string();
}
inline std::optional<std::filesystem::path> prepare_vtk_dir(const RunConfig& run) {
    if (run.vtkDir.empty() || run.vtkEvery <= 0) return std::nullopt;
    std::filesystem::path p(run.vtkDir);
    std::error_code ec;
    std::filesystem::create_directories(p, ec);
    if (ec) {
        std::fprintf(stderr, "[vtk] 出力ディレクトリ作成に失敗: %s (%s)\n",
                     run.vtkDir.c_str(), ec.message().c_str());
        return std::nullopt;
    }
    return p;
}
