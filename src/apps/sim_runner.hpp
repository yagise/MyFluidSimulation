// sim_runner.hpp
//
// 従来法 / HOME 法 / Hybrid(B0) を同じ入出力形式で実験できるようにするための
// 実行ループ共通化ファイルです。
//
// ここでは以下を提供します:
// - run_single_solver<...>() : 単一ソルバ(従来/HOME)を headless で回して VTK を出す
// - run_hybrid_b0() : Hybrid(B0) を headless で回す
// - B0 の band(d0) 用に distance-to-solid を BFS で作る処理
//
// 論文用の整理ポイント:
// - fan/teardrop の自動投入はしない (必要なら --fan/--teardrop を明示)
// - 回転体や移動壁などのデバッグ機能はここには入れない
//

#pragma once

#include <cmath>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

#include "apps/sim_common.hpp"

// summary: launch_speed の処理を行う
// param u: 入力パラメータ
// param v: 入力パラメータ
// param w: 入力パラメータ
// param out: 入力パラメータ
// param N: 入力パラメータ
// return: なし
extern "C" void launch_speed(const float* u, const float* v, const float* w, float* out, int N);

//
// 近傍探索（Hybrid B0 用）
//

// summary: build_distance_to_solid_6n の処理を行う
// param solidMask: 入力パラメータ
// param Nx: 入力パラメータ
// param Ny: 入力パラメータ
// param Nz: 入力パラメータ
// param maxDist: 入力パラメータ
// return: 戻り値
inline std::vector<int> build_distance_to_solid_6n(const std::vector<unsigned char>& solidMask,
                                                   int Nx, int Ny, int Nz,
                                                   int maxDist)
{
    const int N = Nx * Ny * Nz;
    std::vector<int> dist(N, -1);
    if (maxDist <= 0) return dist;

    std::vector<int> q;
    q.reserve((size_t)N / 4);

    // solid を全て距離0としてキューに投入
    for (int i = 0; i < N; ++i) {
        if (solidMask[(size_t)i]) {
            dist[i] = 0;
            q.push_back(i);
        }
    }

    const int strideY = Nx;
    const int strideZ = Nx * Ny;

    auto pushIfUnvisited = [&](int nid, int nd) {
        if (dist[nid] == -1) {
            dist[nid] = nd;
            q.push_back(nid);
        }
    };

    // BFS
    size_t head = 0;
    while (head < q.size()) {
        const int id = q[head++];
        const int dcur = dist[id];
        if (dcur >= maxDist) continue;

        const int z = id / strideZ;
        const int rem = id - z * strideZ;
        const int y = rem / strideY;
        const int x = rem - y * strideY;
        const int nd = dcur + 1;

        if (x > 0)      pushIfUnvisited(id - 1,        nd);
        if (x < Nx - 1) pushIfUnvisited(id + 1,        nd);
        if (y > 0)      pushIfUnvisited(id - strideY,  nd);
        if (y < Ny - 1) pushIfUnvisited(id + strideY,  nd);
        if (z > 0)      pushIfUnvisited(id - strideZ,  nd);
        if (z < Nz - 1) pushIfUnvisited(id + strideZ,  nd);
    }
    return dist;
}

// summary: build_legacy_mask_from_distance の処理を行う
// param solidMask: 入力パラメータ
// param distToSolid: 入力パラメータ
// param d0: 入力パラメータ
// return: 戻り値
inline std::vector<unsigned char> build_legacy_mask_from_distance(const std::vector<unsigned char>& solidMask,
                                                                  const std::vector<int>& distToSolid,
                                                                  int d0)
{
    const size_t N = solidMask.size();
    std::vector<unsigned char> isLegacy(N, 0);
    for (size_t i = 0; i < N; ++i) {
        if (solidMask[i]) continue; // solid は対象外
        const int di = distToSolid[i];
        if (di >= 0 && di <= d0) isLegacy[i] = 1;
    }
    return isLegacy;
}

//
// 単一ソルバ実行
//

// / 単一ソルバ（従来法 or HOME 法）を回す
// /
// / - 初期条件: run.initMode に従って rho を作り、u=0 として平衡分布へ再初期化
// / - 外力: dom.fx,fy,fz を Domain にセット（各ソルバ内部で適用）
// / - VTK 出力: rho / |u| / vel を一定間隔で出力
template <class Solver>
// summary: 処理を実行する
// param domCfg: 入力パラメータ
// param obsCfg: 入力パラメータ
// param runCfg: 入力パラメータ
// param tag: 入力パラメータ
// return: 戻り値
inline int run_single_solver(const DomainConfig& domCfg,
                             const ObstacleConfig& obsCfg,
                             const RunConfig& runCfg,
                             const char* tag)
{
    //
    // 1) 障害物(voxel)の作成
    //
    ObstacleData obstacle = build_obstacle(domCfg, obsCfg);

    //
    // 2) ソルバ初期化
    //
    Domain d;
    d.Nx = domCfg.Nx;
    d.Ny = domCfg.Ny;
    d.Nz = domCfg.Nz;
    d.tau = domCfg.tau;
    d.forceX = domCfg.fx;
    d.forceY = domCfg.fy;
    d.forceZ = domCfg.fz;

    Solver solver;
    solver.init(d);
    solver.setSolidMask(obstacle.solidMask.data());

    const int N = d.Nx * d.Ny * d.Nz;

    //
    // 3) 初期条件（rho を作って平衡へ）
    //
    // 論文用整理:
    // - 初期条件の作り方を全ソルバで統一するため、
    // rho,u を与えて平衡へAPI を用意している。
    // - Legacy は分布 f_i を平衡に再構成する。
    // - HOME(moment-encoded) は moments を (rho,u,S=0) で再構成する。
    float* d_rhoInit = build_initial_rho_device(domCfg, runCfg);
    solver.reinitEquilibriumFromMacro(d_rhoInit, nullptr, nullptr, nullptr);
    cudaFree(d_rhoInit);

    //
    // 4) VTK 出力準備
    //
    auto vtkDirOpt = prepare_vtk_dir(runCfg);
    float* d_speed = nullptr;
    if (vtkDirOpt.has_value()) {
        cudaMalloc(&d_speed, sizeof(float) * static_cast<size_t>(N));
    }

    auto dump_vtk = [&](int step) {
        if (!vtkDirOpt.has_value()) return;
        const std::filesystem::path& dir = *vtkDirOpt;

        // |u|
        launch_speed(solver.d_u(), solver.d_v(), solver.d_w(), d_speed, N);

        // ρ
        write_scalar_vtk(vtk_path(dir, std::string("rho_") + tag, step).c_str(),
                         solver.d_rho(), d.Nx, d.Ny, d.Nz);
        // |u|
        write_scalar_vtk(vtk_path(dir, std::string("speed_") + tag, step).c_str(),
                         d_speed, d.Nx, d.Ny, d.Nz);
        // u
        write_vector_vtk(vtk_path(dir, std::string("vel_") + tag, step).c_str(),
                         solver.d_u(), solver.d_v(), solver.d_w(), d.Nx, d.Ny, d.Nz);
    };

    if (vtkDirOpt.has_value()) {
        std::printf("[vtk] %s: dir=%s every=%d\n", tag, vtkDirOpt->string().c_str(), runCfg.vtkEvery);
        dump_vtk(0);
    }

    //
    // 5) 実行ループ
    //
    std::printf("[run] %s  Nx=%d Ny=%d Nz=%d  tau=%.4g  force=(%.3g,%.3g,%.3g)  steps=%d substeps=%d\n",
                tag, d.Nx, d.Ny, d.Nz, d.tau, d.forceX, d.forceY, d.forceZ,
                runCfg.steps, runCfg.substeps);

    for (int step = 1; step <= runCfg.steps; ++step) {
        solver.step(runCfg.substeps);

        if (vtkDirOpt.has_value() && runCfg.vtkEvery > 0 && (step % runCfg.vtkEvery) == 0) {
            dump_vtk(step);
        }
    }

    if (d_speed) cudaFree(d_speed);
    return 0;
}

//
// Hybrid(B0) 実行
//

// / Hybrid(B0) を回す
// /
// / Hybrid は "setLegacyMapping" が必要なため別関数に分ける。
template <class HybridSolver>
// summary: 処理を実行する
// param domCfg: 入力パラメータ
// param obsCfg: 入力パラメータ
// param runCfg: 入力パラメータ
// param hybCfg: 入力パラメータ
// return: 戻り値
inline int run_hybrid_b0(const DomainConfig& domCfg,
                         const ObstacleConfig& obsCfg,
                         const RunConfig& runCfg,
                         const HybridConfig& hybCfg)
{
    ObstacleData obstacle = build_obstacle(domCfg, obsCfg);

    Domain d;
    d.Nx = domCfg.Nx;
    d.Ny = domCfg.Ny;
    d.Nz = domCfg.Nz;
    d.tau = domCfg.tau;
    d.forceX = domCfg.fx;
    d.forceY = domCfg.fy;
    d.forceZ = domCfg.fz;

    const int N = d.Nx * d.Ny * d.Nz;

    // dist と isLegacy を作る（solid が無い場合は全部0になる）
    const int maxDist = std::max(1, hybCfg.bandD0);
    std::vector<int> dist = build_distance_to_solid_6n(obstacle.solidMask, d.Nx, d.Ny, d.Nz, maxDist);
    std::vector<unsigned char> isLegacy = build_legacy_mask_from_distance(obstacle.solidMask, dist, hybCfg.bandD0);

    // Hybrid ソルバ
    HybridSolver solver;
    solver.init(d, /*nLegacyCells*/ 0);
    solver.setSolidMask(obstacle.solidMask.data());
    solver.setLegacyMapping(isLegacy.data(), nullptr);

    // 初期条件
    float* d_rhoInit = build_initial_rho_device(domCfg, runCfg);
    solver.reinitEquilibriumFromMacro(d_rhoInit, nullptr, nullptr, nullptr);
    cudaFree(d_rhoInit);

    // VTK
    auto vtkDirOpt = prepare_vtk_dir(runCfg);
    float* d_speed = nullptr;
    if (vtkDirOpt.has_value()) cudaMalloc(&d_speed, sizeof(float) * static_cast<size_t>(N));

    auto dump_vtk = [&](int step) {
        if (!vtkDirOpt.has_value()) return;
        const std::filesystem::path& dir = *vtkDirOpt;
        launch_speed(solver.d_u(), solver.d_v(), solver.d_w(), d_speed, N);
        write_scalar_vtk(vtk_path(dir, "rho_hybrid", step).c_str(), solver.d_rho(), d.Nx, d.Ny, d.Nz);
        write_scalar_vtk(vtk_path(dir, "speed_hybrid", step).c_str(), d_speed, d.Nx, d.Ny, d.Nz);
        write_vector_vtk(vtk_path(dir, "vel_hybrid", step).c_str(),
                         solver.d_u(), solver.d_v(), solver.d_w(), d.Nx, d.Ny, d.Nz);
    };

    if (vtkDirOpt.has_value()) {
        std::printf("[vtk] hybrid: dir=%s every=%d\n", vtkDirOpt->string().c_str(), runCfg.vtkEvery);
        dump_vtk(0);
    }

    std::printf("[run] hybrid(B0)  d0=%d  Nx=%d Ny=%d Nz=%d  tau=%.4g  force=(%.3g,%.3g,%.3g)  steps=%d substeps=%d\n",
                hybCfg.bandD0, d.Nx, d.Ny, d.Nz, d.tau, d.forceX, d.forceY, d.forceZ,
                runCfg.steps, runCfg.substeps);

    for (int step = 1; step <= runCfg.steps; ++step) {
        solver.step(runCfg.substeps);
        if (vtkDirOpt.has_value() && runCfg.vtkEvery > 0 && (step % runCfg.vtkEvery) == 0) {
            dump_vtk(step);
        }
    }

    if (d_speed) cudaFree(d_speed);
    return 0;
}
