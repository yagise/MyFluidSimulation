#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>

#include "lbm/lbm3d_legacy.hpp"
#include "lbm/lbm3d_home.hpp"
#include "lbm/lbm_common.hpp"
#include "common/cuda_utils.hpp"

// 初期条件をrho,u を与えて平衡へ統一するため、
// 各ソルバは reinitEquilibriumFromMacro(...) を持つ。
extern "C" void launch_speed(const float* u, const float* v, const float* w, float* out, int N);
extern "C" void write_scalar_vtk(const char* filename, const float* d_field, int Nx, int Ny, int Nz);
extern "C" void write_vector_vtk(const char* filename,
                                 const float* d_fx,
                                 const float* d_fy,
                                 const float* d_fz,
                                 int Nx, int Ny, int Nz);

static constexpr double kPi = 3.14159265358979323846;

struct Config {
    int Nx = 128;
    int Ny = 128;
    int Nz = 16;          // z 方向は一定（2D TGV を z に押し出し）
    int steps = 400;
    int sampleEvery = 10;
    double tau = 0.58;
    double u0 = 0.06;
    double kx = 1.0;
    double ky = 1.0;
    double cs2 = 1.0 / 3.0;
    double k2 = 0.0;
    double nu = 0.0;
    double kxWave = 0.0;
    double kyWave = 0.0;
    std::string vtkDir;
    int vtkEvery = 0;
    std::string csvPath = "out_vtk/tgv_results.csv";
    int numCells = 0;
};

struct AnalyticState {
    double u, v, w, rho;
};

struct SolverSample {
    double energy = 0.0;
    double l2 = 0.0;
    double linf = 0.0;
};

struct SampleRow {
    int step = 0;
    double time = 0.0;
    double energyExact = 0.0;
    SolverSample legacy;
    SolverSample home;
};
static AnalyticState analytic_tgv(int ix, int iy, int /*iz*/, double t, const Config& cfg) {
    const double tx = (static_cast<double>(ix) + 0.5) * cfg.kxWave;
    const double ty = (static_cast<double>(iy) + 0.5) * cfg.kyWave;

    const double decayVel = std::exp(-cfg.nu * cfg.k2 * t);
    const double decayRho = std::exp(-2.0 * cfg.nu * cfg.k2 * t);

    const double sinX = std::sin(tx);
    const double cosX = std::cos(tx);
    const double sinY = std::sin(ty);
    const double cosY = std::cos(ty);

    AnalyticState a{};
    a.u = cfg.u0 * sinX * cosY * decayVel;
    a.v = -cfg.u0 * cosX * sinY * decayVel;
    a.w = 0.0;
    a.rho = 1.0 + (cfg.u0 * cfg.u0 * decayRho / (4.0 * cfg.cs2)) *
                    (std::cos(2.0 * tx) + std::cos(2.0 * ty));
    return a;
}

template <typename Solver>
static void apply_initial_fields(const Config& cfg,
                                 Solver& solver,
                                 const std::vector<float>& rho,
                                 const std::vector<float>& ux,
                                 const std::vector<float>& uy,
                                 const std::vector<float>& uz) {
    const size_t bytesScalar = sizeof(float) * static_cast<size_t>(cfg.numCells);
    CUDA_CHECK(cudaMemcpy(solver.d_rho(), rho.data(), bytesScalar, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(solver.d_u(),   ux.data(),  bytesScalar, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(solver.d_v(),   uy.data(),  bytesScalar, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(solver.d_w(),   uz.data(),  bytesScalar, cudaMemcpyHostToDevice));

    // 平衡状態へ（HOME は moments-only なので内部で (rho,u,S=0) を作る）
    solver.reinitEquilibriumFromMacro(solver.d_rho(), solver.d_u(), solver.d_v(), solver.d_w());
}
static void initialize_solvers(const Config& cfg, LBM3D_Legacy& legacy, LBM3D_Home& home) {
    Domain d;
    d.Nx = cfg.Nx;
    d.Ny = cfg.Ny;
    d.Nz = cfg.Nz;
    d.tau = static_cast<float>(cfg.tau);
    d.forceX = d.forceY = d.forceZ = 0.0f;

    legacy.init(d);
    home.init(d);

    std::vector<unsigned char> mask(cfg.numCells, 0);
    legacy.setSolidMask(mask.data());
    home.setSolidMask(mask.data());

    std::vector<float> rho(cfg.numCells);
    std::vector<float> ux(cfg.numCells);
    std::vector<float> uy(cfg.numCells);
    std::vector<float> uz(cfg.numCells, 0.0f);

    size_t idx = 0;
    for (int z = 0; z < cfg.Nz; ++z) {
        for (int y = 0; y < cfg.Ny; ++y) {
            for (int x = 0; x < cfg.Nx; ++x, ++idx) {
                auto a = analytic_tgv(x, y, z, 0.0, cfg);
                rho[idx] = static_cast<float>(a.rho);
                ux[idx]  = static_cast<float>(a.u);
                uy[idx]  = static_cast<float>(a.v);
            }
        }
    }

    apply_initial_fields(cfg, legacy, rho, ux, uy, uz);
    apply_initial_fields(cfg, home,   rho, ux, uy, uz);
}

template <typename Solver>
static SolverSample sample_solver(const Solver& solver,
                                  std::vector<float>& rho,
                                  std::vector<float>& ux,
                                  std::vector<float>& uy,
                                  std::vector<float>& uz,
                                  const Config& cfg,
                                  double time) {
    const size_t bytesScalar = sizeof(float) * static_cast<size_t>(cfg.numCells);
    CUDA_CHECK(cudaMemcpy(rho.data(), solver.d_rho(), bytesScalar, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ux.data(),  solver.d_u(),   bytesScalar, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(uy.data(),  solver.d_v(),   bytesScalar, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(uz.data(),  solver.d_w(),   bytesScalar, cudaMemcpyDeviceToHost));

    double sumEnergy = 0.0;
    double sumErr2 = 0.0;
    double maxErr = 0.0;

    size_t idx = 0;
    for (int z = 0; z < cfg.Nz; ++z) {
        for (int y = 0; y < cfg.Ny; ++y) {
            for (int x = 0; x < cfg.Nx; ++x, ++idx) {
                auto a = analytic_tgv(x, y, z, time, cfg);
                const double du = static_cast<double>(ux[idx]) - a.u;
                const double dv = static_cast<double>(uy[idx]) - a.v;
                const double dw = static_cast<double>(uz[idx]) - a.w;
                const double err2 = du * du + dv * dv + dw * dw;
                sumErr2 += err2;
                maxErr = std::max(maxErr, std::sqrt(err2));

                const double u = static_cast<double>(ux[idx]);
                const double v = static_cast<double>(uy[idx]);
                const double w = static_cast<double>(uz[idx]);
                sumEnergy += 0.5 * (u * u + v * v + w * w);
            }
        }
    }

    SolverSample s;
    s.energy = sumEnergy / static_cast<double>(cfg.numCells);
    s.l2 = std::sqrt(sumErr2 / static_cast<double>(cfg.numCells));
    s.linf = maxErr;
    return s;
}
static double analytic_energy(double t, const Config& cfg) {
// エネルギーの解析解: E = (u0^2 / 4) * exp(-2 * nu * k^2 * t)
    return 0.25 * cfg.u0 * cfg.u0 * std::exp(-2.0 * cfg.nu * cfg.k2 * t);
}
// 出力データを書き出す
static void write_csv(const std::vector<SampleRow>& rows, const std::string& path) {
    std::ofstream ofs(path);
    ofs << "step,time,energy_exact,energy_legacy,energy_home,l2_legacy,l2_home,linf_legacy,linf_home\n";
    ofs << std::scientific << std::setprecision(8);
    for (const auto& r : rows) {
        ofs << r.step << ","
            << r.time << ","
            << r.energyExact << ","
            << r.legacy.energy << ","
            << r.home.energy << ","
            << r.legacy.l2 << ","
            << r.home.l2 << ","
            << r.legacy.linf << ","
            << r.home.linf << "\n";
    }
    std::cout << "[write] saved " << rows.size() << " samples to " << path << "\n";
}
// 引数や入力設定を解析する
static void parse_args(int argc, char** argv, Config& cfg) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](int remain) { return (i + remain) < argc; };
        if (a == "--nx" && need(1)) {
            cfg.Nx = std::stoi(argv[++i]);
        } else if (a == "--ny" && need(1)) {
            cfg.Ny = std::stoi(argv[++i]);
        } else if (a == "--nz" && need(1)) {
            cfg.Nz = std::stoi(argv[++i]);
        } else if (a == "--steps" && need(1)) {
            cfg.steps = std::stoi(argv[++i]);
        } else if (a == "--sample-every" && need(1)) {
            cfg.sampleEvery = std::stoi(argv[++i]);
        } else if (a == "--u0" && need(1)) {
            cfg.u0 = std::stod(argv[++i]);
        } else if (a == "--tau" && need(1)) {
            cfg.tau = std::stod(argv[++i]);
        } else if ((a == "--k" || a == "--wavenumber") && need(1)) {
            cfg.kx = cfg.ky = std::stod(argv[++i]);
        } else if (a == "--vtk-dir" && need(1)) {
            cfg.vtkDir = argv[++i];
        } else if ((a == "--vtk-every" || a == "--vtk-interval") && need(1)) {
            cfg.vtkEvery = std::stoi(argv[++i]);
        } else if (a == "--csv" && need(1)) {
            cfg.csvPath = argv[++i];
        }
    }
}
// 実行エントリポイントとしてシミュレーションを開始する
int main(int argc, char** argv) {
    Config cfg;
    parse_args(argc, argv, cfg);
    if (cfg.sampleEvery <= 0) cfg.sampleEvery = 1;
    if (cfg.vtkEvery < 0) cfg.vtkEvery = 0;
    if (cfg.Nz <= 0) cfg.Nz = 1;
    if (cfg.Nx <= 0) cfg.Nx = 16;
    if (cfg.Ny <= 0) cfg.Ny = 16;
    if (cfg.steps < 0) cfg.steps = 0;
    cfg.numCells = cfg.Nx * cfg.Ny * cfg.Nz;
    const double twoPi = 2.0 * kPi;
    cfg.kxWave = twoPi * cfg.kx / static_cast<double>(cfg.Nx);
    cfg.kyWave = twoPi * cfg.ky / static_cast<double>(cfg.Ny);
    cfg.k2 = cfg.kxWave * cfg.kxWave + cfg.kyWave * cfg.kyWave;
    cfg.nu = (cfg.tau - 0.5) * cfg.cs2;

    std::cout << "[cfg ] Nx=" << cfg.Nx << " Ny=" << cfg.Ny << " Nz=" << cfg.Nz
              << " steps=" << cfg.steps << " sample=" << cfg.sampleEvery << "\n";
    std::cout << "[cfg ] tau=" << cfg.tau << " nu=" << cfg.nu
              << " u0=" << cfg.u0
              << " kx_wave=" << cfg.kxWave << " ky_wave=" << cfg.kyWave
              << " k^2=" << cfg.k2 << "\n";
    if (!cfg.vtkDir.empty()) {
        std::cout << "[cfg ] vtk dir=" << cfg.vtkDir
                  << " every=" << cfg.vtkEvery << " steps\n";
    }
    std::cout << "[cfg ] csv=" << cfg.csvPath << "\n";

    LBM3D_Legacy legacy;
    LBM3D_Home home;
    initialize_solvers(cfg, legacy, home);

    std::vector<float> rhoBuf(cfg.numCells);
    std::vector<float> uxBuf(cfg.numCells);
    std::vector<float> uyBuf(cfg.numCells);
    std::vector<float> uzBuf(cfg.numCells);

// VTK へ速度を書き出すためのデバイスバッファ
    float* d_speedLegacy = nullptr;
    float* d_speedHome = nullptr;
    if (!cfg.vtkDir.empty() && cfg.vtkEvery > 0) {
        CUDA_CHECK(cudaMalloc(&d_speedLegacy, sizeof(float) * cfg.numCells));
        CUDA_CHECK(cudaMalloc(&d_speedHome,   sizeof(float) * cfg.numCells));
    }

    auto make_vtk_path = [&](const std::string& stem, int step) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%s_%06d.vtk", stem.c_str(), step);
        return std::string(buf);
    };

    auto dump_vtk = [&](int step) {
        if (cfg.vtkDir.empty() || cfg.vtkEvery <= 0) return;
        std::filesystem::create_directories(cfg.vtkDir);

        launch_speed(legacy.d_u(), legacy.d_v(), legacy.d_w(), d_speedLegacy, cfg.numCells);
        launch_speed(home.d_u(),   home.d_v(),   home.d_w(),   d_speedHome,   cfg.numCells);

        auto pathLegacyRho   = std::filesystem::path(cfg.vtkDir) / make_vtk_path("rho_legacy", step);
        auto pathHomeRho     = std::filesystem::path(cfg.vtkDir) / make_vtk_path("rho_home", step);
        auto pathLegacySpeed = std::filesystem::path(cfg.vtkDir) / make_vtk_path("speed_legacy", step);
        auto pathHomeSpeed   = std::filesystem::path(cfg.vtkDir) / make_vtk_path("speed_home", step);
        auto pathLegacyVel   = std::filesystem::path(cfg.vtkDir) / make_vtk_path("vel_legacy", step);
        auto pathHomeVel     = std::filesystem::path(cfg.vtkDir) / make_vtk_path("vel_home", step);

        write_scalar_vtk(pathLegacyRho.string().c_str(), legacy.d_rho(), cfg.Nx, cfg.Ny, cfg.Nz);
        write_scalar_vtk(pathHomeRho.string().c_str(),   home.d_rho(),   cfg.Nx, cfg.Ny, cfg.Nz);
        write_scalar_vtk(pathLegacySpeed.string().c_str(), d_speedLegacy, cfg.Nx, cfg.Ny, cfg.Nz);
        write_scalar_vtk(pathHomeSpeed.string().c_str(),   d_speedHome,   cfg.Nx, cfg.Ny, cfg.Nz);
        write_vector_vtk(pathLegacyVel.string().c_str(), legacy.d_u(), legacy.d_v(), legacy.d_w(),
                         cfg.Nx, cfg.Ny, cfg.Nz);
        write_vector_vtk(pathHomeVel.string().c_str(),   home.d_u(),   home.d_v(),   home.d_w(),
                         cfg.Nx, cfg.Ny, cfg.Nz);
    };

    std::vector<SampleRow> rows;
    auto sample_now = [&](int step) {
        const double t = static_cast<double>(step);
        SampleRow r;
        r.step = step;
        r.time = t;
        r.energyExact = analytic_energy(t, cfg);
        r.legacy = sample_solver(legacy, rhoBuf, uxBuf, uyBuf, uzBuf, cfg, t);
        r.home   = sample_solver(home,   rhoBuf, uxBuf, uyBuf, uzBuf, cfg, t);

        std::cout << "[step " << std::setw(5) << step << "] "
                  << "E_exact=" << std::setw(12) << r.energyExact
                  << "  E_L=" << std::setw(12) << r.legacy.energy
                  << "  E_H=" << std::setw(12) << r.home.energy
                  << "  L2_L=" << r.legacy.l2
                  << "  L2_H=" << r.home.l2 << "\n";
        if (cfg.vtkEvery > 0 && (step % cfg.vtkEvery) == 0) {
            dump_vtk(step);
        }
        rows.push_back(r);
    };

    sample_now(0);
    for (int step = 1; step <= cfg.steps; ++step) {
        legacy.step(1);
        home.step(1);
        if ((step % cfg.sampleEvery) == 0 || step == cfg.steps) {
            sample_now(step);
        }
    }

    write_csv(rows, cfg.csvPath);
    if (d_speedLegacy) cudaFree(d_speedLegacy);
    if (d_speedHome)   cudaFree(d_speedHome);
    return 0;
}
