#pragma once
#include "lbm_common.hpp"
#include <cuda_runtime.h>

//
// LBM3D_Hybrid（従来法 + HOME 法: Hybrid(B0)）
//
//
// ここでの B0 の意味:
// - 壁近傍(dist_to_solid <= d0) では Legacy(BGK) の衝突
// - それ以外 では HOME の衝突
//
// 重要:
// - これは新しい物理モデルではなく、比較のための実装上の切り替えです。
// - streaming / bounce-back の扱いは全域で同一にし、公平に比較できるようにします。
//

class LBM3D_Hybrid {
public:
// 確保したメモリを解放する
    ~LBM3D_Hybrid();
    void init(const Domain& d, int nLegacyCells = 0);
    void setSolidMask(const unsigned char* h_mask);
    void setLegacyMapping(const unsigned char* h_isLegacy,
                          const int*           h_legacySlot_ignored);
    void reset();
    void reinitEquilibriumFromMacro(const float* d_rho,
                                    const float* d_ux,
                                    const float* d_uy,
                                    const float* d_uz);
    void step(int substeps = 1);
    void setForce(float fx, float fy=0.0f, float fz=0.0f){ fx_=fx; fy_=fy; fz_=fz; }
    // Hybrid(B0) でも移動壁補正は不要なため削除。
    float*       d_f()       { return d_f_; }
    const float* d_f() const { return d_f_; }
    float*       d_fnext()       { return d_fnext_; }
    const float* d_fnext() const { return d_fnext_; }
    float*       d_rho()       { return d_rho_; }
    const float* d_rho() const { return d_rho_; }
    float*       d_u()         { return d_u_; }
    const float* d_u() const   { return d_u_; }
    float*       d_v()         { return d_v_; }
    const float* d_v() const   { return d_v_; }
    float*       d_w()         { return d_w_; }
    const float* d_w() const   { return d_w_; }
    const unsigned char* d_solid() const { return d_solid_; }
    const unsigned char* d_isLegacy() const { return d_isLegacy_; }
    int Nx() const { return Nx_; }
    int Ny() const { return Ny_; }
    int Nz() const { return Nz_; }
    int N()  const { return N_; }
// 確保したメモリを解放する
    void release();

private:
// 必要なメモリを確保する
    void allocate();

    int Nx_=0, Ny_=0, Nz_=0, N_=0;
    float tau_ = 0.6f;
    float fx_=0.0f, fy_=0.0f, fz_=0.0f;
    // (移動壁補正は削除)

    float* d_f_ = nullptr;
    float* d_fnext_ = nullptr;
    float* d_rho_ = nullptr;
    float* d_u_ = nullptr;
    float* d_v_ = nullptr;
    float* d_w_ = nullptr;
    unsigned char* d_solid_ = nullptr;
    unsigned char* d_isLegacy_ = nullptr;
};
