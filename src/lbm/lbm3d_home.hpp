
#pragma once
#include "lbm_common.hpp"
#include <cuda_runtime.h>

//
// LBM3D_Home（HOME 法）
//
// このクラスの目的（今回の修正）:
// - 分布関数 f_i(19)×2 バッファではなく、
// 0〜2次モーメント（rho,u,S）を 10 変数/セル ×2 バッファで保持する。
// - その上で、D3Q19 / 静止壁 bounce-back / 簡易外力 など
// 既存の比較条件は極力変えずに動くようにする。

class LBM3D_Home {
public:
// 確保したメモリを解放する
    ~LBM3D_Home();
    void init(const Domain& d);
    void setSolidMask(const unsigned char* h_mask);
    void reset();
    void reinitEquilibriumFromMacro(const float* d_rho,
                                    const float* d_ux,
                                    const float* d_uy,
                                    const float* d_uz);
    void step(int substeps = 1);
    void setForce(float fx, float fy=0.0f, float fz=0.0f){ fx_=fx; fy_=fy; fz_=fz; }
    float*       d_rho();
    const float* d_rho() const;
    float*       d_u();
    const float* d_u() const;
    float*       d_v();
    const float* d_v() const;
    float*       d_w();
    const float* d_w() const;
    const unsigned char* d_solid() const { return d_solid_; }
    int Nx() const { return Nx_; }
    int Ny() const { return Ny_; }
    int Nz() const { return Nz_; }
    int N()  const { return N_; }

private:
// 確保したメモリを解放する
    void release();
// 必要なメモリを確保する
    void allocate();
    int Nx_=0, Ny_=0, Nz_=0, N_=0;
    float tau_ = 0.6f;
    float fx_=0.0f, fy_=0.0f, fz_=0.0f;

    // moments-only バッファ（SoA）
    // m[ 0*N + id] = rho
    // m[ 1*N + id] = ux
    // m[ 2*N + id] = uy
    // m[ 3*N + id] = uz
    // m[ 4*N + id] = Sxx (2次非平衡応力)
    // m[ 5*N + id] = Sxy
    // m[ 6*N + id] = Sxz
    // m[ 7*N + id] = Syy
    // m[ 8*N + id] = Syz
    // m[ 9*N + id] = Szz
    //
    // 2バッファにして stream 後に swap する。
    float* d_m_     = nullptr;
    float* d_mnext_ = nullptr;

    unsigned char* d_solid_ = nullptr;
};
