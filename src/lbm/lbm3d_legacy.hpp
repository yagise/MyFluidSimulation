
#pragma once
#include "lbm_common.hpp"
#include <cuda_runtime.h>

//
// LBM3D_Legacy（従来法）
//
// D3Q19 の標準的な LBM（BGK 単一緩和）実装です。
//
// 特徴:
// - 分布関数 f_i を全方向(19)×全セルで保持
// - 1step = 衝突(BGK) + ストリーミング(push)
// - 障害物境界は静止壁の単純バウンスバック
//
// - 移動壁（回転体など）の補正は比較条件を複雑にするため削除
//

class LBM3D_Legacy {
public:
    // GPU リソースを破棄
    ~LBM3D_Legacy();
    // 格子サイズや緩和時間を設定しデバイスメモリを確保する
    void init(const Domain& d);
    // solid(1)/fluid(0) マスクを GPU にコピー
    void setSolidMask(const unsigned char* h_mask);
    // rho=1, u=0 の平衡状態で分布関数を埋める
    void reset();
    // 既存の rho,u から feq を再構成し、f を平衡に戻す
    void reinitEquilibriumFromMacro(const float* d_rho,
                                    const float* d_ux,
                                    const float* d_uy,
                                    const float* d_uz);
    // 衝突+ストリーミングを substeps 回進める
    void step(int substeps = 1);
    // 簡易外力（速度への加算）を設定
    void setForce(float fx, float fy=0.0f, float fz=0.0f){ fx_=fx; fy_=fy; fz_=fz; }
    // 内部バッファへのポインタを返す（VTK 書き出し等で使用）
    float*       d_f();
    const float* d_f() const;
    float*       d_fnext();
    const float* d_fnext() const;
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
    float* d_f_ = nullptr;
    float* d_fnext_ = nullptr;
    float* d_rho_ = nullptr;
    float* d_u_ = nullptr;
    float* d_v_ = nullptr;
    float* d_w_ = nullptr;
    unsigned char* d_solid_ = nullptr;
};
