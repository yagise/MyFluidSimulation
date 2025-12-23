
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
// 論文用整理:
// - 移動壁（回転体など）の補正は比較条件を複雑にするため削除
//

class LBM3D_Legacy {
public:
    // summary: 確保したメモリを解放する
    // param: なし
    // return: なし
    ~LBM3D_Legacy();
    // summary: 初期化処理を行う
    // param d: 入力パラメータ
    // return: なし
    void init(const Domain& d);

    // summary: setSolidMask の処理を行う
    // param h_mask: 入力パラメータ
    // return: なし
    void setSolidMask(const unsigned char* h_mask);

    // summary: reset の処理を行う
    // param: なし
    // return: なし
    void reset();

    // summary: reinitEquilibriumFromMacro の処理を行う
    // param d_rho: 入力パラメータ
    // param d_ux: 入力パラメータ
    // param d_uy: 入力パラメータ
    // param d_uz: 入力パラメータ
    // return: なし
    void reinitEquilibriumFromMacro(const float* d_rho,
                                    const float* d_ux,
                                    const float* d_uy,
                                    const float* d_uz);

    // summary: step の処理を行う
    // param substeps: 入力パラメータ
    // return: なし
    void step(int substeps = 1);

    // summary: setForce の処理を行う
    // param fx: 入力パラメータ
    // param fy: 入力パラメータ
    // param fz: 入力パラメータ
    // return: なし
    void setForce(float fx, float fy=0.0f, float fz=0.0f){ fx_=fx; fy_=fy; fz_=fz; }

    // summary: d_f の処理を行う
    // param: なし
    // return: 戻り値
    float*       d_f();
    // summary: d_f の処理を行う
    // param: なし
    // return: 戻り値
    const float* d_f() const;
    // summary: d_fnext の処理を行う
    // param: なし
    // return: 戻り値
    float*       d_fnext();
    // summary: d_fnext の処理を行う
    // param: なし
    // return: 戻り値
    const float* d_fnext() const;
    // summary: d_rho の処理を行う
    // param: なし
    // return: 戻り値
    float*       d_rho();
    // summary: d_rho の処理を行う
    // param: なし
    // return: 戻り値
    const float* d_rho() const;
    // summary: d_u の処理を行う
    // param: なし
    // return: 戻り値
    float*       d_u();
    // summary: d_u の処理を行う
    // param: なし
    // return: 戻り値
    const float* d_u() const;
    // summary: d_v の処理を行う
    // param: なし
    // return: 戻り値
    float*       d_v();
    // summary: d_v の処理を行う
    // param: なし
    // return: 戻り値
    const float* d_v() const;
    // summary: d_w の処理を行う
    // param: なし
    // return: 戻り値
    float*       d_w();
    // summary: d_w の処理を行う
    // param: なし
    // return: 戻り値
    const float* d_w() const;

    // summary: d_solid の処理を行う
    // param: なし
    // return: 戻り値
    const unsigned char* d_solid() const { return d_solid_; }

    // summary: Nx の処理を行う
    // param: なし
    // return: 戻り値
    int Nx() const { return Nx_; }
    // summary: Ny の処理を行う
    // param: なし
    // return: 戻り値
    int Ny() const { return Ny_; }
    // summary: Nz の処理を行う
    // param: なし
    // return: 戻り値
    int Nz() const { return Nz_; }
    // summary: N の処理を行う
    // param: なし
    // return: 戻り値
    int N()  const { return N_; }

private:
    // summary: 確保したメモリを解放する
    // param: なし
    // return: なし
    void release();
    // summary: 必要なメモリを確保する
    // param: なし
    // return: なし
    void allocate();
    int Nx_=0, Ny_=0, Nz_=0, N_=0;
    float tau_ = 0.6f;
    float fx_=1e-6f, fy_=0.0f, fz_=0.0f;
    float* d_f_ = nullptr;
    float* d_fnext_ = nullptr;
    float* d_rho_ = nullptr;
    float* d_u_ = nullptr;
    float* d_v_ = nullptr;
    float* d_w_ = nullptr;
    unsigned char* d_solid_ = nullptr;
};
