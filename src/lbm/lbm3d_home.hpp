#pragma once
#include "lbm_common.hpp"
#include <cuda_runtime.h>

//
// LBM3D_Home（HOME 法）
//
// 重要: ここでの "HOME" は、Caltech の論文でいう HOME-LBM のうち
// モーメントを保持する（moment-encoded）というデータ構造面を先に導入した版。
//
// このクラスの目的（今回の修正）:
// - 分布関数 f_i(19)×2 バッファではなく、
//   0〜2次モーメント（rho,u,S）を 10 変数/セル ×2 バッファで保持する。
// - その上で、D3Q19 / 静止壁 bounce-back / 簡易外力 など
//   既存の比較条件は極力変えずに動くようにする。

class LBM3D_Home {
public:
    // 確保したデバイスメモリを解放するデストラクタ。
    ~LBM3D_Home();

    // ドメイン設定を反映し、内部バッファを確保・初期化する。
    void init(const Domain& d);

    // ホスト側の固体マスク（0:流体, 1:固体）を GPU にコピーする。
    void setSolidMask(const unsigned char* h_mask);

    // rho=1, u=0, S=0 の平衡モーメントで全セルを再初期化する。
    void reset();

    // 与えられたマクロ量（rho,u）を用いて m/mnext を平衡状態に再構成する。
    void reinitEquilibriumFromMacro(const float* d_rho,
                                    const float* d_ux,
                                    const float* d_uy,
                                    const float* d_uz);

    // 指定された substeps 回だけ HOME スキームで時間発展させる。
    void step(int substeps = 1);

    // 1 ステップあたりの外力ベクトルを設定する（簡易 Guo 風）。
    void setForce(float fx, float fy=0.0f, float fz=0.0f){ fx_=fx; fy_=fy; fz_=fz; }

    // GPU 上のマクロ量バッファへのポインタを返す（SoA のオフセット込み）。
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

    // 固体マスク（0:流体, 1:固体）と格子サイズへの参照。
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
    // 確保したデバイスメモリを解放する。
    void release();
    // 現在の N_ に応じて必要なデバイスメモリを確保する。
    void allocate();

    // 格子サイズと総セル数。
    int Nx_=0, Ny_=0, Nz_=0, N_=0;
    // BGK 緩和時間と外力ベクトル（速度に加算する形で使用）。
    float tau_ = 0.6f;
    float fx_=1e-6f, fy_=0.0f, fz_=0.0f;

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

    // 固体セルを示す 0/1 マスク。bounce-back 判定に使う。
    unsigned char* d_solid_ = nullptr;
};
