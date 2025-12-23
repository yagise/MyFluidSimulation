
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
// 0〜2次モーメント（rho,u,S）を 10 変数/セル ×2 バッファで保持する。
// - その上で、D3Q19 / 静止壁 bounce-back / 簡易外力 など
// 既存の比較条件は極力変えずに動くようにする。
//
// 注意:
// - 論文の HOME-LBM は D3Q27 や 3次Hermite再構成などを含みますが、
// それらは今回の依頼（1だけ修正）の範囲外として未実装です。
// - ただしモーメントだけを保存するために、ストリーミングに必要な f_i は
// (rho,u,S) からその場で再構成します（2次モーメントまでの regularized 形式）。
//
// 論文用整理:
// - 移動壁（回転体など）の補正は削除（静止壁のみ）
//

class LBM3D_Home {
public:
    // summary: 確保したメモリを解放する
    // param: なし
    // return: なし
    ~LBM3D_Home();
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
