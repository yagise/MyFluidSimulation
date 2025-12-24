#pragma once
#include "lbm_common.hpp"
#include <cuda_runtime.h>

//
// LBM3D_Hybrid（従来法 + HOME 法: Hybrid(B0)）
//
// 論文用の比較を目的としたセルごとの衝突演算の切り替え実装です。
//
// ここでの B0 の意味:
// - 壁近傍(dist_to_solid <= d0) では Legacy(BGK) の衝突
// - それ以外 では HOME の衝突
//
// 重要:
// - これは新しい物理モデルではなく、比較のための実装上の切り替えです。
// - streaming / bounce-back の扱いは全域で同一にし、公平に比較できるようにします。
//

//
// LBM3D_Hybrid ("Correct B0")
//
// This is the *correct* implementation of Plan B0 used for experiments.
//
// What "B0" means in this project:
// - We want to compare two existing solvers fairly:
// * Legacy solver (LBM3D_Legacy)
// * HOME solver (LBM3D_Home)
//
// - "B0" is NOT a new fluid model.
// It is simply a *cell-wise switch*:
// * Near the obstacle (wall-distance d <= d0): use Legacy collision
// * Bulk (d > d0): use HOME collision
//
// Why this class exists:
// - Our earlier "hybrid" prototype stored only moments in the bulk and
// reconstructed f_i (regularized). That changed the bulk solver itself
// and broke the intended comparison.
//
// This implementation is deliberately conservative:
// - It stores full distributions f_i everywhere (like Legacy/Home)
// - It matches the existing kernels as closely as possible
// - When the legacy mask is all-0 => should behave like pure HOME
// When the legacy mask is all-1 (fluid) => should behave like pure Legacy
//
// NOTE ABOUT API COMPATIBILITY
// main_compare.cpp already calls:
// hybrid.init(d, nLegacyCells);
// hybrid.setLegacyMapping(isLegacy, legacySlot);
// For "Correct B0" we only need isLegacy; legacySlot is ignored.
//

class LBM3D_Hybrid {
public:
    // 確保したデバイスメモリを解放するデストラクタ。
    ~LBM3D_Hybrid();

    // ドメイン設定を反映し、必要なバッファを確保する。nLegacyCells は期待セル数のヒント。
    void init(const Domain& d, int nLegacyCells = 0);

    // ホスト側の固体マスク（0:流体,1:固体）を GPU に転送する。
    void setSolidMask(const unsigned char* h_mask);

    // Legacy/HOME をセル単位で切り替えるマスクを設定する（nullptr なら全セル HOME）。
    void setLegacyMapping(const unsigned char* h_isLegacy,
                          const int*           h_legacySlot_ignored);

    // 分布関数とマクロ量を rho=1, u=0 の平衡に初期化する。
    void reset();

    // 与えられたマクロ量(rho,u) から feq を再構築し直す。
    void reinitEquilibriumFromMacro(const float* d_rho,
                                    const float* d_ux,
                                    const float* d_uy,
                                    const float* d_uz);

    // ハイブリッドモデルで substeps 回だけ時間発展させる。
    void step(int substeps = 1);

    // 1 ステップあたりの外力ベクトルを設定する。
    void setForce(float fx, float fy=0.0f, float fz=0.0f){ fx_=fx; fy_=fy; fz_=fz; }

    // GPU 上に保持している各バッファへの生ポインタ（push/pull デバッグ用）。
    float*       d_f()       { return d_f_; }
    // summary: d_f の処理を行う
    // param: なし
    // return: 戻り値
    const float* d_f() const { return d_f_; }
    // summary: d_fnext の処理を行う
    // param: なし
    // return: 戻り値
    float*       d_fnext()       { return d_fnext_; }
    // summary: d_fnext の処理を行う
    // param: なし
    // return: 戻り値
    const float* d_fnext() const { return d_fnext_; }

    // summary: d_rho の処理を行う
    // param: なし
    // return: 戻り値
    float*       d_rho()       { return d_rho_; }
    // summary: d_rho の処理を行う
    // param: なし
    // return: 戻り値
    const float* d_rho() const { return d_rho_; }
    // summary: d_u の処理を行う
    // param: なし
    // return: 戻り値
    float*       d_u()         { return d_u_; }
    // summary: d_u の処理を行う
    // param: なし
    // return: 戻り値
    const float* d_u() const   { return d_u_; }
    // summary: d_v の処理を行う
    // param: なし
    // return: 戻り値
    float*       d_v()         { return d_v_; }
    // summary: d_v の処理を行う
    // param: なし
    // return: 戻り値
    const float* d_v() const   { return d_v_; }
    // summary: d_w の処理を行う
    // param: なし
    // return: 戻り値
    float*       d_w()         { return d_w_; }
    // summary: d_w の処理を行う
    // param: なし
    // return: 戻り値
    const float* d_w() const   { return d_w_; }

    // 固体マスクと Legacy/HOME 切り替えマスク。
    const unsigned char* d_solid() const { return d_solid_; }
    // summary: d_isLegacy の処理を行う
    // param: なし
    // return: 戻り値
    const unsigned char* d_isLegacy() const { return d_isLegacy_; }

    // 格子サイズの参照。
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

    // 確保したメモリを解放する
    void release();

private:
    // 必要なメモリを確保する
    void allocate();

    // 格子サイズとセル数（GPU 側バッファ確保に使用）
    int Nx_=0, Ny_=0, Nz_=0, N_=0;
    // 緩和時間と外力ベクトル（Legacy/HOME 共通で使用）
    float tau_ = 0.6f;
    float fx_=1e-6f, fy_=0.0f, fz_=0.0f;
    // (移動壁補正は削除)

    // 分布関数 f（現在ステップ）と fnext（次ステップ）
    float* d_f_ = nullptr;
    float* d_fnext_ = nullptr;
    // マクロ量（rho, ux, uy, uz）
    float* d_rho_ = nullptr;
    float* d_u_ = nullptr;
    float* d_v_ = nullptr;
    float* d_w_ = nullptr;
    // 固体マスクと Legacy マスク
    unsigned char* d_solid_ = nullptr;
    unsigned char* d_isLegacy_ = nullptr;
};
