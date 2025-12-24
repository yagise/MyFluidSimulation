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
    // 確保したデバイスメモリを解放するデストラクタ。
    ~LBM3D_Legacy();

    // ドメイン設定を反映し、必要なバッファを確保・初期化する。
    void init(const Domain& d);

    // 固体マスク（0:流体,1:固体）を GPU に送る。
    void setSolidMask(const unsigned char* h_mask);

    // rho=1, u=0, feq=w*rho の平衡にリセットする。
    void reset();

    // マクロ量 (rho,u) から平衡分布 feq を再構築して f/fnext を整える。
    void reinitEquilibriumFromMacro(const float* d_rho,
                                    const float* d_ux,
                                    const float* d_uy,
                                    const float* d_uz);

    // Legacy BGK スキームで substeps 回だけ時間発展させる。
    void step(int substeps = 1);

    // 1 ステップあたりの外力ベクトルを設定する。
    void setForce(float fx, float fy=0.0f, float fz=0.0f){ fx_=fx; fy_=fy; fz_=fz; }

    // GPU バッファへの生ポインタ（デバッグ・可視化用）。
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

    // 固体マスクと格子サイズへの参照。
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
    // 確保したメモリを解放する
    void release();
    // 必要なメモリを確保する
    void allocate();

    // 格子サイズとセル数。
    int Nx_=0, Ny_=0, Nz_=0, N_=0;
    // 緩和時間と外力ベクトル。
    float tau_ = 0.6f;
    float fx_=0.0f, fy_=0.0f, fz_=0.0f;
    // 分布関数 f (current) と fnext (next step)
    float* d_f_ = nullptr;
    float* d_fnext_ = nullptr;
    // マクロ量のストレージ
    float* d_rho_ = nullptr;
    float* d_u_ = nullptr;
    float* d_v_ = nullptr;
    float* d_w_ = nullptr;
    // 固体セルマスク（0:流体,1:固体）
    unsigned char* d_solid_ = nullptr;
};
