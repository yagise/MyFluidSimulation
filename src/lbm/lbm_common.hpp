
#pragma once
#include <cuda_runtime.h>
// 計算格子と物理パラメータをひとまとめにした設定構造体。
// シミュレーション開始前にこれを用意し、各 LBM クラスへ渡す。
struct Domain {
    // x, y, z 方向のセル数（周期境界を前提にした格子サイズ）
    int Nx=128, Ny=128, Nz=96;
    // BGK モデルの緩和時間。粘性や安定性を決める主要パラメータ。
    float tau = 0.6f;
    // 単位時間あたりに加える外力ベクトル（Guo 風の速度シフト用）
    float forceX = 0.0f, forceY = 0.0f, forceZ = 0.0f;
};
