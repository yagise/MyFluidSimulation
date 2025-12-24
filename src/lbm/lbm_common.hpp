
#pragma once
#include <cuda_runtime.h>

// シミュレーション領域と基本パラメータをまとめて持つ構造体
struct Domain {
    int Nx=128, Ny=128, Nz=96;
    float tau = 0.6f;
    float forceX = 0.0f, forceY = 0.0f, forceZ = 0.0f;
};
