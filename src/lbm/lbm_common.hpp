
#pragma once
#include <cuda_runtime.h>
struct Domain {
    int Nx=128, Ny=128, Nz=96;
    float tau = 0.6f;
    float forceX = 1e-6f, forceY = 0.0f, forceZ = 0.0f;
};
