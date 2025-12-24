#include <cuda_runtime.h>
#include "lbm_layout.hpp"

// --- D3Q19 (cs^2 = 1/3)
// 0:(0,0,0)
// 1:(+1,0,0) 2:(-1,0,0) 3:(0,+1,0) 4:(0,-1,0) 5:(0,0,+1) 6:(0,0,-1)
// 7:(+1,+1,0) 8:(-1,+1,0) 9:(+1,-1,0) 10:(-1,-1,0)
// 11:(+1,0,+1) 12:(-1,0,+1) 13:(+1,0,-1) 14:(-1,0,-1)
// 15:(0,+1,+1) 16:(0,-1,+1) 17:(0,+1,-1) 18:(0,-1,-1)
// 各方向ベクトル (cx,cy,cz) と重み w を定数メモリに置き、平衡再初期化で使う。
__constant__ int  cxi[19]  = { 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1, 0, 0,  0, 0 };
__constant__ int  cyi[19]  = { 0, 0, 0, 1,-1, 0, 0, 1, 1,-1,-1, 0, 0, 0, 0, 1,-1,  1,-1 };
__constant__ int  czi[19]  = { 0, 0, 0, 0, 0, 1,-1, 0, 0,  0, 0,  1, 1,-1,-1, 1, 1, -1,-1 };
__constant__ float wi[19]  = {
    1.f/3.f,
    1.f/18.f,1.f/18.f,1.f/18.f,1.f/18.f,1.f/18.f,1.f/18.f,
    1.f/36.f,1.f/36.f,1.f/36.f,1.f/36.f,1.f/36.f,1.f/36.f,
    1.f/36.f,1.f/36.f,1.f/36.f,1.f/36.f,1.f/36.f,1.f/36.f
};

// マクロ量 (rho, u) から平衡分布 feq を再構成し、分布関数バッファ f を埋める。
__global__ void kern_reinit_eq(const float*  rho,
                               const float*  ux,
                               const float*  uy,
                               const float*  uz,
                               float*  f,
                               int N)
{
    // 全セルを 1 スレッドで処理するための線形インデックス
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= N) return;

    // 密度と速度の入力（u が無ければ静止とみなす）
    const float r  = rho[i];
    const float u0 = ux ? ux[i] : 0.f;
    const float v0 = uy ? uy[i] : 0.f;
    const float w0 = uz ? uz[i] : 0.f;

    // 音速二乗(cs^2)と |u|^2
    const float cs2 = 1.f/3.f;
    const float uu  = u0*u0 + v0*v0 + w0*w0;

    #pragma unroll
    for(int q=0;q<19;++q){
        const float cx = (float)cxi[q];
        const float cy = (float)cyi[q];
        const float cz = (float)czi[q];
        const float cu = cx*u0 + cy*v0 + cz*w0;

        // 標準の BGK 平衡（二次まで）
        const float feq = wi[q]*r*( 1.f + cu/cs2 + 0.5f*(cu*cu)/(cs2*cs2) - 0.5f*uu/cs2 );

        f[fIndex(q,i,N)] = feq;
    }
}

// CUDA カーネルを呼び出し、全セルの分布関数を平衡状態で再初期化する。
extern "C" void reinit_equilibrium_from_macro(const float* d_rho,
                                              const float* d_ux,
                                              const float* d_uy,
                                              const float* d_uz,
                                              float*       d_f,
                                              int          N)
{
    // 1 ブロック 256 スレッドで全セルをカバーするグリッド構成
    dim3 bs(256), gs((N + bs.x - 1)/bs.x);
    kern_reinit_eq<<<gs,bs>>>(d_rho, d_ux, d_uy, d_uz, d_f, N);
}
