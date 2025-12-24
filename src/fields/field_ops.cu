#include <cuda_runtime.h>
#include "../common/cuda_utils.hpp"
#include <cmath>

// 簡単な場の演算カーネル群。
// - kern_speed  : 速度ベクトルの大きさ |u| を計算する
// - kern_diff   : 2 つのスカラー場の差 a-b を求める
// - kern_shift  : スカラー場にバイアスを足す
// それぞれ launch_* で安全なグリッドを組んで呼び出す。
extern "C" {
// |u| = sqrt(u^2+v^2+w^2) を要素ごとに計算する
__global__ void kern_speed(const float* u, const float* v, const float* w, float* out, int N){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<N){
        float uu = u[i]*u[i] + v[i]*v[i] + w[i]*w[i];
        out[i] = sqrtf(uu);
    }
}
// 差分 out = a - b
__global__ void kern_diff(const float* a, const float* b, float* out, int N){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<N){ out[i] = a[i] - b[i]; }
}
// 上記カーネルのホストラッパー
void launch_speed(const float* u, const float* v, const float* w, float* out, int N){
    dim3 bs(256), gs((N+255)/256);
    kern_speed<<<gs,bs>>>(u,v,w,out,N);
}
void launch_diff(const float* a, const float* b, float* out, int N){
    dim3 bs(256), gs((N+255)/256);
    kern_diff<<<gs,bs>>>(a,b,out,N);
}
// out = a + bias
__global__ void kern_shift(const float* a, float bias, float* out, int N){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<N) out[i] = a[i] + bias;
}
extern "C" void launch_shift(const float* a, float bias, float* out, int N){
    dim3 bs(256), gs((N+255)/256);
    kern_shift<<<gs,bs>>>(a, bias, out, N);
}


} // extern "C"
