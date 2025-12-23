
#include <cuda_runtime.h>
#include "../common/cuda_utils.hpp"
#include <cmath>

extern "C" {

// summary: kern_speed の処理を行う
// param u: 入力パラメータ
// param v: 入力パラメータ
// param w: 入力パラメータ
// param out: 入力パラメータ
// param N: 入力パラメータ
// return: なし
__global__ void kern_speed(const float* u, const float* v, const float* w, float* out, int N){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<N){
        float uu = u[i]*u[i] + v[i]*v[i] + w[i]*w[i];
        out[i] = sqrtf(uu);
    }
}

// summary: kern_diff の処理を行う
// param a: 入力パラメータ
// param b: 入力パラメータ
// param out: 入力パラメータ
// param N: 入力パラメータ
// return: なし
__global__ void kern_diff(const float* a, const float* b, float* out, int N){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<N){ out[i] = a[i] - b[i]; }
}

// summary: launch_speed の処理を行う
// param u: 入力パラメータ
// param v: 入力パラメータ
// param w: 入力パラメータ
// param out: 入力パラメータ
// param N: 入力パラメータ
// return: なし
void launch_speed(const float* u, const float* v, const float* w, float* out, int N){
    dim3 bs(256), gs((N+255)/256);
    kern_speed<<<gs,bs>>>(u,v,w,out,N);
}

// summary: launch_diff の処理を行う
// param a: 入力パラメータ
// param b: 入力パラメータ
// param out: 入力パラメータ
// param N: 入力パラメータ
// return: なし
void launch_diff(const float* a, const float* b, float* out, int N){
    dim3 bs(256), gs((N+255)/256);
    kern_diff<<<gs,bs>>>(a,b,out,N);
}
// summary: kern_shift の処理を行う
// param a: 入力パラメータ
// param bias: 入力パラメータ
// param out: 入力パラメータ
// param N: 入力パラメータ
// return: なし
__global__ void kern_shift(const float* a, float bias, float* out, int N){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<N) out[i] = a[i] + bias;
}
// summary: launch_shift の処理を行う
// param a: 入力パラメータ
// param bias: 入力パラメータ
// param out: 入力パラメータ
// param N: 入力パラメータ
// return: なし
extern "C" void launch_shift(const float* a, float bias, float* out, int N){
    dim3 bs(256), gs((N+255)/256);
    kern_shift<<<gs,bs>>>(a, bias, out, N);
}


} // extern "C"
