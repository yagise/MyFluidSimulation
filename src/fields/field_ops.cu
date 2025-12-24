
#include <cuda_runtime.h>
#include "../common/cuda_utils.hpp"
#include <cmath>

extern "C" {
__global__ void kern_speed(const float* u, const float* v, const float* w, float* out, int N){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<N){
        float uu = u[i]*u[i] + v[i]*v[i] + w[i]*w[i];
        out[i] = sqrtf(uu);
    }
}
__global__ void kern_diff(const float* a, const float* b, float* out, int N){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<N){ out[i] = a[i] - b[i]; }
}
void launch_speed(const float* u, const float* v, const float* w, float* out, int N){
    dim3 bs(256), gs((N+255)/256);
    kern_speed<<<gs,bs>>>(u,v,w,out,N);
}
void launch_diff(const float* a, const float* b, float* out, int N){
    dim3 bs(256), gs((N+255)/256);
    kern_diff<<<gs,bs>>>(a,b,out,N);
}
__global__ void kern_shift(const float* a, float bias, float* out, int N){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<N) out[i] = a[i] + bias;
}
extern "C" void launch_shift(const float* a, float bias, float* out, int N){
    dim3 bs(256), gs((N+255)/256);
    kern_shift<<<gs,bs>>>(a, bias, out, N);
}


} // extern "C"
