
#include "lbm3d_legacy.hpp"
#include "../common/cuda_utils.hpp"
#include <cmath>

// 平衡分布の再初期化カーネル（lbm_reinit.cu）
extern "C" void reinit_equilibrium_from_macro(const float* d_rho,
                                              const float* d_ux,
                                              const float* d_uy,
                                              const float* d_uz,
                                              float*       d_f,
                                              int          N);
// 確保したメモリを解放する
LBM3D_Legacy::~LBM3D_Legacy(){ release(); }

// D3Q19 の標準的な方向並び:
// 0:(0,0,0)
// 1:(+1,0,0) 2:(-1,0,0) 3:(0,+1,0) 4:(0,-1,0) 5:(0,0,+1) 6:(0,0,-1)
// 7:(+1,+1,0) 8:(-1,+1,0) 9:(+1,-1,0) 10:(-1,-1,0)
// 11:(+1,0,+1) 12:(-1,0,+1) 13:(+1,0,-1) 14:(-1,0,-1)
// 15:(0,+1,+1) 16:(0,-1,+1) 17:(0,+1,-1) 18:(0,-1,-1)
__device__ __constant__ int cx19[19] = {0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1,  1,-1, 1,-1, 0, 0,  0, 0};
__device__ __constant__ int cy19[19] = {0, 0, 0, 1,-1, 0, 0, 1, 1,-1,-1,  0, 0, 0, 0, 1,-1,  1,-1};
__device__ __constant__ int cz19[19] = {0, 0, 0, 0, 0, 1,-1, 0, 0,  0, 0,  1, 1,-1,-1, 1, 1, -1,-1};
__device__ __constant__ float w19[19] = {1.0f/3.0f,
    1.0f/18.0f,1.0f/18.0f,1.0f/18.0f,1.0f/18.0f,1.0f/18.0f,1.0f/18.0f,
    1.0f/36.0f,1.0f/36.0f,1.0f/36.0f,1.0f/36.0f,1.0f/36.0f,1.0f/36.0f,
    1.0f/36.0f,1.0f/36.0f,1.0f/36.0f,1.0f/36.0f,1.0f/36.0f,1.0f/36.0f};
__device__ __forceinline__ static int opp(int i){
    const int o[19] = {0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15,18,17};
    return o[i];
}
__device__ __forceinline__ static int index3D(int x,int y,int z,int Nx,int Ny){
    return (z*Ny + y)*Nx + x;
}
// --- GPU kernels ---
// kern_reset           : rho=1,u=0 の平衡分布で f を初期化
// kern_collide_stream  : BGK 衝突 + ストリーミング + バウンスバック
// kern_swap            : ping-pong バッファの入れ替え
__global__ static void kern_reset(float* f, float* rho, float* u, float* v, float* w,
                                  const unsigned char* solid, int Nx,int Ny,int Nz){
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    int iz = blockIdx.z*blockDim.z + threadIdx.z;
    if(ix>=Nx || iy>=Ny || iz>=Nz) return;
    int N = Nx*Ny*Nz;
    int id = index3D(ix,iy,iz,Nx,Ny);
    float r = 1.0f;
    rho[id] = r; u[id]=v[id]=w[id]=0;
    for(int q=0;q<19;++q){
        f[q*N + id] = w19[q] * r;
    }
}
__device__ __forceinline__ static float feq(int q, float rho, float ux, float uy, float uz){
    float eiu = cx19[q]*ux + cy19[q]*uy + cz19[q]*uz;
    float uu = ux*ux + uy*uy + uz*uz;
    return w19[q]*rho*(1.0f + 3.0f*eiu + 4.5f*eiu*eiu - 1.5f*uu);
}
__global__ static void kern_collide_stream(const float* f, float* fnext,
                                           float* rho, float* ux, float* uy, float* uz,
                                           const unsigned char* solid,
                                           int Nx,int Ny,int Nz, float omega,
                                           float fx, float fy, float fz)
{
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    int iz = blockIdx.z*blockDim.z + threadIdx.z;
    if(ix>=Nx || iy>=Ny || iz>=Nz) return;
    int N = Nx*Ny*Nz;
    int id = index3D(ix,iy,iz,Nx,Ny);

    if(solid[id]){
        for(int q=0;q<19;++q) fnext[q*N + id] = f[q*N + id];
        rho[id]=1.0f; ux[id]=uy[id]=uz[id]=0.0f;
        return;
    }

    float r=0, jx=0, jy=0, jz=0;
    for(int q=0;q<19;++q){
        float fq = f[q*N + id];
        r += fq;
        jx += fq*cx19[q];
        jy += fq*cy19[q];
        jz += fq*cz19[q];
    }
    float ux0 = (r>0)? jx/r : 0;
    float uy0 = (r>0)? jy/r : 0;
    float uz0 = (r>0)? jz/r : 0;

    ux0 += fx; uy0 += fy; uz0 += fz;

    for(int q=0;q<19;++q){
        float fe = feq(q, r, ux0, uy0, uz0);
        float fq = f[q*N + id];
        float fc = fq + omega*(fe - fq);
        int x2 = (ix + cx19[q] + Nx) % Nx;
        int y2 = (iy + cy19[q] + Ny) % Ny;
        int z2 = (iz + cz19[q] + Nz) % Nz;
        int id2 = index3D(x2,y2,z2,Nx,Ny);
        if(solid[id2]){
            // 移動壁補正（回転体など）は本比較では不要なので外し、
            // 反射(バウンスバック)のみを行う。
            int qo = opp(q);
            fnext[qo*N + id] = fc;
        }else{
            fnext[q*N + id2] = fc;
        }
    }

    rho[id] = r;
    ux[id]  = ux0;
    uy[id]  = uy0;
    uz[id]  = uz0;
}
__global__ static void kern_swap(float* a, float* b, int n){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<n){ float t=a[i]; a[i]=b[i]; b[i]=t; }
}
// 必要なメモリを確保する
void LBM3D_Legacy::allocate(){
    int N = N_;
    CUDA_CHECK(cudaMalloc(&d_f_,     sizeof(float)*19*N));
    CUDA_CHECK(cudaMalloc(&d_fnext_, sizeof(float)*19*N));
    CUDA_CHECK(cudaMalloc(&d_rho_,   sizeof(float)*N));
    CUDA_CHECK(cudaMalloc(&d_u_,     sizeof(float)*N));
    CUDA_CHECK(cudaMalloc(&d_v_,     sizeof(float)*N));
    CUDA_CHECK(cudaMalloc(&d_w_,     sizeof(float)*N));
    CUDA_CHECK(cudaMalloc(&d_solid_, sizeof(unsigned char)*N));
}
// 確保したメモリを解放する
void LBM3D_Legacy::release(){
    cudaFree(d_f_); d_f_=nullptr;
    cudaFree(d_fnext_); d_fnext_=nullptr;
    cudaFree(d_rho_); d_rho_=nullptr;
    cudaFree(d_u_); d_u_=nullptr;
    cudaFree(d_v_); d_v_=nullptr;
    cudaFree(d_w_); d_w_=nullptr;
    cudaFree(d_solid_); d_solid_=nullptr;
}
// Domain 情報を受け取り、内部状態を構築
void LBM3D_Legacy::init(const Domain& d){
    Nx_=d.Nx; Ny_=d.Ny; Nz_=d.Nz; N_=Nx_*Ny_*Nz_;
    tau_ = d.tau;
    fx_ = d.forceX; fy_ = d.forceY; fz_ = d.forceZ;
    allocate();
    reset();
}
// ホスト側マスクをデバイスへコピー
void LBM3D_Legacy::setSolidMask(const unsigned char* h_mask){
    CUDA_CHECK(cudaMemcpy(d_solid_, h_mask, sizeof(unsigned char)*N_, cudaMemcpyHostToDevice));
}
// rho=1,u=0 の平衡状態で配列を埋める
void LBM3D_Legacy::reset(){
    dim3 bs(8,8,8);
    dim3 gs((Nx_+bs.x-1)/bs.x, (Ny_+bs.y-1)/bs.y, (Nz_+bs.z-1)/bs.z);
    kern_reset<<<gs,bs>>>(d_f_, d_rho_, d_u_, d_v_, d_w_, d_solid_, Nx_,Ny_,Nz_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
void LBM3D_Legacy::reinitEquilibriumFromMacro(const float* d_rho,
                                              const float* d_ux,
                                              const float* d_uy,
                                              const float* d_uz)
{
    //
    // 目的:
    // 初期条件を "rho,u を与えて平衡分布へ" で統一する。
    // (HOME も moments-only の reinit を持つため、API を揃える)
    //
    // 入力:
    // d_rho, d_ux, d_uy, d_uz はいずれも device pointer を想定。
    // u が nullptr の場合は 0 とする。
    //
    if(!d_rho){
        // rho が無い場合は reset 相当（rho=1,u=0）
        reset();
        return;
    }

    const size_t bytesScalar = sizeof(float) * static_cast<size_t>(N_);

    // マクロ量は出力にも使うので内部配列へコピーしておく
    CUDA_CHECK(cudaMemcpy(d_rho_, d_rho, bytesScalar, cudaMemcpyDeviceToDevice));
    if(d_ux){
        CUDA_CHECK(cudaMemcpy(d_u_, d_ux, bytesScalar, cudaMemcpyDeviceToDevice));
    }else{
        CUDA_CHECK(cudaMemset(d_u_, 0, bytesScalar));
    }
    if(d_uy){
        CUDA_CHECK(cudaMemcpy(d_v_, d_uy, bytesScalar, cudaMemcpyDeviceToDevice));
    }else{
        CUDA_CHECK(cudaMemset(d_v_, 0, bytesScalar));
    }
    if(d_uz){
        CUDA_CHECK(cudaMemcpy(d_w_, d_uz, bytesScalar, cudaMemcpyDeviceToDevice));
    }else{
        CUDA_CHECK(cudaMemset(d_w_, 0, bytesScalar));
    }

    // 平衡分布を構築
    reinit_equilibrium_from_macro(d_rho_, d_u_, d_v_, d_w_, d_f_, N_);
    CUDA_CHECK(cudaGetLastError());

    // fnext も整合させる（step 内で swap するため）
    const size_t bytesF = sizeof(float) * 19ull * static_cast<size_t>(N_);
    CUDA_CHECK(cudaMemcpy(d_fnext_, d_f_, bytesF, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
}
// BGK 衝突+ストリーミングを substeps 回分進める
void LBM3D_Legacy::step(int substeps){
    dim3 bs(8,8,8);
    dim3 gs((Nx_+bs.x-1)/bs.x, (Ny_+bs.y-1)/bs.y, (Nz_+bs.z-1)/bs.z);
    float omega = 1.0f/tau_;
    for(int s=0;s<substeps;++s){
        kern_collide_stream<<<gs,bs>>>(d_f_, d_fnext_, d_rho_, d_u_, d_v_, d_w_,
                                       d_solid_, Nx_,Ny_,Nz_, omega, fx_,fy_,fz_);
        CUDA_CHECK(cudaGetLastError());
        int n = 19*N_;
        dim3 bs1(256); dim3 gs1((n+255)/256);
        kern_swap<<<gs1,bs1>>>(d_f_, d_fnext_, n);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}
float*       LBM3D_Legacy::d_f()       { return d_f_; }
const float* LBM3D_Legacy::d_f() const { return d_f_; }
float*       LBM3D_Legacy::d_fnext()       { return d_fnext_; }
const float* LBM3D_Legacy::d_fnext() const { return d_fnext_; }
float*       LBM3D_Legacy::d_rho()       { return d_rho_; }
const float* LBM3D_Legacy::d_rho() const { return d_rho_; }
float*       LBM3D_Legacy::d_u()         { return d_u_; }
const float* LBM3D_Legacy::d_u()   const { return d_u_; }
float*       LBM3D_Legacy::d_v()         { return d_v_; }
const float* LBM3D_Legacy::d_v()   const { return d_v_; }
float*       LBM3D_Legacy::d_w()         { return d_w_; }
const float* LBM3D_Legacy::d_w()   const { return d_w_; }
