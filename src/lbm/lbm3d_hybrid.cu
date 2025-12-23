#include "lbm3d_hybrid.hpp"
#include "../common/cuda_utils.hpp"
#include <cmath>

// summary: 確保したメモリを解放する
// param: なし
// return: なし
LBM3D_Hybrid::~LBM3D_Hybrid(){ release(); }

//
// Correct B0 implementation (full-f, cell-wise switch)
//
// This file implements a hybrid collide-stream kernel that chooses between:
// - Legacy collision (BGK)
// - HOME collision (moment/regularized style in LBM3D_Home)
// per-cell according to a mask d_isLegacy_[id].
//
// IMPORTANT:
// - We keep the *streaming* and *bounce-back* identical to Legacy/Home.
// - This file is meant for *fair comparison* experiments.
// - It is not the memory-saving "moment-cache" hybrid.
//
// Direction ordering must match lbm3d_legacy.cu / lbm3d_home.cu.

// Standard D3Q19 ordering:
// 0:(0,0,0)
// 1:(+1,0,0) 2:(-1,0,0) 3:(0,+1,0) 4:(0,-1,0) 5:(0,0,+1) 6:(0,0,-1)
// 7:(+1,+1,0) 8:(-1,+1,0) 9:(+1,-1,0) 10:(-1,-1,0)
// 11:(+1,0,+1) 12:(-1,0,+1) 13:(+1,0,-1) 14:(-1,0,-1)
// 15:(0,+1,+1) 16:(0,-1,+1) 17:(0,+1,-1) 18:(0,-1,-1)
__device__ __constant__ int cx19_b0[19] = {0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1,  1,-1, 1,-1, 0, 0,  0, 0};
__device__ __constant__ int cy19_b0[19] = {0, 0, 0, 1,-1, 0, 0, 1, 1,-1,-1,  0, 0, 0, 0, 1,-1,  1,-1};
__device__ __constant__ int cz19_b0[19] = {0, 0, 0, 0, 0, 1,-1, 0, 0,  0, 0,  1, 1,-1,-1, 1, 1, -1,-1};
__device__ __constant__ float w19_b0[19] = {1.0f/3.0f,
    1.0f/18.0f,1.0f/18.0f,1.0f/18.0f,1.0f/18.0f,1.0f/18.0f,1.0f/18.0f,
    1.0f/36.0f,1.0f/36.0f,1.0f/36.0f,1.0f/36.0f,1.0f/36.0f,1.0f/36.0f,
    1.0f/36.0f,1.0f/36.0f,1.0f/36.0f,1.0f/36.0f,1.0f/36.0f,1.0f/36.0f};

// summary: opp19 の処理を行う
// param i: 入力パラメータ
// return: 戻り値
__device__ __forceinline__ static int opp19(int i){
    const int o[19] = {0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15,18,17};
    return o[i];
}

// summary: index3D の処理を行う
// param x: 入力パラメータ
// param y: 入力パラメータ
// param z: 入力パラメータ
// param Nx: 入力パラメータ
// param Ny: 入力パラメータ
// return: 戻り値
__device__ __forceinline__ static int index3D(int x,int y,int z,int Nx,int Ny){
    return (z*Ny + y)*Nx + x;
}

// summary: feq19 の処理を行う
// param q: 入力パラメータ
// param rho: 入力パラメータ
// param ux: 入力パラメータ
// param uy: 入力パラメータ
// param uz: 入力パラメータ
// return: 戻り値
__device__ __forceinline__ static float feq19(int q, float rho, float ux, float uy, float uz){
    float eiu = cx19_b0[q]*ux + cy19_b0[q]*uy + cz19_b0[q]*uz;
    float uu = ux*ux + uy*uy + uz*uz;
    return w19_b0[q]*rho*(1.0f + 3.0f*eiu + 4.5f*eiu*eiu - 1.5f*uu);
}

// summary: kern_reset の処理を行う
// param f: 入力パラメータ
// param rho: 入力パラメータ
// param u: 入力パラメータ
// param v: 入力パラメータ
// param w: 入力パラメータ
// param Nx: 入力パラメータ
// param Ny: 入力パラメータ
// param Nz: 入力パラメータ
// return: なし
__global__ static void kern_reset(float* f, float* rho, float* u, float* v, float* w,
                                  int Nx,int Ny,int Nz){
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    int iz = blockIdx.z*blockDim.z + threadIdx.z;
    if(ix>=Nx || iy>=Ny || iz>=Nz) return;
    int N = Nx*Ny*Nz;
    int id = index3D(ix,iy,iz,Nx,Ny);
    float r = 1.0f;
    rho[id] = r; u[id]=v[id]=w[id]=0.0f;
    for(int q=0;q<19;++q){
        f[q*N + id] = w19_b0[q] * r;
    }
}

// summary: kern_reinit_equilibrium の処理を行う
// param rhoIn: 入力パラメータ
// param uxIn: 入力パラメータ
// param uyIn: 入力パラメータ
// param uzIn: 入力パラメータ
// param f: 入力パラメータ
// param rhoOut: 入力パラメータ
// param uxOut: 入力パラメータ
// param uyOut: 入力パラメータ
// param uzOut: 入力パラメータ
// param Nx: 入力パラメータ
// param Ny: 入力パラメータ
// param Nz: 入力パラメータ
// return: なし
__global__ static void kern_reinit_equilibrium(const float* rhoIn,
                                               const float* uxIn,
                                               const float* uyIn,
                                               const float* uzIn,
                                               float* f,
                                               float* rhoOut,
                                               float* uxOut,
                                               float* uyOut,
                                               float* uzOut,
                                               int Nx,int Ny,int Nz)
{
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    int iz = blockIdx.z*blockDim.z + threadIdx.z;
    if(ix>=Nx || iy>=Ny || iz>=Nz) return;
    int N = Nx*Ny*Nz;
    int id = index3D(ix,iy,iz,Nx,Ny);

    float r  = rhoIn ? rhoIn[id] : 1.0f;
    float ux = uxIn  ? uxIn[id]  : 0.0f;
    float uy = uyIn  ? uyIn[id]  : 0.0f;
    float uz = uzIn  ? uzIn[id]  : 0.0f;
    rhoOut[id] = r;
    uxOut[id]  = ux;
    uyOut[id]  = uy;
    uzOut[id]  = uz;
    for(int q=0;q<19;++q){
        f[q*N + id] = feq19(q, r, ux, uy, uz);
    }
}

// summary: kern_collide_stream_b0 の処理を行う
// param f: 入力パラメータ
// param fnext: 入力パラメータ
// param rho: 入力パラメータ
// param ux: 入力パラメータ
// param uy: 入力パラメータ
// param uz: 入力パラメータ
// param solid: 入力パラメータ
// param isLegacy: 入力パラメータ
// param Nx: 入力パラメータ
// param Ny: 入力パラメータ
// param Nz: 入力パラメータ
// param omegaLegacy: 入力パラメータ
// param omegaHydro: 入力パラメータ
// param omegaShear: 入力パラメータ
// param fx: 入力パラメータ
// param fy: 入力パラメータ
// param fz: 入力パラメータ
// return: なし
__global__ static void kern_collide_stream_b0(const float* f, float* fnext,
                                              float* rho, float* ux, float* uy, float* uz,
                                              const unsigned char* solid,
                                              const unsigned char* isLegacy,
                                              int Nx,int Ny,int Nz,
                                              float omegaLegacy,
                                              float omegaHydro, float omegaShear,
                                              float fx, float fy, float fz)
{
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    int iz = blockIdx.z*blockDim.z + threadIdx.z;
    if(ix>=Nx || iy>=Ny || iz>=Nz) return;

    int N = Nx*Ny*Nz;
    int id = index3D(ix,iy,iz,Nx,Ny);

    if(solid[id]){
        // Keep solid f unchanged (like legacy/home), and clamp macros.
        for(int q=0;q<19;++q) fnext[q*N + id] = f[q*N + id];
        rho[id]=1.0f; ux[id]=uy[id]=uz[id]=0.0f;
        return;
    }

    // Macro compute (same as legacy/home)
    float r=0, jx=0, jy=0, jz=0;
    for(int q=0;q<19;++q){
        float fq = f[q*N + id];
        r  += fq;
        jx += fq * cx19_b0[q];
        jy += fq * cy19_b0[q];
        jz += fq * cz19_b0[q];
    }

    float ux0 = (r>0.0f) ? (jx/r) : 0.0f;
    float uy0 = (r>0.0f) ? (jy/r) : 0.0f;
    float uz0 = (r>0.0f) ? (jz/r) : 0.0f;
    ux0 += fx; uy0 += fy; uz0 += fz;

    // Precompute equilibrium for all q.
    float uu = ux0*ux0 + uy0*uy0 + uz0*uz0;
    float feq[19];
    #pragma unroll
    for(int q=0;q<19;++q){
        float eiu = cx19_b0[q]*ux0 + cy19_b0[q]*uy0 + cz19_b0[q]*uz0;
        feq[q] = w19_b0[q]*r*(1.0f + 3.0f*eiu + 4.5f*eiu*eiu - 1.5f*uu);
    }

    const bool useLegacy = (isLegacy != nullptr) ? (isLegacy[id] != 0) : false;

    // Collide + stream (push), identical bounce-back handling.
    for(int q=0;q<19;++q){
        const float fq = f[q*N + id];
        float fpost;

        if(useLegacy){
            // Legacy BGK
            fpost = fq + omegaLegacy*(feq[q] - fq);
        }else{
            // HOME collide (same algebraic form as LBM3D_Home)
            // Keep it explicit for future modifications.
            float dh = feq[q];
            float ds = fq - feq[q];
            fpost = fq + omegaHydro*(dh - fq) + (omegaShear - omegaHydro) * (-ds);
        }

        int x2 = (ix + cx19_b0[q] + Nx) % Nx;
        int y2 = (iy + cy19_b0[q] + Ny) % Ny;
        int z2 = (iz + cz19_b0[q] + Nz) % Nz;
        int id2 = index3D(x2,y2,z2,Nx,Ny);

        if(solid[id2]){
            // 論文用整理:
            // Hybrid(B0) でも移動壁の補正は不要。
            // 反射(バウンスバック)のみを適用する。
            int qo = opp19(q);
            fnext[qo*N + id] = fpost;
        }else{
            // Normal streaming
            fnext[q*N + id2] = fpost;
        }
    }

    rho[id] = r;
    ux[id]  = ux0;
    uy[id]  = uy0;
    uz[id]  = uz0;
}

// summary: kern_swap の処理を行う
// param a: 入力パラメータ
// param b: 入力パラメータ
// param n: 入力パラメータ
// return: なし
__global__ static void kern_swap(float* a, float* b, int n){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<n){ float t=a[i]; a[i]=b[i]; b[i]=t; }
}

//
// LBM3D_Hybrid methods
//

// summary: 必要なメモリを確保する
// param: なし
// return: 戻り値
void LBM3D_Hybrid::allocate(){
    const int N = N_;
    CUDA_CHECK(cudaMalloc(&d_f_,     sizeof(float)*19ull*N));
    CUDA_CHECK(cudaMalloc(&d_fnext_, sizeof(float)*19ull*N));
    CUDA_CHECK(cudaMalloc(&d_rho_,   sizeof(float)*N));
    CUDA_CHECK(cudaMalloc(&d_u_,     sizeof(float)*N));
    CUDA_CHECK(cudaMalloc(&d_v_,     sizeof(float)*N));
    CUDA_CHECK(cudaMalloc(&d_w_,     sizeof(float)*N));
    CUDA_CHECK(cudaMalloc(&d_solid_, sizeof(unsigned char)*N));
    CUDA_CHECK(cudaMalloc(&d_isLegacy_, sizeof(unsigned char)*N));
}

// summary: 確保したメモリを解放する
// param: なし
// return: 戻り値
void LBM3D_Hybrid::release(){
    cudaFree(d_f_); d_f_=nullptr;
    cudaFree(d_fnext_); d_fnext_=nullptr;
    cudaFree(d_rho_); d_rho_=nullptr;
    cudaFree(d_u_); d_u_=nullptr;
    cudaFree(d_v_); d_v_=nullptr;
    cudaFree(d_w_); d_w_=nullptr;
    cudaFree(d_solid_); d_solid_=nullptr;
    cudaFree(d_isLegacy_); d_isLegacy_=nullptr;
    Nx_=Ny_=Nz_=N_=0;
}

// summary: 初期化処理を行う
// param d: 入力パラメータ
// param nLegacyCells: 入力パラメータ
// return: 戻り値
void LBM3D_Hybrid::init(const Domain& d, int /*nLegacyCells*/){
    Nx_=d.Nx; Ny_=d.Ny; Nz_=d.Nz; N_=Nx_*Ny_*Nz_;
    tau_ = d.tau;
    fx_  = d.forceX; fy_ = d.forceY; fz_ = d.forceZ;
    allocate();

    // Default: no legacy cells => pure HOME behavior.
    CUDA_CHECK(cudaMemset(d_isLegacy_, 0, sizeof(unsigned char)*N_));

    // Reset (safe even before solid mask is uploaded)
    reset();
}

// summary: setSolidMask の処理を行う
// param h_mask: 入力パラメータ
// return: 戻り値
void LBM3D_Hybrid::setSolidMask(const unsigned char* h_mask){
    CUDA_CHECK(cudaMemcpy(d_solid_, h_mask, sizeof(unsigned char)*N_, cudaMemcpyHostToDevice));
}

// summary: setLegacyMapping の処理を行う
// param h_isLegacy: 入力パラメータ
// param h_legacySlot_ignored: 入力パラメータ
// return: 戻り値
void LBM3D_Hybrid::setLegacyMapping(const unsigned char* h_isLegacy,
                                    const int* /*h_legacySlot_ignored*/)
{
    if(h_isLegacy){
        CUDA_CHECK(cudaMemcpy(d_isLegacy_, h_isLegacy, sizeof(unsigned char)*N_, cudaMemcpyHostToDevice));
    }else{
        CUDA_CHECK(cudaMemset(d_isLegacy_, 0, sizeof(unsigned char)*N_));
    }
}

// summary: reset の処理を行う
// param: なし
// return: 戻り値
void LBM3D_Hybrid::reset(){
    dim3 bs(8,8,8);
    dim3 gs((Nx_+bs.x-1)/bs.x, (Ny_+bs.y-1)/bs.y, (Nz_+bs.z-1)/bs.z);
    kern_reset<<<gs,bs>>>(d_f_, d_rho_, d_u_, d_v_, d_w_, Nx_,Ny_,Nz_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// summary: reinitEquilibriumFromMacro の処理を行う
// param d_rho: 入力パラメータ
// param d_ux: 入力パラメータ
// param d_uy: 入力パラメータ
// param d_uz: 入力パラメータ
// return: 戻り値
void LBM3D_Hybrid::reinitEquilibriumFromMacro(const float* d_rho,
                                              const float* d_ux,
                                              const float* d_uy,
                                              const float* d_uz)
{
    dim3 bs(8,8,8);
    dim3 gs((Nx_+bs.x-1)/bs.x, (Ny_+bs.y-1)/bs.y, (Nz_+bs.z-1)/bs.z);
    kern_reinit_equilibrium<<<gs,bs>>>(d_rho, d_ux, d_uy, d_uz,
                                       d_f_, d_rho_, d_u_, d_v_, d_w_,
                                       Nx_,Ny_,Nz_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// summary: step の処理を行う
// param substeps: 入力パラメータ
// return: 戻り値
void LBM3D_Hybrid::step(int substeps){
    dim3 bs(8,8,8);
    dim3 gs((Nx_+bs.x-1)/bs.x, (Ny_+bs.y-1)/bs.y, (Nz_+bs.z-1)/bs.z);

    // 既存ソルバ(Legacy/Home)と同じ tau から緩和率を作る。
    float omegaLegacy = 1.0f / tau_;
    float omegaHydro  = 1.0f / tau_;
    // 公平な比較のため、HOME 側の shear も同じ緩和率に揃える。
    float omegaShear  = omegaHydro;

    for(int s=0;s<substeps;++s){
        kern_collide_stream_b0<<<gs,bs>>>(d_f_, d_fnext_, d_rho_, d_u_, d_v_, d_w_,
                                          d_solid_, d_isLegacy_,
                                          Nx_,Ny_,Nz_,
                                          omegaLegacy, omegaHydro, omegaShear,
                                          fx_,fy_,fz_);
        CUDA_CHECK(cudaGetLastError());
        int n = 19*N_;
        dim3 bs1(256);
        dim3 gs1((n+255)/256);
        kern_swap<<<gs1,bs1>>>(d_f_, d_fnext_, n);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}
