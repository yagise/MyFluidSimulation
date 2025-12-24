#include "lbm3d_hybrid.hpp"
#include "../common/cuda_utils.hpp"
#include <cmath>
// 確保したメモリを解放する
LBM3D_Hybrid::~LBM3D_Hybrid(){ release(); }

//
// 正式版の B0 実装（フル f、セル単位で Legacy/HOME を切り替え）
//
// d_isLegacy_ マスクに応じて Legacy か HOME の衝突をセル単位で選ぶハイブリッドカーネル。
// ストリーミングとバウンスバックは Legacy/Home と同じ扱いで、比較実験用の実装（メモリ節約版ではない）。
// 方向の並びは lbm3d_legacy.cu / lbm3d_home.cu と一致させる。

// D3Q19 の標準的な並び:
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
// 方向 q の反対方向インデックスを返す
__device__ __forceinline__ static int opp19(int i){
    const int o[19] = {0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15,18,17};
    return o[i];
}
// 3D グリッドを一次元インデックスに変換する
__device__ __forceinline__ static int index3D(int x,int y,int z,int Nx,int Ny){
    return (z*Ny + y)*Nx + x;
}
// D3Q19 の平衡分布 feq(q) を計算する
__device__ __forceinline__ static float feq19(int q, float rho, float ux, float uy, float uz){
    float eiu = cx19_b0[q]*ux + cy19_b0[q]*uy + cz19_b0[q]*uz;
    float uu = ux*ux + uy*uy + uz*uz;
    return w19_b0[q]*rho*(1.0f + 3.0f*eiu + 4.5f*eiu*eiu - 1.5f*uu);
}
// 分布関数とマクロ量を平衡状態へリセットする
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
// マクロ入力(rho,u)から平衡分布 feq を構築し直す
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
// Legacy/HOME をセル単位で切り替えつつ衝突・ストリーミングを行う
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
        // solid セルの f は変更せず（Legacy/HOME と同様）、マクロ量を固定する。
        for(int q=0;q<19;++q) fnext[q*N + id] = f[q*N + id];
        rho[id]=1.0f; ux[id]=uy[id]=uz[id]=0.0f;
        return;
    }

    // マクロ量の計算（Legacy/HOME と同じ）
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

    // 全 q の平衡分布を事前計算
    float uu = ux0*ux0 + uy0*uy0 + uz0*uz0;
    float feq[19];
    #pragma unroll
    for(int q=0;q<19;++q){
        float eiu = cx19_b0[q]*ux0 + cy19_b0[q]*uy0 + cz19_b0[q]*uz0;
        feq[q] = w19_b0[q]*r*(1.0f + 3.0f*eiu + 4.5f*eiu*eiu - 1.5f*uu);
    }

    const bool useLegacy = (isLegacy != nullptr) ? (isLegacy[id] != 0) : false;

    // 衝突 + ストリーミング（プッシュ）。バウンスバックは同一処理。
    for(int q=0;q<19;++q){
        const float fq = f[q*N + id];
        float fpost;

        if(useLegacy){
            // Legacy の BGK
            fpost = fq + omegaLegacy*(feq[q] - fq);
        }else{
            // HOME の衝突（LBM3D_Home と同じ代数形）
            // 将来の変更に備えて明示的に書く。
            float dh = feq[q];
            float ds = fq - feq[q];
            fpost = fq + omegaHydro*(dh - fq) + (omegaShear - omegaHydro) * (-ds);
        }

        int x2 = (ix + cx19_b0[q] + Nx) % Nx;
        int y2 = (iy + cy19_b0[q] + Ny) % Ny;
        int z2 = (iz + cz19_b0[q] + Nz) % Nz;
        int id2 = index3D(x2,y2,z2,Nx,Ny);

        if(solid[id2]){
            // Hybrid(B0) でも移動壁の補正は不要。
            // 反射(バウンスバック)のみを適用する。
            int qo = opp19(q);
            fnext[qo*N + id] = fpost;
        }else{
            // 通常ストリーミング
            fnext[q*N + id2] = fpost;
        }
    }

    rho[id] = r;
    ux[id]  = ux0;
    uy[id]  = uy0;
    uz[id]  = uz0;
}
// 2 本の配列を要素ごとに入れ替える
__global__ static void kern_swap(float* a, float* b, int n){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<n){ float t=a[i]; a[i]=b[i]; b[i]=t; }
}

//
// LBM3D_Hybrid のメソッド
//
// 必要なデバイスメモリを確保する
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
// 確保したデバイスメモリを解放する
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
// ドメイン設定と外力を反映してバッファを初期化する
void LBM3D_Hybrid::init(const Domain& d, int /*nLegacyCells*/){
    Nx_=d.Nx; Ny_=d.Ny; Nz_=d.Nz; N_=Nx_*Ny_*Nz_;
    tau_ = d.tau;
    fx_  = d.forceX; fy_ = d.forceY; fz_ = d.forceZ;
    allocate();

    // 既定: Legacy セルなし => 純粋な HOME 挙動。
    CUDA_CHECK(cudaMemset(d_isLegacy_, 0, sizeof(unsigned char)*N_));

    // リセット（solid マスク未転送でも安全）
    reset();
}
// 固体マスクをデバイスへ送る
void LBM3D_Hybrid::setSolidMask(const unsigned char* h_mask){
    CUDA_CHECK(cudaMemcpy(d_solid_, h_mask, sizeof(unsigned char)*N_, cudaMemcpyHostToDevice));
}
// Legacy 領域マスクを設定する（未指定なら全セル HOME）
void LBM3D_Hybrid::setLegacyMapping(const unsigned char* h_isLegacy,
                                    const int* /*h_legacySlot_ignored*/)
{
    if(h_isLegacy){
        CUDA_CHECK(cudaMemcpy(d_isLegacy_, h_isLegacy, sizeof(unsigned char)*N_, cudaMemcpyHostToDevice));
    }else{
        CUDA_CHECK(cudaMemset(d_isLegacy_, 0, sizeof(unsigned char)*N_));
    }
}
// 分布関数とマクロ量を平衡状態にリセットする
void LBM3D_Hybrid::reset(){
    dim3 bs(8,8,8);
    dim3 gs((Nx_+bs.x-1)/bs.x, (Ny_+bs.y-1)/bs.y, (Nz_+bs.z-1)/bs.z);
    kern_reset<<<gs,bs>>>(d_f_, d_rho_, d_u_, d_v_, d_w_, Nx_,Ny_,Nz_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
// 与えられたマクロ量(rho,u)から feq を再構築する
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
// substeps 回だけ Hybrid(B0) のステップを進める
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
