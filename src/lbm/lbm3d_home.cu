#include "lbm3d_home.hpp"

#include "../common/cuda_utils.hpp"
// デバイスメモリ解放（init/reset 後でも安全に呼べるよう nullptr 初期化する）
LBM3D_Home::~LBM3D_Home(){ release(); }

//
// HOME(moments-only) 実装
//
// 保存する量（1セルあたり10変数）:
// rho, (ux,uy,uz), Sxx,Sxy,Sxz,Syy,Syz,Szz
// ここで S は 2次の非平衡応力（Pi^neq）を表す。
//
// f_i は保存しない。
// ストリーミングで必要になる f_i^* は (rho,u,S) から
// 2次までの regularized 形式でその場再構成する。
//

// D3Q19 の標準順序（lbm3d_legacy.cu と同じ）
__device__ __constant__ int cx19_h[19] = {0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1,  1,-1, 1,-1, 0, 0,  0, 0};
__device__ __constant__ int cy19_h[19] = {0, 0, 0, 1,-1, 0, 0, 1, 1,-1,-1,  0, 0, 0, 0, 1,-1,  1,-1};
__device__ __constant__ int cz19_h[19] = {0, 0, 0, 0, 0, 1,-1, 0, 0,  0, 0,  1, 1,-1,-1, 1, 1, -1,-1};
__device__ __constant__ float w19_h[19] = {1.0f/3.0f,
    1.0f/18.0f,1.0f/18.0f,1.0f/18.0f,1.0f/18.0f,1.0f/18.0f,1.0f/18.0f,
    1.0f/36.0f,1.0f/36.0f,1.0f/36.0f,1.0f/36.0f,1.0f/36.0f,1.0f/36.0f,
    1.0f/36.0f,1.0f/36.0f,1.0f/36.0f,1.0f/36.0f,1.0f/36.0f,1.0f/36.0f};
__device__ __forceinline__ static int opp19(int i){
    const int o[19] = {0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15,18,17};
    return o[i];
}
__device__ __forceinline__ static int index3D(int x,int y,int z,int Nx,int Ny){
    return (z*Ny + y)*Nx + x;
}

//
// moments <-> distribution 再構成（2次まで）
__device__ __forceinline__ static float reconstruct_f_post(
    int q,
    float rho,
    float ux,
    float uy,
    float uz,
    float Sxx,
    float Sxy,
    float Sxz,
    float Syy,
    float Syz,
    float Szz,
    float oneMinusOmega)
{
    // --- 平衡分布（2次まで）---
    const float cx = (float)cx19_h[q];
    const float cy = (float)cy19_h[q];
    const float cz = (float)cz19_h[q];
    const float w  = w19_h[q];

    const float eu = cx*ux + cy*uy + cz*uz;
    const float uu = ux*ux + uy*uy + uz*uz;
    const float feq = w * rho * (1.0f + 3.0f*eu + 4.5f*eu*eu - 1.5f*uu);

    // --- 2次の non-equilibrium（regularized）---
    // fneq_i = (w_i / (2 cs^4)) * ( (e_i e_i - cs^2 I) : Pi^neq )
    // cs^2 = 1/3 => 1/(2 cs^4) = 9/2 = 4.5
    const float cs2 = 1.0f/3.0f;
    const float Qxx = cx*cx - cs2;
    const float Qyy = cy*cy - cs2;
    const float Qzz = cz*cz - cs2;
    const float Qxy = cx*cy;
    const float Qxz = cx*cz;
    const float Qyz = cy*cz;

    // 対称テンソルなので off-diagonal は 2倍して二重収縮に合わせる
    const float contraction =
        Qxx*Sxx + Qyy*Syy + Qzz*Szz +
        2.0f*(Qxy*Sxy + Qxz*Sxz + Qyz*Syz);

    const float fneq = 4.5f * w * contraction;

    // 衝突後（streaming に流す値）
    return feq + oneMinusOmega * fneq;
}
__global__ static void kern_reset_mom(float* m, float* mnext,
                                      int Nx,int Ny,int Nz)
{
    const int ix = blockIdx.x*blockDim.x + threadIdx.x;
    const int iy = blockIdx.y*blockDim.y + threadIdx.y;
    const int iz = blockIdx.z*blockDim.z + threadIdx.z;
    if(ix>=Nx || iy>=Ny || iz>=Nz) return;

    const int N  = Nx*Ny*Nz;
    const int id = index3D(ix,iy,iz,Nx,Ny);

    // rho=1, u=0, S=0
    m[0*N + id] = 1.0f;
    m[1*N + id] = 0.0f;
    m[2*N + id] = 0.0f;
    m[3*N + id] = 0.0f;
    for(int k=4;k<10;++k) m[k*N + id] = 0.0f;

    // 次バッファも同じで埋めておく（初回 step の安定用）
    mnext[0*N + id] = 1.0f;
    mnext[1*N + id] = 0.0f;
    mnext[2*N + id] = 0.0f;
    mnext[3*N + id] = 0.0f;
    for(int k=4;k<10;++k) mnext[k*N + id] = 0.0f;
}
__global__ static void kern_reinit_from_macro(float* m, float* mnext,
                                              const float* rhoIn,
                                              const float* uxIn,
                                              const float* uyIn,
                                              const float* uzIn,
                                              int N)
{
    const int id = blockIdx.x*blockDim.x + threadIdx.x;
    if(id>=N) return;

    const float rho = rhoIn ? rhoIn[id] : 1.0f;
    const float ux  = uxIn  ? uxIn[id]  : 0.0f;
    const float uy  = uyIn  ? uyIn[id]  : 0.0f;
    const float uz  = uzIn  ? uzIn[id]  : 0.0f;

    // m
    m[0*N + id] = rho;
    m[1*N + id] = ux;
    m[2*N + id] = uy;
    m[3*N + id] = uz;
    for(int k=4;k<10;++k) m[k*N + id] = 0.0f;

    // mnext
    mnext[0*N + id] = rho;
    mnext[1*N + id] = ux;
    mnext[2*N + id] = uy;
    mnext[3*N + id] = uz;
    for(int k=4;k<10;++k) mnext[k*N + id] = 0.0f;
}

//
// collide + stream（gather）
__global__ static void kern_step_moments_only(const float* m, float* mnext,
                                              const unsigned char* solid,
                                              int Nx,int Ny,int Nz,
                                              float omega,
                                              float fx, float fy, float fz)
{
    const int ix = blockIdx.x*blockDim.x + threadIdx.x;
    const int iy = blockIdx.y*blockDim.y + threadIdx.y;
    const int iz = blockIdx.z*blockDim.z + threadIdx.z;
    if(ix>=Nx || iy>=Ny || iz>=Nz) return;

    const int N  = Nx*Ny*Nz;
    const int id = index3D(ix,iy,iz,Nx,Ny);

    // components
    const float* rhoA = m + 0*N;
    const float* uxA  = m + 1*N;
    const float* uyA  = m + 2*N;
    const float* uzA  = m + 3*N;
    const float* SxxA = m + 4*N;
    const float* SxyA = m + 5*N;
    const float* SxzA = m + 6*N;
    const float* SyyA = m + 7*N;
    const float* SyzA = m + 8*N;
    const float* SzzA = m + 9*N;

    float* rhoB = mnext + 0*N;
    float* uxB  = mnext + 1*N;
    float* uyB  = mnext + 2*N;
    float* uzB  = mnext + 3*N;
    float* SxxB = mnext + 4*N;
    float* SxyB = mnext + 5*N;
    float* SxzB = mnext + 6*N;
    float* SyyB = mnext + 7*N;
    float* SyzB = mnext + 8*N;
    float* SzzB = mnext + 9*N;

    // solid セルはそのままでも良いが、出力の安定性のため平衡に固定
    if(solid[id]){
        rhoB[id] = 1.0f;
        uxB[id]  = 0.0f;
        uyB[id]  = 0.0f;
        uzB[id]  = 0.0f;
        SxxB[id] = 0.0f; SxyB[id] = 0.0f; SxzB[id] = 0.0f;
        SyyB[id] = 0.0f; SyzB[id] = 0.0f; SzzB[id] = 0.0f;
        return;
    }

    // bounce-back 用に自セルの moments を取っておく
    const float rhoC = rhoA[id];
    const float uxC0 = uxA[id];
    const float uyC0 = uyA[id];
    const float uzC0 = uzA[id];
    const float SxxC = SxxA[id];
    const float SxyC = SxyA[id];
    const float SxzC = SxzA[id];
    const float SyyC = SyyA[id];
    const float SyzC = SyzA[id];
    const float SzzC = SzzA[id];

    const float oneMinusOmega = 1.0f - omega;

    // incoming 分布からマクロ量を計算
    float r = 0.0f;
    float jx = 0.0f, jy = 0.0f, jz = 0.0f;
    float Mxx = 0.0f, Mxy = 0.0f, Mxz = 0.0f;
    float Myy = 0.0f, Myz = 0.0f, Mzz = 0.0f;

    #pragma unroll
    for(int q=0;q<19;++q){
        // source cell = x - e_q
        const int xs = (ix - cx19_h[q] + Nx) % Nx;
        const int ys = (iy - cy19_h[q] + Ny) % Ny;
        const int zs = (iz - cz19_h[q] + Nz) % Nz;
        const int ids = index3D(xs,ys,zs,Nx,Ny);

        float fq;
        if(solid[ids]){
            // 近傍が solid の場合: 自セルから出て行く opp(q) が反射して q として戻る
            const int qo = opp19(q);
            fq = reconstruct_f_post(qo,
                                    rhoC, uxC0, uyC0, uzC0,
                                    SxxC, SxyC, SxzC, SyyC, SyzC, SzzC,
                                    oneMinusOmega);
        }else{
            // 通常: 近傍セルの post-collision 分布が流入
            fq = reconstruct_f_post(q,
                                    rhoA[ids], uxA[ids], uyA[ids], uzA[ids],
                                    SxxA[ids], SxyA[ids], SxzA[ids], SyyA[ids], SyzA[ids], SzzA[ids],
                                    oneMinusOmega);
        }

        const float cx = (float)cx19_h[q];
        const float cy = (float)cy19_h[q];
        const float cz = (float)cz19_h[q];

        r  += fq;
        jx += fq * cx;
        jy += fq * cy;
        jz += fq * cz;

        // 2次モーメント（対称）
        Mxx += fq * cx * cx;
        Mxy += fq * cx * cy;
        Mxz += fq * cx * cz;
        Myy += fq * cy * cy;
        Myz += fq * cy * cz;
        Mzz += fq * cz * cz;
    }

    // 速度（簡易外力: 従来法と同じく速度に加算）
    float ux = (r > 0.0f) ? (jx / r) : 0.0f;
    float uy = (r > 0.0f) ? (jy / r) : 0.0f;
    float uz = (r > 0.0f) ? (jz / r) : 0.0f;
    ux += fx; uy += fy; uz += fz;

    // 応力 S = Pi - Pi_eq を作る（Pi = 2次モーメント）
    // Pi_eq = rho*cs2*I + rho*u*u
    const float cs2 = 1.0f/3.0f;
    const float PiEq_xx = r*cs2 + r*ux*ux;
    const float PiEq_xy = r*ux*uy;
    const float PiEq_xz = r*ux*uz;
    const float PiEq_yy = r*cs2 + r*uy*uy;
    const float PiEq_yz = r*uy*uz;
    const float PiEq_zz = r*cs2 + r*uz*uz;

    const float Sxx = Mxx - PiEq_xx;
    const float Sxy = Mxy - PiEq_xy;
    const float Sxz = Mxz - PiEq_xz;
    const float Syy = Myy - PiEq_yy;
    const float Syz = Myz - PiEq_yz;
    const float Szz = Mzz - PiEq_zz;

    // 出力
    rhoB[id] = r;
    uxB[id]  = ux;
    uyB[id]  = uy;
    uzB[id]  = uz;
    SxxB[id] = Sxx;
    SxyB[id] = Sxy;
    SxzB[id] = Sxz;
    SyyB[id] = Syy;
    SyzB[id] = Syz;
    SzzB[id] = Szz;
}
// 必要なメモリを確保する
void LBM3D_Home::allocate(){
    const int N = N_;
    CUDA_CHECK(cudaMalloc(&d_m_,     sizeof(float)*10*N));
    CUDA_CHECK(cudaMalloc(&d_mnext_, sizeof(float)*10*N));
    CUDA_CHECK(cudaMalloc(&d_solid_, sizeof(unsigned char)*N));
}
// 確保したメモリを解放する
void LBM3D_Home::release(){
    cudaFree(d_m_);     d_m_ = nullptr;
    cudaFree(d_mnext_); d_mnext_ = nullptr;
    cudaFree(d_solid_); d_solid_ = nullptr;
}
void LBM3D_Home::init(const Domain& d){
    Nx_=d.Nx; Ny_=d.Ny; Nz_=d.Nz; N_=Nx_*Ny_*Nz_;
    tau_ = d.tau;
    fx_ = d.forceX; fy_ = d.forceY; fz_ = d.forceZ;
    allocate();
    reset();
}
void LBM3D_Home::setSolidMask(const unsigned char* h_mask){
    CUDA_CHECK(cudaMemcpy(d_solid_, h_mask, sizeof(unsigned char)*N_, cudaMemcpyHostToDevice));
}
void LBM3D_Home::reset(){
    dim3 bs(8,8,8);
    dim3 gs((Nx_+bs.x-1)/bs.x, (Ny_+bs.y-1)/bs.y, (Nz_+bs.z-1)/bs.z);
    kern_reset_mom<<<gs,bs>>>(d_m_, d_mnext_, Nx_,Ny_,Nz_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
void LBM3D_Home::reinitEquilibriumFromMacro(const float* d_rho,
                                            const float* d_ux,
                                            const float* d_uy,
                                            const float* d_uz)
{
    // NOTE: solidMask はここでは変更しない。
    const int N = N_;
    dim3 bs(256);
    dim3 gs((N + bs.x - 1) / bs.x);
    kern_reinit_from_macro<<<gs,bs>>>(d_m_, d_mnext_, d_rho, d_ux, d_uy, d_uz, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
void LBM3D_Home::step(int substeps){
    dim3 bs(8,8,8);
    dim3 gs((Nx_+bs.x-1)/bs.x, (Ny_+bs.y-1)/bs.y, (Nz_+bs.z-1)/bs.z);

    // ここでは 1緩和（BGK）相当の omega を使用。
    const float omega = 1.0f / tau_;

    for(int s=0;s<substeps;++s){
        kern_step_moments_only<<<gs,bs>>>(d_m_, d_mnext_, d_solid_,
                                          Nx_,Ny_,Nz_, omega,
                                          fx_, fy_, fz_);
        CUDA_CHECK(cudaGetLastError());

        // swap（ポインタの入れ替え）
        float* tmp = d_m_;
        d_m_ = d_mnext_;
        d_mnext_ = tmp;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}
float* LBM3D_Home::d_rho(){ return d_m_ + 0*N_; }
const float* LBM3D_Home::d_rho() const { return d_m_ + 0*N_; }
float* LBM3D_Home::d_u(){ return d_m_ + 1*N_; }
const float* LBM3D_Home::d_u() const { return d_m_ + 1*N_; }
float* LBM3D_Home::d_v(){ return d_m_ + 2*N_; }
const float* LBM3D_Home::d_v() const { return d_m_ + 2*N_; }
float* LBM3D_Home::d_w(){ return d_m_ + 3*N_; }
const float* LBM3D_Home::d_w() const { return d_m_ + 3*N_; }
