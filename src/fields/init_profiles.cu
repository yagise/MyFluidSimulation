#include <cuda_runtime.h>

// 密度場 rho を初期化するための簡易プロファイル生成カーネル。
// - kern_rho_gauss : ガウス分布を重ねた密度場を作る
// - kern_rho_slab  : ある軸方向に slab 状に密度を持たせる
// それぞれ make_rho_* でグリッドを組み、呼び出し側に隠蔽する。
__global__ void kern_rho_gauss(float* rho, int Nx,int Ny,int Nz,
                               float cx,float cy,float cz, float sigma, float amp)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int N = Nx*Ny*Nz; if(i>=N) return;
    int x = i % Nx; int y = (i / Nx) % Ny; int z = i / (Nx*Ny);

    float X = (x+0.5f)/Nx, Y = (y+0.5f)/Ny, Z = (z+0.5f)/Nz;
    float dx=X-cx, dy=Y-cy, dz=Z-cz;
    float r2 = dx*dx + dy*dy + dz*dz;

    rho[i] = 1.0f + amp*__expf(-r2/(2.f*sigma*sigma));
}
__global__ void kern_rho_slab(float* rho, int Nx,int Ny,int Nz,
                              int axis, float amp, float center, float width)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int N = Nx*Ny*Nz; if(i>=N) return;
    int x = i % Nx; int y = (i / Nx) % Ny; int z = i / (Nx*Ny);
    float s = (axis==0)? (x+0.5f)/Nx : (axis==1)? (y+0.5f)/Ny : (z+0.5f)/Nz;
    float val = (fabsf(s-center) < 0.5f*width) ? amp : 0.f;
    rho[i] = 1.0f + val;
}
// ホスト側ラッパー: rho にガウス分布を足し込む
extern "C" void make_rho_gauss(float* d_rho, int Nx,int Ny,int Nz,
                               float cx,float cy,float cz, float sigma, float amp)
{
    int N = Nx*Ny*Nz;
    dim3 bs(256), gs((N+255)/256);
    kern_rho_gauss<<<gs,bs>>>(d_rho, Nx,Ny,Nz, cx,cy,cz, sigma, amp);
}
// ホスト側ラッパー: rho を slab 状の密度で初期化する
extern "C" void make_rho_slab(float* d_rho, int Nx,int Ny,int Nz,
                              int axis, float amp, float center, float width)
{
    int N = Nx*Ny*Nz;
    dim3 bs(256), gs((N+255)/256);
    kern_rho_slab<<<gs,bs>>>(d_rho, Nx,Ny,Nz, axis, amp, center, width);
}
