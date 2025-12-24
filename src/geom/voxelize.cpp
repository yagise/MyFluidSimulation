#include "voxelize.hpp"
#include <queue>
#include <tuple>
#include <algorithm>
#include <cmath>
#include <glm/glm.hpp>

// STL から生成した三角形メッシュをボクセル化し、solid/fluid マスクを作る。
// - triBoxOverlap_SAT : 三角形とボクセルの交差判定（分離軸定理）
// - make_transformed_tris : ユーザー指定のスケール/平行移動を適用
// - voxelize_single_resolution : 単一解像度で表面+内部を塗る
// - downsample_amr : refine^3 の高解像度ボクセルを占有率で粗化する
// - voxelize_mesh_to_mask : 上記を組み合わせた外向き API（voxelize.hpp）
static inline glm::vec3 vmin3(const glm::vec3& a, const glm::vec3& b){
    return glm::vec3(std::min(a.x,b.x), std::min(a.y,b.y), std::min(a.z,b.z));
}
static inline glm::vec3 vmax3(const glm::vec3& a, const glm::vec3& b){
    return glm::vec3(std::max(a.x,b.x), std::max(a.y,b.y), std::max(a.z,b.z));
}
static inline float fmin3(float a,float b,float c){ return std::min(a,std::min(b,c)); }
static inline float fmax3(float a,float b,float c){ return std::max(a,std::max(b,c)); }
// 三角形と軸平行ボックスが交差するかを SAT で判定する
static bool triBoxOverlap_SAT(const glm::vec3& c, const glm::vec3& half,
                              const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2)
{
    // 三角形をボックス中心に平行移動
    glm::vec3 tv0 = v0 - c;
    glm::vec3 tv1 = v1 - c;
    glm::vec3 tv2 = v2 - c;

    // 1) AABB早期除外（三角形AABB vs ボックス）
    glm::vec3 triMin = vmin3(vmin3(tv0, tv1), tv2);
    glm::vec3 triMax = vmax3(vmax3(tv0, tv1), tv2);
    if(triMax.x < -half.x || triMin.x > half.x) return false;
    if(triMax.y < -half.y || triMin.y > half.y) return false;
    if(triMax.z < -half.z || triMin.z > half.z) return false;

    // 2) 三角形平面 vs AABB
    glm::vec3 e0 = tv1 - tv0;
    glm::vec3 e1 = tv2 - tv1;
    glm::vec3 e2 = tv0 - tv2;
    glm::vec3 n  = glm::cross(e0, e1);
    float r = half.x*std::fabs(n.x) + half.y*std::fabs(n.y) + half.z*std::fabs(n.z);
    float s = std::fabs(n.x*tv0.x + n.y*tv0.y + n.z*tv0.z);
    if(s > r) return false;

    // 3) 9本の軸（エッジ×座標軸）
    const glm::vec3 axes[3] = {
        glm::vec3(1,0,0), glm::vec3(0,1,0), glm::vec3(0,0,1)
    };
    const glm::vec3 edges[3] = { e0, e1, e2 };

    for(int ei=0; ei<3; ++ei){
        const glm::vec3& e = edges[ei];
        // エッジがほぼゼロならスキップ（退化三角形対策）
        if(std::fabs(e.x)+std::fabs(e.y)+std::fabs(e.z) < 1e-12f) continue;

        for(int ai=0; ai<3; ++ai){
            // 軸 = e × axis(ai)
            glm::vec3 L = glm::cross(e, axes[ai]);
            float Llen2 = L.x*L.x + L.y*L.y + L.z*L.z;
            if(Llen2 < 1e-20f) continue; // e と座標軸が平行 → 無意味な軸

            // 三角形頂点の投影
            float p0 = L.x*tv0.x + L.y*tv0.y + L.z*tv0.z;
            float p1 = L.x*tv1.x + L.y*tv1.y + L.z*tv1.z;
            float p2 = L.x*tv2.x + L.y*tv2.y + L.z*tv2.z;
            float minp = std::min(p0, std::min(p1,p2));
            float maxp = std::max(p0, std::max(p1,p2));

            // ボックスの投影半径
            float rProj =
                half.x*std::fabs(L.x) +
                half.y*std::fabs(L.y) +
                half.z*std::fabs(L.z);

            if(maxp < -rProj || minp > rProj) return false; // 分離
        }
    }
    return true; // 全テストを通過 → 交差
}
// メッシュを単位立方体に収めるためのスケールと平行移動を計算する
static void compute_fit_transform(const TriangleMesh& m, const VoxelParams& vp,
                                  float& s, glm::vec3& trans)
{
    glm::vec3 bbsize = m.bbmax - m.bbmin;
    float maxDim = std::max(bbsize.x, std::max(bbsize.y, bbsize.z));
    s = (maxDim > 0.0f) ? (vp.uniformScale / maxDim) : 1.0f;
    glm::vec3 center = 0.5f*(m.bbmin + m.bbmax);
    trans = vp.translate - s*center;
}

struct Tri { glm::vec3 v0, v1, v2; };
// 入力メッシュにスケール/平行移動を適用し、三角形配列に展開する
static std::vector<Tri> make_transformed_tris(const TriangleMesh& meshIn, const VoxelParams& params){
    float s = 1.0f; glm::vec3 trans(0.0f);
    compute_fit_transform(meshIn, params, s, trans);

    std::vector<glm::vec3> verts; verts.reserve(meshIn.positions.size());
    for(size_t i=0;i<meshIn.positions.size();++i){
        verts.push_back(s*meshIn.positions[i] + trans);
    }

    std::vector<Tri> tris; tris.reserve(meshIn.indices.size()/3);
    for(size_t i=0;i+2<meshIn.indices.size(); i+=3){
        Tri t;
        t.v0 = verts[ meshIn.indices[i+0] ];
        t.v1 = verts[ meshIn.indices[i+1] ];
        t.v2 = verts[ meshIn.indices[i+2] ];
        tris.push_back(t);
    }
    return tris;
}
// 単一解像度で三角形群を voxelize し、表面/内部を 1 にしたマスクを返す
static std::vector<unsigned char> voxelize_single_resolution(const std::vector<Tri>& tris,
                                                             int Nx,int Ny,int Nz)
{
    const size_t Ntot = (size_t)Nx*Ny*Nz;
    std::vector<unsigned char> surf(Ntot, 0);

    glm::vec3 half(0.5f/Nx, 0.5f/Ny, 0.5f/Nz);
    auto idx = [=](int x,int y,int z){ return (size_t)z*Ny*Nx + (size_t)y*Nx + (size_t)x; };

    const float eps = 1e-6f; // ボクセル境界の数値誤差吸収
    for(size_t it=0; it<tris.size(); ++it){
        const Tri& t = tris[it];
        float minx = fmin3(t.v0.x, t.v1.x, t.v2.x) - eps;
        float miny = fmin3(t.v0.y, t.v1.y, t.v2.y) - eps;
        float minz = fmin3(t.v0.z, t.v1.z, t.v2.z) - eps;
        float maxx = fmax3(t.v0.x, t.v1.x, t.v2.x) + eps;
        float maxy = fmax3(t.v0.y, t.v1.y, t.v2.y) + eps;
        float maxz = fmax3(t.v0.z, t.v1.z, t.v2.z) + eps;

        int x0 = std::max(0,    (int)std::floor(minx * Nx));
        int y0 = std::max(0,    (int)std::floor(miny * Ny));
        int z0 = std::max(0,    (int)std::floor(minz * Nz));
        int x1 = std::min(Nx-1, (int)std::floor(maxx * Nx));
        int y1 = std::min(Ny-1, (int)std::floor(maxy * Ny));
        int z1 = std::min(Nz-1, (int)std::floor(maxz * Nz));

        for(int z=z0; z<=z1; ++z)
        for(int y=y0; y<=y1; ++y)
        for(int x=x0; x<=x1; ++x){
            glm::vec3 c((x+0.5f)/Nx, (y+0.5f)/Ny, (z+0.5f)/Nz);
            if(triBoxOverlap_SAT(c, half, t.v0, t.v1, t.v2)){
                surf[idx(x,y,z)] = 1; // 表面ボクセル
            }
        }
    }

    // 外部BFS（6近傍）で外部をマーキング
    std::vector<unsigned char> vis(Ntot, 0);
    std::queue<size_t> q;
    auto push = [&](int x,int y,int z){
        if(x<0||y<0||z<0||x>=Nx||y>=Ny||z>=Nz) return;
        size_t i = idx(x,y,z);
        if(vis[i] || surf[i]) return;
        vis[i]=1; q.push(i);
    };
    // 6面から外部注入
    for(int y=0;y<Ny;++y) for(int z=0;z<Nz;++z){ push(0,y,z); push(Nx-1,y,z); }
    for(int x=0;x<Nx;++x) for(int z=0;z<Nz;++z){ push(x,0,z); push(x,Ny-1,z); }
    for(int x=0;x<Nx;++x) for(int y=0;y<Ny;++y){ push(x,y,0); push(x,y,Nz-1); }

    // キュー処理（C++14でOK：構造化束縛を使わない）
    auto invIdx = [=](size_t i, int& x, int& y, int& z){
        x = (int)(i % Nx);
        y = (int)((i / Nx) % Ny);
        z = (int)(i / (Nx*Ny));
    };
    while(!q.empty()){
        size_t i = q.front(); q.pop();
        int x,y,z; invIdx(i, x,y,z);
        push(x+1,y,z); push(x-1,y,z);
        push(x,y+1,z); push(x,y-1,z);
        push(x,y,z+1); push(x,y,z-1);
    }

    // solid = 表面 または 内部（vis==0）
    std::vector<unsigned char> solid(Ntot, 0);
    for(size_t i=0;i<Ntot;++i){
        if(surf[i] || !vis[i]) solid[i]=1;
    }
    return solid;
}
// refine^3 のマスクを占有率しきい値で粗化する（壁 AMR 用）
static std::vector<unsigned char> downsample_amr(const std::vector<unsigned char>& fine,
                                                 int refine,
                                                 int Nx, int Ny, int Nz,
                                                 float threshold)
{
    const int NxF = Nx * refine;
    const int NyF = Ny * refine;
    const int NzF = Nz * refine;
    const size_t Ncoarse = (size_t)Nx * Ny * Nz;
    std::vector<unsigned char> coarse(Ncoarse, 0);

    auto idxFine = [=](int x,int y,int z){ return (size_t)z*NyF*NxF + (size_t)y*NxF + (size_t)x; };
    auto idxCoarse = [=](int x,int y,int z){ return (size_t)z*Ny*Nx + (size_t)y*Nx + (size_t)x; };

    const int refine3 = refine*refine*refine;
    const float thr = std::max(0.0f, std::min(1.0f, threshold));

    for(int cz=0; cz<Nz; ++cz){
        for(int cy=0; cy<Ny; ++cy){
            for(int cx=0; cx<Nx; ++cx){
                int solidCount = 0;
                int fx0 = cx * refine;
                int fy0 = cy * refine;
                int fz0 = cz * refine;
                for(int dz=0; dz<refine; ++dz){
                    for(int dy=0; dy<refine; ++dy){
                        size_t base = idxFine(fx0, fy0+dy, fz0+dz);
                        for(int dx=0; dx<refine; ++dx){
                            solidCount += fine[base + dx];
                        }
                    }
                }
                float cov = static_cast<float>(solidCount) / static_cast<float>(refine3);
                bool occupied = (cov > 0.0f) && (cov >= thr);
                if(occupied) coarse[idxCoarse(cx,cy,cz)] = 1;
            }
        }
    }
    return coarse;
}
// 公開 API: メッシュを指定解像度(必要なら refine)でボクセル化し solid マスクを返す
std::vector<unsigned char> voxelize_mesh_to_mask(const TriangleMesh& meshIn, const VoxelParams& params)
{
    auto tris = make_transformed_tris(meshIn, params);

    int refine = (params.refine < 1) ? 1 : params.refine;
    if(refine == 1){
        return voxelize_single_resolution(tris, params.Nx, params.Ny, params.Nz);
    }

    const int NxF = params.Nx * refine;
    const int NyF = params.Ny * refine;
    const int NzF = params.Nz * refine;

    auto fine = voxelize_single_resolution(tris, NxF, NyF, NzF);
    return downsample_amr(fine, refine, params.Nx, params.Ny, params.Nz, params.coverageThreshold);
}
