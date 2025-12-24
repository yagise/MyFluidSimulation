
#pragma once
#include <vector>
#include <glm/vec3.hpp>
#include "stl_loader.hpp"

//
// ボクセル化
//
// TriangleMesh をセルが solid(1) か fluid(0) かの 3D マスクへ変換する。
//
// - Legacy/HOME/Hybrid の境界条件（どのセルを solid とみなすか）を揃える
// - AMR(refine) は境界付近のボクセル化誤差を減らすための補助機能
//

struct VoxelParams {
    int Nx=128, Ny=128, Nz=128;
    float uniformScale = 0.9f; // ボックス一辺に対する比率
    glm::vec3 translate = glm::vec3(0.5f,0.5f,0.5f); // 単位立方体の中心
    // AMR: refine^3 でスーパーサンプリングしてから被覆率でダウンサンプルする。
    // coverageThreshold=0.5 なら半分以上埋まっていれば solid。
    int refine = 1;
    float coverageThreshold = 0.5f;
};

// mesh を params の格子へ voxelize して solidMask を返す。
std::vector<unsigned char> voxelize_mesh_to_mask(const TriangleMesh& mesh, const VoxelParams& params);
