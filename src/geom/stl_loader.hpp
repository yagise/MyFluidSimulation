
#pragma once
#include <vector>
#include <string>
#include <glm/vec3.hpp>

//
// STL ローダ / 手続きメッシュ
//
// 目的:
// - --stl <path> で読み込んだ三角形メッシュを voxelize に渡す
// - 以前はデバッグ用に fan/teardrop を自動で挿入していたが、
// 手続きメッシュは明示指定されたときのみ使う方針に変更。
//

// 三角形メッシュ（インデックスは 3 の倍数: (i0,i1,i2) の並び）
struct TriangleMesh {
    std::vector<glm::vec3> positions;
    std::vector<unsigned>  indices;
    glm::vec3 bbmin, bbmax;
};
// 入力データを読み込む
bool load_stl(const std::string& path, TriangleMesh& out);
TriangleMesh make_teardrop(unsigned nu = 64, unsigned nv = 32);
TriangleMesh make_fan(unsigned blades = 5, float radius = 0.3f, float hub = 0.1f, float thickness = 0.02f);
