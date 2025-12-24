
#pragma once
#include <vector>
#include <glm/vec3.hpp>
#include <glad/glad.h>

// シンプルなメッシュ描画用の薄いラッパー（VAO/VBO/EBO と最小限のシェーダを持つ）
class MeshRenderer {
public:
    MeshRenderer() = default;
    ~MeshRenderer();

    // 頂点とインデックスを GPU に転送し、描画準備をする
    void upload(const std::vector<glm::vec3>& positions, const std::vector<unsigned>& indices);
    void setAlpha(float a){ alpha_ = a; }

    // viewProj 行列と色を渡してメッシュを描画する
    void draw(const float* viewProj /* 4x4 列優先 */, const float* color /* RGB */);

private:
    GLuint vao_ = 0, vbo_ = 0, ebo_ = 0, prog_ = 0;
    GLsizei indexCount_ = 0;
    float alpha_ = 0.2f;
};
