
#pragma once
#include <vector>
#include <glm/vec3.hpp>
#include <glad/glad.h>

class MeshRenderer {
public:
    // summary: MeshRenderer の処理を行う
    // param: なし
    // return: 戻り値
    MeshRenderer() = default;
    // summary: ~MeshRenderer の処理を行う
    // param: なし
    // return: 戻り値
    ~MeshRenderer();
    // summary: upload の処理を行う
    // param positions: 入力パラメータ
    // param indices: 入力パラメータ
    // return: なし
    void upload(const std::vector<glm::vec3>& positions, const std::vector<unsigned>& indices);
    // summary: setAlpha の処理を行う
    // param a: 入力パラメータ
    // return: なし
    void setAlpha(float a){ alpha_ = a; }
    // summary: 描画処理を行う
    // param major: 入力パラメータ
    // param rgb: 入力パラメータ
    // return: なし
    void draw(const float* viewProj /* 4x4 column-major */, const float* color /* rgb */);
private:
    GLuint vao_ = 0, vbo_ = 0, ebo_ = 0, prog_ = 0;
    GLsizei indexCount_ = 0;
    float alpha_ = 0.2f;
};
