
#pragma once
#include <vector>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <cuda_runtime.h>
#include <glad/glad.h>

class Vis3D {
public:
    // summary: Vis3D の処理を行う
    // param nx: 入力パラメータ
    // param ny: 入力パラメータ
    // param nz: 入力パラメータ
    // return: 戻り値
    Vis3D(int nx, int ny, int nz);
    // summary: ~Vis3D の処理を行う
    // param: なし
    // return: 戻り値
    ~Vis3D() = default;
    // summary: uploadScalar の処理を行う
    // param d_field: 入力パラメータ
    // return: なし
    void uploadScalar(const float* d_field);
    // summary: 描画処理を行う
    // param P: 入力パラメータ
    // param V: 入力パラメータ
    // param camPos: 入力パラメータ
    // param step: 入力パラメータ
    // param scale: 入力パラメータ
    // param threshold: 入力パラメータ
    // return: なし
    void renderVolume(const glm::mat4& P, const glm::mat4& V,
                      const glm::vec3& camPos, float step, float scale, float threshold);
    // summary: nx の処理を行う
    // param: なし
    // return: 戻り値
    int nx() const { return Nx; }
    // summary: ny の処理を行う
    // param: なし
    // return: 戻り値
    int ny() const { return Ny; }
    // summary: nz の処理を行う
    // param: なし
    // return: 戻り値
    int nz() const { return Nz; }
private:
    // summary: ensureHostBuffers の処理を行う
    // param: なし
    // return: なし
    void ensureHostBuffers();
    int Nx, Ny, Nz;
    GLuint tex3d = 0;
    GLuint vao   = 0;
    GLuint vbo   = 0;
    GLuint progVol = 0;
    std::vector<float> h_scalar;
};
