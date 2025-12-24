
#pragma once
#include <vector>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <cuda_runtime.h>
#include <glad/glad.h>

// 3D スカラー場をテクスチャ化し、シンプルなボリュームレイマーチで描画する
class Vis3D {
public:
    // グリッドサイズを指定して初期化
    Vis3D(int nx, int ny, int nz);
    ~Vis3D() = default;

    // デバイス上のスカラー場を 3D テクスチャに転送する
    void uploadScalar(const float* d_field);
    // 射影・ビュー行列とカメラ情報を渡してボリューム描画する
    void renderVolume(const glm::mat4& P, const glm::mat4& V,
                      const glm::vec3& camPos, float step, float scale, float threshold);

    int nx() const { return Nx; }
    int ny() const { return Ny; }
    int nz() const { return Nz; }
private:
    // ホスト側の一時バッファを必要分だけ確保する
    void ensureHostBuffers();
    int Nx, Ny, Nz;
    GLuint tex3d = 0;
    GLuint vao   = 0;
    GLuint vbo   = 0;
    GLuint progVol = 0;
    std::vector<float> h_scalar;
};
