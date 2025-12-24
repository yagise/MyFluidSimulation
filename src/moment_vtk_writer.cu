
#include <cuda_runtime.h>
#include "common/cuda_utils.hpp"
#include <vector>
#include <fstream>
#include <string>
#include <stdexcept>
// スカラー場を VTK (STRUCTURED_POINTS, ASCII) としてホスト経由で書き出す
extern "C" void write_scalar_vtk(const char* filename, const float* d_field, int Nx, int Ny, int Nz){
    const size_t N = static_cast<size_t>(Nx) * static_cast<size_t>(Ny) * static_cast<size_t>(Nz);
    std::vector<float> h(N);
    CUDA_CHECK(cudaMemcpy(h.data(), d_field, N * sizeof(float), cudaMemcpyDeviceToHost));
    std::ofstream f(filename);
    if(!f) throw std::runtime_error("failed to open VTK file");
    f << "# vtk DataFile Version 3.0\nscalar\nASCII\n"
      << "DATASET STRUCTURED_POINTS\n"
      << "DIMENSIONS " << Nx << " " << Ny << " " << Nz << "\n"
      << "ORIGIN 0 0 0\n"
      << "SPACING 1 1 1\n"
      << "POINT_DATA " << N << "\n"
      << "SCALARS s float 1\n"
      << "LOOKUP_TABLE default\n";
    for(size_t i=0;i<N;++i) f << h[i] << "\n";
}
// ベクトル場 (fx,fy,fz) を VTK (STRUCTURED_POINTS, ASCII) としてホスト経由で書き出す
extern "C" void write_vector_vtk(const char* filename,
                                 const float* d_fx,
                                 const float* d_fy,
                                 const float* d_fz,
                                 int Nx, int Ny, int Nz){
    const size_t N = static_cast<size_t>(Nx) * static_cast<size_t>(Ny) * static_cast<size_t>(Nz);
    std::vector<float> h(3 * N);
    CUDA_CHECK(cudaMemcpy(h.data(),        d_fx, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h.data() + N,    d_fy, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h.data() + 2*N,  d_fz, N * sizeof(float), cudaMemcpyDeviceToHost));

    std::ofstream f(filename);
    if(!f) throw std::runtime_error("failed to open VTK file");
    f << "# vtk DataFile Version 3.0\nvector\nASCII\n"
      << "DATASET STRUCTURED_POINTS\n"
      << "DIMENSIONS " << Nx << " " << Ny << " " << Nz << "\n"
      << "ORIGIN 0 0 0\n"
      << "SPACING 1 1 1\n"
      << "POINT_DATA " << N << "\n"
      << "VECTORS vec float\n";
    for(size_t i=0;i<N;++i){
        f << h[i] << " " << h[i + N] << " " << h[i + 2*N] << "\n";
    }
}
