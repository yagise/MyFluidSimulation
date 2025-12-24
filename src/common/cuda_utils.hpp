
#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <sstream>
inline void cudaCheckImpl(cudaError_t e, const char* expr, const char* file, int line){
    if(e != cudaSuccess){
        std::ostringstream oss;
        oss << "CUDA error: " << cudaGetErrorString(e) << " for " << expr
            << " at " << file << ":" << line;
        throw std::runtime_error(oss.str());
    }
}
#define CUDA_CHECK(x) cudaCheckImpl((x), #x, __FILE__, __LINE__)
