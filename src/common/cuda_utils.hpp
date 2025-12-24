#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <sstream>

// CUDA 呼び出しが失敗した際に式名と呼び出し位置を含む例外を投げるユーティリティ。
// 成功時は何もしないので、CUDA_CHECK(...) マクロ経由で包んで使う。
inline void cudaCheckImpl(cudaError_t e, const char* expr, const char* file, int line){
    if(e != cudaSuccess){
        std::ostringstream oss;
        oss << "CUDA error: " << cudaGetErrorString(e) << " for " << expr
            << " at " << file << ":" << line;
        throw std::runtime_error(oss.str());
    }
}
// 呼び出し側で CUDA_CHECK(cudaMalloc(...)) のように使うワンライナーマクロ
#define CUDA_CHECK(x) cudaCheckImpl((x), #x, __FILE__, __LINE__)
