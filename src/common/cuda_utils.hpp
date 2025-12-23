
#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <sstream>
// summary: cudaCheckImpl の処理を行う
// param e: 入力パラメータ
// param expr: 入力パラメータ
// param file: 入力パラメータ
// param line: 入力パラメータ
// return: なし
inline void cudaCheckImpl(cudaError_t e, const char* expr, const char* file, int line){
    if(e != cudaSuccess){
        std::ostringstream oss;
        oss << "CUDA error: " << cudaGetErrorString(e) << " for " << expr
            << " at " << file << ":" << line;
        throw std::runtime_error(oss.str());
    }
}
#define CUDA_CHECK(x) cudaCheckImpl((x), #x, __FILE__, __LINE__)
