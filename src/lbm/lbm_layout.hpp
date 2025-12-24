#pragma once

// AoS/SoA いずれのメモリ配置でも (q, i) からフラット配列のインデックスを返す
inline __host__ __device__ int fIndex(int q, int i, int N){
#ifdef LBM_LAYOUT_AOS
    return i*19 + q;    // AoS: 各セルに 19 方向が連続
#else
    return q*N + i;     // SoA: 各方向の配列が N 長
#endif
}
