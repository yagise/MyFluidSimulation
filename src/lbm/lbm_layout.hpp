#pragma once

// 分布関数配列の 1 次元インデックスを返すヘルパー。
// q: 方向インデックス(0-18)、i: セル番号、N: 総セル数を受け取る。
// AoS と SoA のどちらの並びで確保しているかをマクロで切り替える。
inline __host__ __device__ int fIndex(int q, int i, int N){
#ifdef LBM_LAYOUT_AOS
    return i*19 + q;    // AoS: 各セルに 19 方向が連続
#else
    return q*N + i;     // SoA: 各方向の配列が N 長
#endif
}
