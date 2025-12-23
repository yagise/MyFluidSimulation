#pragma once
// summary: fIndex の処理を行う
// param q: 入力パラメータ
// param i: 入力パラメータ
// param N: 入力パラメータ
// return: 戻り値
inline __host__ __device__ int fIndex(int q, int i, int N){
#ifdef LBM_LAYOUT_AOS
    return i*19 + q;    // AoS: 各セルに 19 方向が連続
#else
    return q*N + i;     // SoA: 各方向の配列が N 長
#endif
}
