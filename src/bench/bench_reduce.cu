#include <cuda_runtime.h>

// ベンチマーク用プレースホルダー。
// 実際のリダクション実装を差し込むまで、シンボルを維持するためのダミーカーネル。
__global__ void bench_reduce_stub(const float*, float*, int) {}
