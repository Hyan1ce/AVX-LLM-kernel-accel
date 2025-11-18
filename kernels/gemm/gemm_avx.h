#ifndef GEMM_AVX_H
#define GEMM_AVX_H

#include <cstdint>

// GEMM: C = alpha * A * B + beta * C
// A: [M, K], B: [K, N], C: [M, N]

// Scalar baseline implementation
void gemm_scalar(
    const float* A, const float* B, float* C,
    int64_t M, int64_t N, int64_t K,
    float alpha = 1.0f, float beta = 0.0f
);

// AVX-optimized implementation (single-threaded)
void gemm_avx(
    const float* A, const float* B, float* C,
    int64_t M, int64_t N, int64_t K,
    float alpha = 1.0f, float beta = 0.0f
);

// AVX-optimized implementation (multi-threaded with OpenMP)
void gemm_avx_parallel(
    const float* A, const float* B, float* C,
    int64_t M, int64_t N, int64_t K,
    float alpha = 1.0f, float beta = 0.0f,
    int num_threads = 0
);

// Main entry point (auto-selects best implementation)
void gemm(
    const float* A, const float* B, float* C,
    int64_t M, int64_t N, int64_t K,
    float alpha = 1.0f, float beta = 0.0f,
    bool use_parallel = true
);

#endif // GEMM_AVX_H

