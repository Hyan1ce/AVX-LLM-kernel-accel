#include "gemm_avx.h"
#include "../common/avx_utils.h"
#include "../common/threading.h"
#include <cstring>
#include <algorithm>

// Scalar baseline
void gemm_scalar(
    const float* A, const float* B, float* C,
    int64_t M, int64_t N, int64_t K,
    float alpha, float beta
) {
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = alpha * sum + beta * C[i * N + j];
        }
    }
}

// AVX-optimized single-threaded GEMM
void gemm_avx(
    const float* A, const float* B, float* C,
    int64_t M, int64_t N, int64_t K,
    float alpha, float beta
) {
    const int64_t vec_size = 8;
    
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; j += vec_size) {
            int64_t j_end = std::min(j + vec_size, N);
            int64_t remaining = j_end - j;
            
            __m256 c_vec = _mm256_setzero_ps();
            
            for (int64_t k = 0; k < K; ++k) {
                __m256 a_broadcast = _mm256_set1_ps(A[i * K + k]);
                if (remaining == vec_size) {
                    __m256 b_vec = loadu_8_float(&B[k * N + j]);
                    c_vec = _mm256_fmadd_ps(a_broadcast, b_vec, c_vec);
                } else {
                    // Handle partial vector
                    float b_vals[8] = {0};
                    for (int64_t r = 0; r < remaining; ++r) {
                        b_vals[r] = B[k * N + j + r];
                    }
                    __m256 b_vec = loadu_8_float(b_vals);
                    c_vec = _mm256_fmadd_ps(a_broadcast, b_vec, c_vec);
                }
            }
            
            // Store result
            if (remaining == vec_size) {
                __m256 c_old = loadu_8_float(&C[i * N + j]);
                __m256 c_new = _mm256_fmadd_ps(_mm256_set1_ps(alpha), c_vec,
                                                _mm256_mul_ps(_mm256_set1_ps(beta), c_old));
                storeu_8_float(&C[i * N + j], c_new);
            } else {
                // Handle partial vector
                float c_vals[8], c_old_vals[8];
                _mm256_storeu_ps(c_vals, c_vec);
                for (int64_t r = 0; r < remaining; ++r) {
                    c_old_vals[r] = C[i * N + j + r];
                    C[i * N + j + r] = alpha * c_vals[r] + beta * c_old_vals[r];
                }
            }
        }
    }
}

// AVX-optimized parallel GEMM
void gemm_avx_parallel(
    const float* A, const float* B, float* C,
    int64_t M, int64_t N, int64_t K,
    float alpha, float beta,
    int num_threads
) {
    if (num_threads > 0) {
        set_num_threads(num_threads);
    }
    
    #pragma omp parallel for
    for (int64_t i = 0; i < M; ++i) {
        const int64_t vec_size = 8;
        
        for (int64_t j = 0; j < N; j += vec_size) {
            int64_t j_end = std::min(j + vec_size, N);
            int64_t remaining = j_end - j;
            
            __m256 c_vec = _mm256_setzero_ps();
            
            for (int64_t k = 0; k < K; ++k) {
                __m256 a_broadcast = _mm256_set1_ps(A[i * K + k]);
                if (remaining == vec_size) {
                    __m256 b_vec = loadu_8_float(&B[k * N + j]);
                    c_vec = _mm256_fmadd_ps(a_broadcast, b_vec, c_vec);
                } else {
                    // Handle partial vector
                    float b_vals[8] = {0};
                    for (int64_t r = 0; r < remaining; ++r) {
                        b_vals[r] = B[k * N + j + r];
                    }
                    __m256 b_vec = loadu_8_float(b_vals);
                    c_vec = _mm256_fmadd_ps(a_broadcast, b_vec, c_vec);
                }
            }
            
            // Store result
            if (remaining == vec_size) {
                __m256 c_old = loadu_8_float(&C[i * N + j]);
                __m256 c_new = _mm256_fmadd_ps(_mm256_set1_ps(alpha), c_vec,
                                                _mm256_mul_ps(_mm256_set1_ps(beta), c_old));
                storeu_8_float(&C[i * N + j], c_new);
            } else {
                // Handle partial vector
                float c_vals[8], c_old_vals[8];
                _mm256_storeu_ps(c_vals, c_vec);
                for (int64_t r = 0; r < remaining; ++r) {
                    c_old_vals[r] = C[i * N + j + r];
                    C[i * N + j + r] = alpha * c_vals[r] + beta * c_old_vals[r];
                }
            }
        }
    }
}

// Main entry point
void gemm(
    const float* A, const float* B, float* C,
    int64_t M, int64_t N, int64_t K,
    float alpha, float beta,
    bool use_parallel
) {
    if (has_avx2()) {
        if (use_parallel) {
            gemm_avx_parallel(A, B, C, M, N, K, alpha, beta, 0);
        } else {
            gemm_avx(A, B, C, M, N, K, alpha, beta);
        }
    } else {
        gemm_scalar(A, B, C, M, N, K, alpha, beta);
    }
}

