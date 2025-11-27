#include "gemm_avx.h"
#include "../common/avx_utils.h"
#include "../common/threading.h"
#include <cstring>
#include <algorithm>
#include <omp.h>

// Scalar baseline
void gemm_scalar(
    const float* A, const float* B, float* C,
    int64_t M, int64_t N, int64_t K,
    float alpha, float beta
) {
    // Apply beta scaling first
    if (beta == 0.0f) {
        std::memset(C, 0, M * N * sizeof(float));
    } else if (beta != 1.0f) {
        for (int64_t i = 0; i < M * N; ++i) {
            C[i] *= beta;
        }
    }
    
    // C = alpha * A * B + C
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] += alpha * sum;
        }
    }
}

// Pack a panel of B (KC x NR) into contiguous memory for better cache utilization
static inline void pack_B_panel(
    const float* B, float* B_packed,
    int64_t K, int64_t N, int64_t j_start, int64_t nr
) {
    for (int64_t k = 0; k < K; ++k) {
        for (int64_t j = 0; j < nr; ++j) {
            B_packed[k * nr + j] = B[k * N + j_start + j];
        }
    }
}

// Optimized micro-kernel with packed B: compute 4 rows x 8 cols, accumulate to C
static inline void gemm_micro_kernel_4x8_packed_accum(
    const float* A, const float* B_packed, float* C,
    int64_t K, int64_t lda, int64_t ldc, float alpha
) {
    __m256 c0 = _mm256_setzero_ps();
    __m256 c1 = _mm256_setzero_ps();
    __m256 c2 = _mm256_setzero_ps();
    __m256 c3 = _mm256_setzero_ps();
    
    for (int64_t k = 0; k < K; ++k) {
        __m256 b = loadu_8_float(&B_packed[k * 8]);
        
        __m256 a0 = _mm256_set1_ps(A[0 * lda + k]);
        __m256 a1 = _mm256_set1_ps(A[1 * lda + k]);
        __m256 a2 = _mm256_set1_ps(A[2 * lda + k]);
        __m256 a3 = _mm256_set1_ps(A[3 * lda + k]);
        
        c0 = _mm256_fmadd_ps(a0, b, c0);
        c1 = _mm256_fmadd_ps(a1, b, c1);
        c2 = _mm256_fmadd_ps(a2, b, c2);
        c3 = _mm256_fmadd_ps(a3, b, c3);
    }
    
    // Apply alpha and accumulate to C
    __m256 alpha_vec = _mm256_set1_ps(alpha);
    __m256 c_val;
    
    c_val = loadu_8_float(&C[0 * ldc]);
    c_val = _mm256_fmadd_ps(alpha_vec, c0, c_val);
    storeu_8_float(&C[0 * ldc], c_val);
    
    c_val = loadu_8_float(&C[1 * ldc]);
    c_val = _mm256_fmadd_ps(alpha_vec, c1, c_val);
    storeu_8_float(&C[1 * ldc], c_val);
    
    c_val = loadu_8_float(&C[2 * ldc]);
    c_val = _mm256_fmadd_ps(alpha_vec, c2, c_val);
    storeu_8_float(&C[2 * ldc], c_val);
    
    c_val = loadu_8_float(&C[3 * ldc]);
    c_val = _mm256_fmadd_ps(alpha_vec, c3, c_val);
    storeu_8_float(&C[3 * ldc], c_val);
}

// AVX-optimized single-threaded GEMM with cache blocking and B-packing
void gemm_avx(
    const float* A, const float* B, float* C,
    int64_t M, int64_t N, int64_t K,
    float alpha, float beta
) {
    // Special case: alpha == 0, just scale C by beta
    if (alpha == 0.0f) {
        if (beta == 0.0f) {
            std::memset(C, 0, M * N * sizeof(float));
        } else if (beta != 1.0f) {
            __m256 beta_vec = _mm256_set1_ps(beta);
            int64_t i = 0;
            for (; i + 8 <= M * N; i += 8) {
                __m256 c = loadu_8_float(&C[i]);
                c = _mm256_mul_ps(c, beta_vec);
                storeu_8_float(&C[i], c);
            }
            for (; i < M * N; ++i) {
                C[i] *= beta;
            }
        }
        return;
    }
    
    // Cache blocking parameters
    const int64_t MC = 256;
    const int64_t KC = 512;
    const int64_t NC = 4096;
    const int64_t MR = 4;
    const int64_t NR = 8;
    
    // Allocate buffer for packing B
    float* B_packed = new float[KC * NR];
    
    bool first_k_block = true;
    
    /* 
        这里使用了缓存分块，所以循环展开了很多层
        不使用的话会导致很多Cache Miss，性能可能会降低很多；下面并行版本同理
    */
    for (int64_t pc = 0; pc < K; pc += KC) {
        int64_t kc = std::min(KC, K - pc);
        
        for (int64_t jc = 0; jc < N; jc += NC) {
            int64_t nc = std::min(NC, N - jc);
            
            for (int64_t ic = 0; ic < M; ic += MC) {
                int64_t mc = std::min(MC, M - ic);
                
                // Process micro-tiles
                for (int64_t jr = 0; jr < nc; jr += NR) {
                    int64_t nr = std::min(NR, nc - jr);
                    
                    // Pack B panel if full width
                    if (nr == NR) {
                        pack_B_panel(&B[pc * N], B_packed, kc, N, jc + jr, NR);
                    }
                    
                    for (int64_t ir = 0; ir < mc; ir += MR) {
                        int64_t mr = std::min(MR, mc - ir);
                        
                        int64_t i_base = ic + ir;
                        int64_t j_base = jc + jr;
                        
                        if (mr == MR && nr == NR) {
                            // Handle beta scaling only on first K-block
                            if (first_k_block) {
                                if (beta == 0.0f) {
                                    for (int64_t i = 0; i < MR; ++i) {
                                        storeu_8_float(&C[(i_base + i) * N + j_base], 
                                                      _mm256_setzero_ps());
                                    }
                                } else if (beta != 1.0f) {
                                    __m256 beta_vec = _mm256_set1_ps(beta);
                                    for (int64_t i = 0; i < MR; ++i) {
                                        __m256 c = loadu_8_float(&C[(i_base + i) * N + j_base]);
                                        c = _mm256_mul_ps(c, beta_vec);
                                        storeu_8_float(&C[(i_base + i) * N + j_base], c);
                                    }
                                }
                            }
                            
                            // Use packed micro-kernel
                            gemm_micro_kernel_4x8_packed_accum(
                                &A[i_base * K + pc], B_packed, 
                                &C[i_base * N + j_base],
                                kc, K, N, alpha
                            );
                        } else {
                            // Boundary case: scalar code
                            for (int64_t i = 0; i < mr; ++i) {
                                for (int64_t j = 0; j < nr; ++j) {
                                    float sum = 0.0f;
                                    for (int64_t k = 0; k < kc; ++k) {
                                        sum += A[(i_base + i) * K + pc + k] * 
                                               B[(pc + k) * N + j_base + j];
                                    }
                                    int64_t idx = (i_base + i) * N + j_base + j;
                                    if (first_k_block) {
                                        C[idx] = alpha * sum + beta * C[idx];
                                    } else {
                                        C[idx] += alpha * sum;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        first_k_block = false;
    }
    
    delete[] B_packed;
}

// AVX-optimized parallel GEMM with cache tiling and B-packing
void gemm_avx_parallel(
    const float* A, const float* B, float* C,
    int64_t M, int64_t N, int64_t K,
    float alpha, float beta,
    int num_threads
) {
    if (num_threads > 0) {
        set_num_threads(num_threads);
    }
    
    // Special case: alpha == 0, just scale C by beta
    if (alpha == 0.0f) {
        if (beta == 0.0f) {
            std::memset(C, 0, M * N * sizeof(float));
        } else if (beta != 1.0f) {
            __m256 beta_vec = _mm256_set1_ps(beta);
            #pragma omp parallel for
            for (int64_t idx = 0; idx < M * N; idx += 8) {
                int64_t remaining = std::min((int64_t)8, M * N - idx);
                if (remaining == 8) {
                    __m256 c = loadu_8_float(&C[idx]);
                    c = _mm256_mul_ps(c, beta_vec);
                    storeu_8_float(&C[idx], c);
                } else {
                    for (int64_t i = 0; i < remaining; ++i) {
                        C[idx + i] *= beta;
                    }
                }
            }
        }
        return;
    }
    
    // For small/medium matrices, use serial version to avoid OpenMP overhead
    // Serial is faster when the computation is too small to amortize thread overhead
    int64_t total_ops = M * N * K;
    if (total_ops < 512 * 1024 * 1024) {  // < 512M operations (between 512^3 and 1024^3)
        gemm_avx(A, B, C, M, N, K, alpha, beta);
        return;
    }
    
    // Cache blocking parameters
    const int64_t MC = 256;
    const int64_t KC = 512;
    const int64_t NC = 4096;
    const int64_t MR = 4;
    const int64_t NR = 8;
    
    // Allocate shared B-packing buffer (one per potential thread)
    const int max_threads = omp_get_max_threads();
    float** B_packed_buffers = new float*[max_threads];
    for (int t = 0; t < max_threads; ++t) {
        B_packed_buffers[t] = new float[KC * NR];
    }
    
    bool first_k_block = true;
    
    /* 
        这里使用了缓存分块，所以循环展开了很多层
        不使用的话会导致很多Cache Miss，性能可能会降低很多
    */

    for (int64_t pc = 0; pc < K; pc += KC) {
        int64_t kc = std::min(KC, K - pc);
        
        for (int64_t jc = 0; jc < N; jc += NC) {
            int64_t nc = std::min(NC, N - jc);
            
            // Parallelize over M blocks only for better granularity
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                float* B_packed = B_packed_buffers[tid];
                
                #pragma omp for schedule(dynamic)
                for (int64_t ic = 0; ic < M; ic += MC) {
                    int64_t mc = std::min(MC, M - ic);
                    
                    for (int64_t jr = 0; jr < nc; jr += NR) {
                        int64_t nr = std::min(NR, nc - jr);
                        
                        // Pack B panel once per jr (within this M-block)
                        if (nr == NR) {
                            pack_B_panel(&B[pc * N], B_packed, kc, N, jc + jr, NR);
                        }
                        
                        for (int64_t ir = 0; ir < mc; ir += MR) {
                            int64_t mr = std::min(MR, mc - ir);
                            
                            int64_t i_base = ic + ir;
                            int64_t j_base = jc + jr;
                            
                            if (mr == MR && nr == NR) {
                                // Handle beta scaling only on first K-block
                                if (first_k_block) {
                                    if (beta == 0.0f) {
                                        for (int64_t i = 0; i < MR; ++i) {
                                            storeu_8_float(&C[(i_base + i) * N + j_base], 
                                                          _mm256_setzero_ps());
                                        }
                                    } else if (beta != 1.0f) {
                                        __m256 beta_vec = _mm256_set1_ps(beta);
                                        for (int64_t i = 0; i < MR; ++i) {
                                            __m256 c = loadu_8_float(&C[(i_base + i) * N + j_base]);
                                            c = _mm256_mul_ps(c, beta_vec);
                                            storeu_8_float(&C[(i_base + i) * N + j_base], c);
                                        }
                                    }
                                }
                                
                                // Use packed micro-kernel
                                gemm_micro_kernel_4x8_packed_accum(
                                    &A[i_base * K + pc], B_packed, 
                                    &C[i_base * N + j_base],
                                    kc, K, N, alpha
                                );
                            } else {
                                // Boundary case: scalar code
                                for (int64_t i = 0; i < mr; ++i) {
                                    for (int64_t j = 0; j < nr; ++j) {
                                        float sum = 0.0f;
                                        for (int64_t k = 0; k < kc; ++k) {
                                            sum += A[(i_base + i) * K + pc + k] * 
                                                   B[(pc + k) * N + j_base + j];
                                        }
                                        int64_t idx = (i_base + i) * N + j_base + j;
                                        if (first_k_block) {
                                            C[idx] = alpha * sum + beta * C[idx];
                                        } else {
                                            C[idx] += alpha * sum;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        first_k_block = false;
    }
    
    // Clean up B-packing buffers
    for (int t = 0; t < max_threads; ++t) {
        delete[] B_packed_buffers[t];
    }
    delete[] B_packed_buffers;
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


