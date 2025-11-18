#include "softmax_avx.h"
#include "../common/avx_utils.h"
#include "../common/threading.h"
#include <cmath>
#include <algorithm>

// Scalar implementation
void softmax_forward_scalar(
    const float* input, float* output,
    int64_t N, int64_t seq_len
) {
    for (int64_t i = 0; i < N; ++i) {
        // Find max
        float max_val = input[i * seq_len];
        for (int64_t j = 1; j < seq_len; ++j) {
            max_val = std::max(max_val, input[i * seq_len + j]);
        }
        
        // Compute exp and sum
        float sum = 0.0f;
        for (int64_t j = 0; j < seq_len; ++j) {
            float val = std::exp(input[i * seq_len + j] - max_val);
            output[i * seq_len + j] = val;
            sum += val;
        }
        
        // Normalize
        float inv_sum = 1.0f / sum;
        for (int64_t j = 0; j < seq_len; ++j) {
            output[i * seq_len + j] *= inv_sum;
        }
    }
}

// AVX implementation
void softmax_forward_avx(
    const float* input, float* output,
    int64_t N, int64_t seq_len
) {
    for (int64_t i = 0; i < N; ++i) {
        const float* row_input = &input[i * seq_len];
        float* row_output = &output[i * seq_len];
        
        // Find max using AVX
        __m256 max_vec = _mm256_set1_ps(-1e30f);
        int64_t j = 0;
        for (; j + 8 <= seq_len; j += 8) {
            __m256 x = loadu_8_float(&row_input[j]);
            max_vec = _mm256_max_ps(max_vec, x);
        }
        float max_val = horizontal_max_8(max_vec);
        for (; j < seq_len; ++j) {
            max_val = std::max(max_val, row_input[j]);
        }
        __m256 max_vec_broadcast = _mm256_set1_ps(max_val);
        
        // Compute exp and sum
        __m256 sum_vec = _mm256_setzero_ps();
        j = 0;
        for (; j + 8 <= seq_len; j += 8) {
            __m256 x = loadu_8_float(&row_input[j]);
            __m256 x_sub_max = _mm256_sub_ps(x, max_vec_broadcast);
            // Approximate exp using Taylor series or lookup (simplified)
            // For now, use scalar exp for accuracy
            float vals[8];
            _mm256_storeu_ps(vals, x_sub_max);
            float exps[8];
            for (int k = 0; k < 8; ++k) {
                exps[k] = std::exp(vals[k]);
            }
            __m256 exp_vec = loadu_8_float(exps);
            sum_vec = _mm256_add_ps(sum_vec, exp_vec);
            storeu_8_float(&row_output[j], exp_vec);
        }
        float sum = horizontal_sum_8(sum_vec);
        for (; j < seq_len; ++j) {
            float val = std::exp(row_input[j] - max_val);
            row_output[j] = val;
            sum += val;
        }
        
        // Normalize
        __m256 inv_sum_vec = _mm256_set1_ps(1.0f / sum);
        j = 0;
        for (; j + 8 <= seq_len; j += 8) {
            __m256 out = loadu_8_float(&row_output[j]);
            out = _mm256_mul_ps(out, inv_sum_vec);
            storeu_8_float(&row_output[j], out);
        }
        for (; j < seq_len; ++j) {
            row_output[j] /= sum;
        }
    }
}

// AVX parallel implementation
void softmax_forward_avx_parallel(
    const float* input, float* output,
    int64_t N, int64_t seq_len,
    int num_threads
) {
    if (num_threads > 0) {
        set_num_threads(num_threads);
    }
    
    #pragma omp parallel for
    for (int64_t i = 0; i < N; ++i) {
        const float* row_input = &input[i * seq_len];
        float* row_output = &output[i * seq_len];
        
        // Find max
        __m256 max_vec = _mm256_set1_ps(-1e30f);
        int64_t j = 0;
        for (; j + 8 <= seq_len; j += 8) {
            __m256 x = loadu_8_float(&row_input[j]);
            max_vec = _mm256_max_ps(max_vec, x);
        }
        float max_val = horizontal_max_8(max_vec);
        for (; j < seq_len; ++j) {
            max_val = std::max(max_val, row_input[j]);
        }
        __m256 max_vec_broadcast = _mm256_set1_ps(max_val);
        
        // Compute exp and sum
        __m256 sum_vec = _mm256_setzero_ps();
        j = 0;
        for (; j + 8 <= seq_len; j += 8) {
            __m256 x = loadu_8_float(&row_input[j]);
            __m256 x_sub_max = _mm256_sub_ps(x, max_vec_broadcast);
            float vals[8];
            _mm256_storeu_ps(vals, x_sub_max);
            float exps[8];
            for (int k = 0; k < 8; ++k) {
                exps[k] = std::exp(vals[k]);
            }
            __m256 exp_vec = loadu_8_float(exps);
            sum_vec = _mm256_add_ps(sum_vec, exp_vec);
            storeu_8_float(&row_output[j], exp_vec);
        }
        float sum = horizontal_sum_8(sum_vec);
        for (; j < seq_len; ++j) {
            float val = std::exp(row_input[j] - max_val);
            row_output[j] = val;
            sum += val;
        }
        
        // Normalize
        __m256 inv_sum_vec = _mm256_set1_ps(1.0f / sum);
        j = 0;
        for (; j + 8 <= seq_len; j += 8) {
            __m256 out = loadu_8_float(&row_output[j]);
            out = _mm256_mul_ps(out, inv_sum_vec);
            storeu_8_float(&row_output[j], out);
        }
        for (; j < seq_len; ++j) {
            row_output[j] /= sum;
        }
    }
}

void softmax_forward(
    const float* input, float* output,
    int64_t N, int64_t seq_len,
    bool use_parallel
) {
    if (has_avx2()) {
        if (use_parallel) {
            softmax_forward_avx_parallel(input, output, N, seq_len, 0);
        } else {
            softmax_forward_avx(input, output, N, seq_len);
        }
    } else {
        softmax_forward_scalar(input, output, N, seq_len);
    }
}

