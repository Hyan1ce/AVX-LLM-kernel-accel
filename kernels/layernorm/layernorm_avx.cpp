#include "layernorm_avx.h"
#include "../common/avx_utils.h"
#include "../common/threading.h"
#include <cmath>
#include <cstring>

// Scalar forward - Two-pass algorithm for better numerical stability
void layernorm_forward_scalar(
    const float* input, const float* gamma, const float* beta,
    float* output, float* mean, float* var,
    int64_t N, int64_t hidden_size, float eps
) {
    for (int64_t i = 0; i < N; ++i) {
        // Pass 1: Mean
        float sum = 0.0f;
        for (int64_t j = 0; j < hidden_size; ++j) {
            sum += input[i * hidden_size + j];
        }
        float m = sum / hidden_size;
        mean[i] = m;
        
        // Pass 2: Variance and Normalize
        float sum_sq_diff = 0.0f;
        for (int64_t j = 0; j < hidden_size; ++j) {
            float diff = input[i * hidden_size + j] - m;
            sum_sq_diff += diff * diff;
        }
        float v = sum_sq_diff / hidden_size;
        var[i] = v;
        
        float inv_std = 1.0f / std::sqrt(v + eps);
        
        for (int64_t j = 0; j < hidden_size; ++j) {
            float normalized = (input[i * hidden_size + j] - m) * inv_std;
            output[i * hidden_size + j] = normalized * gamma[j] + beta[j];
        }
    }
}

// AVX forward - Two-pass algorithm
void layernorm_forward_avx(
    const float* input, const float* gamma, const float* beta,
    float* output, float* mean, float* var,
    int64_t N, int64_t hidden_size, float eps
) {
    for (int64_t i = 0; i < N; ++i) {
        // Pass 1: Mean
        __m256 sum_vec = _mm256_setzero_ps();
        int64_t j = 0;
        for (; j + 8 <= hidden_size; j += 8) {
            __m256 x = loadu_8_float(&input[i * hidden_size + j]);
            sum_vec = _mm256_add_ps(sum_vec, x);
        }
        float sum = horizontal_sum_8(sum_vec);
        for (; j < hidden_size; ++j) {
            sum += input[i * hidden_size + j];
        }
        float m = sum / hidden_size;
        mean[i] = m;
        
        // Pass 2: Variance
        __m256 m_vec = _mm256_set1_ps(m);
        __m256 sum_sq_diff_vec = _mm256_setzero_ps();
        
        j = 0;
        for (; j + 8 <= hidden_size; j += 8) {
            __m256 x = loadu_8_float(&input[i * hidden_size + j]);
            __m256 diff = _mm256_sub_ps(x, m_vec);
            sum_sq_diff_vec = _mm256_fmadd_ps(diff, diff, sum_sq_diff_vec);
        }
        float sum_sq_diff = horizontal_sum_8(sum_sq_diff_vec);
        for (; j < hidden_size; ++j) {
            float diff = input[i * hidden_size + j] - m;
            sum_sq_diff += diff * diff;
        }
        float v = sum_sq_diff / hidden_size;
        var[i] = v;
        
        // Normalize and Output
        float inv_std = 1.0f / std::sqrt(v + eps);
        __m256 inv_std_vec = _mm256_set1_ps(inv_std);
        
        j = 0;
        for (; j + 8 <= hidden_size; j += 8) {
            __m256 x = loadu_8_float(&input[i * hidden_size + j]);
            __m256 g = loadu_8_float(&gamma[j]);
            __m256 b = loadu_8_float(&beta[j]);
            
            __m256 normalized = _mm256_mul_ps(_mm256_sub_ps(x, m_vec), inv_std_vec);
            __m256 out = _mm256_fmadd_ps(normalized, g, b);
            storeu_8_float(&output[i * hidden_size + j], out);
        }
        
        for (; j < hidden_size; ++j) {
            float normalized = (input[i * hidden_size + j] - m) * inv_std;
            output[i * hidden_size + j] = normalized * gamma[j] + beta[j];
        }
    }
}

// AVX parallel forward - Two-pass algorithm
void layernorm_forward_avx_parallel(
    const float* input, const float* gamma, const float* beta,
    float* output, float* mean, float* var,
    int64_t N, int64_t hidden_size, float eps,
    int num_threads
) {
    if (num_threads > 0) {
        set_num_threads(num_threads);
    }
    
    #pragma omp parallel for
    for (int64_t i = 0; i < N; ++i) {
        // Pass 1: Mean
        __m256 sum_vec = _mm256_setzero_ps();
        int64_t j = 0;
        for (; j + 8 <= hidden_size; j += 8) {
            __m256 x = loadu_8_float(&input[i * hidden_size + j]);
            sum_vec = _mm256_add_ps(sum_vec, x);
        }
        float sum = horizontal_sum_8(sum_vec);
        for (; j < hidden_size; ++j) {
            sum += input[i * hidden_size + j];
        }
        float m = sum / hidden_size;
        mean[i] = m;
        
        // Pass 2: Variance
        __m256 m_vec = _mm256_set1_ps(m);
        __m256 sum_sq_diff_vec = _mm256_setzero_ps();
        
        j = 0;
        for (; j + 8 <= hidden_size; j += 8) {
            __m256 x = loadu_8_float(&input[i * hidden_size + j]);
            __m256 diff = _mm256_sub_ps(x, m_vec);
            sum_sq_diff_vec = _mm256_fmadd_ps(diff, diff, sum_sq_diff_vec);
        }
        float sum_sq_diff = horizontal_sum_8(sum_sq_diff_vec);
        for (; j < hidden_size; ++j) {
            float diff = input[i * hidden_size + j] - m;
            sum_sq_diff += diff * diff;
        }
        float v = sum_sq_diff / hidden_size;
        var[i] = v;
        
        // Normalize and Output
        float inv_std = 1.0f / std::sqrt(v + eps);
        __m256 inv_std_vec = _mm256_set1_ps(inv_std);
        
        j = 0;
        for (; j + 8 <= hidden_size; j += 8) {
            __m256 x = loadu_8_float(&input[i * hidden_size + j]);
            __m256 g = loadu_8_float(&gamma[j]);
            __m256 b = loadu_8_float(&beta[j]);
            
            __m256 normalized = _mm256_mul_ps(_mm256_sub_ps(x, m_vec), inv_std_vec);
            __m256 out = _mm256_fmadd_ps(normalized, g, b);
            storeu_8_float(&output[i * hidden_size + j], out);
        }
        
        for (; j < hidden_size; ++j) {
            float normalized = (input[i * hidden_size + j] - m) * inv_std;
            output[i * hidden_size + j] = normalized * gamma[j] + beta[j];
        }
    }
}

void layernorm_forward(
    const float* input, const float* gamma, const float* beta,
    float* output, float* mean, float* var,
    int64_t N, int64_t hidden_size, float eps,
    bool use_parallel
) {
    // 为了数值精度，与PyTorch保持严格一致，这里不再依赖手写AVX归约，
    // 而是使用标量实现 + OpenMP 并行（编译器仍会在 -mavx2 下自动向量化）。
    if (!use_parallel) {
        layernorm_forward_scalar(input, gamma, beta, output, mean, var,
                                 N, hidden_size, eps);
        return;
    }

    // 自适应选择：小任务时使用串行避免OpenMP开销
    // 每个任务处理一行，工作量 = N * hidden_size
    // 阈值：当总工作量 < 64K元素时使用串行（避免线程创建开销）
    int64_t total_elements = N * hidden_size;
    if (total_elements < 64 * 1024) {
        layernorm_forward_scalar(input, gamma, beta, output, mean, var,
                                 N, hidden_size, eps);
        return;
    }

    // 并行版本：按 batch 维度拆分，不共享中间状态，线程安全
    #pragma omp parallel for
    for (int64_t i = 0; i < N; ++i) {
        const int64_t base = i * hidden_size;

        // Pass 1: Mean
        float sum = 0.0f;
        for (int64_t j = 0; j < hidden_size; ++j) {
            sum += input[base + j];
        }
        float m = sum / hidden_size;
        mean[i] = m;

        // Pass 2: Variance
        float sum_sq_diff = 0.0f;
        for (int64_t j = 0; j < hidden_size; ++j) {
            float diff = input[base + j] - m;
            sum_sq_diff += diff * diff;
        }
        float v = sum_sq_diff / hidden_size;
        var[i] = v;

        float inv_std = 1.0f / std::sqrt(v + eps);

        // Normalize + affine
        for (int64_t j = 0; j < hidden_size; ++j) {
            float normalized = (input[base + j] - m) * inv_std;
            output[base + j] = normalized * gamma[j] + beta[j];
        }
    }
}

// Scalar backward - Fixed logic for gradients
void layernorm_backward_scalar(
    const float* grad_output, const float* input,
    const float* gamma, const float* mean, const float* var,
    float* grad_input, float* grad_gamma, float* grad_beta,
    int64_t N, int64_t hidden_size, float eps
) {
    // Initialize gradients
    if (grad_gamma) {
        std::memset(grad_gamma, 0, hidden_size * sizeof(float));
    }
    if (grad_beta) {
        std::memset(grad_beta, 0, hidden_size * sizeof(float));
    }
    
    for (int64_t i = 0; i < N; ++i) {
        float inv_std = 1.0f / std::sqrt(var[i] + eps);
        float inv_hidden_size = 1.0f / hidden_size;
        
        float sum_dy = 0.0f;
        float sum_dy_xhat = 0.0f;
        
        // Pass 1: Compute sums
        for (int64_t j = 0; j < hidden_size; ++j) {
            float normalized = (input[i * hidden_size + j] - mean[i]) * inv_std;
            float dy = grad_output[i * hidden_size + j] * gamma[j];
            
            sum_dy += dy;
            sum_dy_xhat += dy * normalized;
            
            if (grad_gamma) {
                grad_gamma[j] += normalized * grad_output[i * hidden_size + j];
            }
            if (grad_beta) {
                grad_beta[j] += grad_output[i * hidden_size + j];
            }
        }
        
        // Pass 2: Compute grad_input
        for (int64_t j = 0; j < hidden_size; ++j) {
            float normalized = (input[i * hidden_size + j] - mean[i]) * inv_std;
            float dy = grad_output[i * hidden_size + j] * gamma[j];
            
            // dx = (1/sigma) * (dy - mean(dy) - x_hat * mean(dy * x_hat))
            float grad_input_val = inv_std * (dy - sum_dy * inv_hidden_size - normalized * sum_dy_xhat * inv_hidden_size);
            grad_input[i * hidden_size + j] = grad_input_val;
        }
    }
}

// AVX backward - Fixed logic for gradients
void layernorm_backward_avx(
    const float* grad_output, const float* input,
    const float* gamma, const float* mean, const float* var,
    float* grad_input, float* grad_gamma, float* grad_beta,
    int64_t N, int64_t hidden_size, float eps
) {
    if (grad_gamma) {
        std::memset(grad_gamma, 0, hidden_size * sizeof(float));
    }
    if (grad_beta) {
        std::memset(grad_beta, 0, hidden_size * sizeof(float));
    }
    
    for (int64_t i = 0; i < N; ++i) {
        float inv_std = 1.0f / std::sqrt(var[i] + eps);
        float inv_hidden_size = 1.0f / hidden_size;
        __m256 mean_vec = _mm256_set1_ps(mean[i]);
        __m256 inv_std_vec = _mm256_set1_ps(inv_std);
        
        // Pass 1: Compute sums (sum_dy and sum_dy_xhat)
        __m256 sum_dy_vec = _mm256_setzero_ps();
        __m256 sum_dy_xhat_vec = _mm256_setzero_ps();
        
        int64_t j = 0;
        for (; j + 8 <= hidden_size; j += 8) {
            __m256 x = loadu_8_float(&input[i * hidden_size + j]);
            __m256 go = loadu_8_float(&grad_output[i * hidden_size + j]);
            __m256 g = loadu_8_float(&gamma[j]);
            
            __m256 normalized = _mm256_mul_ps(_mm256_sub_ps(x, mean_vec), inv_std_vec);
            __m256 dy = _mm256_mul_ps(go, g);
            
            sum_dy_vec = _mm256_add_ps(sum_dy_vec, dy);
            sum_dy_xhat_vec = _mm256_fmadd_ps(dy, normalized, sum_dy_xhat_vec);
            
            if (grad_gamma) {
                __m256 gg = loadu_8_float(&grad_gamma[j]);
                gg = _mm256_fmadd_ps(normalized, go, gg);
                storeu_8_float(&grad_gamma[j], gg);
            }
            if (grad_beta) {
                __m256 gb = loadu_8_float(&grad_beta[j]);
                gb = _mm256_add_ps(gb, go);
                storeu_8_float(&grad_beta[j], gb);
            }
        }
        
        float sum_dy = horizontal_sum_8(sum_dy_vec);
        float sum_dy_xhat = horizontal_sum_8(sum_dy_xhat_vec);
        
        for (; j < hidden_size; ++j) {
            float normalized = (input[i * hidden_size + j] - mean[i]) * inv_std;
            float dy = grad_output[i * hidden_size + j] * gamma[j];
            
            sum_dy += dy;
            sum_dy_xhat += dy * normalized;
            
            if (grad_gamma) {
                grad_gamma[j] += normalized * grad_output[i * hidden_size + j];
            }
            if (grad_beta) {
                grad_beta[j] += grad_output[i * hidden_size + j];
            }
        }
        
        __m256 mean_dy_vec = _mm256_set1_ps(sum_dy * inv_hidden_size);
        __m256 mean_dy_xhat_vec = _mm256_set1_ps(sum_dy_xhat * inv_hidden_size);
        
        // Pass 2: Compute grad_input
        j = 0;
        for (; j + 8 <= hidden_size; j += 8) {
            __m256 x = loadu_8_float(&input[i * hidden_size + j]);
            __m256 go = loadu_8_float(&grad_output[i * hidden_size + j]);
            __m256 g = loadu_8_float(&gamma[j]);
            
            __m256 normalized = _mm256_mul_ps(_mm256_sub_ps(x, mean_vec), inv_std_vec);
            __m256 dy = _mm256_mul_ps(go, g);
            
            // grad = inv_std * (dy - mean_dy - normalized * mean_dy_xhat)
            __m256 term2 = _mm256_mul_ps(normalized, mean_dy_xhat_vec);
            __m256 inside = _mm256_sub_ps(dy, mean_dy_vec);
            inside = _mm256_sub_ps(inside, term2);
            __m256 gi = _mm256_mul_ps(inv_std_vec, inside);
            
            storeu_8_float(&grad_input[i * hidden_size + j], gi);
        }
        
        for (; j < hidden_size; ++j) {
            float normalized = (input[i * hidden_size + j] - mean[i]) * inv_std;
            float dy = grad_output[i * hidden_size + j] * gamma[j];
            
            float grad_input_val = inv_std * (dy - sum_dy * inv_hidden_size - normalized * sum_dy_xhat * inv_hidden_size);
            grad_input[i * hidden_size + j] = grad_input_val;
        }
    }
}

void layernorm_backward_avx_parallel(
    const float* grad_output, const float* input,
    const float* gamma, const float* mean, const float* var,
    float* grad_input, float* grad_gamma, float* grad_beta,
    int64_t N, int64_t hidden_size, float eps,
    int num_threads
) {
    if (num_threads > 0) {
        set_num_threads(num_threads);
    }
    
    if (grad_gamma) {
        std::memset(grad_gamma, 0, hidden_size * sizeof(float));
    }
    if (grad_beta) {
        std::memset(grad_beta, 0, hidden_size * sizeof(float));
    }
    
    #pragma omp parallel for
    for (int64_t i = 0; i < N; ++i) {
        float inv_std = 1.0f / std::sqrt(var[i] + eps);
        float inv_hidden_size = 1.0f / hidden_size;
        __m256 mean_vec = _mm256_set1_ps(mean[i]);
        __m256 inv_std_vec = _mm256_set1_ps(inv_std);
        
        // Pass 1
        __m256 sum_dy_vec = _mm256_setzero_ps();
        __m256 sum_dy_xhat_vec = _mm256_setzero_ps();
        
        int64_t j = 0;
        for (; j + 8 <= hidden_size; j += 8) {
            __m256 x = loadu_8_float(&input[i * hidden_size + j]);
            __m256 go = loadu_8_float(&grad_output[i * hidden_size + j]);
            __m256 g = loadu_8_float(&gamma[j]);
            
            __m256 normalized = _mm256_mul_ps(_mm256_sub_ps(x, mean_vec), inv_std_vec);
            __m256 dy = _mm256_mul_ps(go, g);
            
            sum_dy_vec = _mm256_add_ps(sum_dy_vec, dy);
            sum_dy_xhat_vec = _mm256_fmadd_ps(dy, normalized, sum_dy_xhat_vec);
            
            #pragma omp critical
            {
                if (grad_gamma) {
                    __m256 gg = loadu_8_float(&grad_gamma[j]);
                    gg = _mm256_fmadd_ps(normalized, go, gg);
                    storeu_8_float(&grad_gamma[j], gg);
                }
                if (grad_beta) {
                    __m256 gb = loadu_8_float(&grad_beta[j]);
                    gb = _mm256_add_ps(gb, go);
                    storeu_8_float(&grad_beta[j], gb);
                }
            }
        }
        
        float sum_dy = horizontal_sum_8(sum_dy_vec);
        float sum_dy_xhat = horizontal_sum_8(sum_dy_xhat_vec);
        
        for (; j < hidden_size; ++j) {
            float normalized = (input[i * hidden_size + j] - mean[i]) * inv_std;
            float dy = grad_output[i * hidden_size + j] * gamma[j];
            
            sum_dy += dy;
            sum_dy_xhat += dy * normalized;
            
            #pragma omp critical
            {
                if (grad_gamma) {
                    grad_gamma[j] += normalized * grad_output[i * hidden_size + j];
                }
                if (grad_beta) {
                    grad_beta[j] += grad_output[i * hidden_size + j];
                }
            }
        }
        
        __m256 mean_dy_vec = _mm256_set1_ps(sum_dy * inv_hidden_size);
        __m256 mean_dy_xhat_vec = _mm256_set1_ps(sum_dy_xhat * inv_hidden_size);
        
        // Pass 2
        j = 0;
        for (; j + 8 <= hidden_size; j += 8) {
            __m256 x = loadu_8_float(&input[i * hidden_size + j]);
            __m256 go = loadu_8_float(&grad_output[i * hidden_size + j]);
            __m256 g = loadu_8_float(&gamma[j]);
            
            __m256 normalized = _mm256_mul_ps(_mm256_sub_ps(x, mean_vec), inv_std_vec);
            __m256 dy = _mm256_mul_ps(go, g);
            
            __m256 term2 = _mm256_mul_ps(normalized, mean_dy_xhat_vec);
            __m256 inside = _mm256_sub_ps(dy, mean_dy_vec);
            inside = _mm256_sub_ps(inside, term2);
            __m256 gi = _mm256_mul_ps(inv_std_vec, inside);
            
            storeu_8_float(&grad_input[i * hidden_size + j], gi);
        }
        
        for (; j < hidden_size; ++j) {
            float normalized = (input[i * hidden_size + j] - mean[i]) * inv_std;
            float dy = grad_output[i * hidden_size + j] * gamma[j];
            
            float grad_input_val = inv_std * (dy - sum_dy * inv_hidden_size - normalized * sum_dy_xhat * inv_hidden_size);
            grad_input[i * hidden_size + j] = grad_input_val;
        }
    }
}

void layernorm_backward(
    const float* grad_output, const float* input,
    const float* gamma, const float* mean, const float* var,
    float* grad_input, float* grad_gamma, float* grad_beta,
    int64_t N, int64_t hidden_size, float eps,
    bool use_parallel
) {
    // 自适应选择：小任务时避免并行开销
    int64_t total_elements = N * hidden_size;
    bool should_parallel = use_parallel && (total_elements >= 64 * 1024);
    
    if (has_avx2()) {
        if (should_parallel) {
            layernorm_backward_avx_parallel(grad_output, input, gamma, mean, var,
                                           grad_input, grad_gamma, grad_beta,
                                           N, hidden_size, eps, 0);
        } else {
            layernorm_backward_avx(grad_output, input, gamma, mean, var,
                                 grad_input, grad_gamma, grad_beta,
                                 N, hidden_size, eps);
        }
    } else {
        layernorm_backward_scalar(grad_output, input, gamma, mean, var,
                                grad_input, grad_gamma, grad_beta,
                                N, hidden_size, eps);
    }
}
