#include "layernorm_avx.h"
#include "../common/avx_utils.h"
#include "../common/threading.h"
#include <cmath>
#include <cstring>

// Scalar forward
void layernorm_forward_scalar(
    const float* input, const float* gamma, const float* beta,
    float* output, float* mean, float* var,
    int64_t N, int64_t hidden_size, float eps
) {
    for (int64_t i = 0; i < N; ++i) {
        float sum = 0.0f;
        float sum_sq = 0.0f;
        
        for (int64_t j = 0; j < hidden_size; ++j) {
            float val = input[i * hidden_size + j];
            sum += val;
            sum_sq += val * val;
        }
        
        float m = sum / hidden_size;
        float v = (sum_sq / hidden_size) - (m * m);
        mean[i] = m;
        var[i] = v;
        
        float inv_std = 1.0f / std::sqrt(v + eps);
        
        for (int64_t j = 0; j < hidden_size; ++j) {
            float normalized = (input[i * hidden_size + j] - m) * inv_std;
            output[i * hidden_size + j] = normalized * gamma[j] + beta[j];
        }
    }
}

// AVX forward
void layernorm_forward_avx(
    const float* input, const float* gamma, const float* beta,
    float* output, float* mean, float* var,
    int64_t N, int64_t hidden_size, float eps
) {
    for (int64_t i = 0; i < N; ++i) {
        __m256 sum_vec = _mm256_setzero_ps();
        __m256 sum_sq_vec = _mm256_setzero_ps();
        
        int64_t j = 0;
        for (; j + 8 <= hidden_size; j += 8) {
            __m256 x = loadu_8_float(&input[i * hidden_size + j]);
            sum_vec = _mm256_add_ps(sum_vec, x);
            sum_sq_vec = _mm256_fmadd_ps(x, x, sum_sq_vec);
        }
        
        float sum = horizontal_sum_8(sum_vec);
        float sum_sq = horizontal_sum_8(sum_sq_vec);
        
        for (; j < hidden_size; ++j) {
            float val = input[i * hidden_size + j];
            sum += val;
            sum_sq += val * val;
        }
        
        float m = sum / hidden_size;
        float v = (sum_sq / hidden_size) - (m * m);
        mean[i] = m;
        var[i] = v;
        
        float inv_std = 1.0f / std::sqrt(v + eps);
        __m256 mean_vec = _mm256_set1_ps(m);
        __m256 inv_std_vec = _mm256_set1_ps(inv_std);
        
        j = 0;
        for (; j + 8 <= hidden_size; j += 8) {
            __m256 x = loadu_8_float(&input[i * hidden_size + j]);
            __m256 g = loadu_8_float(&gamma[j]);
            __m256 b = loadu_8_float(&beta[j]);
            
            __m256 normalized = _mm256_mul_ps(_mm256_sub_ps(x, mean_vec), inv_std_vec);
            __m256 out = _mm256_fmadd_ps(normalized, g, b);
            storeu_8_float(&output[i * hidden_size + j], out);
        }
        
        for (; j < hidden_size; ++j) {
            float normalized = (input[i * hidden_size + j] - m) * inv_std;
            output[i * hidden_size + j] = normalized * gamma[j] + beta[j];
        }
    }
}

// AVX parallel forward
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
        __m256 sum_vec = _mm256_setzero_ps();
        __m256 sum_sq_vec = _mm256_setzero_ps();
        
        int64_t j = 0;
        for (; j + 8 <= hidden_size; j += 8) {
            __m256 x = loadu_8_float(&input[i * hidden_size + j]);
            sum_vec = _mm256_add_ps(sum_vec, x);
            sum_sq_vec = _mm256_fmadd_ps(x, x, sum_sq_vec);
        }
        
        float sum = horizontal_sum_8(sum_vec);
        float sum_sq = horizontal_sum_8(sum_sq_vec);
        
        for (; j < hidden_size; ++j) {
            float val = input[i * hidden_size + j];
            sum += val;
            sum_sq += val * val;
        }
        
        float m = sum / hidden_size;
        float v = (sum_sq / hidden_size) - (m * m);
        mean[i] = m;
        var[i] = v;
        
        float inv_std = 1.0f / std::sqrt(v + eps);
        __m256 mean_vec = _mm256_set1_ps(m);
        __m256 inv_std_vec = _mm256_set1_ps(inv_std);
        
        j = 0;
        for (; j + 8 <= hidden_size; j += 8) {
            __m256 x = loadu_8_float(&input[i * hidden_size + j]);
            __m256 g = loadu_8_float(&gamma[j]);
            __m256 b = loadu_8_float(&beta[j]);
            
            __m256 normalized = _mm256_mul_ps(_mm256_sub_ps(x, mean_vec), inv_std_vec);
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
    if (has_avx2()) {
        if (use_parallel) {
            layernorm_forward_avx_parallel(input, gamma, beta, output, mean, var,
                                          N, hidden_size, eps, 0);
        } else {
            layernorm_forward_avx(input, gamma, beta, output, mean, var,
                                N, hidden_size, eps);
        }
    } else {
        layernorm_forward_scalar(input, gamma, beta, output, mean, var,
                               N, hidden_size, eps);
    }
}

// Scalar backward
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
        
        float sum_grad_norm = 0.0f;
        for (int64_t j = 0; j < hidden_size; ++j) {
            float normalized = (input[i * hidden_size + j] - mean[i]) * inv_std;
            float grad_norm = grad_output[i * hidden_size + j] * gamma[j];
            sum_grad_norm += grad_norm;
            
            if (grad_gamma) {
                grad_gamma[j] += normalized * grad_output[i * hidden_size + j];
            }
            if (grad_beta) {
                grad_beta[j] += grad_output[i * hidden_size + j];
            }
        }
        
        for (int64_t j = 0; j < hidden_size; ++j) {
            float normalized = (input[i * hidden_size + j] - mean[i]) * inv_std;
            float grad_norm = grad_output[i * hidden_size + j] * gamma[j];
            
            float grad_input_val = inv_std * grad_norm;
            grad_input_val -= inv_hidden_size * sum_grad_norm;
            grad_input_val -= inv_hidden_size * normalized * sum_grad_norm;
            
            grad_input[i * hidden_size + j] = grad_input_val;
        }
    }
}

// AVX backward (simplified, can be optimized further)
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
        __m256 inv_hs_vec = _mm256_set1_ps(inv_hidden_size);
        
        // Compute sum_grad_norm
        __m256 sum_grad_norm_vec = _mm256_setzero_ps();
        int64_t j = 0;
        for (; j + 8 <= hidden_size; j += 8) {
            __m256 go = loadu_8_float(&grad_output[i * hidden_size + j]);
            __m256 g = loadu_8_float(&gamma[j]);
            sum_grad_norm_vec = _mm256_fmadd_ps(go, g, sum_grad_norm_vec);
        }
        float sum_grad_norm = horizontal_sum_8(sum_grad_norm_vec);
        for (; j < hidden_size; ++j) {
            sum_grad_norm += grad_output[i * hidden_size + j] * gamma[j];
        }
        __m256 sum_gn_vec = _mm256_set1_ps(sum_grad_norm);
        
        // Compute gradients
        j = 0;
        for (; j + 8 <= hidden_size; j += 8) {
            __m256 x = loadu_8_float(&input[i * hidden_size + j]);
            __m256 go = loadu_8_float(&grad_output[i * hidden_size + j]);
            __m256 g = loadu_8_float(&gamma[j]);
            
            __m256 normalized = _mm256_mul_ps(_mm256_sub_ps(x, mean_vec), inv_std_vec);
            __m256 grad_norm = _mm256_mul_ps(go, g);
            
            // grad_gamma and grad_beta
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
            
            // grad_input
            __m256 gi = _mm256_mul_ps(inv_std_vec, grad_norm);
            gi = _mm256_fnmadd_ps(inv_hs_vec, sum_gn_vec, gi);
            __m256 norm_sum = _mm256_mul_ps(normalized, sum_gn_vec);
            gi = _mm256_fnmadd_ps(inv_hs_vec, norm_sum, gi);
            storeu_8_float(&grad_input[i * hidden_size + j], gi);
        }
        
        // Handle remainder
        for (; j < hidden_size; ++j) {
            float normalized = (input[i * hidden_size + j] - mean[i]) * inv_std;
            float grad_norm = grad_output[i * hidden_size + j] * gamma[j];
            
            if (grad_gamma) {
                grad_gamma[j] += normalized * grad_output[i * hidden_size + j];
            }
            if (grad_beta) {
                grad_beta[j] += grad_output[i * hidden_size + j];
            }
            
            float grad_input_val = inv_std * grad_norm;
            grad_input_val -= inv_hidden_size * sum_grad_norm;
            grad_input_val -= inv_hidden_size * normalized * sum_grad_norm;
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
        __m256 inv_hs_vec = _mm256_set1_ps(inv_hidden_size);
        
        __m256 sum_grad_norm_vec = _mm256_setzero_ps();
        int64_t j = 0;
        for (; j + 8 <= hidden_size; j += 8) {
            __m256 go = loadu_8_float(&grad_output[i * hidden_size + j]);
            __m256 g = loadu_8_float(&gamma[j]);
            sum_grad_norm_vec = _mm256_fmadd_ps(go, g, sum_grad_norm_vec);
        }
        float sum_grad_norm = horizontal_sum_8(sum_grad_norm_vec);
        for (; j < hidden_size; ++j) {
            sum_grad_norm += grad_output[i * hidden_size + j] * gamma[j];
        }
        __m256 sum_gn_vec = _mm256_set1_ps(sum_grad_norm);
        
        j = 0;
        for (; j + 8 <= hidden_size; j += 8) {
            __m256 x = loadu_8_float(&input[i * hidden_size + j]);
            __m256 go = loadu_8_float(&grad_output[i * hidden_size + j]);
            __m256 g = loadu_8_float(&gamma[j]);
            
            __m256 normalized = _mm256_mul_ps(_mm256_sub_ps(x, mean_vec), inv_std_vec);
            __m256 grad_norm = _mm256_mul_ps(go, g);
            
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
            
            __m256 gi = _mm256_mul_ps(inv_std_vec, grad_norm);
            gi = _mm256_fnmadd_ps(inv_hs_vec, sum_gn_vec, gi);
            __m256 norm_sum = _mm256_mul_ps(normalized, sum_gn_vec);
            gi = _mm256_fnmadd_ps(inv_hs_vec, norm_sum, gi);
            storeu_8_float(&grad_input[i * hidden_size + j], gi);
        }
        
        for (; j < hidden_size; ++j) {
            float normalized = (input[i * hidden_size + j] - mean[i]) * inv_std;
            float grad_norm = grad_output[i * hidden_size + j] * gamma[j];
            
            #pragma omp critical
            {
                if (grad_gamma) {
                    grad_gamma[j] += normalized * grad_output[i * hidden_size + j];
                }
                if (grad_beta) {
                    grad_beta[j] += grad_output[i * hidden_size + j];
                }
            }
            
            float grad_input_val = inv_std * grad_norm;
            grad_input_val -= inv_hidden_size * sum_grad_norm;
            grad_input_val -= inv_hidden_size * normalized * sum_grad_norm;
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
    if (has_avx2()) {
        if (use_parallel) {
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

