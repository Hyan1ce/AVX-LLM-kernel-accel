#ifndef LAYERNORM_AVX_H
#define LAYERNORM_AVX_H

#include <cstdint>

// LayerNorm forward: output = (input - mean) / sqrt(var + eps) * gamma + beta
// input: [N, hidden_size], output: [N, hidden_size]
// gamma, beta: [hidden_size]

void layernorm_forward_scalar(
    const float* input, const float* gamma, const float* beta,
    float* output, float* mean, float* var,
    int64_t N, int64_t hidden_size, float eps = 1e-5f
);

void layernorm_forward_avx(
    const float* input, const float* gamma, const float* beta,
    float* output, float* mean, float* var,
    int64_t N, int64_t hidden_size, float eps = 1e-5f
);

void layernorm_forward_avx_parallel(
    const float* input, const float* gamma, const float* beta,
    float* output, float* mean, float* var,
    int64_t N, int64_t hidden_size, float eps = 1e-5f,
    int num_threads = 0
);

void layernorm_forward(
    const float* input, const float* gamma, const float* beta,
    float* output, float* mean, float* var,
    int64_t N, int64_t hidden_size, float eps = 1e-5f,
    bool use_parallel = true
);

// LayerNorm backward
void layernorm_backward_scalar(
    const float* grad_output, const float* input,
    const float* gamma, const float* mean, const float* var,
    float* grad_input, float* grad_gamma, float* grad_beta,
    int64_t N, int64_t hidden_size, float eps = 1e-5f
);

void layernorm_backward_avx(
    const float* grad_output, const float* input,
    const float* gamma, const float* mean, const float* var,
    float* grad_input, float* grad_gamma, float* grad_beta,
    int64_t N, int64_t hidden_size, float eps = 1e-5f
);

void layernorm_backward_avx_parallel(
    const float* grad_output, const float* input,
    const float* gamma, const float* mean, const float* var,
    float* grad_input, float* grad_gamma, float* grad_beta,
    int64_t N, int64_t hidden_size, float eps = 1e-5f,
    int num_threads = 0
);

void layernorm_backward(
    const float* grad_output, const float* input,
    const float* gamma, const float* mean, const float* var,
    float* grad_input, float* grad_gamma, float* grad_beta,
    int64_t N, int64_t hidden_size, float eps = 1e-5f,
    bool use_parallel = true
);

#endif // LAYERNORM_AVX_H

