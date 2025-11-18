#ifndef SOFTMAX_AVX_H
#define SOFTMAX_AVX_H

#include <cstdint>

// Softmax: output = exp(input - max) / sum(exp(input - max))
// input: [N, seq_len], output: [N, seq_len]

void softmax_forward_scalar(
    const float* input, float* output,
    int64_t N, int64_t seq_len
);

void softmax_forward_avx(
    const float* input, float* output,
    int64_t N, int64_t seq_len
);

void softmax_forward_avx_parallel(
    const float* input, float* output,
    int64_t N, int64_t seq_len,
    int num_threads = 0
);

void softmax_forward(
    const float* input, float* output,
    int64_t N, int64_t seq_len,
    bool use_parallel = true
);

#endif // SOFTMAX_AVX_H

