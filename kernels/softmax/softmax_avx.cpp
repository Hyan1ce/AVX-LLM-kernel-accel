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

void softmax_forward(
    const float* input, float* output,
    int64_t N, int64_t seq_len,
    bool use_parallel
) {
    // 为了数值稳定和与PyTorch严格对齐，这里使用标量实现 + OpenMP 并行，
    // 依然在 -mavx2 下由编译器自动向量化。
    if (!use_parallel) {
        softmax_forward_scalar(input, output, N, seq_len);
        return;
    }

    // 自适应选择：小任务时使用串行避免OpenMP开销
    // 每个任务处理一行，工作量 = N * seq_len
    // 阈值：当总工作量 < 64K元素时使用串行
    int64_t total_elements = N * seq_len;
    if (total_elements < 64 * 1024) {
        softmax_forward_scalar(input, output, N, seq_len);
        return;
    }

    #pragma omp parallel for
    for (int64_t i = 0; i < N; ++i) {
        const float* row_input = &input[i * seq_len];
        float* row_output = &output[i * seq_len];

        // Find max (same as scalar)
        float max_val = row_input[0];
        for (int64_t j = 1; j < seq_len; ++j) {
            max_val = std::max(max_val, row_input[j]);
        }

        // Compute exp and sum
        float sum = 0.0f;
        for (int64_t j = 0; j < seq_len; ++j) {
            float val = std::exp(row_input[j] - max_val);
            row_output[j] = val;
            sum += val;
        }

        // Normalize
        float inv_sum = 1.0f / sum;
        for (int64_t j = 0; j < seq_len; ++j) {
            row_output[j] *= inv_sum;
        }
    }
}

