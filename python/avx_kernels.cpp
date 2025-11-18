#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <vector>
#include "../kernels/gemm/gemm_avx.h"
#include "../kernels/layernorm/layernorm_avx.h"
#include "../kernels/softmax/softmax_avx.h"

namespace py = pybind11;

// GEMM wrapper
torch::Tensor gemm_avx_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    float alpha = 1.0f,
    float beta = 0.0f,
    bool use_parallel = true
) {
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D tensors");
    TORCH_CHECK(A.size(1) == B.size(0), "Matrix dimensions mismatch");
    
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(1);
    
    if (C.numel() == 0) {
        C = torch::zeros({M, N}, A.options());
    } else {
        TORCH_CHECK(C.size(0) == M && C.size(1) == N, "Output tensor size mismatch");
    }
    
    const float* A_data = A.data_ptr<float>();
    const float* B_data = B.data_ptr<float>();
    float* C_data = C.data_ptr<float>();
    
    gemm(A_data, B_data, C_data, M, N, K, alpha, beta, use_parallel);
    
    return C;
}

// LayerNorm wrapper
std::vector<torch::Tensor> layernorm_avx_forward(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps = 1e-5f,
    bool use_parallel = true
) {
    TORCH_CHECK(input.dim() == 2, "Input must be 2D tensor [N, hidden_size]");
    TORCH_CHECK(gamma.dim() == 1, "Gamma must be 1D tensor");
    TORCH_CHECK(beta.dim() == 1, "Beta must be 1D tensor");
    
    int64_t N = input.size(0);
    int64_t hidden_size = input.size(1);
    
    TORCH_CHECK(gamma.size(0) == hidden_size, "Gamma size mismatch");
    TORCH_CHECK(beta.size(0) == hidden_size, "Beta size mismatch");
    
    auto output = torch::empty_like(input);
    auto mean = torch::empty({N}, input.options());
    auto var = torch::empty({N}, input.options());
    
    const float* input_data = input.data_ptr<float>();
    const float* gamma_data = gamma.data_ptr<float>();
    const float* beta_data = beta.data_ptr<float>();
    float* output_data = output.data_ptr<float>();
    float* mean_data = mean.data_ptr<float>();
    float* var_data = var.data_ptr<float>();
    
    layernorm_forward(input_data, gamma_data, beta_data, output_data,
                     mean_data, var_data, N, hidden_size, eps, use_parallel);
    
    return {output, mean, var};
}

std::vector<torch::Tensor> layernorm_avx_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor mean,
    torch::Tensor var,
    float eps = 1e-5f,
    bool use_parallel = true
) {
    TORCH_CHECK(grad_output.dim() == 2, "Grad output must be 2D tensor");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D tensor");
    
    int64_t N = input.size(0);
    int64_t hidden_size = input.size(1);
    
    auto grad_input = torch::empty_like(input);
    auto grad_gamma = torch::zeros_like(gamma);
    auto grad_beta = torch::zeros_like(beta);
    
    const float* grad_output_data = grad_output.data_ptr<float>();
    const float* input_data = input.data_ptr<float>();
    const float* gamma_data = gamma.data_ptr<float>();
    const float* mean_data = mean.data_ptr<float>();
    const float* var_data = var.data_ptr<float>();
    float* grad_input_data = grad_input.data_ptr<float>();
    float* grad_gamma_data = grad_gamma.data_ptr<float>();
    float* grad_beta_data = grad_beta.data_ptr<float>();
    
    layernorm_backward(grad_output_data, input_data, gamma_data,
                      mean_data, var_data, grad_input_data,
                      grad_gamma_data, grad_beta_data,
                      N, hidden_size, eps, use_parallel);
    
    return {grad_input, grad_gamma, grad_beta};
}

// Softmax wrapper
torch::Tensor softmax_avx_forward(
    torch::Tensor input,
    bool use_parallel = true
) {
    TORCH_CHECK(input.dim() == 2, "Input must be 2D tensor [N, seq_len]");
    
    int64_t N = input.size(0);
    int64_t seq_len = input.size(1);
    
    auto output = torch::empty_like(input);
    
    const float* input_data = input.data_ptr<float>();
    float* output_data = output.data_ptr<float>();
    
    softmax_forward(input_data, output_data, N, seq_len, use_parallel);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_forward", &gemm_avx_forward, "GEMM AVX forward",
          py::arg("A"), py::arg("B"), py::arg("C") = torch::Tensor(),
          py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f,
          py::arg("use_parallel") = true);
    
    m.def("layernorm_forward", &layernorm_avx_forward, "LayerNorm AVX forward",
          py::arg("input"), py::arg("gamma"), py::arg("beta"),
          py::arg("eps") = 1e-5f, py::arg("use_parallel") = true);
    
    m.def("layernorm_backward", &layernorm_avx_backward, "LayerNorm AVX backward",
          py::arg("grad_output"), py::arg("input"), py::arg("gamma"),
          py::arg("mean"), py::arg("var"), py::arg("eps") = 1e-5f,
          py::arg("use_parallel") = true);
    
    m.def("softmax_forward", &softmax_avx_forward, "Softmax AVX forward",
          py::arg("input"), py::arg("use_parallel") = true);
}

