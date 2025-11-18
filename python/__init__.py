import torch

try:
    from . import avx_kernels_cpp
    _has_avx_kernels = True
except ImportError:
    _has_avx_kernels = False
    print("Warning: AVX kernels not compiled. Please run 'python setup.py build_ext --inplace'")


def gemm(A, B, C=None, alpha=1.0, beta=0.0, use_parallel=True):
    """GEMM operation with AVX optimization.
    
    Args:
        A: [M, K] tensor
        B: [K, N] tensor
        C: [M, N] tensor (optional, will be created if None)
        alpha: scaling factor for A*B
        beta: scaling factor for C
        use_parallel: whether to use parallel computation
    
    Returns:
        Result tensor [M, N]
    """
    if not _has_avx_kernels:
        # Fallback to PyTorch
        if C is None:
            return alpha * torch.mm(A, B)
        else:
            return alpha * torch.mm(A, B) + beta * C
    
    return avx_kernels_cpp.gemm_forward(A, B, C, alpha, beta, use_parallel)


def layernorm(input, gamma, beta, eps=1e-5, use_parallel=True):
    """LayerNorm operation with AVX optimization.
    
    Args:
        input: [N, hidden_size] tensor
        gamma: [hidden_size] tensor
        beta: [hidden_size] tensor
        eps: epsilon for numerical stability
        use_parallel: whether to use parallel computation
    
    Returns:
        (output, mean, var) tuple
    """
    if not _has_avx_kernels:
        # Fallback to PyTorch
        mean = input.mean(dim=1, keepdim=True)
        var = input.var(dim=1, keepdim=True, unbiased=False)
        output = (input - mean) / torch.sqrt(var + eps)
        output = output * gamma.unsqueeze(0) + beta.unsqueeze(0)
        return output, mean.squeeze(1), var.squeeze(1)
    
    return avx_kernels_cpp.layernorm_forward(input, gamma, beta, eps, use_parallel)


def softmax(input, use_parallel=True):
    """Softmax operation with AVX optimization.
    
    Args:
        input: [N, seq_len] tensor
        use_parallel: whether to use parallel computation
    
    Returns:
        Output tensor [N, seq_len]
    """
    if not _has_avx_kernels:
        # Fallback to PyTorch
        return torch.nn.functional.softmax(input, dim=1)
    
    return avx_kernels_cpp.softmax_forward(input, use_parallel)

