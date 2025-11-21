import torch
import sys
import os

# Ensure project root (where .so is located) is importable
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    import avx_kernels_cpp
    _has_avx_kernels = True
except ImportError:
    _has_avx_kernels = False
    print("Warning: AVX kernels not compiled. Please run 'python setup.py build_ext --inplace'")

# 控制是否启用AVX实现
# 默认：如果C++扩展存在，则三种kernel都启用AVX/OpenMP实现；
# 如需强制退回PyTorch，可显式设置环境变量为0。
_USE_AVX_GEMM = os.getenv("USE_AVX_GEMM", "1") == "1"
_USE_AVX_LAYERNORM = os.getenv("USE_AVX_LAYERNORM", "1") == "1"
_USE_AVX_SOFTMAX = os.getenv("USE_AVX_SOFTMAX", "1") == "1"


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
    # 确保张量连续
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous() if C is not None else None

    if not _has_avx_kernels or not _USE_AVX_GEMM:
        # Fallback to PyTorch
        if C is None:
            return alpha * torch.mm(A, B)
        else:
            return alpha * torch.mm(A, B) + beta * C

    if C is None:
        C = torch.empty(0)  # Pass empty tensor instead of None
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
    # Fallback 或未显式启用AVX时，使用PyTorch实现（保证数值完全一致）
    if (not _has_avx_kernels) or (not _USE_AVX_LAYERNORM):
        mean = input.mean(dim=1, keepdim=True)
        var = input.var(dim=1, keepdim=True, unbiased=False)
        output = (input - mean) / torch.sqrt(var + eps)
        output = output * gamma.unsqueeze(0) + beta.unsqueeze(0)
        return output, mean.squeeze(1), var.squeeze(1)

    # AVX实现分支（仅在明确启用时使用）
    input = input.contiguous()
    gamma = gamma.contiguous()
    beta = beta.contiguous()
    return avx_kernels_cpp.layernorm_forward(input, gamma, beta, eps, use_parallel)


def softmax(input, use_parallel=True):
    """Softmax operation with AVX optimization.
    
    Args:
        input: [N, seq_len] tensor
        use_parallel: whether to use parallel computation
    
    Returns:
        Output tensor [N, seq_len]
    """
    # 默认使用PyTorch实现，只有在显式启用AVX时才走C++路径
    if (not _has_avx_kernels) or (not _USE_AVX_SOFTMAX):
        return torch.nn.functional.softmax(input, dim=1)

    input = input.contiguous()
    return avx_kernels_cpp.softmax_forward(input, use_parallel)


