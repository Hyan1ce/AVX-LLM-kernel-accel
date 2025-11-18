import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from python import gemm

def test_gemm():
    """Test GEMM correctness."""
    print("Testing GEMM...")
    
    M, N, K = 128, 256, 512
    A = torch.randn(M, K, dtype=torch.float32)
    B = torch.randn(K, N, dtype=torch.float32)
    
    # PyTorch reference
    result_torch = torch.mm(A, B)
    
    # AVX implementation
    result_avx = gemm(A, B, use_parallel=True)
    
    # Check
    max_diff = torch.max(torch.abs(result_torch - result_avx)).item()
    mean_diff = torch.mean(torch.abs(result_torch - result_avx)).item()
    
    print(f"  Max difference: {max_diff:.2e}")
    print(f"  Mean difference: {mean_diff:.2e}")
    
    assert max_diff < 1e-4, f"GEMM test failed: max_diff={max_diff}"
    print("  ✓ GEMM test passed!")

if __name__ == "__main__":
    test_gemm()

