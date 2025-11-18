import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from python import layernorm

def test_layernorm():
    """Test LayerNorm correctness."""
    print("Testing LayerNorm...")
    
    N, hidden_size = 32, 512
    input_tensor = torch.randn(N, hidden_size, dtype=torch.float32)
    gamma = torch.ones(hidden_size, dtype=torch.float32)
    beta = torch.zeros(hidden_size, dtype=torch.float32)
    
    # PyTorch reference
    mean = input_tensor.mean(dim=1, keepdim=True)
    var = input_tensor.var(dim=1, keepdim=True, unbiased=False)
    result_torch = (input_tensor - mean) / torch.sqrt(var + 1e-5)
    result_torch = result_torch * gamma.unsqueeze(0) + beta.unsqueeze(0)
    
    # AVX implementation
    result_avx, mean_avx, var_avx = layernorm(input_tensor, gamma, beta, 
                                            use_parallel=True)
    
    # Check
    max_diff = torch.max(torch.abs(result_torch - result_avx)).item()
    mean_diff = torch.mean(torch.abs(result_torch - result_avx)).item()
    
    print(f"  Max difference: {max_diff:.2e}")
    print(f"  Mean difference: {mean_diff:.2e}")
    
    assert max_diff < 1e-4, f"LayerNorm test failed: max_diff={max_diff}"
    print("  ✓ LayerNorm test passed!")

if __name__ == "__main__":
    test_layernorm()

