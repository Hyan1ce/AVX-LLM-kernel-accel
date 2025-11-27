
import torch
import sys
import os

# 确保测试时优先走 C++/AVX 路径（如果已成功编译）
os.environ.setdefault("USE_AVX_SOFTMAX", "1")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from python import softmax

def test_softmax():
    """Test Softmax correctness."""
    print("Testing Softmax...")
    
    N, seq_len = 32, 128
    input_tensor = torch.randn(N, seq_len, dtype=torch.float32)
    
    # PyTorch reference
    result_torch = torch.nn.functional.softmax(input_tensor, dim=1)
    
    # AVX implementation
    result_avx = softmax(input_tensor, use_parallel=True)
    
    # Check
    max_diff = torch.max(torch.abs(result_torch - result_avx)).item()
    mean_diff = torch.mean(torch.abs(result_torch - result_avx)).item()
    
    print(f"  Max difference: {max_diff:.2e}")
    print(f"  Mean difference: {mean_diff:.2e}")
    
    assert max_diff < 1e-4, f"Softmax test failed: max_diff={max_diff}"
    print("Softmax test passed!")

if __name__ == "__main__":
    test_softmax()

