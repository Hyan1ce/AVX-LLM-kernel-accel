import torch
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from python import layernorm

def benchmark_layernorm(N, hidden_size, num_iterations=100, warmup=10):
    """Benchmark LayerNorm operation."""
    print(f"\nBenchmarking LayerNorm: N={N}, hidden_size={hidden_size}")
    print("=" * 60)
    
    # Create random tensors
    input_tensor = torch.randn(N, hidden_size, dtype=torch.float32)
    gamma = torch.ones(hidden_size, dtype=torch.float32)
    beta = torch.zeros(hidden_size, dtype=torch.float32)
    
    # PyTorch baseline
    print("\n1. PyTorch baseline:")
    start = time.time()
    for _ in range(warmup):
        mean = input_tensor.mean(dim=1, keepdim=True)
        var = input_tensor.var(dim=1, keepdim=True, unbiased=False)
        result_torch = (input_tensor - mean) / torch.sqrt(var + 1e-5)
        result_torch = result_torch * gamma.unsqueeze(0) + beta.unsqueeze(0)
    
    start = time.time()
    for _ in range(num_iterations):
        mean = input_tensor.mean(dim=1, keepdim=True)
        var = input_tensor.var(dim=1, keepdim=True, unbiased=False)
        result_torch = (input_tensor - mean) / torch.sqrt(var + 1e-5)
        result_torch = result_torch * gamma.unsqueeze(0) + beta.unsqueeze(0)
    torch_time = (time.time() - start) / num_iterations * 1000  # ms
    
    print(f"   Time: {torch_time:.4f} ms")
    
    # AVX implementation
    print("\n2. AVX implementation (single-threaded):")
    start = time.time()
    for _ in range(warmup):
        _ = layernorm(input_tensor, gamma, beta, use_parallel=False)
    
    start = time.time()
    for _ in range(num_iterations):
        result_avx, _, _ = layernorm(input_tensor, gamma, beta, use_parallel=False)
    avx_time = (time.time() - start) / num_iterations * 1000  # ms
    
    print(f"   Time: {avx_time:.4f} ms")
    print(f"   Speedup: {torch_time / avx_time:.2f}x")
    
    # AVX parallel implementation
    print("\n3. AVX implementation (multi-threaded):")
    start = time.time()
    for _ in range(warmup):
        _ = layernorm(input_tensor, gamma, beta, use_parallel=True)
    
    start = time.time()
    for _ in range(num_iterations):
        result_avx_parallel, _, _ = layernorm(input_tensor, gamma, beta, use_parallel=True)
    avx_parallel_time = (time.time() - start) / num_iterations * 1000  # ms
    
    print(f"   Time: {avx_parallel_time:.4f} ms")
    print(f"   Speedup: {torch_time / avx_parallel_time:.2f}x")
    
    # Check correctness
    max_diff = torch.max(torch.abs(result_torch - result_avx_parallel)).item()
    print(f"\n4. Numerical accuracy:")
    print(f"   Max difference: {max_diff:.2e}")
    
    return {
        'torch_time': torch_time,
        'avx_time': avx_time,
        'avx_parallel_time': avx_parallel_time,
        'speedup': torch_time / avx_parallel_time,
        'max_diff': max_diff
    }

if __name__ == "__main__":
    print("LayerNorm Benchmark")
    print("=" * 60)
    
    test_cases = [
        (32, 512),
        (128, 512),
        (256, 512),
        (512, 512),
        (1024, 512),
        (32, 1024),
        (128, 1024),
    ]
    
    results = []
    for N, hidden_size in test_cases:
        result = benchmark_layernorm(N, hidden_size, num_iterations=50, warmup=5)
        results.append((N, hidden_size, result))
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"{'N':<8} {'hidden_size':<12} {'PyTorch(ms)':<15} {'AVX-Parallel(ms)':<18} {'Speedup':<10} {'Max Diff':<12}")
    print("-" * 60)
    for N, hidden_size, r in results:
        print(f"{N:<8} {hidden_size:<12} {r['torch_time']:<15.4f} {r['avx_parallel_time']:<18.4f} {r['speedup']:<10.2f} {r['max_diff']:<12.2e}")

