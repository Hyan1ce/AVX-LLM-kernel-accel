import torch
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from python import softmax

def benchmark_softmax(N, seq_len, num_iterations=100, warmup=10):
    """Benchmark Softmax operation."""
    print(f"\nBenchmarking Softmax: N={N}, seq_len={seq_len}")
    print("=" * 60)
    
    # Create random tensor
    input_tensor = torch.randn(N, seq_len, dtype=torch.float32)
    
    # PyTorch baseline
    print("\n1. PyTorch baseline:")
    start = time.time()
    for _ in range(warmup):
        _ = torch.nn.functional.softmax(input_tensor, dim=1)
    
    start = time.time()
    for _ in range(num_iterations):
        result_torch = torch.nn.functional.softmax(input_tensor, dim=1)
    torch_time = (time.time() - start) / num_iterations * 1000  # ms
    
    print(f"   Time: {torch_time:.4f} ms")
    
    # AVX implementation
    print("\n2. AVX implementation (single-threaded):")
    start = time.time()
    for _ in range(warmup):
        _ = softmax(input_tensor, use_parallel=False)
    
    start = time.time()
    for _ in range(num_iterations):
        result_avx = softmax(input_tensor, use_parallel=False)
    avx_time = (time.time() - start) / num_iterations * 1000  # ms
    
    print(f"   Time: {avx_time:.4f} ms")
    print(f"   Speedup: {torch_time / avx_time:.2f}x")
    
    # AVX parallel implementation
    print("\n3. AVX implementation (multi-threaded):")
    start = time.time()
    for _ in range(warmup):
        _ = softmax(input_tensor, use_parallel=True)
    
    start = time.time()
    for _ in range(num_iterations):
        result_avx_parallel = softmax(input_tensor, use_parallel=True)
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
    print("Softmax Benchmark")
    print("=" * 60)
    
    test_cases = [
        (32, 128),
        (32, 512),
        (128, 128),
        (128, 512),
        (256, 512),
        (512, 512),
        (128, 1024),
    ]
    
    results = []
    for N, seq_len in test_cases:
        result = benchmark_softmax(N, seq_len, num_iterations=50, warmup=5)
        results.append((N, seq_len, result))
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"{'N':<8} {'seq_len':<10} {'PyTorch(ms)':<15} {'AVX-Parallel(ms)':<18} {'Speedup':<10} {'Max Diff':<12}")
    print("-" * 60)
    for N, seq_len, r in results:
        print(f"{N:<8} {seq_len:<10} {r['torch_time']:<15.4f} {r['avx_parallel_time']:<18.4f} {r['speedup']:<10.2f} {r['max_diff']:<12.2e}")

