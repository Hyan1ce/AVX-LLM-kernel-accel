import torch
import time
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from python import gemm

def benchmark_gemm(M, N, K, num_iterations=100, warmup=10):
    """Benchmark GEMM operation."""
    print(f"\nBenchmarking GEMM: M={M}, N={N}, K={K}")
    print("=" * 60)
    
    # Create random tensors
    A = torch.randn(M, K, dtype=torch.float32)
    B = torch.randn(K, N, dtype=torch.float32)
    C = torch.zeros(M, N, dtype=torch.float32)
    
    # PyTorch baseline
    print("\n1. PyTorch baseline:")
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    for _ in range(warmup):
        _ = torch.mm(A, B)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    start = time.time()
    for _ in range(num_iterations):
        result_torch = torch.mm(A, B)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    torch_time = (time.time() - start) / num_iterations * 1000  # ms
    
    print(f"   Time: {torch_time:.4f} ms")
    print(f"   Throughput: {2 * M * N * K / torch_time / 1e6:.2f} GFLOPS")
    
    # AVX implementation
    print("\n2. AVX implementation (single-threaded):")
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    for _ in range(warmup):
        _ = gemm(A, B, use_parallel=False)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    start = time.time()
    for _ in range(num_iterations):
        result_avx = gemm(A, B, use_parallel=False)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    avx_time = (time.time() - start) / num_iterations * 1000  # ms
    
    print(f"   Time: {avx_time:.4f} ms")
    print(f"   Throughput: {2 * M * N * K / avx_time / 1e6:.2f} GFLOPS")
    print(f"   Speedup: {torch_time / avx_time:.2f}x")
    
    # AVX parallel implementation
    print("\n3. AVX implementation (multi-threaded):")
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    for _ in range(warmup):
        _ = gemm(A, B, use_parallel=True)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    start = time.time()
    for _ in range(num_iterations):
        result_avx_parallel = gemm(A, B, use_parallel=True)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    avx_parallel_time = (time.time() - start) / num_iterations * 1000  # ms
    
    print(f"   Time: {avx_parallel_time:.4f} ms")
    print(f"   Throughput: {2 * M * N * K / avx_parallel_time / 1e6:.2f} GFLOPS")
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
    print("GEMM Benchmark")
    print("=" * 60)
    
    # Test different sizes
    test_cases = [
        (128, 128, 128),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (512, 2048, 1024),
    ]
    
    results = []
    for M, N, K in test_cases:
        result = benchmark_gemm(M, N, K, num_iterations=50, warmup=5)
        results.append((M, N, K, result))
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"{'M':<8} {'N':<8} {'K':<8} {'PyTorch(ms)':<15} {'AVX-Parallel(ms)':<18} {'Speedup':<10} {'Max Diff':<12}")
    print("-" * 60)
    for M, N, K, r in results:
        print(f"{M:<8} {N:<8} {K:<8} {r['torch_time']:<15.4f} {r['avx_parallel_time']:<18.4f} {r['speedup']:<10.2f} {r['max_diff']:<12.2e}")

