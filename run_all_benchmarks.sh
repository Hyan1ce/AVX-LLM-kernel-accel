#!/bin/bash
# 运行所有benchmark脚本

echo "=========================================="
echo "AVX-LLM-Kernel-Accel Benchmark Suite"
echo "=========================================="

echo ""
echo "1. Benchmarking GEMM..."
python benchmarks/benchmark_gemm.py

echo ""
echo "2. Benchmarking LayerNorm..."
python benchmarks/benchmark_layernorm.py

echo ""
echo "3. Benchmarking Softmax..."
python benchmarks/benchmark_softmax.py

echo ""
echo "=========================================="
echo "All benchmarks completed!"
echo "=========================================="

