#!/bin/bash
# 运行所有测试脚本

echo "=========================================="
echo "AVX-LLM-Kernel-Accel Test Suite"
echo "=========================================="

echo ""
echo "1. Testing GEMM..."
python tests/test_gemm.py

echo ""
echo "2. Testing LayerNorm..."
python tests/test_layernorm.py

echo ""
echo "3. Testing Softmax..."
python tests/test_softmax.py

echo ""
echo "=========================================="
echo "All tests completed!"
echo "=========================================="

