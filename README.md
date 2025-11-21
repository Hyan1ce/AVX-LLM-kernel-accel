# AVX-LLM-Kernel-Accel

一个使用AVX指令集和多线程优化的Transformer训练/微调程序，支持在CPU上高效运行: https://github.com/Hyan1ce/AVX-LLM-kernel-accel

## 项目概述

本项目实现了三个关键kernel的 **CPU 优化版本**，在支持 AVX2 的 CPU 上配合 OpenMP 可以获得显著加速：
- **GEMM** (General Matrix Multiply): 矩阵乘法核心操作（显式使用 AVX2 intrinsics）
- **LayerNorm**: 层归一化（包含 forward 和 backward，使用数值稳定的两遍算法 + OpenMP，并在 backward 中使用 AVX2 intrinsics）
- **Softmax**: 注意力机制中的 softmax 操作（数值稳定实现 + OpenMP）

所有 kernel 至少提供：
- **标量实现（baseline）**
- **基于 OpenMP 的多线程实现**（在开启 `-mavx2 -O3` 时由编译器自动向量化，结合 AVX2 指令）

## 项目结构

```
AVX-LLM-kernel-accel/
├── README.md                 # 本文件
├── CMakeLists.txt           # CMake构建配置
├── setup.py                  # Python扩展构建脚本
│
├── kernels/                  # C++ kernel实现
│   ├── common/              # 公共工具
│   │   ├── avx_utils.h      # AVX工具函数
│   │   └── threading.h      # 多线程工具
│   ├── gemm/                # GEMM kernel
│   │   ├── gemm_avx.h
│   │   └── gemm_avx.cpp
│   ├── layernorm/            # LayerNorm kernel
│   │   ├── layernorm_avx.h
│   │   └── layernorm_avx.cpp
│   └── softmax/              # Softmax kernel
│       ├── softmax_avx.h
│       └── softmax_avx.cpp
│
├── python/                   # Python接口
│   ├── __init__.py          # Python包装
│   ├── avx_kernels.cpp      # PyTorch C++ extension
│   └── model.py             # Transformer模型定义
│
├── benchmarks/               # 性能测试脚本
│   ├── benchmark_gemm.py
│   ├── benchmark_layernorm.py
│   └── benchmark_softmax.py
│
├── tests/                    # 正确性测试
│   ├── test_gemm.py
│   ├── test_layernorm.py
│   └── test_softmax.py
│
└── data/                     # 数据生成
    └── generate_synthetic.py
```

## 环境要求

### 系统要求
- Linux/macOS (推荐Linux)
- CPU支持AVX2指令集
- 多核CPU（用于并行优化）

### 软件依赖
- Python 3.7+
- PyTorch 1.8+ (CPU版本)
- pybind11
- OpenMP
- GCC/G++ 7+ (支持C++17和AVX2)
- CMake 3.18+ (可选，用于CMake构建)

### 安装依赖（推荐使用 conda 环境）

```bash
# 创建并激活环境（示例）
conda create -n avx-llm-kernel python=3.9 -y
conda activate avx-llm-kernel

# 安装 Python 依赖
pip install -r requirements.txt

# 如果系统缺少 OpenMP（部分 Linux/macOS）
# Ubuntu/Debian:
#   sudo apt-get install libomp-dev
# macOS (Homebrew):
#   brew install libomp
```

## 编译和安装

### 方法1: 使用 setup.py 构建 PyTorch 扩展（推荐）

```bash
# 进入项目目录
cd AVX-LLM-kernel-accel

# 编译并在当前目录生成 avx_kernels_cpp*.so
python setup.py build_ext --inplace

# （可选）以开发模式安装
pip install -e .
```

### 方法2: 使用CMake (可选)

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## 使用方法

### 1. 运行测试

首先验证kernel的正确性：

```bash
# 确认当前在项目根目录
cd AVX-LLM-kernel-accel
conda activate avx-llm-kernel  # 或者你自己的环境名

# 测试 GEMM（默认使用 AVX + OpenMP 实现）
python tests/test_gemm.py

# 测试 LayerNorm / Softmax（默认使用 PyTorch 实现，保证数值完全一致）
python tests/test_layernorm.py
python tests/test_softmax.py

# 如果你想显式测试 C++/AVX/OpenMP 路径，可设置环境变量：
USE_AVX_LAYERNORM=1 USE_AVX_SOFTMAX=1 python tests/test_layernorm.py
USE_AVX_LAYERNORM=1 USE_AVX_SOFTMAX=1 python tests/test_softmax.py
```

### 2. 运行Benchmark

性能对比测试：

```bash
# GEMM性能测试
python benchmarks/benchmark_gemm.py

# LayerNorm性能测试
python benchmarks/benchmark_layernorm.py

# Softmax性能测试
python benchmarks/benchmark_softmax.py
```

### 3. 在代码中使用

```python
import torch
from python import gemm, layernorm, softmax

# GEMM 示例（默认走 AVX+OpenMP 实现）
A = torch.randn(128, 512)
B = torch.randn(512, 256)
C = gemm(A, B, use_parallel=True)

# LayerNorm 示例
input_tensor = torch.randn(32, 512)
gamma = torch.ones(512)
beta = torch.zeros(512)

# 默认：使用 PyTorch 实现（数值与 PyTorch 完全一致）
output, mean, var = layernorm(input_tensor, gamma, beta, use_parallel=True)

# 如果你想强制使用 C++/AVX/OpenMP 实现：
# import os
# os.environ["USE_AVX_LAYERNORM"] = "1"
# output, mean, var = layernorm(input_tensor, gamma, beta, use_parallel=True)

# Softmax 示例（同理，默认 PyTorch，可通过 USE_AVX_SOFTMAX=1 启用 C++ 实现）
input_tensor = torch.randn(32, 128)
output = softmax(input_tensor, use_parallel=True)
```

### 4. 使用Transformer模型

```python
from python.model import SimpleTransformer

# 创建模型（使用AVX优化）
model = SimpleTransformer(
    vocab_size=10000,
    d_model=512,
    nhead=8,
    num_layers=6,
    use_avx=True  # 启用AVX优化
)

# 前向传播
input_ids = torch.randint(0, 10000, (32, 128))
logits = model(input_ids)
```

## 性能优化说明

### 编译优化标志

项目使用以下编译优化：
- `-O3`: 最高级别优化
- `-march=native`: 针对当前CPU架构优化
- `-mavx2`: 启用AVX2指令集
- `-fopenmp`: 启用OpenMP并行

### 并行化策略

- **GEMM**: 使用显式 AVX2 intrinsics + OpenMP 对矩阵行进行并行化
- **LayerNorm**:
  - Forward: 使用两遍扫描（计算 mean / var）+ OpenMP，数学公式与 PyTorch 完全一致，依赖编译器在 AVX2 下自动向量化
  - Backward: 使用 AVX2 intrinsics 实现梯度计算，并对 batch 维度并行
- **Softmax**: 对 batch 维度使用 OpenMP 并行，内部使用数值稳定的 `exp(x - max)` 实现，在 AVX2 下同样可由编译器自动向量化

### 性能调优建议

1. **线程数设置**: 默认使用所有可用CPU核心，可通过环境变量控制：
   ```bash
   export OMP_NUM_THREADS=8  # 使用8个线程
   ```

2. **内存对齐**: 代码中已处理内存对齐，确保AVX指令高效执行

3. **批量大小**: 较大的batch size通常能获得更好的性能

## 实现细节

### GEMM 实现
- 使用 AVX2 的 `_mm256_fmadd_ps` 进行融合乘加操作
- 8 元素向量化处理
- 支持 alpha / beta 参数

### LayerNorm 实现
- Forward:
  - 使用两遍扫描算法：先计算每一行的 mean，再计算 `(x - mean)^2` 的平均得到 var
  - 数值上与 `input.var(dim=1, unbiased=False)` 完全一致
  - 使用 OpenMP 对 batch 维度并行，编译器在 AVX2 下自动向量化循环
- Backward:
  - 实现完整的梯度计算（对 input、gamma、beta）
  - 使用 AVX2 intrinsics 加速 + OpenMP 并行

### Softmax 实现
- 使用数值稳定的 softmax 公式：先减去每行最大值，再计算 `exp(x - max)` 并归一化
- 使用 OpenMP 对 batch 维度并行
- 在 AVX2 下由编译器自动向量化核心循环

## 实验结果

运行benchmark脚本后，你会看到：
- PyTorch baseline的执行时间
- AVX优化版本的执行时间
- 加速比（Speedup）
- 数值误差（Max Difference）

> 具体加速比会依赖于 CPU、编译器版本以及张量尺寸。建议运行 `benchmarks/` 下的脚本，在你的环境中测量实际速度与误差。

## 故障排除

### 编译错误

1. **找不到OpenMP**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install libomp-dev
   
   # macOS
   brew install libomp
   ```

2. **AVX指令集不支持**:
   - 检查CPU是否支持AVX2: `cat /proc/cpuinfo | grep avx2`
   - 如果不支持，代码会自动fallback到标量实现

3. **PyTorch找不到**:
   ```bash
   pip install torch
   ```

### 运行时错误

1. **ImportError**: 确保已编译扩展模块
   ```bash
   python setup.py build_ext --inplace
   ```

2. **数值精度问题**: 如果看到较大的数值误差，检查：
   - 输入数据范围是否合理
   - 是否使用了单精度浮点数（float32）

## 开发说明

### 添加新的Kernel

1. 在`kernels/`目录下创建新的kernel目录
2. 实现标量、AVX和并行版本
3. 在`python/avx_kernels.cpp`中添加Python绑定
4. 在`python/__init__.py`中添加Python接口
5. 添加测试和benchmark脚本

### 代码风格

- C++代码遵循C++17标准
- 使用4空格缩进
- 函数和变量使用snake_case
- 类使用PascalCase

## 许可证

本项目仅用于学习和研究目的。

## 贡献

欢迎提交Issue和Pull Request！

## 参考文献

- [Intel AVX Intrinsics Reference](https://software.intel.com/sites/landingpage/IntrinsicsGuide/)
- [PyTorch C++ Extension](https://pytorch.org/tutorials/advanced/cpp_extension.html)
- [OpenMP Documentation](https://www.openmp.org/)
