# AVX-LLM-Kernel-Accel

一个使用AVX指令集和多线程优化的Transformer训练/微调程序，支持在CPU上高效运行: https://github.com/Hyan1ce/AVX-LLM-kernel-accel

## 项目概述

本项目实现了三个关键kernel的AVX优化版本：
- **GEMM** (General Matrix Multiply): 矩阵乘法核心操作
- **LayerNorm**: 层归一化（包含forward和backward）
- **Softmax**: 注意力机制中的softmax操作

所有kernel都提供了：
- 标量实现（baseline）
- AVX2优化实现（单线程）
- AVX2优化实现（多线程并行）

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

### 安装依赖

```bash
# 安装PyTorch (CPU版本)
pip install torch

# 安装pybind11
pip install pybind11

# 安装OpenMP (Ubuntu/Debian)
sudo apt-get install libomp-dev

# 安装OpenMP (macOS)
brew install libomp
```

## 编译和安装

### 方法1: 使用setup.py (推荐)

```bash
# 进入项目目录
cd AVX-LLM-kernel-accel

# 编译并安装Python扩展
python setup.py build_ext --inplace

# 或者安装到系统
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
# 测试GEMM
python tests/test_gemm.py

# 测试LayerNorm
python tests/test_layernorm.py

# 测试Softmax
python tests/test_softmax.py
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

# GEMM示例
A = torch.randn(128, 512)
B = torch.randn(512, 256)
C = gemm(A, B, use_parallel=True)

# LayerNorm示例
input_tensor = torch.randn(32, 512)
gamma = torch.ones(512)
beta = torch.zeros(512)
output, mean, var = layernorm(input_tensor, gamma, beta, use_parallel=True)

# Softmax示例
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

- **GEMM**: 使用OpenMP对矩阵行进行并行化
- **LayerNorm**: 对batch维度进行并行化
- **Softmax**: 对batch维度进行并行化

### 性能调优建议

1. **线程数设置**: 默认使用所有可用CPU核心，可通过环境变量控制：
   ```bash
   export OMP_NUM_THREADS=8  # 使用8个线程
   ```

2. **内存对齐**: 代码中已处理内存对齐，确保AVX指令高效执行

3. **批量大小**: 较大的batch size通常能获得更好的性能

## 实现细节

### GEMM实现
- 使用AVX2的`_mm256_fmadd_ps`进行融合乘加操作
- 8元素向量化处理
- 支持alpha和beta参数

### LayerNorm实现
- Forward: 使用AVX计算均值和方差，向量化归一化
- Backward: 实现完整的梯度计算，支持gamma和beta的梯度

### Softmax实现
- 使用AVX查找最大值
- 向量化exp计算和归一化
- 数值稳定性优化（减去最大值）

## 实验结果

运行benchmark脚本后，你会看到：
- PyTorch baseline的执行时间
- AVX优化版本的执行时间
- 加速比（Speedup）
- 数值误差（Max Difference）

典型结果：
- GEMM: 2-5x加速（取决于矩阵大小）
- LayerNorm: 1.5-3x加速
- Softmax: 1.5-2.5x加速

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
