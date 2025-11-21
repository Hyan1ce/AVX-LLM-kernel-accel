from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
import torch
import os
import glob

# Get include directories
torch_include = [
    torch.__path__[0] + '/include',
    torch.__path__[0] + '/include/torch/csrc/api/include',
]
pybind11_include = pybind11.get_include()

# Get library directories
torch_lib = [torch.__path__[0] + '/lib']

ext_modules = [
    Pybind11Extension(
        "avx_kernels_cpp",
        [
            "python/avx_kernels.cpp",
            "kernels/gemm/gemm_avx.cpp",
            "kernels/layernorm/layernorm_avx.cpp",
            "kernels/softmax/softmax_avx.cpp",
        ],
        include_dirs=[
            "kernels",
            "kernels/common",
            "kernels/gemm",
            "kernels/layernorm",
            "kernels/softmax",
        ] + torch_include + [pybind11_include],
        libraries=["torch", "torch_cpu", "torch_python", "c10"],
        library_dirs=torch_lib,
        language='c++',
        cxx_std=17,
        extra_compile_args=[
            '-O3',
            '-march=native',
            '-mavx2',
            '-fopenmp',
            '-Wall',
            '-Wextra',
            f'-D_GLIBCXX_USE_CXX11_ABI={1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0}',
        ],
        extra_link_args=[
            '-fopenmp',
            '-Wl,-rpath,' + torch.__path__[0] + '/lib',
        ],
    ),
]

setup(
    name="avx_llm_kernels",
    version="0.1.0",
    author="",
    description="AVX-optimized kernels for LLM training",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
)

