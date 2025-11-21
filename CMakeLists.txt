cmake_minimum_required(VERSION 3.18)
project(AVX_LLM_Kernels)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mavx2 -fopenmp")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/common
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/gemm
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/layernorm
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/softmax
)

# Source files
set(GEMM_SOURCES
    kernels/gemm/gemm_avx.cpp
)

set(LAYERNORM_SOURCES
    kernels/layernorm/layernorm_avx.cpp
)

set(SOFTMAX_SOURCES
    kernels/softmax/softmax_avx.cpp
)

# Libraries
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    link_libraries(${OpenMP_CXX_LIBRARIES})
endif()

# Test executable (optional - commented out as test_gemm.cpp doesn't exist)
# Uncomment and create test_gemm.cpp if you want to build a standalone test
# add_executable(test_gemm test_gemm.cpp)
# target_link_libraries(test_gemm ${OpenMP_CXX_LIBRARIES})
# target_sources(test_gemm PRIVATE ${GEMM_SOURCES})

