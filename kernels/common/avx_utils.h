#ifndef AVX_UTILS_H
#define AVX_UTILS_H

#include <immintrin.h>
#include <cstdint>
#include <cstring>

// AVX feature detection
inline bool has_avx2() {
    return __builtin_cpu_supports("avx2");
}

inline bool has_avx512() {
    return __builtin_cpu_supports("avx512f");
}

// Memory alignment helpers
inline void* aligned_malloc(size_t size, size_t alignment = 32) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return nullptr;
    }
    return ptr;
}

inline void aligned_free(void* ptr) {
    free(ptr);
}

// AVX load/store helpers
inline __m256 load_8_float(const float* ptr) {
    return _mm256_load_ps(ptr);
}

inline __m256 loadu_8_float(const float* ptr) {
    return _mm256_loadu_ps(ptr);
}

inline void store_8_float(float* ptr, __m256 val) {
    _mm256_store_ps(ptr, val);
}

inline void storeu_8_float(float* ptr, __m256 val) {
    _mm256_storeu_ps(ptr, val);
}

// Horizontal sum of 8 floats
inline float horizontal_sum_8(__m256 v) {
    __m128 vlow = _mm256_extractf128_ps(v, 0);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    vlow = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_shuffle_ps(vlow, vlow, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 sums = _mm_add_ps(vlow, shuf);
    shuf = _mm_movehdup_ps(sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

// Horizontal max of 8 floats
inline float horizontal_max_8(__m256 v) {
    __m128 vlow = _mm256_extractf128_ps(v, 0);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    vlow = _mm_max_ps(vlow, vhigh);
    __m128 shuf = _mm_shuffle_ps(vlow, vlow, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 maxs = _mm_max_ps(vlow, shuf);
    shuf = _mm_movehdup_ps(maxs);
    maxs = _mm_max_ss(maxs, shuf);
    return _mm_cvtss_f32(maxs);
}

#endif // AVX_UTILS_H

