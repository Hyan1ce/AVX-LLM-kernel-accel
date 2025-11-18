#ifndef THREADING_H
#define THREADING_H

#include <omp.h>
#include <cstdint>

// Threading utilities
inline int get_num_threads() {
    return omp_get_max_threads();
}

inline void set_num_threads(int n) {
    omp_set_num_threads(n);
}

inline int get_thread_num() {
    return omp_get_thread_num();
}

// Parallel for loop helper
template<typename Func>
void parallel_for(int64_t start, int64_t end, Func func) {
    #pragma omp parallel for
    for (int64_t i = start; i < end; ++i) {
        func(i);
    }
}

#endif // THREADING_H

