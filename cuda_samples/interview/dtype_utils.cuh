#pragma once

#include <cuda_runtime.h>

// Get maximum value
template <typename T> __device__ __inline__ T Inf();

template <> 
__inline__ __device__ float Inf<float>() { 
    return FLT_MAX; 
}

template <> 
__inline__ __device__ double Inf<double>() { 
    return DBL_MAX; 
}

// Vectorized load / store
#define VECTORIZED(ptr, VEC_TYPE) (*reinterpret_cast<VEC_TYPE*>(ptr))

template<typename T, int VEC_SIZE>
__inline__ __device__ void VecAssign(T* src, T* dst);

template<>
__inline__ __device__ void VecAssign<float, 4>(float* src, float* dst) {
    VECTORIZED(dst, float4) = VECTORIZED(src, float4);
}
