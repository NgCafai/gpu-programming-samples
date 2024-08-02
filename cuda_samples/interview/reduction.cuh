#pragma once

#include <cuda_runtime.h>
#include <float.h>
#include "dtype_utils.cuh"

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

/*
 * Sums `val` across all threads in a warp.
 */
template <typename T> 
__inline__ __device__ T WarpReduceSum(T val) {
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

/*
 * Picks the maximum `val` across all threads in a warp.
 */
template <typename T> 
__inline__ __device__ T WarpReduceMax(T val) {
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        val = max(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

/*
 * Sums `val` across all threads in a block.
 *
 * Warning: the return value is only valid for thread 0.
 */
template <typename T> 
__inline__ __device__ T BlockReduceSum(T val) {
    const int thread_num = blockDim.x * blockDim.y * blockDim.z;
    const int warp_num = thread_num / WARP_SIZE;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    __shared__ T buf[WARP_SIZE];
    __shared__ T result_broadcast;

    // reduce on each warp
    val = WarpReduceSum(val);
    if (lane_id == 0) {
        buf[warp_id] = val;
    }
    __syncthreads();

    val = (tid < warp_num) ? buf[tid] : T(0);
    if (warp_id == 0) {
        val = WarpReduceSum(val);
        if (tid == 0) {
            result_broadcast = val;
        }
    }
    __syncthreads();
    return result_broadcast;
}


/*
 * Picks out the maximum `val` across all threads in a block.
 */
template <typename T> 
__inline__ __device__ T BlockReduceMax(T val) {
    const int thread_num = blockDim.x * blockDim.y * blockDim.z;
    const int warp_num = thread_num / WARP_SIZE;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    __shared__ T buf[WARP_SIZE];
    __shared__ T result_broadcast;

    // reduce on each warp
    val = WarpReduceMax(val);
    if (lane_id == 0) {
        buf[warp_id] = val;
    }
    __syncthreads();

    val = (tid < warp_num) ? buf[tid] : -Inf<T>();
    if (warp_id == 0) {
        val = WarpReduceMax(val);
        if (tid == 0) {
            result_broadcast = val;
        }
    }
    __syncthreads();
    return result_broadcast;
}