#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <cmath>

// define some error checking macros
#define cudaErrCheck(stat) \
    { cudaErrCheck_((stat), __FILE__, __LINE__); }
inline void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
    if (stat != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
    }
}

#define cublasErrCheck(stat) \
    { cublasErrCheck_((stat), __FILE__, __LINE__); }
inline void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
    }
}

#define curandErrCheck(stat) \
    { curandErrCheck_((stat), __FILE__, __LINE__); }
inline void curandErrCheck_(curandStatus_t stat, const char *file, int line) {
    if (stat != CURAND_STATUS_SUCCESS) {
        fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
    }
}

// Check if two matrices are equal
inline bool IsMatrixEqual(float *A, float *B, int M, int N, float tol = 1.e-5) {
    for (int i = 0; i < M * N; i++) {
        if (fabs(A[i] - B[i]) / fabs(B[i]) > tol) {
            return false;
        }
    }
    return true;
}

inline void PrintMatrix(float *A, int M, int N, std::ofstream &out) {
    // for (int i = 0; i < M; i++) {
    //     for (int j = 0; j < N; j++) {
    //         printf("%5.3f ", A[i * N + j]);
    //     }
    //     printf("\n");
    // }
    out << std::setprecision(5) << std::fixed; 
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            out << A[i * N + j] << " ";
        }
        out << std::endl;
    }
    out << std::endl << std::endl;
}