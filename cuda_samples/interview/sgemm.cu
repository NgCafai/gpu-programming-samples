#include <cuda_runtime.h>

#define WARP_SIZE 32;
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(data) (*reinterpret_cast<float4 *>(&(data)))

/*
 * C = A * B, row-major
 *
 * Parameters:
 *     A: the input matrix A of shape (M, K), row-major
 *     B: the input matrix B of shape (K, N), row-major
 *     C: the output matrix C of shape (M, N), row-major
 *
 * Default size of tiling:
 *     thread block tile: m 128, n 128, k 8
 *     warp tile:         m 32,  n 64,  k 8
 *     thread tile:       m 8,   n 8,   k 8
 *     thread fragment:
 *         matrix A: 8x1 FP32
 *         matrix B: 1x8 FP32
 *
 * ----------------------------------------------------------------
 * thread block tile map:
 *
 *                                128
 *                    --|---------------------|
 *             B_tile  8|                     |
 *                    --|---------------------|
 *
 *  A_tile   | 8 |      |    64    |
 *         --|---|    --|----------|----------|
 *           |   |    32|  warp_0  |  warp_1  |
 *           |   |    --|----------|----------|
 *           |   |      |  warp_2  |  warp_3  |
 *        128|   |      |----------|----------|
 *           |   |      |  warp_4  |  warp_5  |
 *           |   |      |----------|----------|
 *           |   |      |  warp_6  |  warp_7  |
 *         --|---|      |----------|----------|
 *
 * ----------------------------------------------------------------
 * warp tile map:
 *
 * 'z' thread map to avoid LDS.128 shared memory broadcast limitation.
 *
 *              |              32               ||
 *     B_frag --|---|---|---|---|---|---|---|---||---|---|---|---|---|---|---|---|
 *             1|///|   |   |   |   |   |   |   ||///|   |   |   |   |   |   |   |
 *            --|---|---|---|---|---|---|---|---||---|---|---|---|---|---|---|---|
 * A_frag       | 4 |                           ||
 *    | 1 |                                     ||
 *  --|---|--   |---|---|---|---|---|---|---|---||---|---------------------------|
 *    |///|4    |t0 |t2 |t4 |t6 |t8 |t10|t12|t14||t0 |                           |
 *    |---|--   |---|---|---|---|---|---|---|---||---|                           |
 *    |   |     |t1 |t3 |t5 |t7 |t9 |t11|t13|t15||                               |
 *  16|---|     |---|---|---|---|---|---|---|---||                               |
 *    |   |     |t16|t18|t20|t22|t24|t26|t28|t30||                               |
 *    |---|     |---|---|---|---|---|---|---|---||                               |
 *    |   |     |t17|t19|t21|t23|t25|t27|t29|t31||                               |
 *  ==|===|=====|===|===|===|===|===|===|===|===||===|============================
 *    |///|     |t0 |                           ||t0 |                           |
 *    |---|     |---|                           ||---|                           |
 *    |   |     |                               ||                               |
 *    |---|     |                               ||                               |
 *    |   |     |                               ||                               |
 *    |---|     |                               ||                               |
 *    |   |     |                               ||                               |
 *    |---|     |-------------------------------||-------------------------------|
 *
 */
template <const int BM = 128, const int BN = 128, const int BK = 8, const int TM = 8,
          const int TN = 8>
__global__ void Sgemm(int M, int N, int K, float *A, float *B, float *C) {
    // block index
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int tid = threadIdx.x;

    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    constexpr int THREAD_NUM_PER_BLOCK = (BM / TM) * (BN / TN);

    // for block tile
    __shared__ float A_smem[BK][BM];
    __shared__ float B_smem[BK][BN];

    // for thread tile
    float A_frag[TM];
    float B_frag[TN];
    // accumulator for C
    float accum[TM][TN] = {0.0f};

    // Calculate index for: 1) read from A_smem and B_smem to register;
    //                      2) write to C;
    // warp tile: 4 * 8 threads, 32 * 64 elements
    const int t_N_offset = (lane_id / 2) % 8;
    const int t_M_offset = (lane_id / 16) * 2 + (lane_id % 2);

    constexpr int WARP_NUM_N = BM / TM / 8;
    const int A_smem_load_offset = (warp_id / WARP_NUM_N) * 32 + t_M_offset * 4;
    const int B_smem_load_offset = (warp_id % WARP_NUM_N) * 64 + t_N_offset * 4;

    // Calculate index for loading data from global mem to shared mem
    // Assume that BM * BK == BN * BK == THREAD_NUM_PER_BLOCK * 4
    constexpr int A_COPY_THREAD_PER_ROW = BK / 4;
    constexpr int B_COPY_THREAD_PER_ROW = BN / 4;

    const int A_copy_row_start = tid / A_COPY_THREAD_PER_ROW;
    const int A_copy_col_start = (tid % A_COPY_THREAD_PER_ROW) * 4;
    const int B_copy_row_start = tid / B_COPY_THREAD_PER_ROW;
    const int B_copy_col_start = (tid % B_COPY_THREAD_PER_ROW) * 4;

    // move A, B
    A += by * BM * K;
    B += bx * BN;

    for (int block_tile_idx = 0; block_tile_idx < K; block_tile_idx += BK) {
        // Copy data from global to shared
        float4 A_ldg_reg;
        A_ldg_reg = FLOAT4(A[OFFSET(A_copy_row_start, A_copy_col_start + block_tile_idx, K)]);
        // Transpose
        A_smem[A_copy_col_start][A_copy_row_start] = A_ldg_reg.x;
        A_smem[A_copy_col_start + 1][A_copy_row_start] = A_ldg_reg.y;
        A_smem[A_copy_col_start + 2][A_copy_row_start] = A_ldg_reg.z;
        A_smem[A_copy_col_start + 3][A_copy_row_start] = A_ldg_reg.w;

        FLOAT4(B_smem[B_copy_row_start][B_copy_col_start]) =
            FLOAT4(B[OFFSET(B_copy_row_start + block_tile_idx, B_copy_col_start, N)]);
        __syncthreads();

        // Compute in a outer product way
#pragma unroll
        for (int frag_k = 0; frag_k < BK; ++frag_k) {
            // load from shared to registerp
            FLOAT4(A_frag[0]) = FLOAT4(A_smem[frag_k][A_smem_load_offset]);
            FLOAT4(A_frag[4]) = FLOAT4(A_smem[frag_k][A_smem_load_offset + 4 * 4]);

            FLOAT4(B_frag[0]) = FLOAT4(B_smem[frag_k][B_smem_load_offset]);
            FLOAT4(B_frag[4]) = FLOAT4(B_smem[frag_k][B_smem_load_offset + 8 * 4]);

#pragma unroll
            for (int idx_m = 0; idx_m < TM; ++idx_m) {
#pragma unroll
                for (int idx_n = 0; idx_n < TN; ++idx_n) {
                    accum[idx_m][idx_n] += A_frag[frag_k][idx] * B_frag[frag][idx_n];
                }
            }
        }
        __syncthreads();
    }

    // Store the result to global
    C += OFFSET(by * BM + A_smem_load_offset, bx * BN + B_smem_load_offset, N);
#pragma unroll
    for (int idx_m = 0; idx_m < TM; idx_m += 4) {
#pragma unroll
        for (int idx_n = 0; idx_n < TN; idx_n += 4) {
#pragma unroll
            for (int i = 0; i < 4; i++) {
                FLOAT4(C[OFFSET(idx_m * 4, idx_n * 8, N)]) = FLOAT4(accum[idx_m + i][idx_n]);
            }
        }
    }
}
