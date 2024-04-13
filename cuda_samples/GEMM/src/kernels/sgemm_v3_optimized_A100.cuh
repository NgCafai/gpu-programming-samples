#pragma once

#include <cooperative_groups.h>
#include <cuda/barrier>

// cal offset from row col and ld , in row-major matrix, ld is the width of the matrix
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// transfer float4
#define FETCH_FLOAT4(data) (reinterpret_cast<float4 *>(&(data))[0])

template <const int BLOCK_ITEMS_M, const int BLOCK_ITEMS_N, const int BLOCK_ITEMS_K,
          const int A_TILE_ROW_STRIDE, const int B_TILE_ROW_STRIDE, typename BarrierType>
__device__ __inline__ void LoadFromGmem(int N, int K, float *A, float *B,
                                        float A_smem[BLOCK_ITEMS_K][BLOCK_ITEMS_M],
                                        float B_smem[BLOCK_ITEMS_K][BLOCK_ITEMS_N], int A_tile_row,
                                        int A_tile_col, int B_tile_row, int B_tile_col,
                                        BarrierType &barrier)
{
    // load into A_smem
#pragma unroll
    for (int offset = 0; offset + A_TILE_ROW_STRIDE <= BLOCK_ITEMS_M; offset += A_TILE_ROW_STRIDE)
    {
        cuda::memcpy_async(&A_smem[A_tile_col][A_tile_row + offset],
                           &A[OFFSET(A_tile_row + offset, A_tile_col, K)], sizeof(float), barrier);
    }

    // load into B_smem
#pragma unroll
    for (int offset = 0; offset + B_TILE_ROW_STRIDE <= BLOCK_ITEMS_K; offset += B_TILE_ROW_STRIDE)
    {
        cuda::memcpy_async(&B_smem[B_tile_row + offset][B_tile_col],
                           &B[OFFSET(B_tile_row + offset, B_tile_col, N)], sizeof(float), barrier);
    }
}

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
template <const int BLOCK_ITEMS_M,  // Height in rows of a block-wide tile in matrix C
          const int BLOCK_ITEMS_N,  // Width in columns of a block-wide tile in matrix C
          const int BLOCK_ITEMS_K,  // Width in columns of a block-wide tile in matrix A and \
                                       height in rows of a block-wide tile in matrix B
          const int THREAD_ITEMS_M, // Height in rows of a thread tile in C
          const int THREAD_ITEMS_N  // Width in columns of a thread tile in C
          >
__global__ void SgemmV3(int M, int N, int K, float *A, float *B, float *C)
{
    // block index - use 2D gridDim
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // thread index - use 1D blockDim, and map threads to correpsonding data positions
    const int tid = threadIdx.x;

    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int THREAD_NUM_PER_BLOCK =
        (BLOCK_ITEMS_M / THREAD_ITEMS_M) * (BLOCK_ITEMS_N / THREAD_ITEMS_N);

    // map the threads in a warp tile to a 4x8 layout
    const int mma_tid_x = (lane_id / 2) % 8;
    const int mma_tid_y = (lane_id / 16) * 2 + (lane_id % 2);

    // map the warps in a block tile to a Nx2 layout
    const int A_smem_load_offset = (warp_id / 2) * 32 + mma_tid_y * 4;
    const int B_smem_load_offset = (warp_id % 2) * 64 + mma_tid_x * 4;

    // allocate shared memory for block tile
    __shared__ float A_smem[2][BLOCK_ITEMS_K][BLOCK_ITEMS_M];
    __shared__ float B_smem[2][BLOCK_ITEMS_K][BLOCK_ITEMS_N];

    // allocate register for A and B for thread tile
    float A_frag[2][THREAD_ITEMS_M];
    float B_frag[2][THREAD_ITEMS_N];
    // allocate register for C for thread tile
    float accum[THREAD_ITEMS_M][THREAD_ITEMS_N] = {0.0f};

    // create barriers for synchronizing cuda::memcpy_async()
    // __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> barriers[2];
    using BarrierType = cuda::barrier<cuda::thread_scope::thread_scope_block>;
    __shared__ char barrier_space[sizeof(BarrierType) * 2];
    BarrierType *barriers = reinterpret_cast<BarrierType *>(barrier_space);
    auto group = cooperative_groups::this_thread_block();
    if (group.thread_rank() == 0)
    {
        init(&barriers[0], group.size());
        init(&barriers[1], group.size());
    }
    __syncthreads();

    // -----------------------------------------------------------------------------
    // calculate the indices that this thread will load from globa memory into shared
    // memory, and each thread will load 1 elements(4 bytes) at each step

    // thread number required to load data for one row of <block tile> of A and B
    constexpr int A_TILE_THREAD_PER_ROW = BLOCK_ITEMS_K;
    constexpr int B_TILE_THREAD_PER_ROW = BLOCK_ITEMS_N;

    // row and col index in <block tile> of A and B that this thread is responsible for
    const int A_tile_row = tid / A_TILE_THREAD_PER_ROW;
    const int A_tile_col = tid % A_TILE_THREAD_PER_ROW;
    const int B_tile_row = tid / B_TILE_THREAD_PER_ROW;
    const int B_tile_col = tid % B_TILE_THREAD_PER_ROW;

    // stride is used to move to the next target row
    constexpr int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW;
    constexpr int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW;

    // -----------------------------------------------------------------------------
    // main logic

    // move block tile to beginning of A's row and B's column
    A += by * BLOCK_ITEMS_M * K;
    B += bx * BLOCK_ITEMS_N;

    // --------------------------------------------------------
    // load first block tile from global mem to shared mem
    LoadFromGmem<BLOCK_ITEMS_M, BLOCK_ITEMS_N, BLOCK_ITEMS_K, A_TILE_ROW_STRIDE, B_TILE_ROW_STRIDE>(
        N, K, A, B, A_smem[0], B_smem[0], A_tile_row, A_tile_col, B_tile_row, B_tile_col,
        barriers[0]);
    barriers[0].arrive_and_wait();

    // --------------------------------------------------------
    // load first fragment from shared memory to register for computation
    // Note: the thread layout in a warp is 4x8
    FETCH_FLOAT4(A_frag[0][0]) = FETCH_FLOAT4(A_smem[0][0][A_smem_load_offset]);
    FETCH_FLOAT4(A_frag[0][4]) = FETCH_FLOAT4(A_smem[0][0][A_smem_load_offset + 4 * 4]);

    FETCH_FLOAT4(B_frag[0][0]) = FETCH_FLOAT4(B_smem[0][0][B_smem_load_offset]);
    FETCH_FLOAT4(B_frag[0][4]) = FETCH_FLOAT4(B_smem[0][0][B_smem_load_offset + 8 * 4]);

    int write_stage_idx = 1;
    int block_tile_idx = 0;
    do
    {
        // --------------------------------------------------------
        // load next block tile from global mem to shared mem asynchronously

        // next block tile index
        block_tile_idx += BLOCK_ITEMS_K;

        if (block_tile_idx < K)
        {
            LoadFromGmem<BLOCK_ITEMS_M, BLOCK_ITEMS_N, BLOCK_ITEMS_K, A_TILE_ROW_STRIDE,
                         B_TILE_ROW_STRIDE>(N, K, A + block_tile_idx, B + block_tile_idx * N,
                                            A_smem[write_stage_idx], B_smem[write_stage_idx],
                                            A_tile_row, A_tile_col, B_tile_row, B_tile_col,
                                            barriers[write_stage_idx]);
        }

        // --------------------------------------------------------
        // read data from current block tile(already on shared memory) to register for
        // computation

        int read_stage_idx = write_stage_idx ^ 1;
#pragma unroll
        for (int frag_k = 0; frag_k < BLOCK_ITEMS_K - 1; ++frag_k)
        {
            // load next fragment from shared memory to register
            // load A fragment
            FETCH_FLOAT4(A_frag[(frag_k + 1) % 2][0]) =
                FETCH_FLOAT4(A_smem[read_stage_idx][frag_k + 1][A_smem_load_offset]);
            FETCH_FLOAT4(A_frag[(frag_k + 1) % 2][4]) =
                FETCH_FLOAT4(A_smem[read_stage_idx][frag_k + 1][A_smem_load_offset + 4 * 4]);
            // load B fragment
            FETCH_FLOAT4(B_frag[(frag_k + 1) % 2][0]) =
                FETCH_FLOAT4(B_smem[read_stage_idx][frag_k + 1][B_smem_load_offset]);
            FETCH_FLOAT4(B_frag[(frag_k + 1) % 2][4]) =
                FETCH_FLOAT4(B_smem[read_stage_idx][frag_k + 1][B_smem_load_offset + 8 * 4]);

            // compute on current fragment in a outter product way
#pragma unroll
            for (int thread_y = 0; thread_y < THREAD_ITEMS_M; ++thread_y)
            {
#pragma unroll
                for (int thread_x = 0; thread_x < THREAD_ITEMS_N; ++thread_x)
                {
                    accum[thread_y][thread_x] +=
                        A_frag[frag_k % 2][thread_y] * B_frag[frag_k % 2][thread_x];
                }
            }
        }

        // --------------------------------------------------------
        // wait for the next block tile to be loaded
        if (block_tile_idx < K)
        {
            barriers[write_stage_idx].arrive_and_wait();

            // switch read and write stage
            __syncthreads();
            write_stage_idx ^= 1;
        }

        // --------------------------------------------------------
        // load the first fragment of next block tile from shared memory to register
        FETCH_FLOAT4(A_frag[BLOCK_ITEMS_K % 2][0]) =
            FETCH_FLOAT4(A_smem[read_stage_idx ^ 1][0][A_smem_load_offset]);
        FETCH_FLOAT4(A_frag[BLOCK_ITEMS_K % 2][4]) =
            FETCH_FLOAT4(A_smem[read_stage_idx ^ 1][0][A_smem_load_offset + 4 * 4]);
        FETCH_FLOAT4(B_frag[BLOCK_ITEMS_K % 2][0]) =
            FETCH_FLOAT4(B_smem[read_stage_idx ^ 1][0][B_smem_load_offset]);
        FETCH_FLOAT4(B_frag[BLOCK_ITEMS_K % 2][4]) =
            FETCH_FLOAT4(B_smem[read_stage_idx ^ 1][0][B_smem_load_offset + 8 * 4]);

        // --------------------------------------------------------
        // compute on the last fragment of current block tile
#pragma unroll
        for (int thread_y = 0; thread_y < THREAD_ITEMS_M; ++thread_y)
        {
#pragma unroll
            for (int thread_x = 0; thread_x < THREAD_ITEMS_N; ++thread_x)
            {
                accum[thread_y][thread_x] += A_frag[(BLOCK_ITEMS_K - 1) % 2][thread_y] *
                                             B_frag[(BLOCK_ITEMS_K - 1) % 2][thread_x];
            }
        }

    } while (block_tile_idx < K);

    // --------------------------------------------------------
    // store the result to global memory
    // that is, store the 4 4x4 thread sub tiles in accum to C
    int C_row_start = by * BLOCK_ITEMS_M + A_smem_load_offset;
    int C_col_start = bx * BLOCK_ITEMS_N + B_smem_load_offset;
    C += OFFSET(C_row_start, C_col_start, N);

    for (int idx_m = 0; idx_m < THREAD_ITEMS_M; idx_m += 4)
    {
        for (int idx_n = 0; idx_n < THREAD_ITEMS_N; idx_n += 4)
        {
            for (int i = 0; i < 4; i++)
            {
                FETCH_FLOAT4(C[OFFSET(idx_m * 4 + i, // row
                                      idx_n * 8,     // col
                                      N)]) = FETCH_FLOAT4(accum[idx_m + i][idx_n]);
            }
        }
    }

    // Done!!!
}