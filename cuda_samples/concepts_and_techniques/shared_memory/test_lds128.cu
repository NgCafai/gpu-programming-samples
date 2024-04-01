#include <cstdint>
#include <cstdio>

__global__ void smem_1(uint32_t *a) {
  __shared__ uint32_t smem[128];
  uint32_t tid = threadIdx.x;
  for (int i = 0; i < 4; i++) {
    smem[i * 32 + tid] = tid;
  }
  __syncthreads();
  if (tid == 15 || tid == 16) {
    reinterpret_cast<uint4 *>(a)[tid] =
        reinterpret_cast<const uint4 *>(smem)[4];
  }
}

__global__ void smem_2(uint32_t *a) {
  __shared__ uint32_t smem[128];
  uint32_t tid = threadIdx.x;
  for (int i = 0; i < 4; i++) {
    smem[i * 32 + tid] = tid;
  }
  __syncthreads();
  if (tid == 0 || tid == 15) {
    reinterpret_cast<uint4 *>(a)[tid] =
        reinterpret_cast<const uint4 *>(smem)[4];
  }
}

__global__ void smem_3(uint32_t *a) {
  __shared__ uint32_t smem[128];
  uint32_t tid = threadIdx.x;
  for (int i = 0; i < 4; i++) {
    smem[i * 32 + tid] = tid;
  }
  __syncthreads();
  reinterpret_cast<uint4 *>(a)[tid] = reinterpret_cast<const uint4 *>(
      smem)[(tid / 8) * 2 + ((tid % 8) / 2) % 2];
}

__global__ void smem_4(uint32_t *a) {
  __shared__ uint32_t smem[128];
  uint32_t tid = threadIdx.x;
  for (int i = 0; i < 4; i++) {
    smem[i * 32 + tid] = tid;
  }
  __syncthreads();
  uint32_t addr;
  if (tid < 16) {
    addr = (tid / 8) * 2 + ((tid % 8) / 2) % 2;
  } else {
    addr = (tid / 8) * 2 + ((tid % 8) % 2);
  }
  reinterpret_cast<uint4 *>(a)[tid] =
      reinterpret_cast<const uint4 *>(smem)[addr];
  // printf("tid: %d, addr: %d\n", tid, addr);
}

__global__ void smem_5(uint32_t *a) {
  __shared__ uint32_t smem[128];
  uint32_t tid = threadIdx.x;
  for (int i = 0; i < 4; i++) {
    smem[i * 32 + tid] = tid;
  }
  __syncthreads();
  uint32_t addr = (tid / 16) * 4 + (tid % 16) / 8 + (tid % 8) / 4 * 8;
  reinterpret_cast<uint4 *>(a)[tid] =
      reinterpret_cast<const uint4 *>(smem)[addr];
  printf("tid: %d, addr: %d\n", tid, addr);
}


int main() {
  uint32_t *d_a;
  cudaMalloc(&d_a, sizeof(uint32_t) * 128);
  // smem_1<<<1, 32>>>(d_a);
  // smem_2<<<1, 32>>>(d_a);
  // smem_3<<<1, 32>>>(d_a);
  // smem_4<<<1, 32>>>(d_a);
  smem_5<<<1, 32>>>(d_a);
  cudaFree(d_a);
  cudaDeviceSynchronize();
  return 0;
}