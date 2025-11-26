#include "matmul.h"
#include "util.h"

#include <stdio.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#define CUDA_CALL(f)                                                           \
  {                                                                            \
    cudaError_t err = (f);                                                     \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at [%s:%d] %d %s\n", __FILE__, __LINE__,     \
              err, cudaGetErrorString(err));                                   \
      exit(1);                                                                 \
    }                                                                          \
  }

#define TILE 16
#define CHUNK 8
#define NUM_STAGE 2
#define MAX_NUM_GPU 4
int num_devices = 0;

__global__ void matmul_kernel(float *A, float *B, float *C, int M, int N,
                              int K) {
  __shared__ float A_tile[NUM_STAGE][TILE*CHUNK][CHUNK];
  __shared__ float B_tile[NUM_STAGE][CHUNK][TILE*CHUNK];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int block_row = blockIdx.y * TILE * CHUNK;
  int block_col = blockIdx.x * TILE * CHUNK;

  int row = blockIdx.y * TILE * CHUNK + ty * CHUNK;
  int col = blockIdx.x * TILE * CHUNK + tx * CHUNK;

  float C_val[CHUNK][CHUNK];
#pragma unroll
  for (int i = 0; i < CHUNK; ++i) {
#pragma unroll
    for (int j = 0; j < CHUNK; ++j) {
      C_val[i][j] = 0.0f;
    }
  }

  int num_chunks = (K + CHUNK - 1) / CHUNK;
  if (num_chunks == 0) return;

  int tid = ty * blockDim.x + tx;
  int total_threads = blockDim.x * blockDim.y;

  int curr = 0;
  int next = 1;
  for (int i=tid; i<TILE*CHUNK*CHUNK; i+=total_threads) {
    int local_row_A = i / CHUNK;
    int local_col_A = i % CHUNK;
    int local_row_B = i / (TILE * CHUNK);
    int local_col_B = i % (TILE * CHUNK);

    int global_row_A = block_row + local_row_A;
    int global_col_A = local_col_A;
    int global_row_B = local_row_B;
    int global_col_B = block_col + local_col_B;

    float val_A = 0.0f, val_B = 0.0f;
    if (global_row_A < M && global_col_A < K)
      val_A = A[global_row_A * K + global_col_A];

    if (global_row_B < K && global_col_B < N)
      val_B = B[global_row_B * N + global_col_B];

    A_tile[curr][local_row_A][local_col_A] = val_A;
    B_tile[curr][local_row_B][local_col_B] = val_B;
  }

  __syncthreads();

  // double buffering
  for (int t = 0; t < num_chunks; ++t) {
    // load next tile
    if (t + 1 < num_chunks) {
      int next_k = (t + 1) * CHUNK;

      for (int i=tid; i<TILE*CHUNK*CHUNK; i+=total_threads) {
        int local_row_A = i / CHUNK;
        int local_col_A = i % CHUNK;
        int local_row_B = i / (TILE * CHUNK);
        int local_col_B = i % (TILE * CHUNK);

        int global_row_A = block_row + local_row_A;
        int global_col_A = next_k + local_col_A;
        int global_row_B = next_k + local_row_B;
        int global_col_B = block_col + local_col_B;

        float val_A = 0.0f, val_B = 0.0f;
        if (global_row_A < M && global_col_A < K)
          val_A = A[global_row_A * K + global_col_A];

        if (global_row_B < K && global_col_B < N)
          val_B = B[global_row_B * N + global_col_B];

        A_tile[next][local_row_A][local_col_A] = val_A;
        B_tile[next][local_row_B][local_col_B] = val_B;
      }
    }

    float A_reg[CHUNK];
#pragma unroll
    for (int k = 0; k < CHUNK; ++k) {
#pragma unroll
      for (int i = 0; i < CHUNK; ++i) {
        A_reg[i] = A_tile[curr][ty * CHUNK + i][k];
      }
#pragma unroll
      for (int j = 0; j < CHUNK; ++j) {
        float B_val = B_tile[curr][k][tx * CHUNK + j];
#pragma unroll
        for (int i = 0; i < CHUNK; ++i) {
          C_val[i][j] += A_reg[i] * B_val;
        }
      }
    }

    __syncthreads();
    curr ^= 1;
    next ^= 1;
  }

  for (int i = 0; i < CHUNK; ++i) {
    int global_row = row + i;
    if (global_row >= M) continue;
    for (int j = 0; j < CHUNK; ++j) {
      int global_col = col + j;
      if (global_col < N) {
        C[global_row * N + global_col] = C_val[i][j];
      }
    }
  }
}

// Array of device (GPU) pointers
static float *a_d[MAX_NUM_GPU];
static float *b_d[MAX_NUM_GPU];
static float *c_d[MAX_NUM_GPU];
static int Mbegin[MAX_NUM_GPU], Mend[MAX_NUM_GPU];

static bool data_transferred[MAX_NUM_GPU] = {false};
static cudaStream_t streams[MAX_NUM_GPU];

void matmul(const float *A, const float *B, float *C, int M, int N, int K) {

  // #pragma omp parallel for num_threads(num_devices)
  for (int i = 0; i < num_devices; i++) {

    CUDA_CALL(cudaSetDevice(i));
    int M_local = Mend[i] - Mbegin[i];

    // Upload A and B matrix to every GPU
    if (!data_transferred[i]) {
      CUDA_CALL(cudaMemcpyAsync(a_d[i], A + Mbegin[i] * K,
                          M_local * K * sizeof(float),
                          cudaMemcpyHostToDevice, streams[i]));
      CUDA_CALL(cudaMemcpyAsync(b_d[i], B,
                          K * N * sizeof(float),
                          cudaMemcpyHostToDevice, streams[i]));
      data_transferred[i] = true;
    }

    // Launch kernel on every GPU
    dim3 blockDim(TILE, TILE, 1);
    dim3 gridDim((N + (TILE * CHUNK) - 1) / (TILE * CHUNK),
                 (M_local + TILE * CHUNK - 1) / (TILE * CHUNK),
                 1);

    matmul_kernel<<<gridDim, blockDim, 0, streams[i]>>>(a_d[i], b_d[i], c_d[i], M_local, N, K);

    // Download C matrix from GPUs
    CUDA_CALL(cudaMemcpyAsync(C + Mbegin[i] * N, c_d[i],
                         M_local * N * sizeof(float),
                         cudaMemcpyDeviceToHost, streams[i]));
  }

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaDeviceSynchronize());
  }
}

void matmul_initialize(int M, int N, int K) {

  CUDA_CALL(cudaGetDeviceCount(&num_devices));

  printf("Using %d devices\n", num_devices);
  for (int i = 0; i < num_devices; i++) {
    cudaDeviceProp prop;
    CUDA_CALL(cudaGetDeviceProperties(&prop, i));

    // Try printing more detailed information here
    printf("GPU %d: %s\n", i, prop.name);
  }

  if (num_devices <= 0) {
    printf("No CUDA device found. Aborting\n");
    exit(1);
  }

  // Setup problem size for each GPU
  for (int i = 0; i < num_devices; i++) {
    Mbegin[i] = (M / num_devices) * i;
    Mend[i] = (M / num_devices) * (i + 1);
  }
  Mend[num_devices - 1] = M;

  // Allocate device memory for each GPU
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaSetDevice(i));
    CUDA_CALL(cudaMalloc(&a_d[i], (Mend[i] - Mbegin[i]) * K * sizeof(float)));
    CUDA_CALL(cudaMalloc(&b_d[i], K * N * sizeof(float)));
    CUDA_CALL(cudaMalloc(&c_d[i], (Mend[i] - Mbegin[i]) * N * sizeof(float)));
    CUDA_CALL(cudaStreamCreate(&streams[i]));
  }
}

void matmul_finalize() {

  // Free all GPU memory
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaFree(a_d[i]));
    CUDA_CALL(cudaFree(b_d[i]));
    CUDA_CALL(cudaFree(c_d[i]));
    CUDA_CALL(cudaStreamDestroy(streams[i]));
  }
}
