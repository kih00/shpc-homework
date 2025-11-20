#include "matmul_single.h"
#include "util.h"

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CALL(f)                                                           \
  {                                                                            \
    cudaError_t err = (f);                                                     \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at [%s:%d] %d %s\n", __FILE__, __LINE__,     \
              err, cudaGetErrorString(err));                                   \
      exit(1);                                                                 \
    }                                                                          \
  }

#define TILE_WIDTH 32

__global__ void matmul_kernel(float *A, float *B, float *C, int M, int N, int K) {

  // shared memory for double buffering
  __shared__ float A_tile[2][TILE_WIDTH][TILE_WIDTH];
  __shared__ float B_tile[2][TILE_WIDTH][TILE_WIDTH];

  // calculate global row and column indices
  int tx = threadIdx.x; // column within thread block
  int ty = threadIdx.y; // row within thread block
  int row = blockIdx.y * TILE_WIDTH + ty; // global row index
  int col = blockIdx.x * TILE_WIDTH + tx; // global column index

  // register to hold the computed value
  float C_val = 0;

  // number of tile loops
  int num_tiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;

  // pre-load the first tile into shared memory
  if (row < M && tx < K)
      A_tile[0][ty][tx] = A[row * K + tx];
  else
      A_tile[0][ty][tx] = 0;

  if (ty < K && col < N)
      B_tile[0][ty][tx] = B[ty * N + col];
  else
      B_tile[0][ty][tx] = 0;

  __syncthreads();

  // simultaneously load and compute tiles with double buffering
  for (int t = 0; t < num_tiles; ++t) {
    int curr = t % 2;       // for multiplication
    int next = (t + 1) % 2; // for loading

    // 1. pre-load (t+1)th tile into "next" shared memory
    if (t < num_tiles - 1) { 
      int A_col = (t + 1) * TILE_WIDTH + tx;
      if (row < M && A_col < K) {
        A_tile[next][ty][tx] = A[row * K + A_col];
      } else {
        A_tile[next][ty][tx] = 0.0f;
      }

      int B_row = (t + 1) * TILE_WIDTH + ty;
      if (B_row < K && col < N) {
        B_tile[next][ty][tx] = B[B_row * N + col];
      } else {
        B_tile[next][ty][tx] = 0.0f;
      }
    }

    __syncthreads();

    // 2. compute (t)th tile with "current" shared memory
    for (int k = 0; k < TILE_WIDTH; ++k) {
      C_val += A_tile[curr][ty][k] * B_tile[curr][k][tx];
    }

    // 3. synchronize to make sure loading is done
    __syncthreads();
  }

  // write back the computed value
  if (row < M && col < N) C[row * N + col] = C_val;
}


// Array of device (GPU) pointers
static float *a_d;
static float *b_d;
static float *c_d;

static bool data_transferred = false;

void matmul(const float *A, const float *B, float *C, int M, int N, int K) {

  // Upload A and B matrix to every GPU
  if (!data_transferred) {
    CUDA_CALL(cudaMemcpy(a_d, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(b_d, B, K * N * sizeof(float), cudaMemcpyHostToDevice));
    data_transferred = true;
  }

  // Launch kernel on every GPU
  dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 gridDim((N + TILE_WIDTH - 1) / TILE_WIDTH,
               (M + TILE_WIDTH - 1) / TILE_WIDTH,
               1);

  matmul_kernel<<<gridDim, blockDim>>>(a_d, b_d, c_d, M, N, K);

  CUDA_CALL(cudaDeviceSynchronize());

  // Download C matrix from GPUs
  CUDA_CALL(cudaMemcpy(C, c_d, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CUDA_CALL(cudaDeviceSynchronize());
}

void matmul_initialize(int M, int N, int K) {
  
  int num_devices;
  // Only root process do something
  CUDA_CALL(cudaGetDeviceCount(&num_devices));

  if (num_devices <= 0) {
    printf("No CUDA device found. Aborting\n");
    exit(1);
  }

  // Allocate device memory 
  CUDA_CALL(cudaMalloc(&a_d, M * K * sizeof(float)));
  CUDA_CALL(cudaMalloc(&b_d, K * N * sizeof(float)));
  CUDA_CALL(cudaMalloc(&c_d, M * N * sizeof(float)));
}

void matmul_finalize() {

  // Free GPU memory
  CUDA_CALL(cudaFree(a_d));
  CUDA_CALL(cudaFree(b_d));
  CUDA_CALL(cudaFree(c_d));
}
