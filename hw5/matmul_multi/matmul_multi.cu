#include "matmul_multi.h"
#include "util.h"

#include <stdio.h>
#include <cuda_runtime.h>
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

#define TILE_WIDTH 32
#define MAX_NUM_GPU 4
int num_devices = 0;

__global__ void matmul_kernel(float *A, float *B, float *C, int M, int N,
                              int K) {
  // shared memory for double buffering
  __shared__ float A_tile[2][TILE_WIDTH][TILE_WIDTH];
  __shared__ float B_tile[2][TILE_WIDTH][TILE_WIDTH];

  // calculate global row and column index
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int row = blockIdx.y * TILE_WIDTH + ty;
  int col = blockIdx.x * TILE_WIDTH + tx;

  // register to hold the computed value
  float C_val = 0.0f;

  // number of tile loops
  int num_tiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;

  // pre-load first tile
  if (row < M && tx < K)
    A_tile[0][ty][tx] = A[row * K + tx];
  else
    A_tile[0][ty][tx] = 0.0f;
  
  if (ty < K && col < N)
    B_tile[0][ty][tx] = B[ty * N + col];
  else
    B_tile[0][ty][tx] = 0.0f;

  __syncthreads();

  // simultaneously load and compute tiles
  for (int t = 0; t < num_tiles; ++t) {
    int curr = t % 2;       // for multiplication
    int next = (t + 1) % 2; // for loading

    // load next tile
    if (t + 1 < num_tiles) {
      int A_col = (t + 1) * TILE_WIDTH + tx;
      if (row < M && A_col < K)
        A_tile[next][ty][tx] = A[row * K + A_col];
      else
        A_tile[next][ty][tx] = 0.0f;
      
      int B_row = (t + 1) * TILE_WIDTH + ty;
      if (B_row < K && col < N)
        B_tile[next][ty][tx] = B[B_row * N + col];
      else
        B_tile[next][ty][tx] = 0.0f;
    }

    // compute current tile
    for (int k = 0; k < TILE_WIDTH; ++k) {
      C_val += A_tile[curr][ty][k] * B_tile[curr][k][tx];
    } 

    // synchronize to make sure the loading is done
    __syncthreads();
  }

  // write back the result
  if (row < M && col < N) C[row * N + col] = C_val;
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
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim((N + TILE_WIDTH - 1) / TILE_WIDTH,
                 (M_local + TILE_WIDTH - 1) / TILE_WIDTH,
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
