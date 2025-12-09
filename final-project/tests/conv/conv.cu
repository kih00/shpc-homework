#include "conv.h"
#include "util.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Tunable launch params
constexpr int BLOCK_MM = 32;     // tile size for matmul
constexpr int BLOCK_SEQ = 256;   // threads along sequence dimension

#define CHECK_CUDA(call)                                              \
  do {                                                                \
    cudaError_t status_ = call;                                       \
    if (status_ != cudaSuccess) {                                     \
      fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(status_));                           \
      exit(EXIT_FAILURE);                                             \
    }                                                                 \
  } while (0)

static float *x_gpu, *conv_weight_gpu, *in_proj_weight_gpu, *out_proj_weight_gpu;
static float *in_proj_out_gpu, *output_gpu;
static cudaStream_t stream;

// ============================================================================
// CUDA kernels
// ============================================================================

// Matrix multiply: out[m, n] = a[m, k] @ w[n, k]^T
__global__ void matmul_transposed_kernel(const float* __restrict__ a,
           const float* __restrict__ w, float* __restrict__ out,
           int m, int k, int n) {
  __shared__ float a_tile[BLOCK_MM][BLOCK_MM];
  __shared__ float w_tile[BLOCK_MM][BLOCK_MM];

  int row = blockIdx.y * BLOCK_MM + threadIdx.y;
  int col = blockIdx.x * BLOCK_MM + threadIdx.x;

  float sum = 0.0f;
  int tiles = (k + BLOCK_MM - 1) / BLOCK_MM;
  for (int t = 0; t < tiles; ++t) {
    int a_col = t * BLOCK_MM + threadIdx.x;
    int w_k = t * BLOCK_MM + threadIdx.y;

    a_tile[threadIdx.y][threadIdx.x] = (row < m && a_col < k)
                        ? a[row * k + a_col]
                        : 0.0f;
    w_tile[threadIdx.y][threadIdx.x] = (col < n && w_k < k)
                        ? w[col * k + w_k]
                        : 0.0f;
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < BLOCK_MM; ++i) {
      sum += a_tile[threadIdx.y][i] * w_tile[i][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < m && col < n) out[row * n + col] = sum;
}

// Compute y_pre = C * conv(B * gate), following ShortConv logic
// in_proj_out layout: (bs, 3*H) where bs = batch*seq_len
// conv_weight layout: (H, K)
// y_pre_out layout: (batch, H, seq_len)
__global__ void shortconv_kernel(const float* __restrict__ in_proj_out,
         const float* __restrict__ conv_w, float* __restrict__ y_pre_out,
         int batch, int seq_len, int hidden, int kernel) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= seq_len) return;

  extern __shared__ float shared_w[];
  for (int k_idx = threadIdx.x; k_idx < kernel; k_idx += blockDim.x) {
    shared_w[k_idx] = conv_w[h * kernel + k_idx];
  }
  __syncthreads();

    int base = (b * seq_len + s) * (3 * hidden) + h;

    // B at +0, C at +hidden, gate at +2*hidden
    // B = in_proj_out[base];
    // gate = in_proj_out[base + 2 * hidden];
    float C = in_proj_out[base + hidden];

    float conv_sum = 0.0f;
    for (int k = 0; k < kernel; k++) {
      int in_pos = s - (kernel - 1) + k;
      if (in_pos >= 0) {
        int base_in = (b * seq_len + in_pos) * (3 * hidden) + h;
        float B_in = in_proj_out[base_in];
        float gate_in = in_proj_out[base_in + 2 * hidden];
        conv_sum += B_in * gate_in * shared_w[k];
      }
    }
    float y = C * conv_sum;
    y_pre_out[(b * hidden + h) * seq_len + s] = y; // (B,H,S) packed
}

// Flatten (B,H,S) -> (B*S, H) for output projection
__global__ void flatten_kernel(const float* in, float* out,
                 int batch, int hidden, int seq_len) {
    int b = blockIdx.z;
    int s = blockIdx.y;
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= hidden) return;
    float val = in[(b * hidden + h) * seq_len + s];
    out[(b * seq_len + s) * hidden + h] = val;
}

void conv_initialize(int batch, int seq_len, int hidden_size, int kernel_size,
                     float *conv_weight, float *in_proj_weight, float *out_proj_weight) {
    CHECK_CUDA(cudaMalloc(&x_gpu, batch * seq_len * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&conv_weight_gpu, hidden_size * kernel_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&in_proj_weight_gpu, 3 * hidden_size * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out_proj_weight_gpu, hidden_size * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&in_proj_out_gpu, batch * seq_len * 3 * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&output_gpu, batch * seq_len * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaStreamCreate(&stream));
    
    // Copy static weights to GPU
    CHECK_CUDA(cudaMemcpyAsync(conv_weight_gpu, conv_weight, hidden_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(in_proj_weight_gpu, in_proj_weight, 3 * hidden_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(out_proj_weight_gpu, out_proj_weight, hidden_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice, stream));
}

void conv(float *x, float *conv_weight, float *in_proj_weight, float *out_proj_weight,
          float *output, int batch, int seq_len, int hidden_size, int kernel_size) {
    
    // Copy input data to GPU
    CHECK_CUDA(cudaMemcpyAsync(x_gpu, x, batch * seq_len * hidden_size * sizeof(float), cudaMemcpyHostToDevice, stream));

    int bs = batch * seq_len;

    // 1) in_proj: (bs, hidden) @ (3*hidden, hidden)^T -> (bs, 3*hidden)
    dim3 block_mm(BLOCK_MM, BLOCK_MM);
    dim3 grid_mm((3 * hidden_size + block_mm.x - 1) / block_mm.x,
                 (bs + block_mm.y - 1) / block_mm.y);
    matmul_transposed_kernel<<<grid_mm, block_mm, 0, stream>>>(x_gpu, in_proj_weight_gpu, in_proj_out_gpu,
                            bs, hidden_size, 3 * hidden_size);

    // 2) ShortConv core: compute y_pre = C * conv(B * gate)
    dim3 grid_conv((seq_len + BLOCK_SEQ - 1) / BLOCK_SEQ, hidden_size, batch);
    shortconv_kernel<<<grid_conv, BLOCK_SEQ, 0, stream>>>(in_proj_out_gpu, conv_weight_gpu, output_gpu,
                           batch, seq_len, hidden_size, kernel_size);

    // 3) Flatten (B,H,S) -> (B*S, H)
    dim3 grid_flat((hidden_size + BLOCK_SEQ - 1) / BLOCK_SEQ, seq_len, batch);
    flatten_kernel<<<grid_flat, BLOCK_SEQ, 0, stream>>>(output_gpu, x_gpu, batch, hidden_size, seq_len);

    // 4) Out proj: (bs, hidden) @ (hidden, hidden)^T -> (bs, hidden)
    dim3 grid_out((hidden_size + block_mm.x - 1) / block_mm.x,
            (bs + block_mm.y - 1) / block_mm.y);
    matmul_transposed_kernel<<<grid_out, block_mm, 0, stream>>>(x_gpu, out_proj_weight_gpu, output_gpu,
                                                     bs, hidden_size, hidden_size);

    // Copy result to host
    CHECK_CUDA(cudaMemcpyAsync(output, output_gpu, batch * seq_len * hidden_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
}

void conv_finalize() {
    CHECK_CUDA(cudaFree(x_gpu));
    CHECK_CUDA(cudaFree(in_proj_out_gpu));
    CHECK_CUDA(cudaFree(conv_weight_gpu));
    CHECK_CUDA(cudaFree(in_proj_weight_gpu));
    CHECK_CUDA(cudaFree(out_proj_weight_gpu));
    CHECK_CUDA(cudaFree(output_gpu));
}
