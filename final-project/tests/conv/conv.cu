#include "conv.h"
#include "util.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK_CUDA(call)                                              \
  do {                                                                \
    cudaError_t status_ = call;                                       \
    if (status_ != cudaSuccess) {                                     \
      fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(status_));                           \
      exit(EXIT_FAILURE);                                             \
    }                                                                 \
  } while (0)

static float *x_gpu, *conv_weight_gpu, *in_proj_weight_gpu, *out_proj_weight_gpu, *output_gpu;

// Intermediate buffers
static float *in_proj_out_gpu;
static float *Bx_gpu;
static float *C_out_gpu;
static float *conv_out_gpu;
static float *y_pre_gpu;

void conv_initialize(int batch, int seq_len, int hidden_size, int kernel_size,
                     float *conv_weight, float *in_proj_weight, float *out_proj_weight) {
    CHECK_CUDA(cudaMalloc(&x_gpu, batch * seq_len * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&conv_weight_gpu, hidden_size * kernel_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&in_proj_weight_gpu, 3 * hidden_size * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out_proj_weight_gpu, hidden_size * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&output_gpu, batch * seq_len * hidden_size * sizeof(float)));
    
    // Allocate intermediates
    CHECK_CUDA(cudaMalloc(&in_proj_out_gpu, batch * seq_len * 3 * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&Bx_gpu, batch * hidden_size * seq_len * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&C_out_gpu, batch * hidden_size * seq_len * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&conv_out_gpu, batch * hidden_size * seq_len * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&y_pre_gpu, batch * seq_len * hidden_size * sizeof(float)));
    
    // Copy static weights to GPU
    CHECK_CUDA(cudaMemcpy(conv_weight_gpu, conv_weight, hidden_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(in_proj_weight_gpu, in_proj_weight, 3 * hidden_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(out_proj_weight_gpu, out_proj_weight, hidden_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
}

// --- Kernels ---

__global__ void matmul_kernel(float *A, float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) sum += A[row * K + i] * B[i * N + col];
        C[row * N + col] = sum;
    }
}

__global__ void pre_conv_gating_kernel(float *in_proj_out, float *Bx, float *C_out, int batch, int seq_len, int hidden_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seq_len * hidden_size;
    if (idx < total) {
        int h = idx % hidden_size;
        int rem = idx / hidden_size;
        int s = rem % seq_len;
        int b = rem / seq_len;
        
        int in_base = b * (seq_len * 3 * hidden_size) + s * (3 * hidden_size);
        float val_B = in_proj_out[in_base + h];
        float val_C = in_proj_out[in_base + hidden_size + h];
        float val_X = in_proj_out[in_base + 2 * hidden_size + h];
        
        int out_idx = b * (hidden_size * seq_len) + h * seq_len + s;
        Bx[out_idx] = val_B * val_X;
        C_out[out_idx] = val_C;
    }
}

__global__ void depthwise_conv1d_kernel(float *Bx, float *W, float *ConvOut, int batch, int seq_len, int hidden_size, int kernel_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * hidden_size * seq_len;
    if (idx < total) {
        int s = idx % seq_len;
        int rem = idx / seq_len;
        int h = rem % hidden_size;
        int b = rem / hidden_size;
        
        int channel_base = b * (hidden_size * seq_len) + h * seq_len;
        float sum = 0.0f;
        for (int k = 0; k < kernel_size; ++k) {
            int input_s = s - (kernel_size - 1) + k;
            if (input_s >= 0 && input_s < seq_len) {
                sum += Bx[channel_base + input_s] * W[h * kernel_size + k];
            }
        }
        ConvOut[idx] = sum;
    }
}

__global__ void post_conv_gating_kernel(float *ConvOut, float *C, float *Y_pre, int batch, int seq_len, int hidden_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * hidden_size * seq_len;
    if (idx < total) {
        int s = idx % seq_len;
        int rem = idx / seq_len;
        int h = rem % hidden_size;
        int b = rem / hidden_size;
        
        int in_idx = b * (hidden_size * seq_len) + h * seq_len + s;
        float val = ConvOut[in_idx] * C[in_idx];
        
        int out_idx = b * (seq_len * hidden_size) + s * hidden_size + h;
        Y_pre[out_idx] = val;
    }
}

void conv(float *x, float *conv_weight, float *in_proj_weight, float *out_proj_weight,
          float *output, int batch, int seq_len, int hidden_size, int kernel_size) {
    
    // Copy input data to GPU
    // Copy input data to GPU
    CHECK_CUDA(cudaMemcpy(x_gpu, x, batch * seq_len * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // 1. Input Projection
    dim3 block(16, 16);
    dim3 grid_proj((3 * hidden_size + 15)/16, (batch * seq_len + 15)/16);
    matmul_kernel<<<grid_proj, block>>>(x_gpu, in_proj_weight_gpu, in_proj_out_gpu, batch * seq_len, 3 * hidden_size, hidden_size);
    
    // 2. Pre-Conv Gating
    int total = batch * seq_len * hidden_size;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    pre_conv_gating_kernel<<<blocks, threads>>>(in_proj_out_gpu, Bx_gpu, C_out_gpu, batch, seq_len, hidden_size);
    
    // 3. Depthwise Conv
    depthwise_conv1d_kernel<<<blocks, threads>>>(Bx_gpu, conv_weight_gpu, conv_out_gpu, batch, seq_len, hidden_size, kernel_size);
    
    // 4. Post-Conv Gating
    post_conv_gating_kernel<<<blocks, threads>>>(conv_out_gpu, C_out_gpu, y_pre_gpu, batch, seq_len, hidden_size);
    
    // 5. Output Projection
    dim3 grid_out((hidden_size + 15)/16, (batch * seq_len + 15)/16);
    matmul_kernel<<<grid_out, block>>>(y_pre_gpu, out_proj_weight_gpu, output_gpu, batch * seq_len, hidden_size, hidden_size);

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(output, output_gpu, batch * seq_len * hidden_size * sizeof(float), cudaMemcpyDeviceToHost));
}

void conv_finalize() {
    CHECK_CUDA(cudaFree(x_gpu));
    CHECK_CUDA(cudaFree(conv_weight_gpu));
    CHECK_CUDA(cudaFree(in_proj_weight_gpu));
    CHECK_CUDA(cudaFree(out_proj_weight_gpu));
    CHECK_CUDA(cudaFree(output_gpu));
    
    CHECK_CUDA(cudaFree(in_proj_out_gpu));
    CHECK_CUDA(cudaFree(Bx_gpu));
    CHECK_CUDA(cudaFree(C_out_gpu));
    CHECK_CUDA(cudaFree(conv_out_gpu));
    CHECK_CUDA(cudaFree(y_pre_gpu));
}
