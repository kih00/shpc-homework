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
static float *in_proj_out_gpu, *BCx_gpu, *Bx_gpu, *conv_out_gpu, *y_pre_gpu, *y_pre_transposed_gpu;

// ============================================================================
// Kernels
// ============================================================================

// GEMM: C = A @ B^T
__global__ void matmul_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, 
                           int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[col * K + k];
        }
        C[row * N + col] = sum;
    }
}

// Transpose: (B, S, C) -> (B, C, S)
// Input: (batch, seq_len, 3*hidden_size)
// Output: (batch, 3*hidden_size, seq_len)
__global__ void transpose_bsc_to_bcs_kernel(const float* __restrict__ input, float* __restrict__ output,
                                            int B, int S, int C) {
    int b = blockIdx.z;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (c < C && s < S) {
        int in_idx = ((b * S + s) * C + c);
        int out_idx = ((b * C + c) * S + s);
        output[out_idx] = input[in_idx];
    }
}

// Split and Mul: Bx = B * x_gate
// Input: BCx (B, 3*H, S)
// Output: Bx (B, H, S)
// B is BCx[:, 0:H, :]
// C is BCx[:, H:2H, :] (saved for later? No, we need C later)
// x_gate is BCx[:, 2H:3H, :]
// We only compute Bx here.
__global__ void split_and_mul_kernel(const float* __restrict__ BCx, float* __restrict__ Bx,
                                     int B, int H, int S) {
    int b = blockIdx.z;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (h < H && s < S) {
        int idx_B = ((b * 3 * H + h) * S + s);
        int idx_X = ((b * 3 * H + (h + 2 * H)) * S + s);
        int idx_out = ((b * H + h) * S + s);
        
        Bx[idx_out] = BCx[idx_B] * BCx[idx_X];
    }
}

// Causal Conv1d
// Input: Bx (B, H, S)
// Weight: (H, K) -> grouped conv, 1 group per channel
// Output: (B, H, S)
// y[b, h, s] = sum(Bx[b, h, s - (K-1) + k] * weight[h, k])
__global__ void causal_conv1d_kernel(const float* __restrict__ input, const float* __restrict__ weight, float* __restrict__ output,
                                     int B, int H, int S, int K) {
    int b = blockIdx.z;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (h < H && s < S) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            int input_pos = s - (K - 1) + k;
            if (input_pos >= 0) {
                int in_idx = ((b * H + h) * S + input_pos);
                int w_idx = h * K + k;
                sum += input[in_idx] * weight[w_idx];
            }
        }
        int out_idx = ((b * H + h) * S + s);
        output[out_idx] = sum;
    }
}

// Mul and Transpose Back
// y_pre = C * conv_out
// C is BCx[:, H:2H, :]
// Output: y_pre_transposed (B, S, H)
__global__ void mul_and_transpose_back_kernel(const float* __restrict__ BCx, const float* __restrict__ conv_out, float* __restrict__ output,
                                              int B, int H, int S) {
    int b = blockIdx.z;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (h < H && s < S) {
        int idx_C = ((b * 3 * H + (h + H)) * S + s);
        int idx_conv = ((b * H + h) * S + s);
        
        float val = BCx[idx_C] * conv_out[idx_conv];
        
        // Transpose: (B, H, S) -> (B, S, H)
        int out_idx = ((b * S + s) * H + h);
        output[out_idx] = val;
    }
}

// ============================================================================
// Initialize / Finalize
// ============================================================================

void conv_initialize(int batch, int seq_len, int hidden_size, int kernel_size,
                     float *conv_weight, float *in_proj_weight, float *out_proj_weight) {
    CHECK_CUDA(cudaMalloc(&x_gpu, batch * seq_len * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&conv_weight_gpu, hidden_size * kernel_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&in_proj_weight_gpu, 3 * hidden_size * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out_proj_weight_gpu, hidden_size * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&output_gpu, batch * seq_len * hidden_size * sizeof(float)));
    
    // Intermediate buffers
    CHECK_CUDA(cudaMalloc(&in_proj_out_gpu, batch * seq_len * 3 * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&BCx_gpu, batch * 3 * hidden_size * seq_len * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&Bx_gpu, batch * hidden_size * seq_len * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&conv_out_gpu, batch * hidden_size * seq_len * sizeof(float)));
    // y_pre_gpu not needed if we fuse mul and transpose
    // But wait, mul_and_transpose_back_kernel writes to y_pre_transposed_gpu directly.
    CHECK_CUDA(cudaMalloc(&y_pre_transposed_gpu, batch * seq_len * hidden_size * sizeof(float)));
    
    // Copy static weights to GPU
    CHECK_CUDA(cudaMemcpy(conv_weight_gpu, conv_weight, hidden_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(in_proj_weight_gpu, in_proj_weight, 3 * hidden_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(out_proj_weight_gpu, out_proj_weight, hidden_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
}

void conv(float *x, float *conv_weight, float *in_proj_weight, float *out_proj_weight,
          float *output, int batch, int seq_len, int hidden_size, int kernel_size) {
    
    int num_tokens = batch * seq_len;
    
    // Copy input data to GPU
    CHECK_CUDA(cudaMemcpy(x_gpu, x, num_tokens * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    
    dim3 block(32, 32);
    
    // 1. Input Projection
    // x: (num_tokens, hidden_size)
    // in_proj: (3*hidden_size, hidden_size)
    // Output: (num_tokens, 3*hidden_size)
    dim3 grid_in((3 * hidden_size + 31) / 32, (num_tokens + 31) / 32);
    matmul_kernel<<<grid_in, block>>>(x_gpu, in_proj_weight_gpu, in_proj_out_gpu, num_tokens, 3 * hidden_size, hidden_size);
    
    // 2. Transpose (B, S, 3H) -> (B, 3H, S)
    dim3 grid_trans((seq_len + 31) / 32, (3 * hidden_size + 31) / 32, batch);
    transpose_bsc_to_bcs_kernel<<<grid_trans, block>>>(in_proj_out_gpu, BCx_gpu, batch, seq_len, 3 * hidden_size);
    
    // 3. Split and Mul: Bx = B * X_gate
    dim3 grid_split((seq_len + 31) / 32, (hidden_size + 31) / 32, batch);
    split_and_mul_kernel<<<grid_split, block>>>(BCx_gpu, Bx_gpu, batch, hidden_size, seq_len);
    
    // 4. Causal Conv1d
    // Bx: (B, H, S)
    // Weight: (H, K)
    // Output: (B, H, S)
    causal_conv1d_kernel<<<grid_split, block>>>(Bx_gpu, conv_weight_gpu, conv_out_gpu, batch, hidden_size, seq_len, kernel_size);
    
    // 5. Mul and Transpose Back
    // y_pre = C * conv_out
    // Output: y_pre_transposed (B, S, H)
    mul_and_transpose_back_kernel<<<grid_split, block>>>(BCx_gpu, conv_out_gpu, y_pre_transposed_gpu, batch, hidden_size, seq_len);
    
    // 6. Output Projection
    // y_pre_transposed: (num_tokens, hidden_size)
    // out_proj: (hidden_size, hidden_size)
    // Output: (num_tokens, hidden_size)
    dim3 grid_out((hidden_size + 31) / 32, (num_tokens + 31) / 32);
    matmul_kernel<<<grid_out, block>>>(y_pre_transposed_gpu, out_proj_weight_gpu, output_gpu, num_tokens, hidden_size, hidden_size);

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(output, output_gpu, num_tokens * hidden_size * sizeof(float), cudaMemcpyDeviceToHost));
}

void conv_finalize() {
    CHECK_CUDA(cudaFree(x_gpu));
    CHECK_CUDA(cudaFree(conv_weight_gpu));
    CHECK_CUDA(cudaFree(in_proj_weight_gpu));
    CHECK_CUDA(cudaFree(out_proj_weight_gpu));
    CHECK_CUDA(cudaFree(output_gpu));
    CHECK_CUDA(cudaFree(in_proj_out_gpu));
    CHECK_CUDA(cudaFree(BCx_gpu));
    CHECK_CUDA(cudaFree(Bx_gpu));
    CHECK_CUDA(cudaFree(conv_out_gpu));
    CHECK_CUDA(cudaFree(y_pre_transposed_gpu));
}
