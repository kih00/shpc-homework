#include "moe.h"
#include "util.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <vector>
#include <cmath>

#define CHECK_CUDA(call)                                              \
  do {                                                                \
    cudaError_t status_ = call;                                       \
    if (status_ != cudaSuccess) {                                     \
      fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(status_));                           \
      exit(EXIT_FAILURE);                                             \
    }                                                                 \
  } while (0)

static float *x_gpu, *gate_gpu, *expert_bias_gpu, *output_gpu;
static float **expert_w1_gpu, **expert_w2_gpu, **expert_w3_gpu;
static float **expert_w1_gpu_ptrs, **expert_w2_gpu_ptrs, **expert_w3_gpu_ptrs;
static int g_num_experts = 0;

// launch params
constexpr int BLOCK_MM = 16; // tile size for matmul
constexpr int BLOCK_TOPK = 32; // threads per token for top-k
constexpr int BLOCK_OUT = 256; // threads for output hidden dim work

// MoE configuration flags (match src/model.cu behavior)
static const float ROUTED_SCALING_FACTOR = 1.0f;
static const bool NORM_TOPK_PROB = true;
static const bool USE_EXPERT_BIAS = true;

// ============================================================================
// CUDA kernels
// ============================================================================

// Matrix multiply: out[m, n] = x[m, k] @ w[n, k]^T
__global__ void matmul_transposed_kernel(
    const float* x, const float* w, float* out,
    int m, int k, int n) {

    __shared__ float x_tile[BLOCK_MM][BLOCK_MM];
    __shared__ float w_tile[BLOCK_MM][BLOCK_MM];

    int row = blockIdx.y * BLOCK_MM + threadIdx.y;
    int col = blockIdx.x * BLOCK_MM + threadIdx.x;

    float sum = 0.0f;
    int tiles = (k + BLOCK_MM - 1) / BLOCK_MM;
    for (int t = 0; t < tiles; ++t) {
        int a_k = t * BLOCK_MM + threadIdx.x;
        int w_k = t * BLOCK_MM + threadIdx.y;

        x_tile[threadIdx.y][threadIdx.x] = 0.0f;
        w_tile[threadIdx.y][threadIdx.x] = 0.0f;
        if (row < m && a_k < k)
            x_tile[threadIdx.y][threadIdx.x] = x[row * k + a_k];
        if (col < n && w_k < k)
            w_tile[threadIdx.y][threadIdx.x] = w[col * k + w_k];
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < BLOCK_MM; ++i)
            sum += x_tile[threadIdx.y][i] * w_tile[i][threadIdx.x];
        __syncthreads();
    }

    if (row < m && col < n)
        out[row * n + col] = sum;
}

// Compute sigmoid(router_logits) + optional bias, then select top-k per token
__global__ void topk_kernel(
    const float* router_logits, const float* expert_bias,
    int* topk_indices, float* topk_weights,
    int num_experts, int num_experts_per_tok) {

    int t = blockIdx.x; // token index
    if (threadIdx.x != 0) return; // only one thread does work per token

    // Keep small local buffers for selected results
    // Assume num_experts_per_tok is small (e.g., 4)
    const int MAX_K = 32; // safety cap
    int sel_idx[MAX_K];
    float sel_w[MAX_K];

    for (int k = 0; k < num_experts_per_tok; k++) {
        float best_score = -1e30f;
        int best_idx = 0;

        for (int j = 0; j < num_experts; j++) {
            // skip already selected experts
            bool used = false;
            for (int kk = 0; kk < k; kk++) {
                if (sel_idx[kk] == j) { used = true; break; }
            }
            if (used) continue;

            float logit = router_logits[t * num_experts + j];
            float w = 1.0f / (1.0f + expf(-logit));
            float score = USE_EXPERT_BIAS ? (w + expert_bias[j]) : w;
            if (score > best_score) {
                best_score = score;
                best_idx = j;
            }
        }

        sel_idx[k] = best_idx;
        float routed_w = 1.0f / (1.0f + expf(-router_logits[t * num_experts + best_idx]));
        sel_w[k] = routed_w;
    }

    // Normalize selected weights if needed and apply scaling
    if (NORM_TOPK_PROB) {
        float sum = 0.0f;
        for (int k = 0; k < num_experts_per_tok; k++) sum += sel_w[k];
        float inv = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
        for (int k = 0; k < num_experts_per_tok; k++) sel_w[k] *= inv;
    }

    for (int k = 0; k < num_experts_per_tok; k++) {
        topk_indices[t * num_experts_per_tok + k] = sel_idx[k];
        topk_weights[t * num_experts_per_tok + k] = sel_w[k] * ROUTED_SCALING_FACTOR;
    }
}

// Compute hidden = silu(x @ w1^T) * (x @ w3^T) for selected (token, expert)
__global__ void compute_hidden_kernel(
    const float* x, float* const* w1_ptrs, float* const* w3_ptrs,
    const int* topk_indices, const float* topk_weights, float* hidden_buf,
    int hidden_size, int expert_hidden_size, int num_experts_per_tok) {

    int t = blockIdx.x; // token
    int k = blockIdx.y; // top-k slot
    int e = topk_indices[t * num_experts_per_tok + k];
    float weight = topk_weights[t * num_experts_per_tok + k];

    const float* x_vec = x + t * hidden_size;
    const float* w1 = w1_ptrs[e];
    const float* w3 = w3_ptrs[e];
    float* hidden = hidden_buf + (t * num_experts_per_tok + k) * expert_hidden_size;

    __shared__ float x_tile[BLOCK_MM];

    for (int j = threadIdx.x; j < expert_hidden_size; j += blockDim.x) {
        float g1 = 0.0f;
        float g3 = 0.0f;
        for (int tile = 0; tile < hidden_size; tile += BLOCK_MM) {
            int idx = tile + threadIdx.x;
            if (threadIdx.x < BLOCK_MM) {
                x_tile[threadIdx.x] = (idx < hidden_size) ? x_vec[idx] : 0.0f;
            }
            __syncthreads();

            int tile_len = min(BLOCK_MM, hidden_size - tile);
            const float* w1_row = w1 + j * hidden_size + tile;
            const float* w3_row = w3 + j * hidden_size + tile;
            #pragma unroll
            for (int i = 0; i < BLOCK_MM; i++) {
                if (i < tile_len) {
                    float xv = x_tile[i];
                    g1 += xv * w1_row[i];
                    g3 += xv * w3_row[i];
                }
            }
            __syncthreads();
        }
        float silu = g1 / (1.0f + expf(-g1));
        hidden[j] = silu * g3 * weight;
    }
}

// Project hidden back to hidden_size and accumulate to output
__global__ void project_down_kernel(
    const float* hidden_buf, float* const* w2_ptrs,
    const int* topk_indices, float* output,
    int hidden_size, int expert_hidden_size, int num_experts_per_tok) {

    int h = blockIdx.x * blockDim.x + threadIdx.x; // output dim
    if (h >= hidden_size) return;

    int t = blockIdx.y; // token
    int k = blockIdx.z; // top-k slot
    int e = topk_indices[t * num_experts_per_tok + k];
    const float* w2 = w2_ptrs[e];
    const float* hidden = hidden_buf + (t * num_experts_per_tok + k) * expert_hidden_size;

    float sum = 0.0f;
    const float* w2_row = w2 + h * expert_hidden_size;
    for (int j = 0; j < expert_hidden_size; j++)
        sum += hidden[j] * w2_row[j];

    atomicAdd(output + t * hidden_size + h, sum);
}

void moe_initialize(int batch, int seq_len, int hidden_size, int num_experts, 
                   int num_experts_per_tok, int expert_hidden_size,
                   float *gate, float **expert_w1, float **expert_w2, float **expert_w3, float *expert_bias) {
    g_num_experts = num_experts;
    
    CHECK_CUDA(cudaMalloc(&x_gpu, batch * seq_len * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gate_gpu, num_experts * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&expert_bias_gpu, num_experts * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&output_gpu, batch * seq_len * hidden_size * sizeof(float)));
    
    // Allocate expert weights
    expert_w1_gpu = (float**)malloc(num_experts * sizeof(float*));
    expert_w2_gpu = (float**)malloc(num_experts * sizeof(float*));
    expert_w3_gpu = (float**)malloc(num_experts * sizeof(float*));
    
    for (int i = 0; i < num_experts; i++) {
        CHECK_CUDA(cudaMalloc(&expert_w1_gpu[i], expert_hidden_size * hidden_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&expert_w2_gpu[i], hidden_size * expert_hidden_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&expert_w3_gpu[i], expert_hidden_size * hidden_size * sizeof(float)));
    }
    
    // Allocate device array of pointers
    CHECK_CUDA(cudaMalloc(&expert_w1_gpu_ptrs, num_experts * sizeof(float*)));
    CHECK_CUDA(cudaMalloc(&expert_w2_gpu_ptrs, num_experts * sizeof(float*)));
    CHECK_CUDA(cudaMalloc(&expert_w3_gpu_ptrs, num_experts * sizeof(float*)));
    
    CHECK_CUDA(cudaMemcpy(expert_w1_gpu_ptrs, expert_w1_gpu, num_experts * sizeof(float*), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(expert_w2_gpu_ptrs, expert_w2_gpu, num_experts * sizeof(float*), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(expert_w3_gpu_ptrs, expert_w3_gpu, num_experts * sizeof(float*), cudaMemcpyHostToDevice));
    
    // Copy static weights to GPU
    CHECK_CUDA(cudaMemcpy(gate_gpu, gate, num_experts * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(expert_bias_gpu, expert_bias, num_experts * sizeof(float), cudaMemcpyHostToDevice));
    
    for (int i = 0; i < num_experts; i++) {
        CHECK_CUDA(cudaMemcpy(expert_w1_gpu[i], expert_w1[i], expert_hidden_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(expert_w2_gpu[i], expert_w2[i], hidden_size * expert_hidden_size * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(expert_w3_gpu[i], expert_w3[i], expert_hidden_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void moe(float *x, float *gate, float **expert_w1, float **expert_w2, float **expert_w3,
         float *expert_bias, float *output, int batch, int seq_len, int hidden_size, 
         int num_experts, int num_experts_per_tok, int expert_hidden_size) {
    
    int num_tokens = batch * seq_len;
    
    // Initialize output to zero
    memset(output, 0, num_tokens * hidden_size * sizeof(float));
    
    // Copy input data to GPU
    CHECK_CUDA(cudaMemcpy(x_gpu, x, num_tokens * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(output_gpu, 0, num_tokens * hidden_size * sizeof(float)));
    
    // Temporary buffers
    float *router_logits_gpu = nullptr;
    float *hidden_buf_gpu = nullptr;
    float *topk_weights_gpu = nullptr;
    int *topk_indices_gpu = nullptr;

    CHECK_CUDA(cudaMalloc(&router_logits_gpu, num_tokens * num_experts * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&topk_indices_gpu, num_tokens * num_experts_per_tok * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&topk_weights_gpu, num_tokens * num_experts_per_tok * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&hidden_buf_gpu, (size_t)num_tokens * num_experts_per_tok * expert_hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(hidden_buf_gpu, 0, (size_t)num_tokens * num_experts_per_tok * expert_hidden_size * sizeof(float)));

    // 1) router logits: (num_tokens, hidden) @ (num_experts, hidden)^T -> (num_tokens, num_experts)
    dim3 block_mm(BLOCK_MM, BLOCK_MM);
    dim3 grid_mm((num_experts + block_mm.x - 1) / block_mm.x,
                 (num_tokens + block_mm.y - 1) / block_mm.y);
    matmul_transposed_kernel<<<grid_mm, block_mm>>>(
        x_gpu, gate_gpu, router_logits_gpu,
        num_tokens, hidden_size, num_experts);

    // 2) select top-k experts per token
    topk_kernel<<<num_tokens, BLOCK_TOPK>>>(
        router_logits_gpu, expert_bias_gpu,
        topk_indices_gpu, topk_weights_gpu,
        num_experts, num_experts_per_tok);

    // 3) compute hidden for selected experts
    dim3 grid_hidden(num_tokens, num_experts_per_tok);
    compute_hidden_kernel<<<grid_hidden, BLOCK_MM>>>(
        x_gpu, expert_w1_gpu_ptrs, expert_w3_gpu_ptrs,
        topk_indices_gpu, topk_weights_gpu, hidden_buf_gpu,
        hidden_size, expert_hidden_size, num_experts_per_tok);

    // 4) project down and accumulate outputs
    dim3 grid_out((hidden_size + BLOCK_OUT - 1) / BLOCK_OUT,
                  num_tokens, num_experts_per_tok);
    project_down_kernel<<<grid_out, BLOCK_OUT>>>(
        hidden_buf_gpu, expert_w2_gpu_ptrs, topk_indices_gpu, output_gpu,
        hidden_size, expert_hidden_size, num_experts_per_tok);

    // Clean up temporaries
    CHECK_CUDA(cudaFree(router_logits_gpu));
    CHECK_CUDA(cudaFree(topk_indices_gpu));
    CHECK_CUDA(cudaFree(topk_weights_gpu));
    CHECK_CUDA(cudaFree(hidden_buf_gpu));
    
    // Copy output back to host
    CHECK_CUDA(cudaMemcpy(output, output_gpu, num_tokens * hidden_size * sizeof(float), cudaMemcpyDeviceToHost));
}

void moe_finalize() {
    CHECK_CUDA(cudaFree(x_gpu));
    CHECK_CUDA(cudaFree(gate_gpu));
    CHECK_CUDA(cudaFree(expert_bias_gpu));
    CHECK_CUDA(cudaFree(output_gpu));
    
    for (int i = 0; i < g_num_experts; i++) {
        CHECK_CUDA(cudaFree(expert_w1_gpu[i]));
        CHECK_CUDA(cudaFree(expert_w2_gpu[i]));
        CHECK_CUDA(cudaFree(expert_w3_gpu[i]));
    }
    
    CHECK_CUDA(cudaFree(expert_w1_gpu_ptrs));
    CHECK_CUDA(cudaFree(expert_w2_gpu_ptrs));
    CHECK_CUDA(cudaFree(expert_w3_gpu_ptrs));
    
    free(expert_w1_gpu);
    free(expert_w2_gpu);
    free(expert_w3_gpu);
}
