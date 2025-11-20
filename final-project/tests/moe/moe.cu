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

// Intermediate buffers
static float *router_logits_gpu;
static int *top_k_indices_gpu;
static float *top_k_weights_gpu;
static int *host_top_k_indices;
static float *host_top_k_weights;
static float *expert_in_gpu;
static float *expert_gate_gpu;
static float *expert_up_gpu;
static float *expert_hidden_gpu;
static float *expert_out_gpu;

// MoE configuration flags (match src/model.cu behavior)
static const float ROUTED_SCALING_FACTOR = 1.0f;
static const bool NORM_TOPK_PROB = true;
static const bool USE_EXPERT_BIAS = true;

void moe_initialize(int batch, int seq_len, int hidden_size, int num_experts, 
                   int num_experts_per_tok, int expert_hidden_size,
                   float *gate, float **expert_w1, float **expert_w2, float **expert_w3, float *expert_bias) {
    g_num_experts = num_experts;
    
    CHECK_CUDA(cudaMalloc(&x_gpu, batch * seq_len * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gate_gpu, num_experts * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&expert_bias_gpu, num_experts * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&output_gpu, batch * seq_len * hidden_size * sizeof(float)));
    
    // Allocate intermediates
    int num_tokens = batch * seq_len;
    CHECK_CUDA(cudaMalloc(&router_logits_gpu, num_tokens * num_experts * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&top_k_indices_gpu, num_tokens * num_experts_per_tok * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&top_k_weights_gpu, num_tokens * num_experts_per_tok * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&host_top_k_indices, num_tokens * num_experts_per_tok * sizeof(int)));
    CHECK_CUDA(cudaMallocHost(&host_top_k_weights, num_tokens * num_experts_per_tok * sizeof(float)));
    
    CHECK_CUDA(cudaMalloc(&expert_in_gpu, num_tokens * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&expert_gate_gpu, num_tokens * expert_hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&expert_up_gpu, num_tokens * expert_hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&expert_hidden_gpu, num_tokens * expert_hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&expert_out_gpu, num_tokens * hidden_size * sizeof(float)));
    
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

// --- Kernels ---

__global__ void matmul_transposed_B_kernel(float *A, float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) sum += A[row * K + i] * B[col * K + i];
        C[row * N + col] = sum;
    }
}

__global__ void router_kernel(float *logits, float *bias, int *top_k_indices, float *top_k_weights, 
                              int num_tokens, int num_experts, int k, bool use_bias) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < num_tokens) {
        float *my_logits = logits + t * num_experts;
        unsigned long long mask = 0;
        
        for (int i = 0; i < k; ++i) {
            float max_score = -1e20f;
            int max_idx = -1;
            float max_weight = 0.0f;
            
            for (int e = 0; e < num_experts; ++e) {
                if (!((mask >> e) & 1)) {
                    float logit = my_logits[e];
                    float weight = 1.0f / (1.0f + expf(-logit));
                    float score = weight;
                    if (use_bias) score += bias[e];
                    
                    if (score > max_score) {
                        max_score = score;
                        max_idx = e;
                        max_weight = weight;
                    }
                }
            }
            
            if (max_idx != -1) {
                top_k_indices[t * k + i] = max_idx;
                top_k_weights[t * k + i] = max_weight;
                mask |= (1ULL << max_idx);
            }
        }
        
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) sum += top_k_weights[t * k + i];
        if (sum > 1e-6f) {
            for (int i = 0; i < k; ++i) top_k_weights[t * k + i] /= sum;
        }
    }
}

__global__ void gather_kernel(float *x, float *expert_in, int *indices, int count, int hidden_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        int token_idx = indices[idx];
        for (int h = 0; h < hidden_size; ++h) {
            expert_in[idx * hidden_size + h] = x[token_idx * hidden_size + h];
        }
    }
}

__global__ void silu_mul_kernel(float *gate, float *up, float *out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = gate[idx];
        float u = up[idx];
        float sig = 1.0f / (1.0f + expf(-g));
        out[idx] = (g * sig) * u;
    }
}

__global__ void scatter_add_kernel(float *expert_out, float *output, int *indices, float *weights, int count, int hidden_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        int token_idx = indices[idx];
        float w = weights[idx];
        for (int h = 0; h < hidden_size; ++h) {
            atomicAdd(&output[token_idx * hidden_size + h], expert_out[idx * hidden_size + h] * w);
        }
    }
}

void moe(float *x, float *gate, float **expert_w1, float **expert_w2, float **expert_w3,
         float *expert_bias, float *output, int batch, int seq_len, int hidden_size, 
         int num_experts, int num_experts_per_tok, int expert_hidden_size) {
    
    int num_tokens = batch * seq_len;
    
    // Initialize output to zero
    CHECK_CUDA(cudaMemset(output_gpu, 0, num_tokens * hidden_size * sizeof(float)));
    
    // Copy input data to GPU
    CHECK_CUDA(cudaMemcpy(x_gpu, x, num_tokens * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // 1. Router Logits
    dim3 block(16, 16);
    dim3 grid_logits((num_experts + 15)/16, (num_tokens + 15)/16);
    matmul_transposed_B_kernel<<<grid_logits, block>>>(x_gpu, gate_gpu, router_logits_gpu, num_tokens, num_experts, hidden_size);
    
    // 2. Router Top-K
    int threads = 256;
    int blocks = (num_tokens + threads - 1) / threads;
    router_kernel<<<blocks, threads>>>(router_logits_gpu, expert_bias_gpu, top_k_indices_gpu, top_k_weights_gpu, 
                                       num_tokens, num_experts, num_experts_per_tok, USE_EXPERT_BIAS);
    
    // 3. Copy to Host
    CHECK_CUDA(cudaMemcpy(host_top_k_indices, top_k_indices_gpu, num_tokens * num_experts_per_tok * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(host_top_k_weights, top_k_weights_gpu, num_tokens * num_experts_per_tok * sizeof(float), cudaMemcpyDeviceToHost));
    
    // 4. Host Scheduling
    std::vector<std::vector<int>> expert_token_indices(num_experts);
    std::vector<std::vector<float>> expert_token_weights(num_experts);
    
    for (int t = 0; t < num_tokens; ++t) {
        for (int k = 0; k < num_experts_per_tok; ++k) {
            int expert_idx = host_top_k_indices[t * num_experts_per_tok + k];
            float weight = host_top_k_weights[t * num_experts_per_tok + k];
            expert_token_indices[expert_idx].push_back(t);
            expert_token_weights[expert_idx].push_back(weight);
        }
    }
    
    // 5. Expert Execution
    // Reuse top_k buffers as scratch
    int *scratch_indices = top_k_indices_gpu;
    float *scratch_weights = top_k_weights_gpu;
    
    for (int e = 0; e < num_experts; ++e) {
        int count = expert_token_indices[e].size();
        if (count == 0) continue;
        
        CHECK_CUDA(cudaMemcpy(scratch_indices, expert_token_indices[e].data(), count * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(scratch_weights, expert_token_weights[e].data(), count * sizeof(float), cudaMemcpyHostToDevice));
        
        int gather_blocks = (count + threads - 1) / threads;
        gather_kernel<<<gather_blocks, threads>>>(x_gpu, expert_in_gpu, scratch_indices, count, hidden_size);
        
        dim3 grid_gate((expert_hidden_size + 15)/16, (count + 15)/16);
        dim3 grid_out((hidden_size + 15)/16, (count + 15)/16);
        
        matmul_transposed_B_kernel<<<grid_gate, block>>>(expert_in_gpu, expert_w1_gpu[e], expert_gate_gpu, count, expert_hidden_size, hidden_size);
        matmul_transposed_B_kernel<<<grid_gate, block>>>(expert_in_gpu, expert_w3_gpu[e], expert_up_gpu, count, expert_hidden_size, hidden_size);
        
        int silu_blocks = (count * expert_hidden_size + threads - 1) / threads;
        silu_mul_kernel<<<silu_blocks, threads>>>(expert_gate_gpu, expert_up_gpu, expert_hidden_gpu, count * expert_hidden_size);
        
        matmul_transposed_B_kernel<<<grid_out, block>>>(expert_hidden_gpu, expert_w2_gpu[e], expert_out_gpu, count, hidden_size, expert_hidden_size);
        
        scatter_add_kernel<<<gather_blocks, threads>>>(expert_out_gpu, output_gpu, scratch_indices, scratch_weights, count, hidden_size);
    }
    
    // Copy output back to host
    CHECK_CUDA(cudaMemcpy(output, output_gpu, num_tokens * hidden_size * sizeof(float), cudaMemcpyDeviceToHost));
}

void moe_finalize() {
    CHECK_CUDA(cudaFree(x_gpu));
    CHECK_CUDA(cudaFree(gate_gpu));
    CHECK_CUDA(cudaFree(expert_bias_gpu));
    CHECK_CUDA(cudaFree(output_gpu));
    
    CHECK_CUDA(cudaFree(router_logits_gpu));
    CHECK_CUDA(cudaFree(top_k_indices_gpu));
    CHECK_CUDA(cudaFree(top_k_weights_gpu));
    CHECK_CUDA(cudaFreeHost(host_top_k_indices));
    CHECK_CUDA(cudaFreeHost(host_top_k_weights));
    CHECK_CUDA(cudaFree(expert_in_gpu));
    CHECK_CUDA(cudaFree(expert_gate_gpu));
    CHECK_CUDA(cudaFree(expert_up_gpu));
    CHECK_CUDA(cudaFree(expert_hidden_gpu));
    CHECK_CUDA(cudaFree(expert_out_gpu));
    
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
