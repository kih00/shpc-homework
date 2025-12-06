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
    
    // TODO: Implement MoE operation
    
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
