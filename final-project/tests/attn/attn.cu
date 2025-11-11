#include "attn.h"
#include "util.h"

#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK_CUDA(call)                                              \
  do {                                                                \
    cudaError_t status_ = call;                                       \
    if (status_ != cudaSuccess) {                                     \
      fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(status_));                           \
      exit(EXIT_FAILURE);                                             \
    }                                                                 \
  } while (0)

static float *x_gpu, *cos_gpu, *sin_gpu;
static float *q_proj_gpu, *k_proj_gpu, *v_proj_gpu, *o_proj_gpu;
static float *q_norm_gpu, *k_norm_gpu, *output_gpu;
static float *q_proj_out_gpu, *k_proj_out_gpu, *v_proj_out_gpu;
static float *q_normed_gpu, *k_normed_gpu;
static float *q_transposed_gpu, *k_transposed_gpu, *k_repeated_gpu, *v_transposed_gpu;
static float *attn_scores_gpu, *attn_out_gpu, *attn_out_transposed_gpu;

void attn_initialize(int batch, int seq_len, int num_heads, int head_dim, int num_kv_heads,
                     float *cos, float *sin, float *q_proj, float *k_proj, 
                     float *v_proj, float *o_proj, float *q_norm, float *k_norm) {
    int hidden_size = num_heads * head_dim;
    
    CHECK_CUDA(cudaMalloc(&x_gpu, batch * seq_len * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&cos_gpu, seq_len * head_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&sin_gpu, seq_len * head_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&q_proj_gpu, num_heads * head_dim * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&k_proj_gpu, num_kv_heads * head_dim * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&v_proj_gpu, num_kv_heads * head_dim * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&o_proj_gpu, hidden_size * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&q_norm_gpu, head_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&k_norm_gpu, head_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&output_gpu, batch * seq_len * hidden_size * sizeof(float)));
    
    CHECK_CUDA(cudaMalloc(&q_proj_out_gpu, batch * seq_len * num_heads * head_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&k_proj_out_gpu, batch * seq_len * num_kv_heads * head_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&v_proj_out_gpu, batch * seq_len * num_kv_heads * head_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&q_normed_gpu, batch * seq_len * num_heads * head_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&k_normed_gpu, batch * seq_len * num_kv_heads * head_dim * sizeof(float)));
    
    CHECK_CUDA(cudaMalloc(&q_transposed_gpu, batch * num_heads * seq_len * head_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&k_transposed_gpu, batch * num_kv_heads * seq_len * head_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&k_repeated_gpu, batch * num_heads * seq_len * head_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&v_transposed_gpu, batch * num_kv_heads * seq_len * head_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn_scores_gpu, batch * num_heads * seq_len * seq_len * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn_out_gpu, batch * num_heads * seq_len * head_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn_out_transposed_gpu, batch * seq_len * num_heads * head_dim * sizeof(float)));
    
    // Copy static weights to GPU
    CHECK_CUDA(cudaMemcpy(cos_gpu, cos, seq_len * head_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(sin_gpu, sin, seq_len * head_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(q_proj_gpu, q_proj, num_heads * head_dim * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(k_proj_gpu, k_proj, num_kv_heads * head_dim * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(v_proj_gpu, v_proj, num_kv_heads * head_dim * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(o_proj_gpu, o_proj, hidden_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(q_norm_gpu, q_norm, head_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(k_norm_gpu, k_norm, head_dim * sizeof(float), cudaMemcpyHostToDevice));
}

void attn(float *x, float *cos, float *sin, float *q_proj, float *k_proj, 
          float *v_proj, float *o_proj, float *q_norm, float *k_norm, 
          float *output, int batch, int seq_len, int num_heads, 
          int head_dim, int num_kv_heads) {
    
    int hidden_size = num_heads * head_dim;
    
    // Copy input data to GPU
    CHECK_CUDA(cudaMemcpy(x_gpu, x, batch * seq_len * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // TODO: Implement attention operation
    
    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(output, output_gpu, batch * seq_len * hidden_size * sizeof(float), cudaMemcpyDeviceToHost));
}

void attn_finalize() {
    CHECK_CUDA(cudaFree(x_gpu));
    CHECK_CUDA(cudaFree(cos_gpu));
    CHECK_CUDA(cudaFree(sin_gpu));
    CHECK_CUDA(cudaFree(q_proj_gpu));
    CHECK_CUDA(cudaFree(k_proj_gpu));
    CHECK_CUDA(cudaFree(v_proj_gpu));
    CHECK_CUDA(cudaFree(o_proj_gpu));
    CHECK_CUDA(cudaFree(q_norm_gpu));
    CHECK_CUDA(cudaFree(k_norm_gpu));
    CHECK_CUDA(cudaFree(output_gpu));
    CHECK_CUDA(cudaFree(q_proj_out_gpu));
    CHECK_CUDA(cudaFree(k_proj_out_gpu));
    CHECK_CUDA(cudaFree(v_proj_out_gpu));
    CHECK_CUDA(cudaFree(q_normed_gpu));
    CHECK_CUDA(cudaFree(k_normed_gpu));
    CHECK_CUDA(cudaFree(q_transposed_gpu));
    CHECK_CUDA(cudaFree(k_transposed_gpu));
    CHECK_CUDA(cudaFree(k_repeated_gpu));
    CHECK_CUDA(cudaFree(v_transposed_gpu));
    CHECK_CUDA(cudaFree(attn_scores_gpu));
    CHECK_CUDA(cudaFree(attn_out_gpu));
    CHECK_CUDA(cudaFree(attn_out_transposed_gpu));
}
