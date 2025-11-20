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
static float *q_transposed_gpu, *k_transposed_gpu, *k_repeated_gpu, *v_transposed_gpu, *v_repeated_gpu;
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
    CHECK_CUDA(cudaMalloc(&v_repeated_gpu, batch * num_heads * seq_len * head_dim * sizeof(float)));
    

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

__global__ void rmsnorm_kernel(float *x, float *w, float *out, int rows, int dim) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float sum_sq = 0.0f;
        for (int i = 0; i < dim; ++i) {
            float val = x[row * dim + i];
            sum_sq += val * val;
        }
        float rms = rsqrtf(sum_sq / dim + 1e-6f);
        for (int i = 0; i < dim; ++i) {
            out[row * dim + i] = x[row * dim + i] * rms * w[i];
        }
    }
}

__global__ void rope_kernel(float *data, float *cos, float *sin, int batch, int seq_len, int num_heads, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_dim = head_dim / 2;
    int total = batch * seq_len * num_heads * half_dim;
    if (idx < total) {
        int hd = idx % half_dim;
        int rem = idx / half_dim;
        int h = rem % num_heads;
        rem /= num_heads;
        int s = rem % seq_len;
        int b = rem / seq_len;
        
        int i1 = ((b * seq_len + s) * num_heads + h) * head_dim + hd;
        int i2 = i1 + half_dim;
        
        float x1 = data[i1];
        float x2 = data[i2];
        float c = cos[s * head_dim + hd];
        float sn = sin[s * head_dim + hd];
        
        data[i1] = x1 * c - x2 * sn;
        data[i2] = x2 * c + x1 * sn;
    }
}

__global__ void transpose_kernel(float *in, float *out, int B, int S, int H, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * S * H * D;
    if (idx < total) {
        int d = idx % D;
        int rem = idx / D;
        int h = rem % H;
        rem /= H;
        int s = rem % S;
        int b = rem / S;
        // in: [B, S, H, D] -> out: [B, H, S, D]
        out[b * (H * S * D) + h * (S * D) + s * D + d] = in[idx];
    }
}

__global__ void repeat_kv_kernel(float *in, float *out, int B, int num_heads, int num_kv_heads, int S, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * num_heads * S * D;
    if (idx < total) {
        int d = idx % D;
        int rem = idx / D;
        int s = rem % S;
        rem /= S;
        int h = rem % num_heads;
        int b = rem / num_heads;
        
        int group = num_heads / num_kv_heads;
        int kv_h = h / group;
        
        out[idx] = in[b * (num_kv_heads * S * D) + kv_h * (S * D) + s * D + d];
    }
}

__global__ void batched_matmul_qk_kernel(float *Q, float *K, float *Scores, int S, int D, float scale) {
    int s_k = blockIdx.x * blockDim.x + threadIdx.x;
    int s_q = blockIdx.y * blockDim.y + threadIdx.y;
    int b_h = blockIdx.z;
    
    if (s_q < S && s_k < S) {
        float sum = 0.0f;
        int offset = b_h * S * D;
        for (int i = 0; i < D; ++i) {
            sum += Q[offset + s_q * D + i] * K[offset + s_k * D + i];
        }
        Scores[b_h * S * S + s_q * S + s_k] = sum * scale;
    }
}

__global__ void softmax_kernel(float *scores, int S) {
    int b_h = blockIdx.y;
    int s_q = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (s_q < S) {
        float *row = scores + b_h * S * S + s_q * S;
        float max_val = -1e20f;
        for (int s_k = 0; s_k < S; ++s_k) {
            if (s_k > s_q) row[s_k] = -INFINITY;
            if (row[s_k] > max_val) max_val = row[s_k];
        }
        float sum = 0.0f;
        for (int s_k = 0; s_k < S; ++s_k) {
            if (s_k <= s_q) {
                row[s_k] = expf(row[s_k] - max_val);
                sum += row[s_k];
            } else {
                row[s_k] = 0.0f;
            }
        }
        for (int s_k = 0; s_k < S; ++s_k) row[s_k] /= sum;
    }
}

__global__ void batched_matmul_sv_kernel(float *Scores, float *V, float *Out, int S, int D) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int s_q = blockIdx.y * blockDim.y + threadIdx.y;
    int b_h = blockIdx.z;
    
    if (s_q < S && d < D) {
        float sum = 0.0f;
        int scores_off = b_h * S * S + s_q * S;
        int v_off = b_h * S * D;
        for (int s_k = 0; s_k < S; ++s_k) {
            sum += Scores[scores_off + s_k] * V[v_off + s_k * D + d];
        }
        Out[b_h * S * D + s_q * D + d] = sum;
    }
}

__global__ void transpose_back_kernel(float *in, float *out, int B, int H, int S, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * S * D;
    if (idx < total) {
        int d = idx % D;
        int rem = idx / D;
        int s = rem % S;
        rem /= S;
        int h = rem % H;
        int b = rem / H;
        // in: [B, H, S, D] -> out: [B, S, H, D]
        out[b * (S * H * D) + s * (H * D) + h * D + d] = in[idx];
    }
}

void attn(float *x, float *cos, float *sin, float *q_proj, float *k_proj, 
          float *v_proj, float *o_proj, float *q_norm, float *k_norm, 
          float *output, int batch, int seq_len, int num_heads, 
          int head_dim, int num_kv_heads) {
    
    int hidden_size = num_heads * head_dim;
    int kv_dim = num_kv_heads * head_dim;
    
    // Copy input data to GPU
    CHECK_CUDA(cudaMemcpy(x_gpu, x, batch * seq_len * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    
    dim3 block(16, 16);
    dim3 grid_q((hidden_size + 15)/16, (batch * seq_len + 15)/16);
    dim3 grid_kv((kv_dim + 15)/16, (batch * seq_len + 15)/16);
    
    // 1. Projections
    matmul_kernel<<<grid_q, block>>>(x_gpu, q_proj_gpu, q_proj_out_gpu, batch * seq_len, hidden_size, hidden_size);
    matmul_kernel<<<grid_kv, block>>>(x_gpu, k_proj_gpu, k_proj_out_gpu, batch * seq_len, kv_dim, hidden_size);
    matmul_kernel<<<grid_kv, block>>>(x_gpu, v_proj_gpu, v_proj_out_gpu, batch * seq_len, kv_dim, hidden_size);
    
    // 2. RMSNorm
    int total_q = batch * seq_len * num_heads;
    int total_k = batch * seq_len * num_kv_heads;
    rmsnorm_kernel<<<total_q, 1>>>(q_proj_out_gpu, q_norm_gpu, q_normed_gpu, total_q, head_dim);
    rmsnorm_kernel<<<total_k, 1>>>(k_proj_out_gpu, k_norm_gpu, k_normed_gpu, total_k, head_dim);
    
    // 3. RoPE
    int threads = 256;
    int blocks_q = (total_q * (head_dim/2) + threads - 1) / threads;
    int blocks_k = (total_k * (head_dim/2) + threads - 1) / threads;
    rope_kernel<<<blocks_q, threads>>>(q_normed_gpu, cos_gpu, sin_gpu, batch, seq_len, num_heads, head_dim);
    rope_kernel<<<blocks_k, threads>>>(k_normed_gpu, cos_gpu, sin_gpu, batch, seq_len, num_kv_heads, head_dim);
    
    // 4. Transpose
    transpose_kernel<<<(batch*seq_len*hidden_size + 255)/256, 256>>>(q_normed_gpu, q_transposed_gpu, batch, seq_len, num_heads, head_dim);
    transpose_kernel<<<(batch*seq_len*kv_dim + 255)/256, 256>>>(k_normed_gpu, k_transposed_gpu, batch, seq_len, num_kv_heads, head_dim);
    transpose_kernel<<<(batch*seq_len*kv_dim + 255)/256, 256>>>(v_proj_out_gpu, v_transposed_gpu, batch, seq_len, num_kv_heads, head_dim);
    
    // 5. Repeat KV
    repeat_kv_kernel<<<(batch*num_heads*seq_len*head_dim + 255)/256, 256>>>(k_transposed_gpu, k_repeated_gpu, batch, num_heads, num_kv_heads, seq_len, head_dim);
    repeat_kv_kernel<<<(batch*num_heads*seq_len*head_dim + 255)/256, 256>>>(v_transposed_gpu, v_repeated_gpu, batch, num_heads, num_kv_heads, seq_len, head_dim);
    
    // 6. Scores
    dim3 grid_scores((seq_len + 15)/16, (seq_len + 15)/16, batch * num_heads);
    batched_matmul_qk_kernel<<<grid_scores, block>>>(q_transposed_gpu, k_repeated_gpu, attn_scores_gpu, seq_len, head_dim, 1.0f / sqrtf(head_dim));
    
    // 7. Softmax
    dim3 grid_softmax((seq_len + 255)/256, batch * num_heads);
    softmax_kernel<<<grid_softmax, 256>>>(attn_scores_gpu, seq_len);
    
    // 8. Output
    dim3 grid_out((head_dim + 15)/16, (seq_len + 15)/16, batch * num_heads);
    batched_matmul_sv_kernel<<<grid_out, block>>>(attn_scores_gpu, v_repeated_gpu, attn_out_gpu, seq_len, head_dim);
    
    // 9. Transpose Back
    transpose_back_kernel<<<(batch*seq_len*hidden_size + 255)/256, 256>>>(attn_out_gpu, attn_out_transposed_gpu, batch, num_heads, seq_len, head_dim);
    
    // 10. Final Projection
    matmul_kernel<<<grid_q, block>>>(attn_out_transposed_gpu, o_proj_gpu, output_gpu, batch * seq_len, hidden_size, hidden_size);
    
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
    CHECK_CUDA(cudaFree(v_repeated_gpu));
    
}
