#include "attn.h"
#include "util.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

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

// ============================================================================
// Kernels
// ============================================================================

// GEMM: C = A @ B^T
// A: (M, K), B: (N, K), C: (M, N)
// Grid: (N/32, M/32), Block: (32, 32)
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

// RMS Norm
__global__ void rms_norm_kernel(const float* __restrict__ input, 
                                const float* __restrict__ weight, 
                                float* __restrict__ output, 
                                int N, int head_dim, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Calculate RMS
        float sum_sq = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
            float val = input[idx * head_dim + i];
            sum_sq += val * val;
        }
        float rms = rsqrtf(sum_sq / head_dim + eps);

        // Normalize and scale
        for (int i = 0; i < head_dim; ++i) {
            output[idx * head_dim + i] = input[idx * head_dim + i] * rms * weight[i];
        }
    }
}

// Transpose Kernel: (B, S, H, D) -> (B, H, S, D)
__global__ void transpose_kernel(const float* __restrict__ input, float* __restrict__ output,
                                 int B, int S, int H, int D) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (s < S) {
        for (int d = 0; d < D; ++d) {
            int in_idx = ((b * S + s) * H + h) * D + d;
            int out_idx = ((b * H + h) * S + s) * D + d;
            output[out_idx] = input[in_idx];
        }
    }
}

// RoPE Kernel
__global__ void rope_kernel(float* __restrict__ q, const float* __restrict__ cos, const float* __restrict__ sin,
                            int B, int H, int S, int D) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int s = blockIdx.x * blockDim.x + threadIdx.x;

    if (s < S) {
        int half_D = D / 2;
        for (int d = 0; d < half_D; ++d) {
            int idx = ((b * H + h) * S + s) * D + d;
            int idx2 = idx + half_D;
            
            float q1 = q[idx];
            float q2 = q[idx2];
            float c = cos[s * D + d];
            float si = sin[s * D + d];
            
            q[idx] = q1 * c - q2 * si;
            q[idx2] = q2 * c + q1 * si;
        }
    }
}

// Repeat KV Kernel (GQA)
__global__ void repeat_kv_kernel(const float* __restrict__ input, float* __restrict__ output,
                                 int B, int H_kv, int n_rep, int S, int D) {
    int b = blockIdx.z;
    int h_kv = blockIdx.y;
    int s = blockIdx.x * blockDim.x + threadIdx.x;

    if (s < S) {
        for (int r = 0; r < n_rep; ++r) {
            int h_q = h_kv * n_rep + r;
            for (int d = 0; d < D; ++d) {
                int in_idx = ((b * H_kv + h_kv) * S + s) * D + d;
                int out_idx = ((b * (H_kv * n_rep) + h_q) * S + s) * D + d;
                output[out_idx] = input[in_idx];
            }
        }
    }
}

// BMM: Q @ K^T
__global__ void bmm_qk_kernel(const float* __restrict__ Q, const float* __restrict__ K, float* __restrict__ Scores,
                              int B, int H, int S, int D, float scale) {
    int b = blockIdx.z / H;
    int h = blockIdx.z % H;
    int row = blockIdx.y * blockDim.y + threadIdx.y; // s_q
    int col = blockIdx.x * blockDim.x + threadIdx.x; // s_k

    if (row < S && col < S) {
        float sum = 0.0f;
        int offset_q = ((b * H + h) * S + row) * D;
        int offset_k = ((b * H + h) * S + col) * D;
        
        for (int d = 0; d < D; ++d) {
            sum += Q[offset_q + d] * K[offset_k + d];
        }
        
        // Causal Mask
        if (col > row) {
            Scores[((b * H + h) * S + row) * S + col] = -INFINITY;
        } else {
            Scores[((b * H + h) * S + row) * S + col] = sum * scale;
        }
    }
}

// Softmax
__global__ void softmax_kernel(float* __restrict__ data, int B, int H, int S) {
    int b = blockIdx.z / H;
    int h = blockIdx.z % H;
    int row = blockIdx.x * blockDim.x + threadIdx.x; // s

    if (row < S) {
        int offset = ((b * H + h) * S + row) * S;
        
        // Max
        float max_val = -INFINITY;
        for (int col = 0; col < S; ++col) {
            max_val = fmaxf(max_val, data[offset + col]);
        }
        
        // Exp & Sum
        float sum = 0.0f;
        for (int col = 0; col < S; ++col) {
            float val = expf(data[offset + col] - max_val);
            data[offset + col] = val;
            sum += val;
        }
        
        // Normalize
        for (int col = 0; col < S; ++col) {
            data[offset + col] /= sum;
        }
    }
}

// BMM: Scores @ V
__global__ void bmm_sv_kernel(const float* __restrict__ Scores, const float* __restrict__ V, float* __restrict__ Output,
                              int B, int H, int S, int D) {
    int b = blockIdx.z / H;
    int h = blockIdx.z % H;
    int row = blockIdx.y * blockDim.y + threadIdx.y; // s
    int col = blockIdx.x * blockDim.x + threadIdx.x; // d

    if (row < S && col < D) {
        float sum = 0.0f;
        int offset_s = ((b * H + h) * S + row) * S;
        int offset_v_base = ((b * H + h) * S) * D + col;
        
        for (int k = 0; k < S; ++k) {
            sum += Scores[offset_s + k] * V[offset_v_base + k * D];
        }
        Output[((b * H + h) * S + row) * D + col] = sum;
    }
}

// Transpose Back: (B, H, S, D) -> (B, S, H, D)
__global__ void transpose_back_kernel(const float* __restrict__ input, float* __restrict__ output,
                                      int B, int H, int S, int D) {
    int b = blockIdx.z;
    int s = blockIdx.y;
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (h < H) {
        for (int d = 0; d < D; ++d) {
            int in_idx = ((b * H + h) * S + s) * D + d;
            int out_idx = ((b * S + s) * H + h) * D + d;
            output[out_idx] = input[in_idx];
        }
    }
}

// ============================================================================
// Initialize / Finalize
// ============================================================================

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
    CHECK_CUDA(cudaMalloc(&v_repeated_gpu, batch * num_heads * seq_len * head_dim * sizeof(float)));
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
    int num_tokens = batch * seq_len;
    
    // Copy input data to GPU
    CHECK_CUDA(cudaMemcpy(x_gpu, x, num_tokens * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    
    dim3 block(32, 32);
    
    // 1. Projections: Q, K, V
    // Q Proj
    dim3 grid_q((num_heads * head_dim + 31) / 32, (num_tokens + 31) / 32);
    matmul_kernel<<<grid_q, block>>>(x_gpu, q_proj_gpu, q_proj_out_gpu, num_tokens, num_heads * head_dim, hidden_size);
    
    // K Proj
    dim3 grid_k((num_kv_heads * head_dim + 31) / 32, (num_tokens + 31) / 32);
    matmul_kernel<<<grid_k, block>>>(x_gpu, k_proj_gpu, k_proj_out_gpu, num_tokens, num_kv_heads * head_dim, hidden_size);
    
    // V Proj
    dim3 grid_v((num_kv_heads * head_dim + 31) / 32, (num_tokens + 31) / 32);
    matmul_kernel<<<grid_v, block>>>(x_gpu, v_proj_gpu, v_proj_out_gpu, num_tokens, num_kv_heads * head_dim, hidden_size);
    
    // 2. RMS Norm
    int num_q_vecs = num_tokens * num_heads;
    int num_k_vecs = num_tokens * num_kv_heads;
    
    rms_norm_kernel<<<(num_q_vecs + 255) / 256, 256>>>(q_proj_out_gpu, q_norm_gpu, q_normed_gpu, num_q_vecs, head_dim, 1e-5f);
    rms_norm_kernel<<<(num_k_vecs + 255) / 256, 256>>>(k_proj_out_gpu, k_norm_gpu, k_normed_gpu, num_k_vecs, head_dim, 1e-5f);
    
    // 3. Transpose (B, S, H, D) -> (B, H, S, D)
    dim3 block_trans(32);
    dim3 grid_trans_q((seq_len + 31) / 32, num_heads, batch);
    transpose_kernel<<<grid_trans_q, block_trans>>>(q_normed_gpu, q_transposed_gpu, batch, seq_len, num_heads, head_dim);
    
    dim3 grid_trans_k((seq_len + 31) / 32, num_kv_heads, batch);
    transpose_kernel<<<grid_trans_k, block_trans>>>(k_normed_gpu, k_transposed_gpu, batch, seq_len, num_kv_heads, head_dim);
    
    dim3 grid_trans_v((seq_len + 31) / 32, num_kv_heads, batch);
    transpose_kernel<<<grid_trans_v, block_trans>>>(v_proj_out_gpu, v_transposed_gpu, batch, seq_len, num_kv_heads, head_dim);
    
    // 4. RoPE
    dim3 grid_rope((seq_len + 31) / 32, num_heads, batch);
    rope_kernel<<<grid_rope, block_trans>>>(q_transposed_gpu, cos_gpu, sin_gpu, batch, num_heads, seq_len, head_dim);
    
    dim3 grid_rope_k((seq_len + 31) / 32, num_kv_heads, batch);
    rope_kernel<<<grid_rope_k, block_trans>>>(k_transposed_gpu, cos_gpu, sin_gpu, batch, num_kv_heads, seq_len, head_dim);
    
    // 5. Repeat KV (GQA)
    int n_rep = num_heads / num_kv_heads;
    if (n_rep > 1) {
        dim3 grid_rep((seq_len + 31) / 32, num_kv_heads, batch);
        repeat_kv_kernel<<<grid_rep, block_trans>>>(k_transposed_gpu, k_repeated_gpu, batch, num_kv_heads, n_rep, seq_len, head_dim);
        repeat_kv_kernel<<<grid_rep, block_trans>>>(v_transposed_gpu, v_repeated_gpu, batch, num_kv_heads, n_rep, seq_len, head_dim);
    } else {
        CHECK_CUDA(cudaMemcpy(k_repeated_gpu, k_transposed_gpu, batch * num_heads * seq_len * head_dim * sizeof(float), cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(v_repeated_gpu, v_transposed_gpu, batch * num_heads * seq_len * head_dim * sizeof(float), cudaMemcpyDeviceToDevice));
    }
    
    // 6. Attention Scores: Q @ K^T
    dim3 grid_bmm_scores((seq_len + 31) / 32, (seq_len + 31) / 32, batch * num_heads);
    float scale = 1.0f / sqrtf((float)head_dim);
    bmm_qk_kernel<<<grid_bmm_scores, block>>>(q_transposed_gpu, k_repeated_gpu, attn_scores_gpu, batch, num_heads, seq_len, head_dim, scale);
    
    // 7. Softmax
    dim3 grid_softmax((seq_len + 31) / 32, 1, batch * num_heads);
    softmax_kernel<<<grid_softmax, block_trans>>>(attn_scores_gpu, batch, num_heads, seq_len);
    
    // 8. Attention Output: Scores @ V
    dim3 grid_bmm_out((head_dim + 31) / 32, (seq_len + 31) / 32, batch * num_heads);
    bmm_sv_kernel<<<grid_bmm_out, block>>>(attn_scores_gpu, v_repeated_gpu, attn_out_gpu, batch, num_heads, seq_len, head_dim);
    
    // 9. Transpose Back & Flatten
    dim3 grid_out_trans(num_heads, (seq_len + 31) / 32, batch);
    transpose_back_kernel<<<grid_out_trans, block_trans>>>(attn_out_gpu, attn_out_transposed_gpu, batch, num_heads, seq_len, head_dim);
    
    // 10. Output Projection
    dim3 grid_o((hidden_size + 31) / 32, (num_tokens + 31) / 32);
    matmul_kernel<<<grid_o, block>>>(attn_out_transposed_gpu, o_proj_gpu, output_gpu, num_tokens, hidden_size, hidden_size);
    
    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(output, output_gpu, num_tokens * hidden_size * sizeof(float), cudaMemcpyDeviceToHost));
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
    CHECK_CUDA(cudaFree(v_repeated_gpu));
    CHECK_CUDA(cudaFree(attn_scores_gpu));
    CHECK_CUDA(cudaFree(attn_out_gpu));
    CHECK_CUDA(cudaFree(attn_out_transposed_gpu));
}
