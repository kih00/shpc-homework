#include "layer.h"
#include "tensor.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                              \
  do {                                                                \
    cudaError_t status_ = call;                                       \
    if (status_ != cudaSuccess) {                                     \
      fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(status_));                           \
      exit(EXIT_FAILURE);                                             \
    }                                                                 \
  } while (0)

namespace kernels {

// Matrix multiplication kernel: C = A @ B (regular matmul)
__global__ void matmul_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                           int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];  // A @ B
        }
        C[row * N + col] = sum;
    }
}

// Matrix multiplication kernel: C = A @ B^T (transposed matmul)
__global__ void matmul_transposed_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                           int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[col * K + k];  // A @ B^T
        }
        C[row * N + col] = sum;
    }
}

__global__ void rms_norm_kernel(const float* __restrict__ input, 
                                const float* __restrict__ weight, 
                                float* __restrict__ output, 
                                int N, int head_dim, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float sum_sq = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
            float val = input[idx * head_dim + i];
            sum_sq += val * val;
        }
        float rms = rsqrtf(sum_sq / head_dim + eps);

        for (int i = 0; i < head_dim; ++i) {
            output[idx * head_dim + i] = input[idx * head_dim + i] * rms * weight[i];
        }
    }
}

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

__global__ void bmm_qk_kernel(const float* __restrict__ Q, const float* __restrict__ K, float* __restrict__ Scores,
                              int B, int H, int S, int D, float scale) {
    int b = blockIdx.z / H;
    int h = blockIdx.z % H;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < S && col < S) {
        float sum = 0.0f;
        int offset_q = ((b * H + h) * S + row) * D;
        int offset_k = ((b * H + h) * S + col) * D;
        
        for (int d = 0; d < D; ++d) {
            sum += Q[offset_q + d] * K[offset_k + d];
        }
        
        if (col > row) {
            Scores[((b * H + h) * S + row) * S + col] = -INFINITY;
        } else {
            Scores[((b * H + h) * S + row) * S + col] = sum * scale;
        }
    }
}

__global__ void softmax_kernel(float* __restrict__ data, int B, int H, int S) {
    int b = blockIdx.z / H;
    int h = blockIdx.z % H;
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < S) {
        int offset = ((b * H + h) * S + row) * S;
        
        float max_val = -INFINITY;
        for (int col = 0; col < S; ++col) {
            max_val = fmaxf(max_val, data[offset + col]);
        }
        
        float sum = 0.0f;
        for (int col = 0; col < S; ++col) {
            float val = expf(data[offset + col] - max_val);
            data[offset + col] = val;
            sum += val;
        }
        
        for (int col = 0; col < S; ++col) {
            data[offset + col] /= sum;
        }
    }
}

__global__ void bmm_sv_kernel(const float* __restrict__ Scores, const float* __restrict__ V, float* __restrict__ Output,
                              int B, int H, int S, int D) {
    int b = blockIdx.z / H;
    int h = blockIdx.z % H;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

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

// Conv Kernels

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

__global__ void mul_and_transpose_back_kernel(const float* __restrict__ BCx, const float* __restrict__ conv_out, float* __restrict__ output,
                                              int B, int H, int S) {
    int b = blockIdx.z;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (h < H && s < S) {
        int idx_C = ((b * 3 * H + (h + H)) * S + s);
        int idx_conv = ((b * H + h) * S + s);
        
        float val = BCx[idx_C] * conv_out[idx_conv];
        
        int out_idx = ((b * S + s) * H + h);
        output[out_idx] = val;
    }
}

// MoE Kernels

__global__ void topk_kernel(const float* __restrict__ logits, const float* __restrict__ bias,
                            int* __restrict__ indices, float* __restrict__ weights,
                            int N, int num_experts, int k, bool use_bias, bool norm_prob, float scaling_factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        float scores[32];
        int expert_ids[32];
        float routing_weights[32];
        
        for (int e = 0; e < num_experts; ++e) {
            float logit = logits[idx * num_experts + e];
            float prob = 1.0f / (1.0f + expf(-logit));
            routing_weights[e] = prob;
            
            float score = prob;
            if (use_bias) {
                score += bias[e];
            }
            scores[e] = score;
            expert_ids[e] = e;
        }
        
        for (int i = 0; i < k; ++i) {
            for (int j = i + 1; j < num_experts; ++j) {
                if (scores[j] > scores[i]) {
                    float tmp_s = scores[i]; scores[i] = scores[j]; scores[j] = tmp_s;
                    int tmp_id = expert_ids[i]; expert_ids[i] = expert_ids[j]; expert_ids[j] = tmp_id;
                }
            }
        }
        
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            int expert_idx = expert_ids[i];
            float weight = routing_weights[expert_idx];
            if (norm_prob) {
                sum += weight;
            }
            weights[idx * k + i] = weight;
            indices[idx * k + i] = expert_idx;
        }
        
        if (norm_prob && sum > 1e-6f) {
            for (int i = 0; i < k; ++i) {
                weights[idx * k + i] /= sum;
            }
        }
        
        for (int i = 0; i < k; ++i) {
            weights[idx * k + i] *= scaling_factor;
        }
    }
}


// Element-wise add
__global__ void add_kernel(const float* a, const float* b, float* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Element-wise mul
__global__ void mul_kernel(const float* a, const float* b, float* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

// Silu
__global__ void silu_kernel(const float* x, float* y, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = x[idx] / (1.0f + expf(-x[idx]));
    }
}

__global__ void add_scalar_kernel(const float* a, float b, float* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b;
    }
}

__global__ void mul_scalar_kernel(const float* a, float b, float* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b;
    }
}

__global__ void sigmoid_kernel(const float* x, float* y, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = 1.0f / (1.0f + expf(-x[idx]));
    }
}

__global__ void embedding_kernel(const int* __restrict__ input_ids, const float* __restrict__ embedding_table, float* __restrict__ output,
                                 int num_tokens, int hidden_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_tokens * hidden_size) {
        int token_idx = idx / hidden_size;
        int dim_idx = idx % hidden_size;
        
        int token_id = input_ids[token_idx];
        output[idx] = embedding_table[token_id * hidden_size + dim_idx];
    }
}

// MoE expert dispatch kernel
__global__ void moe_expert_dispatch_kernel(
    const float* __restrict__ x_flat,           // (num_tokens, hidden_size)
    const int* __restrict__ top_k_indices,      // (num_tokens, num_experts_per_tok)
    const float* __restrict__ top_k_weights,    // (num_tokens, num_experts_per_tok)
    float** __restrict__ w1_ptrs,               // (num_experts) pointers
    float** __restrict__ w2_ptrs,               // (num_experts) pointers
    float** __restrict__ w3_ptrs,               // (num_experts) pointers
    float* __restrict__ output,                  // (batch, seq_len, hidden_size)
    int num_tokens, int hidden_size, int intermediate_size, int num_experts_per_tok, int seq_len) {
    
    int token_idx = blockIdx.x;
    int expert_slot = blockIdx.y;  // which of the NUM_EXPERTS_PER_TOK experts
    int h_idx = threadIdx.x;
    
    if (token_idx >= num_tokens || expert_slot >= num_experts_per_tok || h_idx >= hidden_size) return;
    
    int expert_idx = top_k_indices[token_idx * num_experts_per_tok + expert_slot];
    float weight = top_k_weights[token_idx * num_experts_per_tok + expert_slot];
    
    // Get expert weights
    const float* w1 = w1_ptrs[expert_idx];  // (intermediate_size, hidden_size)
    const float* w2 = w2_ptrs[expert_idx];  // (hidden_size, intermediate_size)
    const float* w3 = w3_ptrs[expert_idx];  // (intermediate_size, hidden_size)
    
    // Shared memory for intermediate results
    extern __shared__ float shared_mem[];
    float* gate_proj = shared_mem;                                    // intermediate_size
    float* up_proj = &shared_mem[intermediate_size];                  // intermediate_size
    float* hidden = &shared_mem[2 * intermediate_size];               // intermediate_size
    
    // Compute gate projection: gate = x @ w1^T
    if (h_idx < intermediate_size) {
        float sum = 0.0f;
        for (int k = 0; k < hidden_size; k++) {
            sum += x_flat[token_idx * hidden_size + k] * w1[h_idx * hidden_size + k];
        }
        gate_proj[h_idx] = sum / (1.0f + expf(-sum));  // silu
    }
    __syncthreads();
    
    // Compute up projection: up = x @ w3^T
    if (h_idx < intermediate_size) {
        float sum = 0.0f;
        for (int k = 0; k < hidden_size; k++) {
            sum += x_flat[token_idx * hidden_size + k] * w3[h_idx * hidden_size + k];
        }
        up_proj[h_idx] = sum;
    }
    __syncthreads();
    
    // Element-wise multiply: hidden = gate_silu * up
    if (h_idx < intermediate_size) {
        hidden[h_idx] = gate_proj[h_idx] * up_proj[h_idx];
    }
    __syncthreads();
    
    // Compute output: y = hidden @ w2^T
    if (h_idx < hidden_size) {
        float sum = 0.0f;
        for (int k = 0; k < intermediate_size; k++) {
            sum += hidden[k] * w2[h_idx * intermediate_size + k];
        }
        
        // Add weighted output with atomic operation
        int b = token_idx / seq_len;
        int s = token_idx % seq_len;
        atomicAdd(&output[(b * seq_len + s) * hidden_size + h_idx], weight * sum);
    }
}

} // namespace kernels

namespace tensor_ops {

// MoE operations
void route_tokens(const Tensor& router_logits, const Tensor& expert_bias,
                  Tensor& top_k_indices, Tensor& top_k_weights,
                  int num_tokens, int num_experts, int num_experts_per_tok) {
    dim3 block(256);
    dim3 grid((num_tokens + 255) / 256);
    
    kernels::topk_kernel<<<grid, block>>>(router_logits.data(), expert_bias.data(),
                                          (int*)top_k_indices.data(), top_k_weights.data(),
                                          num_tokens, num_experts, num_experts_per_tok,
                                          true, true, 1.0f);  // use_bias, norm_prob, scaling_factor
}

void moe_expert_dispatch(const Tensor& x_flat, const Tensor& top_k_indices, const Tensor& top_k_weights,
                         float** w1_ptrs_gpu, float** w2_ptrs_gpu, float** w3_ptrs_gpu,
                         Tensor& output,
                         int num_tokens, int hidden_size, int intermediate_size, 
                         int num_experts_per_tok, int seq_len) {
    dim3 block(hidden_size > 1024 ? 1024 : hidden_size);
    dim3 grid(num_tokens, num_experts_per_tok);
    
    size_t shared_mem_size = 3 * intermediate_size * sizeof(float);
    
    kernels::moe_expert_dispatch_kernel<<<grid, block, shared_mem_size>>>(
        x_flat.data(),
        (const int*)top_k_indices.data(),
        top_k_weights.data(),
        w1_ptrs_gpu, w2_ptrs_gpu, w3_ptrs_gpu,
        output.data(),
        num_tokens, hidden_size, intermediate_size, num_experts_per_tok, seq_len
    );
}

// Matrix operations
void matmul(const Tensor& a, const Tensor& b, Tensor& c) {
    // a: (m, k), b: (k, n), c: (m, n)
    size_t m = a.size(0);
    size_t k = a.size(1);
    size_t n = b.size(1);
    
    dim3 block(32, 32);
    dim3 grid((n + 31) / 32, (m + 31) / 32);
    
    kernels::matmul_kernel<<<grid, block>>>(a.data(), b.data(), c.data(), m, n, k);
}

void matmul_transposed(const Tensor& a, const Tensor& b, Tensor& c) {
    // a: (m, k), b: (n, k), c: (m, n)
    size_t m = a.size(0);
    size_t k = a.size(1);
    size_t n = b.size(0);
    
    dim3 block(32, 32);
    dim3 grid((n + 31) / 32, (m + 31) / 32);
    
    kernels::matmul_transposed_kernel<<<grid, block>>>(a.data(), b.data(), c.data(), m, n, k);
}

// Element-wise operations
void add(const Tensor& a, const Tensor& b, Tensor& c) {
    size_t n = a.size();
    kernels::add_kernel<<<(n + 255) / 256, 256>>>(a.data(), b.data(), c.data(), n);
}

void add_scalar(const Tensor& a, float b, Tensor& c) {
    size_t n = a.size();
    kernels::add_scalar_kernel<<<(n + 255) / 256, 256>>>(a.data(), b, c.data(), n);
}

void mul(const Tensor& a, const Tensor& b, Tensor& c) {
    size_t n = a.size();
    kernels::mul_kernel<<<(n + 255) / 256, 256>>>(a.data(), b.data(), c.data(), n);
}

void mul_scalar(const Tensor& a, float b, Tensor& c) {
    size_t n = a.size();
    kernels::mul_scalar_kernel<<<(n + 255) / 256, 256>>>(a.data(), b, c.data(), n);
}

// Activation functions
void silu(const Tensor& x, Tensor& y) {
    size_t n = x.size();
    kernels::silu_kernel<<<(n + 255) / 256, 256>>>(x.data(), y.data(), n);
}

void sigmoid(const Tensor& x, Tensor& y) {
    size_t n = x.size();
    kernels::sigmoid_kernel<<<(n + 255) / 256, 256>>>(x.data(), y.data(), n);
}

void softmax(const Tensor& x, Tensor& y, int dim) {
    size_t outer_size = 1;
    for (size_t i = 0; i < x.ndim() - 1; i++) {
        outer_size *= x.size(i);
    }
    size_t inner_size = x.size(-1);
    
    kernels::softmax_kernel<<<(outer_size + 255) / 256, 256>>>(x.data(), y.data(), outer_size, inner_size);
}

// Normalization
void rms_norm(const Tensor& x, const Tensor& weight, float eps, Tensor& y) {
    size_t outer_size = 1;
    for (size_t i = 0; i < x.ndim() - 1; i++) {
        outer_size *= x.size(i);
    }
    size_t hidden_size = x.size(-1);
    
    kernels::rms_norm_kernel<<<(outer_size + 255) / 256, 256>>>(x.data(), weight.data(), y.data(), outer_size, hidden_size, eps);
}


// RoPE operations
void compute_rope_embeddings(size_t head_dim, size_t max_seq_len, float theta,
                             Tensor& cos, Tensor& sin) {
    // Compute on host and copy to GPU
    std::vector<float> cos_host(max_seq_len * head_dim);
    std::vector<float> sin_host(max_seq_len * head_dim);
    
    // Compute frequency bands
    std::vector<float> inv_freq(head_dim / 2);
    for (size_t i = 0; i < head_dim / 2; i++) {
        inv_freq[i] = 1.0f / std::pow(theta, (float)(2 * i) / head_dim);
    }
    
    // Compute cos and sin for each position
    for (size_t pos = 0; pos < max_seq_len; pos++) {
        for (size_t i = 0; i < head_dim / 2; i++) {
            float angle = pos * inv_freq[i];
            float c = std::cos(angle);
            float s = std::sin(angle);
            
            cos_host[pos * head_dim + i] = c;
            cos_host[pos * head_dim + i + head_dim / 2] = c;
            sin_host[pos * head_dim + i] = s;
            sin_host[pos * head_dim + i + head_dim / 2] = s;
        }
    }
    
    CHECK_CUDA(cudaMemcpy(cos.data(), cos_host.data(), cos_host.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(sin.data(), sin_host.data(), sin_host.size() * sizeof(float), cudaMemcpyHostToDevice));
}

void apply_rotary_pos_emb(Tensor& q, Tensor& k, const Tensor& cos, const Tensor& sin) {
    // q: (batch, num_q_heads, seq_len, head_dim)
    // k: (batch, num_kv_heads, seq_len, head_dim)
    // cos, sin: (seq_len, head_dim)
    // 
    // Apply rotation: q_embed = (q * cos) + (rotate_half(q) * sin)
    // rotate_half: concat([-x2, x1]) where x1=x[..., :head_dim/2], x2=x[..., head_dim/2:]
    
    size_t batch = q.size(0);
    size_t num_q_heads = q.size(1);
    size_t num_kv_heads = k.size(1);
    size_t seq_len = q.size(2);
    size_t head_dim = q.size(3);
    
    dim3 block_q(32);
    dim3 grid_q(seq_len, num_q_heads, batch);
    kernels::rope_kernel<<<grid_q, block_q>>>(q.data(), cos.data(), sin.data(),
                                              batch, num_q_heads, seq_len, head_dim);
    
    dim3 grid_k(seq_len, num_kv_heads, batch);
    kernels::rope_kernel<<<grid_k, block_q>>>(k.data(), cos.data(), sin.data(),
                                              batch, num_kv_heads, seq_len, head_dim);
}

// Grouped Query Attention operations
void repeat_kv(const Tensor& x, size_t n_rep, Tensor& y) {
    // x: (batch, num_kv_heads, seq_len, head_dim)
    // y: (batch, num_q_heads, seq_len, head_dim) where num_q_heads = num_kv_heads * n_rep
    
    size_t batch = x.size(0);
    size_t num_kv_heads = x.size(1);
    size_t seq_len = x.size(2);
    size_t head_dim = x.size(3);
    
    dim3 block(32);
    dim3 grid((seq_len + 31) / 32, num_kv_heads, batch);
    kernels::repeat_kv_kernel<<<grid, block>>>(x.data(), y.data(), batch, num_kv_heads, n_rep, seq_len, head_dim);
}

// Convolution operations
void causal_conv1d(const Tensor& x, const Tensor& weight, const Tensor* bias, Tensor& y) {
    // x: (batch, seq_len, hidden_size) -> needs transpose to (batch, hidden_size, seq_len)
    // weight: (hidden_size, 1, kernel_size)
    // bias: (hidden_size) [optional]
    // y: (batch, seq_len, hidden_size)  
    
    size_t batch = x.size(0);
    size_t seq_len = x.size(1);
    size_t hidden_size = x.size(2);
    size_t kernel_size = weight.size(2);
    
    // Simple host-side computation for conv (can be optimized later)
    for (size_t b = 0; b < batch; b++) {
        for (size_t c = 0; c < hidden_size; c++) {
            for (size_t s = 0; s < seq_len; s++) {
                float sum = 0.0f;
                for (size_t k = 0; k < kernel_size; k++) {
                    int input_pos = (int)s - ((int)kernel_size - 1) + (int)k;
                    if (input_pos >= 0 && input_pos < (int)seq_len) {
                        sum += x.at(b, input_pos, c) * weight.at(c, 0, k);
                    }
                }
                if (bias != nullptr) {
                    sum += (*bias)[c];
                }
                y.at(b, s, c) = sum;
            }
        }
    }
}

// Attention
void attention(const Tensor& x, const Tensor& cos, const Tensor& sin,
               const Tensor& q_proj, const Tensor& k_proj, const Tensor& v_proj, const Tensor& o_proj,
               const Tensor& q_norm, const Tensor& k_norm,
               Tensor& output,
               int batch, int seq_len, int num_heads, int head_dim, int num_kv_heads) {
    
    int hidden_size = num_heads * head_dim;
    int num_tokens = batch * seq_len;
    
    // Allocate temporary tensors
    Tensor q_proj_out({(size_t)num_tokens, (size_t)num_heads * head_dim});
    Tensor k_proj_out({(size_t)num_tokens, (size_t)num_kv_heads * head_dim});
    Tensor v_proj_out({(size_t)num_tokens, (size_t)num_kv_heads * head_dim});
    
    dim3 block(32, 32);
    
    // Projections
    dim3 grid_q((num_heads * head_dim + 31) / 32, (num_tokens + 31) / 32);
    kernels::matmul_kernel<<<grid_q, block>>>(x.data(), q_proj.data(), q_proj_out.data(), num_tokens, num_heads * head_dim, hidden_size);
    
    dim3 grid_k((num_kv_heads * head_dim + 31) / 32, (num_tokens + 31) / 32);
    kernels::matmul_kernel<<<grid_k, block>>>(x.data(), k_proj.data(), k_proj_out.data(), num_tokens, num_kv_heads * head_dim, hidden_size);
    
    dim3 grid_v((num_kv_heads * head_dim + 31) / 32, (num_tokens + 31) / 32);
    kernels::matmul_kernel<<<grid_v, block>>>(x.data(), v_proj.data(), v_proj_out.data(), num_tokens, num_kv_heads * head_dim, hidden_size);
    
    // RMS Norm
    Tensor q_normed({(size_t)num_tokens, (size_t)num_heads * head_dim});
    Tensor k_normed({(size_t)num_tokens, (size_t)num_kv_heads * head_dim});
    
    int num_q_vecs = num_tokens * num_heads;
    int num_k_vecs = num_tokens * num_kv_heads;
    
    kernels::rms_norm_kernel<<<(num_q_vecs + 255) / 256, 256>>>(q_proj_out.data(), q_norm.data(), q_normed.data(), num_q_vecs, head_dim, 1e-5f);
    kernels::rms_norm_kernel<<<(num_k_vecs + 255) / 256, 256>>>(k_proj_out.data(), k_norm.data(), k_normed.data(), num_k_vecs, head_dim, 1e-5f);
    
    // Transpose
    Tensor q_transposed({(size_t)batch, (size_t)num_heads, (size_t)seq_len, (size_t)head_dim});
    Tensor k_transposed({(size_t)batch, (size_t)num_kv_heads, (size_t)seq_len, (size_t)head_dim});
    Tensor v_transposed({(size_t)batch, (size_t)num_kv_heads, (size_t)seq_len, (size_t)head_dim});
    
    dim3 block_trans(32);
    dim3 grid_trans_q((seq_len + 31) / 32, num_heads, batch);
    kernels::transpose_kernel<<<grid_trans_q, block_trans>>>(q_normed.data(), q_transposed.data(), batch, seq_len, num_heads, head_dim);
    
    dim3 grid_trans_k((seq_len + 31) / 32, num_kv_heads, batch);
    kernels::transpose_kernel<<<grid_trans_k, block_trans>>>(k_normed.data(), k_transposed.data(), batch, seq_len, num_kv_heads, head_dim);
    
    dim3 grid_trans_v((seq_len + 31) / 32, num_kv_heads, batch);
    kernels::transpose_kernel<<<grid_trans_v, block_trans>>>(v_proj_out.data(), v_transposed.data(), batch, seq_len, num_kv_heads, head_dim);
    
    // RoPE
    dim3 grid_rope((seq_len + 31) / 32, num_heads, batch);
    kernels::rope_kernel<<<grid_rope, block_trans>>>(q_transposed.data(), cos.data(), sin.data(), batch, num_heads, seq_len, head_dim);
    
    dim3 grid_rope_k((seq_len + 31) / 32, num_kv_heads, batch);
    kernels::rope_kernel<<<grid_rope_k, block_trans>>>(k_transposed.data(), cos.data(), sin.data(), batch, num_kv_heads, seq_len, head_dim);
    
    // Repeat KV
    Tensor k_repeated({(size_t)batch, (size_t)num_heads, (size_t)seq_len, (size_t)head_dim});
    Tensor v_repeated({(size_t)batch, (size_t)num_heads, (size_t)seq_len, (size_t)head_dim});
    
    int n_rep = num_heads / num_kv_heads;
    if (n_rep > 1) {
        dim3 grid_rep((seq_len + 31) / 32, num_kv_heads, batch);
        kernels::repeat_kv_kernel<<<grid_rep, block_trans>>>(k_transposed.data(), k_repeated.data(), batch, num_kv_heads, n_rep, seq_len, head_dim);
        kernels::repeat_kv_kernel<<<grid_rep, block_trans>>>(v_transposed.data(), v_repeated.data(), batch, num_kv_heads, n_rep, seq_len, head_dim);
    } else {
        CHECK_CUDA(cudaMemcpy(k_repeated.data(), k_transposed.data(), k_repeated.size() * sizeof(float), cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(v_repeated.data(), v_transposed.data(), v_repeated.size() * sizeof(float), cudaMemcpyDeviceToDevice));
    }
    
    // Scores
    Tensor attn_scores({(size_t)batch, (size_t)num_heads, (size_t)seq_len, (size_t)seq_len});
    dim3 grid_bmm_scores((seq_len + 31) / 32, (seq_len + 31) / 32, batch * num_heads);
    float scale = 1.0f / sqrtf((float)head_dim);
    kernels::bmm_qk_kernel<<<grid_bmm_scores, block>>>(q_transposed.data(), k_repeated.data(), attn_scores.data(), batch, num_heads, seq_len, head_dim, scale);
    
    // Softmax
    dim3 grid_softmax((seq_len + 31) / 32, 1, batch * num_heads);
    kernels::softmax_kernel<<<grid_softmax, block_trans>>>(attn_scores.data(), batch, num_heads, seq_len);
    
    // Output
    Tensor attn_out({(size_t)batch, (size_t)num_heads, (size_t)seq_len, (size_t)head_dim});
    dim3 grid_bmm_out((head_dim + 31) / 32, (seq_len + 31) / 32, batch * num_heads);
    kernels::bmm_sv_kernel<<<grid_bmm_out, block>>>(attn_scores.data(), v_repeated.data(), attn_out.data(), batch, num_heads, seq_len, head_dim);
    
    // Transpose Back
    Tensor attn_out_transposed({(size_t)batch, (size_t)seq_len, (size_t)num_heads * head_dim});
    dim3 grid_out_trans(num_heads, (seq_len + 31) / 32, batch);
    kernels::transpose_back_kernel<<<grid_out_trans, block_trans>>>(attn_out.data(), attn_out_transposed.data(), batch, num_heads, seq_len, head_dim);
    
    // Output Proj
    dim3 grid_o((hidden_size + 31) / 32, (num_tokens + 31) / 32);
    kernels::matmul_kernel<<<grid_o, block>>>(attn_out_transposed.data(), o_proj.data(), output.data(), num_tokens, hidden_size, hidden_size);
}

// Conv
void conv(const Tensor& x, const Tensor& conv_weight, const Tensor& in_proj_weight, const Tensor& out_proj_weight,
          const Tensor* conv_bias, const Tensor* in_proj_bias, const Tensor* out_proj_bias,
          Tensor& output,
          int batch, int seq_len, int hidden_size, int kernel_size) {
    
    int num_tokens = batch * seq_len;
    
    Tensor in_proj_out({(size_t)num_tokens, (size_t)3 * hidden_size});
    dim3 block(32, 32);
    dim3 grid_in((3 * hidden_size + 31) / 32, (num_tokens + 31) / 32);
    kernels::matmul_kernel<<<grid_in, block>>>(x.data(), in_proj_weight.data(), in_proj_out.data(), num_tokens, 3 * hidden_size, hidden_size);
    
    Tensor BCx({(size_t)batch, (size_t)3 * hidden_size, (size_t)seq_len});
    dim3 grid_trans((seq_len + 31) / 32, (3 * hidden_size + 31) / 32, batch);
    kernels::transpose_bsc_to_bcs_kernel<<<grid_trans, block>>>(in_proj_out.data(), BCx.data(), batch, seq_len, 3 * hidden_size);
    
    Tensor Bx({(size_t)batch, (size_t)hidden_size, (size_t)seq_len});
    dim3 grid_split((seq_len + 31) / 32, (hidden_size + 31) / 32, batch);
    kernels::split_and_mul_kernel<<<grid_split, block>>>(BCx.data(), Bx.data(), batch, hidden_size, seq_len);
    
    Tensor conv_out({(size_t)batch, (size_t)hidden_size, (size_t)seq_len});
    kernels::causal_conv1d_kernel<<<grid_split, block>>>(Bx.data(), conv_weight.data(), conv_out.data(), batch, hidden_size, seq_len, kernel_size);
    
    Tensor y_pre_transposed({(size_t)batch, (size_t)seq_len, (size_t)hidden_size});
    kernels::mul_and_transpose_back_kernel<<<grid_split, block>>>(BCx.data(), conv_out.data(), y_pre_transposed.data(), batch, hidden_size, seq_len);
    
    dim3 grid_out((hidden_size + 31) / 32, (num_tokens + 31) / 32);
    kernels::matmul_kernel<<<grid_out, block>>>(y_pre_transposed.data(), out_proj_weight.data(), output.data(), num_tokens, hidden_size, hidden_size);
}


void embedding_lookup(const int* input_ids, const Tensor& embedding_table, Tensor& output,
                      int batch, int seq_len, int hidden_size) {
    // input_ids: (batch * seq_len) [device pointer]
    // embedding_table: (vocab_size, hidden_size)
    // output: (batch * seq_len, hidden_size)
    
    int num_tokens = batch * seq_len;
    
    dim3 block(256);
    dim3 grid((num_tokens * hidden_size + 255) / 256);
    
    kernels::embedding_kernel<<<grid, block>>>(input_ids, embedding_table.data(), output.data(), num_tokens, hidden_size);
}

} // namespace tensor_ops


RMSNorm::RMSNorm(const std::string& weight_file) {
    weight_ = Tensor::load_from_file(weight_file, nullptr);  // Uses global g_model_loader
}

void RMSNorm::forward(const Tensor& x, Tensor& y) {
    tensor_ops::rms_norm(x, weight_, RMS_NORM_EPS, y);
}

// RotaryEmbedding implementation
RotaryEmbedding::RotaryEmbedding() : max_seq_len_(MAX_POSITION_EMBEDDINGS) {
    cos_cached_ = Tensor({max_seq_len_, HEAD_DIM});
    sin_cached_ = Tensor({max_seq_len_, HEAD_DIM});
    tensor_ops::compute_rope_embeddings(HEAD_DIM, max_seq_len_, ROPE_THETA, 
                                       cos_cached_, sin_cached_);
}

void RotaryEmbedding::forward(size_t seq_len, Tensor& cos, Tensor& sin) {
    // Return cached values for the given sequence length
    // cos, sin should be: (seq_len, head_dim)
    size_t n = seq_len * HEAD_DIM;
    kernels::copy_rope_kernel<<<(n + 255) / 256, 256>>>(cos_cached_.data(), sin_cached_.data(),
                                                        cos.data(), sin.data(), seq_len, HEAD_DIM);
}
