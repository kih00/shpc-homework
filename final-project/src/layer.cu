#include "layer.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

// ============================================================================
// CUDA Kernels
// ============================================================================

__global__ void matmul_kernel(const float *A, const float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) sum += A[row * K + i] * B[i * N + col];
        C[row * N + col] = sum;
    }
}

__global__ void matmul_transposed_kernel(const float *A, const float *B, float *C, int M, int N, int K) {
    // C = A @ B^T
    // A: (M, K), B: (N, K), C: (M, N)
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) sum += A[row * K + i] * B[col * K + i];
        C[row * N + col] = sum;
    }
}

__global__ void add_kernel(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void add_scalar_kernel(const float *a, const float b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b;
    }
}

__global__ void add_bias_kernel(float* a, const float* bias, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int col = idx % cols;
        a[idx] += bias[col];
    }
}

__global__ void mul_kernel(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

__global__ void mul_scalar_kernel(const float *a, const float b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b;
    }
}

__global__ void silu_kernel(const float *x, float *y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx];
        y[idx] = val / (1.0f + expf(-val));
    }
}

__global__ void sigmoid_kernel(const float *x, float *y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = 1.0f / (1.0f + expf(-x[idx]));
    }
}

__global__ void softmax_kernel(const float *x, float *y, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float max_val = x[row * cols];
        for (int j = 0; j < cols; ++j) {
            max_val = max(max_val, x[row * cols + j]);
        }
        
        float sum = 0.0f;
        for (int j = 0; j < cols; ++j) {
            float val = expf(x[row * cols + j] - max_val);
            y[row * cols + j] = val;
            sum += val;
        }
        
        for (int j = 0; j < cols; ++j) {
            y[row * cols + j] /= sum;
        }
    }
}

__global__ void rms_norm_kernel(const float *x, const float *w, float *y, int rows, int dim) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float sum_sq = 0.0f;
        for (int i = 0; i < dim; ++i) {
            float val = x[row * dim + i];
            sum_sq += val * val;
        }
        float rms = rsqrtf(sum_sq / dim + 1e-6f);
        for (int i = 0; i < dim; ++i) {
            y[row * dim + i] = x[row * dim + i] * rms * w[i];
        }
    }
}

__global__ void rope_kernel(float *data, const float *cos, const float *sin, int batch, int seq_len, int num_heads, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_dim = head_dim / 2;
    int total = batch * seq_len * num_heads * half_dim;
    if (idx < total) {
        int d = idx % half_dim;
        int rem = idx / half_dim;
        int h = rem % num_heads;
        rem /= num_heads;
        int s = rem % seq_len;
        int b = rem / seq_len;
        
        int i1 = ((b * seq_len + s) * num_heads + h) * head_dim + d;
        int i2 = i1 + half_dim;
        int deg1 = s * head_dim + d;
        int deg2 = s * head_dim + d + half_dim;
        
        float x1 = data[i1];
        float x2 = data[i2];
        
        // x_rotated = x * cos + rotate_half(x) * sin (x can be q or k)
        // rotate_half(x) = [-x2, x1]
        data[i1] = x1 * cos[deg1] + (-x2) * sin[deg1];
        data[i2] = x2 * cos[deg2] + x1 * sin[deg2];
    }
}

__global__ void compute_rope_embeddings_kernel(float *cos, float *sin, int max_seq_len, int head_dim, float theta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = max_seq_len * (head_dim / 2);
    if (idx < total) {
        int i = idx % (head_dim / 2);
        int pos = idx / (head_dim / 2);
        
        float inv_freq = 1.0f / powf(theta, (2.0f * i) / head_dim);
        float sina, cosa;
        sincosf(pos * inv_freq, &sina, &cosa);
        
        cos[pos * head_dim + i] = cosa;
        cos[pos * head_dim + i + head_dim / 2] = cosa;
        sin[pos * head_dim + i] = sina;
        sin[pos * head_dim + i + head_dim / 2] = sina;
    }
}

__global__ void repeat_kv_kernel(const float *in, float *out, int batch, int num_heads,
                                 int num_kv_heads, int seq_len, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * num_heads * seq_len * head_dim;
    if (idx < total) {
        int d = idx % head_dim;
        int rem = idx / head_dim;
        int s = rem % seq_len;
        rem /= seq_len;
        int out_h = rem % num_heads;
        int b = rem / num_heads;
        
        int n_rep = num_heads / num_kv_heads;
        int h = out_h / n_rep;
        
        out[idx] = in[((b * num_kv_heads + h) * seq_len + s) * head_dim + d];
    }
}

__global__ void causal_conv1d_kernel(const float *x, const float *w, const float *bias, float *y,
                                     int batch, int seq_len, int channels, int kernel_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels * seq_len;
    if (idx < total) {
        int s = idx % seq_len;
        int rem = idx / seq_len;
        int c = rem % channels;
        int b = rem / channels;
        
        int bc = ((b * channels) + c) * seq_len;

        // PyTorch Conv1d with padding=kernel_size-1:
        // At output position s, uses input positions [s-(kernel_size-1), ..., s]
        // kernel[0] multiplies input[s-(kernel_size-1)] (oldest)
        // kernel[kernel_size-1] multiplies input[s] (current)
        float sum = 0.0f;
        for (int k = 0; k < kernel_size; ++k) {
            int input_pos = s - (kernel_size - 1) + k;
            if (input_pos >= 0 && input_pos < seq_len) {
                sum += x[bc + input_pos] * w[c * kernel_size + k];
            }
        }
        if (bias != nullptr) {
            sum += bias[c];
        }
        y[idx] = sum;
    }
}

// ============================================================================
// Shared Kernels from Tests
// ============================================================================

__global__ void embedding_kernel(int *input_ids, float *embedding_table, float *output, int batch, int seq_len, int hidden_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seq_len * hidden_size;
    if (idx < total) {
        int h = idx % hidden_size;
        int rem = idx / hidden_size;
        int s = rem % seq_len;
        int b = rem / seq_len;
        
        int token_id = input_ids[b * seq_len + s];
        output[idx] = embedding_table[token_id * hidden_size + h];
    }
}

// --- Attention Kernels ---

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

__global__ void masked_softmax_kernel(float *scores, int S) {
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

// --- Conv Kernels ---

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

// --- MoE Kernels ---

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

// ============================================================================
// Tensor Operations - Basic operations on tensors
// ============================================================================

const int threads = 256;

namespace tensor_ops {

// Matrix operations
void matmul(const Tensor& a, const Tensor& b, Tensor& c) {
    // a: (m, k), b: (k, n), c: (m, n)
    int m = a.size(0);
    int k = a.size(1);
    int n = b.size(1);
    
    dim3 block(16, 16);
    dim3 grid((n + 15)/16, (m + 15)/16);
    matmul_kernel<<<grid, block>>>(a.data(), b.data(), c.data(), m, n, k);
}

void matmul_transposed(const Tensor& a, const Tensor& b, Tensor& c) {
    // a: (m, k), b: (n, k), c: (m, n)  [c = a @ b^T]
    int m = a.size(0);
    int k = a.size(1);
    int n = b.size(0);
    
    dim3 block(16, 16);
    dim3 grid((n + 15)/16, (m + 15)/16);
    matmul_transposed_kernel<<<grid, block>>>(a.data(), b.data(), c.data(), m, n, k);
}

// Element-wise operations
void add(const Tensor& a, const Tensor& b, Tensor& c) {
    int n = a.size();
    int blocks = (n + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(a.data(), b.data(), c.data(), n);
}

void add_scalar(const Tensor& a, float b, Tensor& c) {
    int n = a.size();
    int blocks = (n + threads - 1) / threads;
    add_scalar_kernel<<<blocks, threads>>>(a.data(), b, c.data(), n);
}

void add_bias(const Tensor& a, const Tensor& bias, Tensor& c) {
    // a: (rows, cols), bias: (cols)
    // Broadcast bias addition: c = a + bias
    
    int rows = a.size(0);
    int cols = a.size(1);
    
    int blocks = (rows * cols + threads - 1) / threads;
    
    // Ensure c contains a's data if not in-place
    if (c.data() != a.data()) {
        CHECK_CUDA(cudaMemcpy(c.data(), a.data(), a.size() * sizeof(float), cudaMemcpyDeviceToDevice));
    }
    add_bias_kernel<<<blocks, threads>>>(c.data(), bias.data(), rows, cols);
}

void mul(const Tensor& a, const Tensor& b, Tensor& c) {
    int n = a.size();
    int blocks = (n + threads - 1) / threads;
    mul_kernel<<<blocks, threads>>>(a.data(), b.data(), c.data(), n);
}

void mul_scalar(const Tensor& a, float b, Tensor& c) {
    int n = a.size();
    int blocks = (n + threads - 1) / threads;
    mul_scalar_kernel<<<blocks, threads>>>(a.data(), b, c.data(), n);
}

// Activation functions
void sigmoid(const Tensor& x, Tensor& y) {
    int n = x.size();
    int blocks = (n + threads - 1) / threads;
    sigmoid_kernel<<<blocks, threads>>>(x.data(), y.data(), n);
}

void silu(const Tensor& x, Tensor& y) {
    int n = x.size();
    int blocks = (n + threads - 1) / threads;
    silu_kernel<<<blocks, threads>>>(x.data(), y.data(), n);
}

void softmax(const Tensor& x, Tensor& y, int dim) {
    // For simplicity, assume dim=-1 (last dimension)
    int outer_size = 1;
    for (size_t i = 0; i < x.ndim() - 1; i++) {
        outer_size *= x.size(i);
    }
    int inner_size = x.size(-1);
    
    int blocks = (outer_size + threads - 1) / threads;
    int blocks = (outer_size + threads - 1) / threads;
    // Naive softmax kernel (one thread per row)
    softmax_kernel<<<blocks, threads>>>(x.data(), y.data(), outer_size, inner_size);
}

// Normalization
void rms_norm(const Tensor& x, const Tensor& weight, float eps, Tensor& y) {
    int outer_size = 1;
    for (size_t i = 0; i < x.ndim() - 1; i++) {
        outer_size *= x.size(i);
    }
    int hidden_size = x.size(-1);
    
    int blocks = (outer_size + threads - 1) / threads;
    rms_norm_kernel<<<blocks, threads>>>(x.data(), weight.data(), y.data(),
                                         outer_size, hidden_size);
}

// RoPE operations
void compute_rope_embeddings(size_t head_dim, size_t max_seq_len, float theta,
                             Tensor& cos, Tensor& sin) {
    int outer_size = max_seq_len * (head_dim / 2);
    int blocks = (outer_size + threads - 1) / threads;

    compute_rope_embeddings_kernel<<<blocks, threads>>>(cos.data(), sin.data(),
                                                        max_seq_len, head_dim, theta);
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
    
    size_t total_q = batch * seq_len * num_q_heads;
    size_t total_k = batch * seq_len * num_kv_heads;
    
    int blocks_q = (total_q * (head_dim/2) + threads - 1) / threads;
    int blocks_k = (total_k * (head_dim/2) + threads - 1) / threads;
    
    rope_kernel<<<blocks_q, threads>>>(q.data(), cos.data(), sin.data(),
                                       batch, seq_len, num_q_heads, head_dim);
    rope_kernel<<<blocks_k, threads>>>(k.data(), cos.data(), sin.data(),
                                       batch, seq_len, num_kv_heads, head_dim);
}

// Grouped Query Attention operations
void repeat_kv(const Tensor& x, size_t n_rep, Tensor& y) {
    if (n_rep == 1) {
        CHECK_CUDA(cudaMemcpy(y.data(), x.data(), x.size() * sizeof(float), cudaMemcpyDeviceToDevice));
        return;
    }
    
    // x: (batch, num_kv_heads, seq_len, head_dim)
    // y: (batch, num_kv_heads * n_rep, seq_len, head_dim)
    size_t batch = x.size(0);
    size_t num_kv_heads = x.size(1);
    size_t seq_len = x.size(2);
    size_t head_dim = x.size(3);
    int num_heads = num_kv_heads * n_rep; // Output heads
    
    int total = batch * num_heads * seq_len * head_dim;
    int blocks = (total + threads - 1) / threads;
    
    repeat_kv_kernel<<<blocks, threads>>>(x.data(), y.data(), batch, num_heads,
                                          num_kv_heads, seq_len, head_dim);
}

// Convolution operations
void causal_conv1d(const Tensor& x, const Tensor& weight, const Tensor* bias, Tensor& y) {
    // x: (batch, channels, seq_len) - Conv1d format
    // weight: (channels, 1, kernel_size) - grouped conv weights
    // bias: (channels) [optional]
    // y: (batch, channels, seq_len)

    size_t batch = x.size(0);
    size_t channels = x.size(1);
    size_t seq_len = x.size(2);
    size_t kernel_size = weight.size(2);
    
    // Allocate y if needed
    if (y.size() == 0) {
        y = Tensor({batch, channels, seq_len});
    }
    y.zero();
    
    int total = batch * channels * seq_len;
    int blocks = (total + threads - 1) / threads;
    
    causal_conv1d_kernel<<<blocks, threads>>>(x.data(), weight.data(),
                                              bias ? bias->data() : nullptr, y.data(),
                                              batch, seq_len, channels, kernel_size);
}

} // namespace tensor_ops

// ============================================================================
// Layer Implementations - Small building blocks
// ============================================================================

// RMSNorm implementation
RMSNorm::RMSNorm(const std::string& weight_file) {
    weight_ = Tensor::load_from_file(weight_file);
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
    // Copy cached tensors to output buffers
    
    // cos: (seq_len, head_dim)
    size_t copy_size = seq_len * HEAD_DIM * sizeof(float);
    CHECK_CUDA(cudaMemcpy(cos.data(), cos_cached_.data(), copy_size, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(sin.data(), sin_cached_.data(), copy_size, cudaMemcpyDeviceToDevice));
}


