#include "layer.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

// ============================================================================
// Tensor Operations - Basic operations on tensors
// ============================================================================

namespace tensor_ops {

// Matrix operations
void matmul(const Tensor& a, const Tensor& b, Tensor& c) {
    // a: (m, k), b: (k, n), c: (m, n)
    size_t m = a.size(0);
    size_t k = a.size(1);
    size_t n = b.size(1);
    
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            float sum = 0.0f;
            for (size_t p = 0; p < k; p++) {
                sum += a.at(i, p) * b.at(p, j);
            }
            c.at(i, j) = sum;
        }
    }
}

void matmul_transposed(const Tensor& a, const Tensor& b, Tensor& c) {
    // a: (m, k), b: (n, k), c: (m, n)  [c = a @ b^T]
    size_t m = a.size(0);
    size_t k = a.size(1);
    size_t n = b.size(0);
    
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            float sum = 0.0f;
            for (size_t p = 0; p < k; p++) {
                sum += a.at(i, p) * b.at(j, p);
            }
            c.at(i, j) = sum;
        }
    }
}

// Element-wise operations
void add(const Tensor& a, const Tensor& b, Tensor& c) {
    #pragma omp parallel for
    for (size_t i = 0; i < a.size(); i++) {
        c[i] = a[i] + b[i];
    }
}

void add_scalar(const Tensor& a, float b, Tensor& c) {
    #pragma omp parallel for
    for (size_t i = 0; i < a.size(); i++) {
        c[i] = a[i] + b;
    }
}

void mul(const Tensor& a, const Tensor& b, Tensor& c) {
    #pragma omp parallel for
    for (size_t i = 0; i < a.size(); i++) {
        c[i] = a[i] * b[i];
    }
}

void mul_scalar(const Tensor& a, float b, Tensor& c) {
    #pragma omp parallel for
    for (size_t i = 0; i < a.size(); i++) {
        c[i] = a[i] * b;
    }
}

// Activation functions
void sigmoid(const Tensor& x, Tensor& y) {
    #pragma omp parallel for
    for (size_t i = 0; i < x.size(); i++) {
        y[i] = 1.0f / (1.0f + std::exp(-x[i]));
    }
}

void silu(const Tensor& x, Tensor& y) {
    #pragma omp parallel for
    for (size_t i = 0; i < x.size(); i++) {
        y[i] = x[i] / (1.0f + std::exp(-x[i]));
    }
}

void softmax(const Tensor& x, Tensor& y, int dim) {
    // For simplicity, assume dim=-1 (last dimension)
    size_t outer_size = 1;
    for (size_t i = 0; i < x.ndim() - 1; i++) {
        outer_size *= x.size(i);
    }
    size_t inner_size = x.size(-1);
    
    #pragma omp parallel for
    for (size_t i = 0; i < outer_size; i++) {
        // Find max for numerical stability
        float max_val = x[i * inner_size];
        for (size_t j = 1; j < inner_size; j++) {
            max_val = std::max(max_val, x[i * inner_size + j]);
        }
        
        // Compute exp and sum
        float sum = 0.0f;
        for (size_t j = 0; j < inner_size; j++) {
            y[i * inner_size + j] = std::exp(x[i * inner_size + j] - max_val);
            sum += y[i * inner_size + j];
        }
        
        // Normalize
        for (size_t j = 0; j < inner_size; j++) {
            y[i * inner_size + j] /= sum;
        }
    }
}

// Normalization
void rms_norm(const Tensor& x, const Tensor& weight, float eps, Tensor& y) {
    size_t outer_size = 1;
    for (size_t i = 0; i < x.ndim() - 1; i++) {
        outer_size *= x.size(i);
    }
    size_t hidden_size = x.size(-1);
    
    #pragma omp parallel for
    for (size_t i = 0; i < outer_size; i++) {
        // Compute RMS
        float sum_sq = 0.0f;
        for (size_t j = 0; j < hidden_size; j++) {
            float val = x[i * hidden_size + j];
            sum_sq += val * val;
        }
        float rms = std::sqrt(sum_sq / hidden_size + eps);
        
        // Normalize and scale
        for (size_t j = 0; j < hidden_size; j++) {
            y[i * hidden_size + j] = (x[i * hidden_size + j] / rms) * weight[j];
        }
    }
}

// RoPE operations
void compute_rope_embeddings(size_t head_dim, size_t max_seq_len, float theta,
                             Tensor& cos, Tensor& sin) {
    // Compute frequency bands
    std::vector<float> inv_freq(head_dim / 2);
    for (size_t i = 0; i < head_dim / 2; i++) {
        inv_freq[i] = 1.0f / std::pow(theta, (float)(2 * i) / head_dim);
    }
    
    // Compute cos and sin for each position
    #pragma omp parallel for
    for (size_t pos = 0; pos < max_seq_len; pos++) {
        for (size_t i = 0; i < head_dim / 2; i++) {
            float angle = pos * inv_freq[i];
            cos.at(pos, i) = std::cos(angle);
            cos.at(pos, i + head_dim / 2) = std::cos(angle);
            sin.at(pos, i) = std::sin(angle);
            sin.at(pos, i + head_dim / 2) = std::sin(angle);
        }
    }
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
    size_t half_dim = head_dim / 2;
    
    // Rotate q
    #pragma omp parallel for collapse(4)
    for (size_t b = 0; b < batch; b++) {
        for (size_t h = 0; h < num_q_heads; h++) {
            for (size_t s = 0; s < seq_len; s++) {
                for (size_t d = 0; d < half_dim; d++) {
                    float q1 = q.at(b, h, s, d);                  // first half
                    float q2 = q.at(b, h, s, d + half_dim);       // second half
                    
                    // q_rotated = q * cos + rotate_half(q) * sin
                    // rotate_half(q) = [-q2, q1]
                    q.at(b, h, s, d) = q1 * cos.at(s, d) + (-q2) * sin.at(s, d);
                    q.at(b, h, s, d + half_dim) = q2 * cos.at(s, d + half_dim) + q1 * sin.at(s, d + half_dim);
                }
            }
        }
    }
    
    // Rotate k (separate loop with correct num_kv_heads)
    #pragma omp parallel for collapse(4)
    for (size_t b = 0; b < batch; b++) {
        for (size_t h = 0; h < num_kv_heads; h++) {
            for (size_t s = 0; s < seq_len; s++) {
                for (size_t d = 0; d < half_dim; d++) {
                    float k1 = k.at(b, h, s, d);
                    float k2 = k.at(b, h, s, d + half_dim);
                    
                    k.at(b, h, s, d) = k1 * cos.at(s, d) + (-k2) * sin.at(s, d);
                    k.at(b, h, s, d + half_dim) = k2 * cos.at(s, d + half_dim) + k1 * sin.at(s, d + half_dim);
                }
            }
        }
    }
}

// Grouped Query Attention operations
void repeat_kv(const Tensor& x, size_t n_rep, Tensor& y) {
    if (n_rep == 1) {
        std::memcpy(y.data(), x.data(), x.size() * sizeof(float));
        return;
    }
    
    // x: (batch, num_kv_heads, seq_len, head_dim)
    // y: (batch, num_kv_heads * n_rep, seq_len, head_dim)
    size_t batch = x.size(0);
    size_t num_kv_heads = x.size(1);
    size_t seq_len = x.size(2);
    size_t head_dim = x.size(3);
    
    #pragma omp parallel for collapse(4)
    for (size_t b = 0; b < batch; b++) {
        for (size_t h = 0; h < num_kv_heads; h++) {
            for (size_t r = 0; r < n_rep; r++) {
                for (size_t s = 0; s < seq_len; s++) {
                    size_t out_h = h * n_rep + r;
                    for (size_t d = 0; d < head_dim; d++) {
                        y.at(b, out_h, s, d) = x.at(b, h, s, d);
                    }
                }
            }
        }
    }
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
    
    #pragma omp parallel for collapse(3)
    for (size_t b = 0; b < batch; b++) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t s = 0; s < seq_len; s++) {
                float sum = 0.0f;
                // PyTorch Conv1d with padding=kernel_size-1:
                // At output position s, uses input positions [s-(kernel_size-1), ..., s]
                // kernel[0] multiplies input[s-(kernel_size-1)] (oldest)
                // kernel[kernel_size-1] multiplies input[s] (current)
                for (size_t k = 0; k < kernel_size; k++) {
                    int input_pos = (int)s - ((int)kernel_size - 1) + (int)k;
                    if (input_pos >= 0) {
                        sum += x.at(b, c, input_pos) * weight.at(c, 0, k);
                    }
                }
                if (bias != nullptr) {
                    sum += (*bias)[c];
                }
                y.at(b, c, s) = sum;
            }
        }
    }
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
    // Return cached values for the given sequence length
    // cos, sin should be: (seq_len, head_dim)
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < HEAD_DIM; j++) {
            cos.at(i, j) = cos_cached_.at(i, j);
            sin.at(i, j) = sin_cached_.at(i, j);
        }
    }
}


// ============================================================================
// Attention Kernels (from tests/attn/attn.cu)
// ============================================================================

// Globals for Attention (renamed to avoid collision)
static float *attn_x_gpu, *attn_cos_gpu, *attn_sin_gpu;
static float *attn_q_proj_gpu, *attn_k_proj_gpu, *attn_v_proj_gpu, *attn_o_proj_gpu;
static float *attn_q_norm_gpu, *attn_k_norm_gpu, *attn_output_gpu;
static float *attn_q_proj_out_gpu, *attn_k_proj_out_gpu, *attn_v_proj_out_gpu;
static float *attn_q_normed_gpu, *attn_k_normed_gpu;
static float *attn_q_transposed_gpu, *attn_k_transposed_gpu, *attn_k_repeated_gpu, *attn_v_transposed_gpu, *attn_v_repeated_gpu;
static float *attn_scores_gpu, *attn_out_gpu, *attn_out_transposed_gpu;

// Kernels (renamed)
__global__ void attn_matmul_kernel(float *A, float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) sum += A[row * K + i] * B[i * N + col];
        C[row * N + col] = sum;
    }
}

__global__ void attn_rmsnorm_kernel(float *x, float *w, float *out, int rows, int dim) {
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

__global__ void attn_rope_kernel(float *data, float *cos, float *sin, int batch, int seq_len, int num_heads, int head_dim) {
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

__global__ void attn_transpose_kernel(float *in, float *out, int B, int S, int H, int D) {
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

__global__ void attn_repeat_kv_kernel(float *in, float *out, int B, int num_heads, int num_kv_heads, int S, int D) {
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

__global__ void attn_batched_matmul_qk_kernel(float *Q, float *K, float *Scores, int S, int D, float scale) {
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

__global__ void attn_softmax_kernel(float *scores, int S) {
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

__global__ void attn_batched_matmul_sv_kernel(float *Scores, float *V, float *Out, int S, int D) {
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

__global__ void attn_transpose_back_kernel(float *in, float *out, int B, int H, int S, int D) {
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

// Initialization helper (internal)
static void attn_initialize_internal(int batch, int seq_len, int num_heads, int head_dim, int num_kv_heads) {
    int hidden_size = num_heads * head_dim;
    
    // Free if already allocated (simple re-allocation strategy)
    if (attn_x_gpu) cudaFree(attn_x_gpu);
    if (attn_cos_gpu) cudaFree(attn_cos_gpu);
    if (attn_sin_gpu) cudaFree(attn_sin_gpu);
    if (attn_q_proj_gpu) cudaFree(attn_q_proj_gpu);
    if (attn_k_proj_gpu) cudaFree(attn_k_proj_gpu);
    if (attn_v_proj_gpu) cudaFree(attn_v_proj_gpu);
    if (attn_o_proj_gpu) cudaFree(attn_o_proj_gpu);
    if (attn_q_norm_gpu) cudaFree(attn_q_norm_gpu);
    if (attn_k_norm_gpu) cudaFree(attn_k_norm_gpu);
    if (attn_output_gpu) cudaFree(attn_output_gpu);
    
    if (attn_q_proj_out_gpu) cudaFree(attn_q_proj_out_gpu);
    if (attn_k_proj_out_gpu) cudaFree(attn_k_proj_out_gpu);
    if (attn_v_proj_out_gpu) cudaFree(attn_v_proj_out_gpu);
    if (attn_q_normed_gpu) cudaFree(attn_q_normed_gpu);
    if (attn_k_normed_gpu) cudaFree(attn_k_normed_gpu);
    
    if (attn_q_transposed_gpu) cudaFree(attn_q_transposed_gpu);
    if (attn_k_transposed_gpu) cudaFree(attn_k_transposed_gpu);
    if (attn_k_repeated_gpu) cudaFree(attn_k_repeated_gpu);
    if (attn_v_transposed_gpu) cudaFree(attn_v_transposed_gpu);
    if (attn_scores_gpu) cudaFree(attn_scores_gpu);
    if (attn_out_gpu) cudaFree(attn_out_gpu);
    if (attn_out_transposed_gpu) cudaFree(attn_out_transposed_gpu);
    if (attn_v_repeated_gpu) cudaFree(attn_v_repeated_gpu);

    CHECK_CUDA(cudaMalloc(&attn_x_gpu, batch * seq_len * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn_cos_gpu, seq_len * head_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn_sin_gpu, seq_len * head_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn_q_proj_gpu, num_heads * head_dim * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn_k_proj_gpu, num_kv_heads * head_dim * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn_v_proj_gpu, num_kv_heads * head_dim * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn_o_proj_gpu, hidden_size * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn_q_norm_gpu, head_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn_k_norm_gpu, head_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn_output_gpu, batch * seq_len * hidden_size * sizeof(float)));
    
    CHECK_CUDA(cudaMalloc(&attn_q_proj_out_gpu, batch * seq_len * num_heads * head_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn_k_proj_out_gpu, batch * seq_len * num_kv_heads * head_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn_v_proj_out_gpu, batch * seq_len * num_kv_heads * head_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn_q_normed_gpu, batch * seq_len * num_heads * head_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn_k_normed_gpu, batch * seq_len * num_kv_heads * head_dim * sizeof(float)));
    
    CHECK_CUDA(cudaMalloc(&attn_q_transposed_gpu, batch * num_heads * seq_len * head_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn_k_transposed_gpu, batch * num_kv_heads * seq_len * head_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn_k_repeated_gpu, batch * num_heads * seq_len * head_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn_v_transposed_gpu, batch * num_kv_heads * seq_len * head_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn_scores_gpu, batch * num_heads * seq_len * seq_len * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn_out_gpu, batch * num_heads * seq_len * head_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn_out_transposed_gpu, batch * seq_len * num_heads * head_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn_v_repeated_gpu, batch * num_heads * seq_len * head_dim * sizeof(float)));
}

void Attention::attn(float *x, float *cos, float *sin, float *q_proj, float *k_proj, 
          float *v_proj, float *o_proj, float *q_norm, float *k_norm, 
          float *output, int batch, int seq_len, int num_heads, 
          int head_dim, int num_kv_heads) {
    
    // Lazy initialization check
    static int cached_batch = -1;
    static int cached_seq_len = -1;
    
    if (batch != cached_batch || seq_len != cached_seq_len) {
        attn_initialize_internal(batch, seq_len, num_heads, head_dim, num_kv_heads);
        cached_batch = batch;
        cached_seq_len = seq_len;
    }

    int hidden_size = num_heads * head_dim;
    int kv_dim = num_kv_heads * head_dim;
    
    // Copy input data to GPU
    CHECK_CUDA(cudaMemcpy(attn_x_gpu, x, batch * seq_len * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Copy weights
    CHECK_CUDA(cudaMemcpy(attn_cos_gpu, cos, seq_len * head_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(attn_sin_gpu, sin, seq_len * head_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(attn_q_proj_gpu, q_proj, num_heads * head_dim * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(attn_k_proj_gpu, k_proj, num_kv_heads * head_dim * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(attn_v_proj_gpu, v_proj, num_kv_heads * head_dim * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(attn_o_proj_gpu, o_proj, hidden_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(attn_q_norm_gpu, q_norm, head_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(attn_k_norm_gpu, k_norm, head_dim * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid_q((hidden_size + 15)/16, (batch * seq_len + 15)/16);
    dim3 grid_kv((kv_dim + 15)/16, (batch * seq_len + 15)/16);
    
    // 1. Projections
    attn_matmul_kernel<<<grid_q, block>>>(attn_x_gpu, attn_q_proj_gpu, attn_q_proj_out_gpu, batch * seq_len, hidden_size, hidden_size);
    attn_matmul_kernel<<<grid_kv, block>>>(attn_x_gpu, attn_k_proj_gpu, attn_k_proj_out_gpu, batch * seq_len, kv_dim, hidden_size);
    attn_matmul_kernel<<<grid_kv, block>>>(attn_x_gpu, attn_v_proj_gpu, attn_v_proj_out_gpu, batch * seq_len, kv_dim, hidden_size);
    
    // 2. RMSNorm
    int total_q = batch * seq_len * num_heads;
    int total_k = batch * seq_len * num_kv_heads;
    attn_rmsnorm_kernel<<<total_q, 1>>>(attn_q_proj_out_gpu, attn_q_norm_gpu, attn_q_normed_gpu, total_q, head_dim);
    attn_rmsnorm_kernel<<<total_k, 1>>>(attn_k_proj_out_gpu, attn_k_norm_gpu, attn_k_normed_gpu, total_k, head_dim);
    
    // 3. RoPE
    int threads = 256;
    int blocks_q = (total_q * (head_dim/2) + threads - 1) / threads;
    int blocks_k = (total_k * (head_dim/2) + threads - 1) / threads;
    attn_rope_kernel<<<blocks_q, threads>>>(attn_q_normed_gpu, attn_cos_gpu, attn_sin_gpu, batch, seq_len, num_heads, head_dim);
    attn_rope_kernel<<<blocks_k, threads>>>(attn_k_normed_gpu, attn_cos_gpu, attn_sin_gpu, batch, seq_len, num_kv_heads, head_dim);
    
    // 4. Transpose
    attn_transpose_kernel<<<(batch*seq_len*hidden_size + 255)/256, 256>>>(attn_q_normed_gpu, attn_q_transposed_gpu, batch, seq_len, num_heads, head_dim);
    attn_transpose_kernel<<<(batch*seq_len*kv_dim + 255)/256, 256>>>(attn_k_normed_gpu, attn_k_transposed_gpu, batch, seq_len, num_kv_heads, head_dim);
    attn_transpose_kernel<<<(batch*seq_len*kv_dim + 255)/256, 256>>>(attn_v_proj_out_gpu, attn_v_transposed_gpu, batch, seq_len, num_kv_heads, head_dim);
    
    // 5. Repeat KV
    attn_repeat_kv_kernel<<<(batch*num_heads*seq_len*head_dim + 255)/256, 256>>>(attn_k_transposed_gpu, attn_k_repeated_gpu, batch, num_heads, num_kv_heads, seq_len, head_dim);
    attn_repeat_kv_kernel<<<(batch*num_heads*seq_len*head_dim + 255)/256, 256>>>(attn_v_transposed_gpu, attn_v_repeated_gpu, batch, num_heads, num_kv_heads, seq_len, head_dim);
    
    // 6. Scores
    dim3 grid_scores((seq_len + 15)/16, (seq_len + 15)/16, batch * num_heads);
    attn_batched_matmul_qk_kernel<<<grid_scores, block>>>(attn_q_transposed_gpu, attn_k_repeated_gpu, attn_scores_gpu, seq_len, head_dim, 1.0f / sqrtf(head_dim));
    
    // 7. Softmax
    dim3 grid_softmax((seq_len + 255)/256, batch * num_heads);
    attn_softmax_kernel<<<grid_softmax, 256>>>(attn_scores_gpu, seq_len);
    
    // 8. Output
    dim3 grid_out((head_dim + 15)/16, (seq_len + 15)/16, batch * num_heads);
    attn_batched_matmul_sv_kernel<<<grid_out, block>>>(attn_scores_gpu, attn_v_repeated_gpu, attn_out_gpu, seq_len, head_dim);
    
    // 9. Transpose Back
    attn_transpose_back_kernel<<<(batch*seq_len*hidden_size + 255)/256, 256>>>(attn_out_gpu, attn_out_transposed_gpu, batch, num_heads, seq_len, head_dim);
    
    // 10. Final Projection
    attn_matmul_kernel<<<grid_q, block>>>(attn_out_transposed_gpu, attn_o_proj_gpu, attn_output_gpu, batch * seq_len, hidden_size, hidden_size);
    
    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(output, attn_output_gpu, batch * seq_len * hidden_size * sizeof(float), cudaMemcpyDeviceToHost));
}

// ============================================================================
// Conv Kernels (from tests/conv/conv.cu)
// ============================================================================

// Globals for Conv (renamed)
static float *conv_x_gpu, *conv_conv_weight_gpu, *conv_in_proj_weight_gpu, *conv_out_proj_weight_gpu, *conv_output_gpu;
static float *conv_in_proj_out_gpu;
static float *conv_Bx_gpu;
static float *conv_C_out_gpu;
static float *conv_conv_out_gpu;
static float *conv_y_pre_gpu;

// Kernels (renamed)
__global__ void conv_matmul_kernel(float *A, float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) sum += A[row * K + i] * B[i * N + col];
        C[row * N + col] = sum;
    }
}

__global__ void conv_pre_conv_gating_kernel(float *in_proj_out, float *Bx, float *C_out, int batch, int seq_len, int hidden_size) {
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

__global__ void conv_depthwise_conv1d_kernel(float *Bx, float *W, float *ConvOut, int batch, int seq_len, int hidden_size, int kernel_size) {
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

__global__ void conv_post_conv_gating_kernel(float *ConvOut, float *C, float *Y_pre, int batch, int seq_len, int hidden_size) {
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

// Initialization helper
static void conv_initialize_internal(int batch, int seq_len, int hidden_size, int kernel_size) {
    if (conv_x_gpu) cudaFree(conv_x_gpu);
    if (conv_conv_weight_gpu) cudaFree(conv_conv_weight_gpu);
    if (conv_in_proj_weight_gpu) cudaFree(conv_in_proj_weight_gpu);
    if (conv_out_proj_weight_gpu) cudaFree(conv_out_proj_weight_gpu);
    if (conv_output_gpu) cudaFree(conv_output_gpu);
    
    if (conv_in_proj_out_gpu) cudaFree(conv_in_proj_out_gpu);
    if (conv_Bx_gpu) cudaFree(conv_Bx_gpu);
    if (conv_C_out_gpu) cudaFree(conv_C_out_gpu);
    if (conv_conv_out_gpu) cudaFree(conv_conv_out_gpu);
    if (conv_y_pre_gpu) cudaFree(conv_y_pre_gpu);

    CHECK_CUDA(cudaMalloc(&conv_x_gpu, batch * seq_len * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&conv_conv_weight_gpu, hidden_size * kernel_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&conv_in_proj_weight_gpu, 3 * hidden_size * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&conv_out_proj_weight_gpu, hidden_size * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&conv_output_gpu, batch * seq_len * hidden_size * sizeof(float)));
    
    CHECK_CUDA(cudaMalloc(&conv_in_proj_out_gpu, batch * seq_len * 3 * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&conv_Bx_gpu, batch * hidden_size * seq_len * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&conv_C_out_gpu, batch * hidden_size * seq_len * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&conv_conv_out_gpu, batch * hidden_size * seq_len * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&conv_y_pre_gpu, batch * seq_len * hidden_size * sizeof(float)));
}

void ShortConv::conv(float *x, float *conv_weight, float *in_proj_weight, float *out_proj_weight,
          float *output, int batch, int seq_len, int hidden_size, int kernel_size) {
    
    static int cached_batch = -1;
    static int cached_seq_len = -1;
    
    if (batch != cached_batch || seq_len != cached_seq_len) {
        conv_initialize_internal(batch, seq_len, hidden_size, kernel_size);
        cached_batch = batch;
        cached_seq_len = seq_len;
    }
    
    // Copy input data to GPU
    CHECK_CUDA(cudaMemcpy(conv_x_gpu, x, batch * seq_len * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Copy weights
    CHECK_CUDA(cudaMemcpy(conv_conv_weight_gpu, conv_weight, hidden_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(conv_in_proj_weight_gpu, in_proj_weight, 3 * hidden_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(conv_out_proj_weight_gpu, out_proj_weight, hidden_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // 1. Input Projection
    dim3 block(16, 16);
    dim3 grid_proj((3 * hidden_size + 15)/16, (batch * seq_len + 15)/16);
    conv_matmul_kernel<<<grid_proj, block>>>(conv_x_gpu, conv_in_proj_weight_gpu, conv_in_proj_out_gpu, batch * seq_len, 3 * hidden_size, hidden_size);
    
    // 2. Pre-Conv Gating
    int total = batch * seq_len * hidden_size;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    conv_pre_conv_gating_kernel<<<blocks, threads>>>(conv_in_proj_out_gpu, conv_Bx_gpu, conv_C_out_gpu, batch, seq_len, hidden_size);
    
    // 3. Depthwise Conv
    conv_depthwise_conv1d_kernel<<<blocks, threads>>>(conv_Bx_gpu, conv_conv_weight_gpu, conv_conv_out_gpu, batch, seq_len, hidden_size, kernel_size);
    
    // 4. Post-Conv Gating
    conv_post_conv_gating_kernel<<<blocks, threads>>>(conv_conv_out_gpu, conv_C_out_gpu, conv_y_pre_gpu, batch, seq_len, hidden_size);
    
    // 5. Output Projection
    dim3 grid_out((hidden_size + 15)/16, (batch * seq_len + 15)/16);
    conv_matmul_kernel<<<grid_out, block>>>(conv_y_pre_gpu, conv_out_proj_weight_gpu, conv_output_gpu, batch * seq_len, hidden_size, hidden_size);

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(output, conv_output_gpu, batch * seq_len * hidden_size * sizeof(float), cudaMemcpyDeviceToHost));
}

// ============================================================================
// MoE Kernels (from tests/moe/moe.cu)
// ============================================================================

// Globals for MoE (renamed)
static float *moe_x_gpu, *moe_gate_gpu, *moe_expert_bias_gpu, *moe_output_gpu;
static float **moe_expert_w1_gpu, **moe_expert_w2_gpu, **moe_expert_w3_gpu;
static float **moe_expert_w1_gpu_ptrs, **moe_expert_w2_gpu_ptrs, **moe_expert_w3_gpu_ptrs;
static int moe_g_num_experts = 0;

// Intermediate buffers
static float *moe_router_logits_gpu;
static int *moe_top_k_indices_gpu;
static float *moe_top_k_weights_gpu;
static int *moe_host_top_k_indices;
static float *moe_host_top_k_weights;
static float *moe_expert_in_gpu;
static float *moe_expert_gate_gpu;
static float *moe_expert_up_gpu;
static float *moe_expert_hidden_gpu;
static float *moe_expert_out_gpu;

// MoE configuration flags
static const float MOE_ROUTED_SCALING_FACTOR = 1.0f;
static const bool MOE_NORM_TOPK_PROB = true;
static const bool MOE_USE_EXPERT_BIAS = true;

// Kernels (renamed)
__global__ void moe_matmul_transposed_B_kernel(float *A, float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) sum += A[row * K + i] * B[col * K + i];
        C[row * N + col] = sum;
    }
}

__global__ void moe_router_kernel(float *logits, float *bias, int *top_k_indices, float *top_k_weights, 
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

__global__ void moe_gather_kernel(float *x, float *expert_in, int *indices, int count, int hidden_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        int token_idx = indices[idx];
        for (int h = 0; h < hidden_size; ++h) {
            expert_in[idx * hidden_size + h] = x[token_idx * hidden_size + h];
        }
    }
}

__global__ void moe_silu_mul_kernel(float *gate, float *up, float *out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = gate[idx];
        float u = up[idx];
        float sig = 1.0f / (1.0f + expf(-g));
        out[idx] = (g * sig) * u;
    }
}

__global__ void moe_scatter_add_kernel(float *expert_out, float *output, int *indices, float *weights, int count, int hidden_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        int token_idx = indices[idx];
        float w = weights[idx];
        for (int h = 0; h < hidden_size; ++h) {
            atomicAdd(&output[token_idx * hidden_size + h], expert_out[idx * hidden_size + h] * w);
        }
    }
}

// Initialization helper
static void moe_initialize_internal(int batch, int seq_len, int hidden_size, int num_experts, 
                   int num_experts_per_tok, int expert_hidden_size) {
    
    // Only re-allocate if dimensions change (or first time)
    // For simplicity, we check if num_experts changed, or batch/seq_len changed
    // But we need to be careful about freeing expert weights
    
    if (moe_g_num_experts > 0) {
        // Free previous allocations
        if (moe_x_gpu) cudaFree(moe_x_gpu);
        if (moe_gate_gpu) cudaFree(moe_gate_gpu);
        if (moe_expert_bias_gpu) cudaFree(moe_expert_bias_gpu);
        if (moe_output_gpu) cudaFree(moe_output_gpu);
        
        if (moe_router_logits_gpu) cudaFree(moe_router_logits_gpu);
        if (moe_top_k_indices_gpu) cudaFree(moe_top_k_indices_gpu);
        if (moe_top_k_weights_gpu) cudaFree(moe_top_k_weights_gpu);
        if (moe_host_top_k_indices) cudaFreeHost(moe_host_top_k_indices);
        if (moe_host_top_k_weights) cudaFreeHost(moe_host_top_k_weights);
        
        if (moe_expert_in_gpu) cudaFree(moe_expert_in_gpu);
        if (moe_expert_gate_gpu) cudaFree(moe_expert_gate_gpu);
        if (moe_expert_up_gpu) cudaFree(moe_expert_up_gpu);
        if (moe_expert_hidden_gpu) cudaFree(moe_expert_hidden_gpu);
        if (moe_expert_out_gpu) cudaFree(moe_expert_out_gpu);
        
        for (int i = 0; i < moe_g_num_experts; i++) {
            if (moe_expert_w1_gpu[i]) cudaFree(moe_expert_w1_gpu[i]);
            if (moe_expert_w2_gpu[i]) cudaFree(moe_expert_w2_gpu[i]);
            if (moe_expert_w3_gpu[i]) cudaFree(moe_expert_w3_gpu[i]);
        }
        if (moe_expert_w1_gpu) free(moe_expert_w1_gpu);
        if (moe_expert_w2_gpu) free(moe_expert_w2_gpu);
        if (moe_expert_w3_gpu) free(moe_expert_w3_gpu);
        
        if (moe_expert_w1_gpu_ptrs) cudaFree(moe_expert_w1_gpu_ptrs);
        if (moe_expert_w2_gpu_ptrs) cudaFree(moe_expert_w2_gpu_ptrs);
        if (moe_expert_w3_gpu_ptrs) cudaFree(moe_expert_w3_gpu_ptrs);
    }

    moe_g_num_experts = num_experts;
    
    CHECK_CUDA(cudaMalloc(&moe_x_gpu, batch * seq_len * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&moe_gate_gpu, num_experts * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&moe_expert_bias_gpu, num_experts * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&moe_output_gpu, batch * seq_len * hidden_size * sizeof(float)));
    
    int num_tokens = batch * seq_len;
    CHECK_CUDA(cudaMalloc(&moe_router_logits_gpu, num_tokens * num_experts * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&moe_top_k_indices_gpu, num_tokens * num_experts_per_tok * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&moe_top_k_weights_gpu, num_tokens * num_experts_per_tok * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&moe_host_top_k_indices, num_tokens * num_experts_per_tok * sizeof(int)));
    CHECK_CUDA(cudaMallocHost(&moe_host_top_k_weights, num_tokens * num_experts_per_tok * sizeof(float)));
    
    CHECK_CUDA(cudaMalloc(&moe_expert_in_gpu, num_tokens * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&moe_expert_gate_gpu, num_tokens * expert_hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&moe_expert_up_gpu, num_tokens * expert_hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&moe_expert_hidden_gpu, num_tokens * expert_hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&moe_expert_out_gpu, num_tokens * hidden_size * sizeof(float)));
    
    moe_expert_w1_gpu = (float**)malloc(num_experts * sizeof(float*));
    moe_expert_w2_gpu = (float**)malloc(num_experts * sizeof(float*));
    moe_expert_w3_gpu = (float**)malloc(num_experts * sizeof(float*));
    
    for (int i = 0; i < num_experts; i++) {
        CHECK_CUDA(cudaMalloc(&moe_expert_w1_gpu[i], expert_hidden_size * hidden_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&moe_expert_w2_gpu[i], hidden_size * expert_hidden_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&moe_expert_w3_gpu[i], expert_hidden_size * hidden_size * sizeof(float)));
    }
    
    CHECK_CUDA(cudaMalloc(&moe_expert_w1_gpu_ptrs, num_experts * sizeof(float*)));
    CHECK_CUDA(cudaMalloc(&moe_expert_w2_gpu_ptrs, num_experts * sizeof(float*)));
    CHECK_CUDA(cudaMalloc(&moe_expert_w3_gpu_ptrs, num_experts * sizeof(float*)));
}

void SparseMoeBlock::moe(float *x, float *gate, float **expert_w1, float **expert_w2, float **expert_w3,
         float *expert_bias, float *output, float *router_logits_out, int batch, int seq_len, int hidden_size, 
         int num_experts, int num_experts_per_tok, int expert_hidden_size) {
    
    static int cached_batch = -1;
    static int cached_seq_len = -1;
    
    // Initialize if needed
    if (batch != cached_batch || seq_len != cached_seq_len || num_experts != moe_g_num_experts) {
        moe_initialize_internal(batch, seq_len, hidden_size, num_experts, num_experts_per_tok, expert_hidden_size);
        cached_batch = batch;
        cached_seq_len = seq_len;
    }
    
    // Copy weights (ALWAYS, because they might change per layer)
    CHECK_CUDA(cudaMemcpy(moe_gate_gpu, gate, num_experts * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(moe_expert_bias_gpu, expert_bias, num_experts * sizeof(float), cudaMemcpyHostToDevice));
    
    for (int i = 0; i < num_experts; i++) {
        CHECK_CUDA(cudaMemcpy(moe_expert_w1_gpu[i], expert_w1[i], expert_hidden_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(moe_expert_w2_gpu[i], expert_w2[i], hidden_size * expert_hidden_size * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(moe_expert_w3_gpu[i], expert_w3[i], expert_hidden_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    }
    
    // Update pointers on device
    CHECK_CUDA(cudaMemcpy(moe_expert_w1_gpu_ptrs, moe_expert_w1_gpu, num_experts * sizeof(float*), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(moe_expert_w2_gpu_ptrs, moe_expert_w2_gpu, num_experts * sizeof(float*), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(moe_expert_w3_gpu_ptrs, moe_expert_w3_gpu, num_experts * sizeof(float*), cudaMemcpyHostToDevice));
    
    int num_tokens = batch * seq_len;
    
    // Initialize output to zero
    CHECK_CUDA(cudaMemset(moe_output_gpu, 0, num_tokens * hidden_size * sizeof(float)));
    
    // Copy input data to GPU
    CHECK_CUDA(cudaMemcpy(moe_x_gpu, x, num_tokens * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // 1. Router Logits
    dim3 block(16, 16);
    dim3 grid_logits((num_experts + 15)/16, (num_tokens + 15)/16);
    moe_matmul_transposed_B_kernel<<<grid_logits, block>>>(moe_x_gpu, moe_gate_gpu, moe_router_logits_gpu, num_tokens, num_experts, hidden_size);
    
    // Copy router logits to output if requested
    if (router_logits_out != nullptr) {
        CHECK_CUDA(cudaMemcpy(router_logits_out, moe_router_logits_gpu, num_tokens * num_experts * sizeof(float), cudaMemcpyDeviceToHost));
    }
    
    // 2. Router Top-K
    int threads = 256;
    int blocks = (num_tokens + threads - 1) / threads;
    moe_router_kernel<<<blocks, threads>>>(moe_router_logits_gpu, moe_expert_bias_gpu, moe_top_k_indices_gpu, moe_top_k_weights_gpu, 
                                       num_tokens, num_experts, num_experts_per_tok, true); // USE_EXPERT_BIAS=true
    
    // 3. Copy to Host
    CHECK_CUDA(cudaMemcpy(moe_host_top_k_indices, moe_top_k_indices_gpu, num_tokens * num_experts_per_tok * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(moe_host_top_k_weights, moe_top_k_weights_gpu, num_tokens * num_experts_per_tok * sizeof(float), cudaMemcpyDeviceToHost));
    
    // 4. Host Scheduling
    std::vector<std::vector<int>> expert_token_indices(num_experts);
    std::vector<std::vector<float>> expert_token_weights(num_experts);
    
    for (int t = 0; t < num_tokens; ++t) {
        for (int k = 0; k < num_experts_per_tok; ++k) {
            int expert_idx = moe_host_top_k_indices[t * num_experts_per_tok + k];
            float weight = moe_host_top_k_weights[t * num_experts_per_tok + k];
            expert_token_indices[expert_idx].push_back(t);
            expert_token_weights[expert_idx].push_back(weight);
        }
    }
    
    // 5. Expert Execution
    // Reuse top_k buffers as scratch
    int *scratch_indices = moe_top_k_indices_gpu;
    float *scratch_weights = moe_top_k_weights_gpu;
    
    for (int e = 0; e < num_experts; ++e) {
        int count = expert_token_indices[e].size();
        if (count == 0) continue;
        
        CHECK_CUDA(cudaMemcpy(scratch_indices, expert_token_indices[e].data(), count * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(scratch_weights, expert_token_weights[e].data(), count * sizeof(float), cudaMemcpyHostToDevice));
        
        int gather_blocks = (count + threads - 1) / threads;
        moe_gather_kernel<<<gather_blocks, threads>>>(moe_x_gpu, moe_expert_in_gpu, scratch_indices, count, hidden_size);
        
        dim3 grid_gate((expert_hidden_size + 15)/16, (count + 15)/16);
        dim3 grid_out((hidden_size + 15)/16, (count + 15)/16);
        
        moe_matmul_transposed_B_kernel<<<grid_gate, block>>>(moe_expert_in_gpu, moe_expert_w1_gpu[e], moe_expert_gate_gpu, count, expert_hidden_size, hidden_size);
        moe_matmul_transposed_B_kernel<<<grid_gate, block>>>(moe_expert_in_gpu, moe_expert_w3_gpu[e], moe_expert_up_gpu, count, expert_hidden_size, hidden_size);
        
        int silu_blocks = (count * expert_hidden_size + threads - 1) / threads;
        moe_silu_mul_kernel<<<silu_blocks, threads>>>(moe_expert_gate_gpu, moe_expert_up_gpu, moe_expert_hidden_gpu, count * expert_hidden_size);
        
        moe_matmul_transposed_B_kernel<<<grid_out, block>>>(moe_expert_hidden_gpu, moe_expert_w2_gpu[e], moe_expert_out_gpu, count, hidden_size, expert_hidden_size);
        
        moe_scatter_add_kernel<<<gather_blocks, threads>>>(moe_expert_out_gpu, moe_output_gpu, scratch_indices, scratch_weights, count, hidden_size);
    }
    
    // Copy output back to host
    CHECK_CUDA(cudaMemcpy(output, moe_output_gpu, num_tokens * hidden_size * sizeof(float), cudaMemcpyDeviceToHost));
}
