#include "layer.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

// ============================================================================
// Tensor Operations - Basic operations on tensors
// ============================================================================

namespace tensor_ops {

// ============================================================================
// CUDA kernels
// ============================================================================

inline dim3 make_grid_2d(size_t x, size_t y, int tx = 16, int ty = 16) {
    return dim3((x + tx - 1) / tx, (y + ty - 1) / ty);
}

__global__ void matmul_kernel(const float* A, const float* B, float* C,
                              size_t m, size_t k, size_t n) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        float sum = 0.0f;
        for (size_t p = 0; p < k; ++p) {
            sum += A[row * k + p] * B[p * n + col];
        }
        C[row * n + col] = sum;
    }
}

__global__ void matmul_transpose_kernel(const float* A, const float* B, float* C,
                                    size_t m, size_t k, size_t n) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        float sum = 0.0f;
        for (size_t p = 0; p < k; ++p) {
            sum += A[row * k + p] * B[col * k + p];
        }
        C[row * n + col] = sum;
    }
}

__global__ void add_kernel(const float* A, const float* B, float* C, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] = A[idx] + B[idx];
}

__global__ void add_scalar_kernel(const float* A, float b, float* C, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] = A[idx] + b;
}

__global__ void mul_kernel(const float* A, const float* B, float* C, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] = A[idx] * B[idx];
}

__global__ void mul_scalar_kernel(const float* A, float b, float* C, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] = A[idx] * b;
}

__global__ void sigmoid_kernel(const float* X, float* Y, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = X[idx];
        Y[idx] = 1.0f / (1.0f + expf(-v));
    }
}

__global__ void silu_kernel(const float* X, float* Y, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = X[idx];
        Y[idx] = v / (1.0f + expf(-v));
    }
}

__global__ void softmax_kernel(const float* X, float* Y, size_t inner) {
    size_t row = blockIdx.x;
    const float* x_row = X + row * inner;
    float* y_row = Y + row * inner;
    float m = x_row[0];
    for (size_t j = 1; j < inner; ++j) m = fmaxf(m, x_row[j]);
    float sum = 0.0f;
    for (size_t j = 0; j < inner; ++j) {
        float e = expf(x_row[j] - m);
        y_row[j] = e;
        sum += e;
    }
    float inv = 1.0f / sum;
    for (size_t j = 0; j < inner; ++j) y_row[j] *= inv;
}

__global__ void rms_norm_kernel(const float* X, const float* W, float eps,
                                float* Y, size_t hidden, size_t rows) {
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        const float* x = X + row * hidden;
        float* y = Y + row * hidden;
        float sum = 0.0f;
        for (size_t j = 0; j < hidden; ++j) sum += x[j] * x[j];
        float rms = rsqrtf(sum / hidden + eps);
        for (size_t j = 0; j < hidden; ++j) y[j] = x[j] * rms * W[j];
    }
}

__global__ void rope_compute_kernel(float* cos_out, float* sin_out,
                                    size_t head_dim, size_t max_seq, float theta) {
    size_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= max_seq) return;
    size_t half = head_dim / 2;
    for (size_t i = 0; i < half; ++i) {
        float inv_freq = 1.0f / powf(theta, (float)(2 * i) / head_dim);
        float angle = pos * inv_freq;
        cos_out[pos * head_dim + i] = cosf(angle);
        cos_out[pos * head_dim + i + half] = cosf(angle);
        sin_out[pos * head_dim + i] = sinf(angle);
        sin_out[pos * head_dim + i + half] = sinf(angle);
    }
}

__global__ void rope_apply_kernel(float* q, float* k, const float* cos, const float* sin,
                                  size_t batch, size_t qh, size_t kh,
                                  size_t seq_len, size_t head_dim) {
    size_t b = blockIdx.x;
    size_t h = blockIdx.y;
    if (b >= batch) return;
    size_t half = head_dim / 2;
    size_t q_stride = seq_len * head_dim;
    size_t k_stride = seq_len * head_dim;
    for (size_t s = threadIdx.x; s < seq_len; s += blockDim.x) {
        float* q_base = q + (b * qh + h) * q_stride + s * head_dim;
        const float* cos_row = cos + s * head_dim;
        const float* sin_row = sin + s * head_dim;
        if (h < qh) {
            for (size_t d = 0; d < half; ++d) {
                float q1 = q_base[d];
                float q2 = q_base[d + half];
                q_base[d] = q1 * cos_row[d] - q2 * sin_row[d];
                q_base[d + half] = q2 * cos_row[d + half] + q1 * sin_row[d + half];
            }
        }
        if (h < kh) {
            float* k_base = k + (b * kh + h) * k_stride + s * head_dim;
            for (size_t d = 0; d < half; ++d) {
                float k1 = k_base[d];
                float k2 = k_base[d + half];
                k_base[d] = k1 * cos_row[d] - k2 * sin_row[d];
                k_base[d + half] = k2 * cos_row[d + half] + k1 * sin_row[d + half];
            }
        }
    }
}

__global__ void repeat_kv_kernel(const float* x, float* y,
                                 size_t batch, size_t kv_heads, size_t n_rep,
                                 size_t seq_len, size_t head_dim) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch * kv_heads * n_rep * seq_len * head_dim;
    if (idx >= total) return;
    size_t d = idx % head_dim;
    size_t s = (idx / head_dim) % seq_len;
    size_t h_rep = (idx / head_dim / seq_len) % (kv_heads * n_rep);
    size_t h = h_rep / n_rep;
    size_t b = idx / head_dim / seq_len / (kv_heads * n_rep);
    size_t x_idx = (((b * kv_heads + h) * seq_len + s) * head_dim + d);
    y[idx] = x[x_idx];
}

__global__ void causal_conv1d_kernel(const float* x, const float* w, const float* bias,
                                     float* y, size_t batch, size_t ch,
                                     size_t seq, size_t ksz) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch * ch * seq;
    if (idx >= total) return;
    size_t s = idx % seq;
    size_t c = (idx / seq) % ch;
    size_t b = idx / (seq * ch);
    float sum = 0.0f;
    const float* x_base = x + (b * ch + c) * seq;
    const float* w_base = w + c * ksz;
    for (size_t k = 0; k < ksz; ++k) {
        int input_pos = (int)s - ((int)ksz - 1) + (int)k;
        if (input_pos >= 0) sum += x_base[input_pos] * w_base[k];
    }
    if (bias) sum += bias[c];
    y[idx] = sum;
}

// Matrix operations
void matmul(const Tensor& a, const Tensor& b, Tensor& c) {
    // a: (m, k), b: (k, n), c: (m, n)
    size_t m = a.size(0);
    size_t k = a.size(1);
    size_t n = b.size(1);
    if (c.size() == 0) c = Tensor({m, n});
    a.to_device(-1);  // Use current device
    b.to_device(-1);
    c.to_device(-1);
    dim3 block(16, 16);
    dim3 grid = make_grid_2d(n, m, block.x, block.y);
    matmul_kernel<<<grid, block>>>(a.device_data(), b.device_data(), c.device_data(), m, k, n);
    CHECK_CUDA(cudaDeviceSynchronize());
    c.sync_host_from_device();
}

void matmul_transposed(const Tensor& a, const Tensor& b, Tensor& c) {
    // a: (m, k), b: (n, k), c: (m, n)  [c = a @ b^T]
    size_t m = a.size(0);
    size_t k = a.size(1);
    size_t n = b.size(0);
    if (c.size() == 0) c = Tensor({m, n});
    a.to_device(-1);
    b.to_device(-1);
    c.to_device(-1);
    dim3 block(16, 16);
    dim3 grid = make_grid_2d(n, m, block.x, block.y);
    matmul_transpose_kernel<<<grid, block>>>(a.device_data(), b.device_data(), c.device_data(), m, k, n);
    CHECK_CUDA(cudaDeviceSynchronize());
    c.sync_host_from_device();
}

// Element-wise operations
void add(const Tensor& a, const Tensor& b, Tensor& c) {
    size_t n = a.size();
    if (c.size() == 0) c = Tensor(a.shape());
    a.to_device(-1);
    b.to_device(-1);
    c.to_device(-1);
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    add_kernel<<<grid, block>>>(a.device_data(), b.device_data(), c.device_data(), n);
    CHECK_CUDA(cudaDeviceSynchronize());
    c.sync_host_from_device();
}

void add_scalar(const Tensor& a, float b, Tensor& c) {
    size_t n = a.size();
    if (c.size() == 0) c = Tensor(a.shape());
    a.to_device(-1);
    c.to_device(-1);
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    add_scalar_kernel<<<grid, block>>>(a.device_data(), b, c.device_data(), n);
    CHECK_CUDA(cudaDeviceSynchronize());
    c.sync_host_from_device();
}

void mul(const Tensor& a, const Tensor& b, Tensor& c) {
    size_t n = a.size();
    if (c.size() == 0) c = Tensor(a.shape());
    a.to_device(-1);
    b.to_device(-1);
    c.to_device(-1);
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    mul_kernel<<<grid, block>>>(a.device_data(), b.device_data(), c.device_data(), n);
    CHECK_CUDA(cudaDeviceSynchronize());
    c.sync_host_from_device();
}

void mul_scalar(const Tensor& a, float b, Tensor& c) {
    size_t n = a.size();
    if (c.size() == 0) c = Tensor(a.shape());
    a.to_device(-1);
    c.to_device(-1);
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    mul_scalar_kernel<<<grid, block>>>(a.device_data(), b, c.device_data(), n);
    CHECK_CUDA(cudaDeviceSynchronize());
    c.sync_host_from_device();
}

// Activation functions
void sigmoid(const Tensor& x, Tensor& y) {
    size_t n = x.size();
    if (y.size() == 0) y = Tensor(x.shape());
    x.to_device(-1);
    y.to_device(-1);
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    sigmoid_kernel<<<grid, block>>>(x.device_data(), y.device_data(), n);
    CHECK_CUDA(cudaDeviceSynchronize());
    y.sync_host_from_device();
}

void silu(const Tensor& x, Tensor& y) {
    size_t n = x.size();
    if (y.size() == 0) y = Tensor(x.shape());
    x.to_device(-1);
    y.to_device(-1);
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    silu_kernel<<<grid, block>>>(x.device_data(), y.device_data(), n);
    CHECK_CUDA(cudaDeviceSynchronize());
    y.sync_host_from_device();
}

void softmax(const Tensor& x, Tensor& y, int dim) {
    // For simplicity, assume dim=-1 (last dimension)
    size_t outer_size = 1;
    for (size_t i = 0; i < x.ndim() - 1; i++) {
        outer_size *= x.size(i);
    }
    size_t inner_size = x.size(-1);
    if (y.size() == 0) y = Tensor(x.shape());
    x.to_device(-1);
    y.to_device(-1);
    dim3 block(inner_size >= 1024 ? 1024 : inner_size);
    dim3 grid(outer_size);
    softmax_kernel<<<grid, block>>>(x.device_data(), y.device_data(), inner_size);
    CHECK_CUDA(cudaDeviceSynchronize());
    y.sync_host_from_device();
}

// Normalization
void rms_norm(const Tensor& x, const Tensor& weight, float eps, Tensor& y) {
    size_t outer_size = 1;
    for (size_t i = 0; i < x.ndim() - 1; i++) {
        outer_size *= x.size(i);
    }
    size_t hidden_size = x.size(-1);
    if (y.size() == 0) y = Tensor(x.shape());
    x.to_device(-1);
    weight.to_device(-1);
    y.to_device(-1);
    dim3 block(128);
    dim3 grid((outer_size + block.x - 1) / block.x);
    rms_norm_kernel<<<grid, block>>>(x.device_data(), weight.device_data(), eps,
                                     y.device_data(), hidden_size, outer_size);
    CHECK_CUDA(cudaDeviceSynchronize());
    y.sync_host_from_device();
}

// RoPE operations
void compute_rope_embeddings(size_t head_dim, size_t max_seq_len, float theta,
                             Tensor& cos, Tensor& sin) {
    if (cos.size() == 0) cos = Tensor({max_seq_len, head_dim});
    if (sin.size() == 0) sin = Tensor({max_seq_len, head_dim});
    cos.to_device(-1);
    sin.to_device(-1);
    dim3 block(256);
    dim3 grid((max_seq_len + block.x - 1) / block.x);
    rope_compute_kernel<<<grid, block>>>(cos.device_data(), sin.device_data(), head_dim, max_seq_len, theta);
    CHECK_CUDA(cudaDeviceSynchronize());
    cos.sync_host_from_device();
    sin.sync_host_from_device();
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
    q.to_device(-1);
    k.to_device(-1);
    cos.to_device(-1);
    sin.to_device(-1);
    size_t h_max = std::max(num_q_heads, num_kv_heads);
    dim3 grid(batch, h_max);
    dim3 block(seq_len);
    if (block.x > 256) block.x = 256;
    rope_apply_kernel<<<grid, block>>>(q.device_data(), k.device_data(),
                                       cos.device_data(), sin.device_data(),
                                       batch, num_q_heads, num_kv_heads,
                                       seq_len, head_dim);
    CHECK_CUDA(cudaDeviceSynchronize());
    q.sync_host_from_device();
    k.sync_host_from_device();
}

// Grouped Query Attention operations
void repeat_kv(const Tensor& x, size_t n_rep, Tensor& y) {
    size_t batch = x.size(0);
    size_t num_kv_heads = x.size(1);
    size_t seq_len = x.size(2);
    size_t head_dim = x.size(3);
    if (y.size() == 0) y = Tensor({batch, num_kv_heads * n_rep, seq_len, head_dim});
    x.to_device(-1);
    y.to_device(-1);
    size_t total = batch * num_kv_heads * n_rep * seq_len * head_dim;
    dim3 block(256);
    dim3 grid((total + block.x - 1) / block.x);
    repeat_kv_kernel<<<grid, block>>>(x.device_data(), y.device_data(),
                                      batch, num_kv_heads, n_rep, seq_len, head_dim);
    CHECK_CUDA(cudaDeviceSynchronize());
    y.sync_host_from_device();
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
    if (y.size() == 0) y = Tensor({batch, channels, seq_len});
    x.to_device(-1);
    weight.to_device(-1);
    if (bias) bias->to_device(-1);
    y.to_device(-1);
    size_t total = batch * channels * seq_len;
    dim3 block(256);
    dim3 grid((total + block.x - 1) / block.x);
    causal_conv1d_kernel<<<grid, block>>>(x.device_data(), weight.device_data(),
                                          bias ? bias->device_data() : nullptr,
                                          y.device_data(), batch, channels, seq_len, kernel_size);
    CHECK_CUDA(cudaDeviceSynchronize());
    y.sync_host_from_device();
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

