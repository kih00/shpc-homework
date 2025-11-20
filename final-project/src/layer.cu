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

__global__ void matmul_kernel(float *A, float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) sum += A[row * K + i] * B[i * N + col];
        C[row * N + col] = sum;
    }
}

__global__ void matmul_transposed_kernel(float *A, float *B, float *C, int M, int N, int K) {
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

__global__ void add_kernel(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void add_scalar_kernel(float *a, float b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b;
    }
}

__global__ void mul_kernel(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

__global__ void mul_scalar_kernel(float *a, float b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b;
    }
}

__global__ void silu_kernel(float *x, float *y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx];
        y[idx] = val / (1.0f + expf(-val));
    }
}

__global__ void sigmoid_kernel(float *x, float *y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = 1.0f / (1.0f + expf(-x[idx]));
    }
}

__global__ void softmax_kernel(float *x, float *y, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float max_val = -1e20f;
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

__global__ void depthwise_conv1d_kernel(float *Bx, float *W, float *bias, float *ConvOut, int batch, int seq_len, int hidden_size, int kernel_size) {
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
        if (bias != nullptr) {
            sum += bias[h];
        }
        ConvOut[idx] = sum;
    }
}

// ============================================================================
// Tensor Operations - Basic operations on tensors
// ============================================================================

__global__ void add_bias_kernel(float* a, const float* bias, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int col = idx % cols;
        a[idx] += bias[col];
    }
}

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
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(a.data(), b.data(), c.data(), n);
}

void add_scalar(const Tensor& a, float b, Tensor& c) {
    int n = a.size();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_scalar_kernel<<<blocks, threads>>>(a.data(), b, c.data(), n);
}

void add_bias(const Tensor& a, const Tensor& bias, Tensor& c) {
    // a: (rows, cols), bias: (cols)
    // Broadcast bias addition: c = a + bias
    
    int rows = a.size(0);
    int cols = a.size(1); // Flattened if > 2D?
    // If a is (B*S, H), bias is (H).
    // If a is (B, S, H), bias is (H).
    // We treat it as (rows, cols) where cols = last dim.
    cols = a.shape().back();
    rows = a.size() / cols;
    
    int threads = 256;
    int blocks = (rows * cols + threads - 1) / threads;
    
    // Ensure c contains a's data if not in-place
    if (c.data() != a.data()) {
        cudaMemcpy(c.data(), a.data(), a.size() * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    add_bias_kernel<<<blocks, threads>>>(c.data(), bias.data(), rows, cols);
}

void mul(const Tensor& a, const Tensor& b, Tensor& c) {
    int n = a.size();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    mul_kernel<<<blocks, threads>>>(a.data(), b.data(), c.data(), n);
}

void mul_scalar(const Tensor& a, float b, Tensor& c) {
    int n = a.size();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    mul_scalar_kernel<<<blocks, threads>>>(a.data(), b, c.data(), n);
}

// Activation functions
void sigmoid(const Tensor& x, Tensor& y) {
    int n = x.size();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    sigmoid_kernel<<<blocks, threads>>>(x.data(), y.data(), n);
}

void silu(const Tensor& x, Tensor& y) {
    int n = x.size();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    silu_kernel<<<blocks, threads>>>(x.data(), y.data(), n);
}

void softmax(const Tensor& x, Tensor& y, int dim) {
    // Assume dim=-1
    int outer_size = 1;
    for (size_t i = 0; i < x.ndim() - 1; i++) {
        outer_size *= x.size(i);
    }
    int inner_size = x.size(-1);
    
    int threads = 256;
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
    
    int threads = 256;
    int blocks = (outer_size + threads - 1) / threads;
    rmsnorm_kernel<<<blocks, threads>>>(x.data(), weight.data(), y.data(), outer_size, hidden_size);
}

// RoPE operations
void compute_rope_embeddings(size_t head_dim, size_t max_seq_len, float theta,
                             Tensor& cos, Tensor& sin) {
    // Compute on Host and copy to Device
    
    std::vector<float> h_cos(max_seq_len * head_dim);
    std::vector<float> h_sin(max_seq_len * head_dim);
    
    std::vector<float> inv_freq(head_dim / 2);
    for (size_t i = 0; i < head_dim / 2; i++) {
        inv_freq[i] = 1.0f / std::pow(theta, (float)(2 * i) / head_dim);
    }
    
    for (size_t pos = 0; pos < max_seq_len; pos++) {
        for (size_t i = 0; i < head_dim / 2; i++) {
            float angle = pos * inv_freq[i];
            h_cos[pos * head_dim + i] = std::cos(angle);
            h_cos[pos * head_dim + i + head_dim / 2] = std::cos(angle);
            h_sin[pos * head_dim + i] = std::sin(angle);
            h_sin[pos * head_dim + i + head_dim / 2] = std::sin(angle);
        }
    }
    
    cudaMemcpy(cos.data(), h_cos.data(), h_cos.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(sin.data(), h_sin.data(), h_sin.size() * sizeof(float), cudaMemcpyHostToDevice);
}

void apply_rotary_pos_emb(Tensor& q, Tensor& k, const Tensor& cos, const Tensor& sin) {
    int batch = q.size(0);
    int num_q_heads = q.size(1);
    int num_kv_heads = k.size(1);
    int seq_len = q.size(2);
    int head_dim = q.size(3);
    
    int total_q = batch * seq_len * num_q_heads;
    int total_k = batch * seq_len * num_kv_heads;
    
    int threads = 256;
    int blocks_q = (total_q * (head_dim/2) + threads - 1) / threads;
    int blocks_k = (total_k * (head_dim/2) + threads - 1) / threads;
    
    rope_kernel<<<blocks_q, threads>>>(q.data(), cos.data(), sin.data(), batch, seq_len, num_q_heads, head_dim);
    rope_kernel<<<blocks_k, threads>>>(k.data(), cos.data(), sin.data(), batch, seq_len, num_kv_heads, head_dim);
}

// Grouped Query Attention operations
void repeat_kv(const Tensor& x, size_t n_rep, Tensor& y) {
    if (n_rep == 1) {
        cudaMemcpy(y.data(), x.data(), x.size() * sizeof(float), cudaMemcpyDeviceToDevice);
        return;
    }
    
    int batch = x.size(0);
    int num_kv_heads = x.size(1);
    int seq_len = x.size(2);
    int head_dim = x.size(3);
    int num_heads = num_kv_heads * n_rep; // Output heads
    
    int threads = 256;
    int total = batch * num_heads * seq_len * head_dim;
    int blocks = (total + threads - 1) / threads;
    
    repeat_kv_kernel<<<blocks, threads>>>(x.data(), y.data(), batch, num_heads, num_kv_heads, seq_len, head_dim);
}

// Convolution operations
void causal_conv1d(const Tensor& x, const Tensor& weight, const Tensor* bias, Tensor& y) {
    int batch = x.size(0);
    int channels = x.size(1); // hidden_size
    int seq_len = x.size(2);
    int kernel_size = weight.size(2);
    
    if (y.size() == 0) {
        y = Tensor({(size_t)batch, (size_t)channels, (size_t)seq_len});
    }
    
    int total = batch * channels * seq_len;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    depthwise_conv1d_kernel<<<blocks, threads>>>(x.data(), weight.data(), bias ? bias->data() : nullptr, y.data(), batch, seq_len, channels, kernel_size);
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
    cudaMemcpy(cos.data(), cos_cached_.data(), copy_size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(sin.data(), sin_cached_.data(), copy_size, cudaMemcpyDeviceToDevice);
}


