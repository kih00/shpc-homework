#include "layer.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

namespace tensor_ops {

constexpr int BLOCK_MM = 32;
constexpr int BLOCK_OPS = 256;
constexpr int BLOCK_SFTMX = 1024;

inline dim3 make_grid_2d(size_t x, size_t y, int tx = 32, int ty = 32) {
    return dim3((x + tx - 1) / tx, (y + ty - 1) / ty);
}

// Matrix multiply: C[m, n] = A[m, k] @ B[k, n]
__global__ void matmul_kernel(
    const float* A, const float* B, float* C,
    size_t m, size_t k, size_t n) {

    __shared__ float A_tile[BLOCK_MM][BLOCK_MM];
    __shared__ float B_tile[BLOCK_MM][BLOCK_MM];

    size_t row = blockIdx.y * BLOCK_MM + threadIdx.y;
    size_t col = blockIdx.x * BLOCK_MM + threadIdx.x;

    float sum = 0.0f;

    for (size_t t = 0; t < (k + BLOCK_MM - 1) / BLOCK_MM; ++t) {
        // Load A tile
        size_t a_col = t * BLOCK_MM + threadIdx.x;
        if (row < m && a_col < k) {
            A_tile[threadIdx.y][threadIdx.x] = A[row * k + a_col];
        } else {
            A_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load B tile
        size_t b_row = t * BLOCK_MM + threadIdx.y;
        if (b_row < k && col < n) {
            B_tile[threadIdx.y][threadIdx.x] = B[b_row * n + col];
        } else {
            B_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (size_t p = 0; p < BLOCK_MM; ++p) {
            sum += A_tile[threadIdx.y][p] * B_tile[p][threadIdx.x];
        }
        
        __syncthreads();
    }

    if (row < m && col < n) {
        C[row * n + col] = sum;
    }
}

// Matrix multiply: C[m, n] = A[m, k] @ B[n, k]^T
__global__ void matmul_transpose_kernel(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    size_t m, size_t k, size_t n) {

    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    __shared__ float As[2][BM][BK + 1];
    __shared__ float Bs[2][BN][BK + 1];

    int tid = threadIdx.x;
    int ty = tid / 16;
    int tx = tid % 16;

    float reg_c[TM][TN] = {0.0f};
    float reg_a[TM];
    float reg_b[TN];
    float4 load_a, load_b;

    size_t a_row_start = blockIdx.y * BM;
    size_t b_row_start = blockIdx.x * BN;

    const float* A_ptr = A + a_row_start * k;
    const float* B_ptr = B + b_row_start * k;

    // Load tile 0
    int t_row = tid / 2;
    int t_col = (tid % 2) * 4;

    // Load A
    if (t_row < BM && (a_row_start + t_row) < m && (t_col < k)) {
            load_a = reinterpret_cast<const float4*>(&A_ptr[t_row * k + t_col])[0];
    } else {
            load_a = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    As[0][t_row][t_col] = load_a.x; As[0][t_row][t_col+1] = load_a.y;
    As[0][t_row][t_col+2] = load_a.z; As[0][t_row][t_col+3] = load_a.w;

    // Load B
    if (t_row < BN && (b_row_start + t_row) < n && (t_col < k)) {
            load_b = reinterpret_cast<const float4*>(&B_ptr[t_row * k + t_col])[0];
    } else {
            load_b = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    Bs[0][t_row][t_col] = load_b.x; Bs[0][t_row][t_col+1] = load_b.y;
    Bs[0][t_row][t_col+2] = load_b.z; Bs[0][t_row][t_col+3] = load_b.w;

    __syncthreads();

    int next = 1;
    int curr = 0;

    for (size_t k_idx = 0; k_idx < k; k_idx += BK) {
        size_t next_k = k_idx + BK;

        // Prefetch next tile
        if (next_k < k) {
            int t_row = tid / 2;
            int t_col = (tid % 2) * 4;

            if (t_row < BM && (a_row_start + t_row) < m && (next_k + t_col < k)) {
                load_a = reinterpret_cast<const float4*>(&A_ptr[t_row * k + next_k + t_col])[0];
            } else {
                load_a = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }

            if (t_row < BN && (b_row_start + t_row) < n && (next_k + t_col < k)) {
                load_b = reinterpret_cast<const float4*>(&B_ptr[t_row * k + next_k + t_col])[0];
            } else {
                load_b = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }

        // Compute
        #pragma unroll
        for (int i = 0; i < BK; ++i) {
            #pragma unroll
            for (int r = 0; r < TM; ++r) reg_a[r] = As[curr][ty * TM + r][i];
            #pragma unroll
            for (int c = 0; c < TN; ++c) reg_b[c] = Bs[curr][tx * TN + c][i];

            #pragma unroll
            for (int r = 0; r < TM; ++r) {
                #pragma unroll
                for (int c = 0; c < TN; ++c) {
                    reg_c[r][c] += reg_a[r] * reg_b[c];
                }
            }
        }

        // Store prefetched tile to shared memory
        if (next_k < k) {
            __syncthreads();
            
            int t_row = tid / 2;
            int t_col = (tid % 2) * 4;

            As[next][t_row][t_col] = load_a.x;
            As[next][t_row][t_col+1] = load_a.y;
            As[next][t_row][t_col+2] = load_a.z;
            As[next][t_row][t_col+3] = load_a.w;

            Bs[next][t_row][t_col] = load_b.x;
            Bs[next][t_row][t_col+1] = load_b.y;
            Bs[next][t_row][t_col+2] = load_b.z;
            Bs[next][t_row][t_col+3] = load_b.w;

            __syncthreads();

            curr ^= 1;
            next ^= 1;
        }
    }

    // Store
    #pragma unroll
    for (int r = 0; r < TM; ++r) {
        int row = a_row_start + ty * TM + r;
        if (row < m) {
            #pragma unroll
            for (int c = 0; c < TN; c += 4) {
                int col = b_row_start + tx * TN + c;
                if (col + 3 < n) {
                    float4 v;
                    v.x = reg_c[r][c]; v.y = reg_c[r][c+1];
                    v.z = reg_c[r][c+2]; v.w = reg_c[r][c+3];
                    reinterpret_cast<float4*>(&C[row * n + col])[0] = v;
                } else {
                    for (int i = 0; i < 4; ++i) {
                        if (col + i < n) C[row * n + col + i] = reg_c[r][c+i];
                    }
                }
            }
        }
    }
}

// Add: C[n] = A[n] + B[n]
__global__ void add_kernel(const float* A, const float* B, float* C, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] = A[idx] + B[idx];
}

// Add scalar: C[n] = A[n] + b
__global__ void add_scalar_kernel(const float* A, float b, float* C, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] = A[idx] + b;
}

// Add bias: Y[n] = X[n] + bias[n % cols]
__global__ void add_bias_kernel(
    const float* X, const float* bias, float* Y, size_t n, size_t cols) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        Y[idx] = X[idx] + bias[idx % cols];
    }
}

// Element-wise multiply: C[n] = A[n] * B[n]
__global__ void mul_kernel(
    const float* A, const float* B, float* C, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] = A[idx] * B[idx];
}

// Multiply by scalar: C[n] = A[n] * b
__global__ void mul_scalar_kernel(
    const float* A, float b, float* C, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] = A[idx] * b;
}

// Sigmoid activation: Y[n] = 1 / (1 + exp(-X[n]))
__global__ void sigmoid_kernel(const float* X, float* Y, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = X[idx];
        Y[idx] = 1.0f / (1.0f + expf(-v));
    }
}

// SiLU activation: Y[n] = X[n] / (1 + exp(-X[n]))
__global__ void silu_kernel(const float* X, float* Y, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = X[idx];
        Y[idx] = v / (1.0f + expf(-v));
    }
}

// Softmax: Y[row, :] = exp(X[row, :] - max) / sum(exp(X[row, :] - max))
__global__ void softmax_kernel(
    const float* X, float* Y, size_t n, size_t rows) {
    // Optimized with warp-level parallelism

    size_t row = blockIdx.x;
    if (row >= rows) return;

    const float* x_row = X + row * n;
    float* y_row = Y + row * n;

    // Compute max: Warp-level -> Block-level
    // Warp-level reduction for max
    float local_max = -INFINITY;
    for (size_t j = threadIdx.x; j < n; j += blockDim.x) {
        local_max = fmaxf(local_max, x_row[j]);
    }
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        local_max = fmaxf(
            local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }

    // Cross-warp reduction using shared memory
    __shared__ float shared_max[32]; // max 32 warps
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    if (lane == 0) shared_max[wid] = local_max;
    __syncthreads();

    // Block-level reduction (First warp)
    if (wid == 0) {
        local_max = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize)
            ? shared_max[lane] : -INFINITY;
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            local_max = fmaxf(
                local_max, __shfl_down_sync(0xffffffff, local_max, offset));
        }
        if (lane == 0) shared_max[0] = local_max;
    }
    __syncthreads();
    float m = shared_max[0];

    // Compute exp and sum
    float local_sum = 0.0f;
    for (size_t j = threadIdx.x; j < n; j += blockDim.x) {
        float e = expf(x_row[j] - m);
        y_row[j] = e;
        local_sum += e;
    }

    // Compute sum: Warp-level -> Block-level
    // Warp shuffle reduction for sum
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    __shared__ float shared_sum[32];
    if (lane == 0) shared_sum[wid] = local_sum;
    __syncthreads();

    if (wid == 0) {
        local_sum = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize)
            ? shared_sum[lane] : 0.0f;
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        }
        if (lane == 0) shared_sum[0] = local_sum;
    }
    __syncthreads();
    float inv = 1.0f / shared_sum[0];

    // Normalize
    for (size_t j = threadIdx.x; j < n; j += blockDim.x) {
        y_row[j] *= inv;
    }
}

// RMS norm: Y[row, :] = X[row, :] * W[:] * rsqrt(mean(X[row, :]^2) + eps)
__global__ void rms_norm_kernel(
    const float* X, const float* W, float eps,
    float* Y, size_t hidden, size_t rows) {
    // optimized with warp-level parallelism

    size_t row = blockIdx.x;
    if (row >= rows) return;

    const float* x = X + row * hidden;
    float* y = Y + row * hidden;

    // Compute partial sum
    float local_sum = 0.0f;
    for (size_t j = threadIdx.x; j < hidden; j += blockDim.x) {
        float val = x[j];
        local_sum += val * val;
    }

    // Warp-level reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    // Cross-warp reduction using shared memory
    __shared__ float warp_sums[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    if (lane == 0) {
        warp_sums[wid] = local_sum;
    }
    __syncthreads();

    // Block-level reduction (first warp)
    float total_sum = 0.0f;
    if (wid == 0) {
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        local_sum = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        }
        if (lane == 0) {
            warp_sums[0] = local_sum;
        }
    }
    __syncthreads();
    total_sum = warp_sums[0];

    // Compute rms and apply norm
    float rms = rsqrtf(total_sum / hidden + eps);
    for (size_t j = threadIdx.x; j < hidden; j += blockDim.x) {
        y[j] = x[j] * rms * W[j];
    }
}

// RoPE compute ... cos_out, sin_out: (max_seq, head_dim)
__global__ void rope_compute_kernel(
    float* cos_out, float* sin_out,
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

// RoPE apply: x[b, h, s, d] = rotate(x[b, h, s, d], cos[s, d], sin[s, d]) (x: q or k)
__global__ void rope_apply_kernel(
    float* q, float* k, const float* cos, const float* sin,
    size_t batch, size_t qh, size_t kh, size_t seq_len, size_t head_dim) {

    size_t b = blockIdx.x;
    size_t h = blockIdx.y;
    if (b >= batch) return;

    size_t half = head_dim / 2;
    size_t q_stride = seq_len * head_dim;
    size_t k_stride = seq_len * head_dim;

    // Each thread processes one sequence position
    for (size_t s = threadIdx.x; s < seq_len; s += blockDim.x) {
        float* q_base = q + (b * qh + h) * q_stride + s * head_dim;
        const float* cos_row = cos + s * head_dim;
        const float* sin_row = sin + s * head_dim;

        // Apply to Q
        if (h < qh) {
            for (size_t d = 0; d < half; ++d) {
                float q1 = q_base[d];
                float q2 = q_base[d + half];
                q_base[d] = q1 * cos_row[d] - q2 * sin_row[d];
                q_base[d + half] = q2 * cos_row[d + half] + q1 * sin_row[d + half];
            }
        }

        // Apply to K
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

// Repeat KV heads: x[b, kv_heads, s, d] ~> y[b, kv_heads * n_rep, s, d]
__global__ void repeat_kv_kernel(
    const float* x, float* y, size_t batch, size_t kv_heads,
    size_t n_rep, size_t seq_len, size_t head_dim) {

    // Output index: y[b, h_rep, s, d]
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch * kv_heads * n_rep * seq_len * head_dim;
    if (idx >= total) return;

    size_t d = idx % head_dim;
    size_t s = (idx / head_dim) % seq_len;
    size_t h_rep = (idx / head_dim / seq_len) % (kv_heads * n_rep);
    size_t h = h_rep / n_rep;
    size_t b = idx / head_dim / seq_len / (kv_heads * n_rep);

    // Input index: x[b, h, s, d]
    size_t x_idx = (((b * kv_heads + h) * seq_len + s) * head_dim + d);
    y[idx] = x[x_idx];
}

// Causal Conv1D
// y[b, ch, s] = sum_{k=0}^{ksz-1} x[b, ch, s - (ksz -1) + k] * w[ch, 0, k] + bias[ch]
__global__ void causal_conv1d_kernel(
    const float* x, const float* w, const float* bias, float* y,
    size_t batch, size_t ch, size_t seq, size_t ksz) {
    // x: (batch, ch, seq)
    // w: (ch, 1, ksz)
    // bias: (ch)
    // y: (batch, ch, seq)

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch * ch * seq;
    if (idx >= total) return;

    size_t s = idx % seq;
    size_t c = (idx / seq) % ch;
    size_t b = idx / (seq * ch);
    float sum = 0.0f;

    // x base: x[b, c, :]
    const float* x_base = x + (b * ch + c) * seq;
    // w base: w[c, 0, :]
    const float* w_base = w + c * ksz;

    // Convolution sum
    for (size_t k = 0; k < ksz; ++k) {
        int input_pos = (int)s - ((int)ksz - 1) + (int)k;
        if (input_pos >= 0) sum += x_base[input_pos] * w_base[k];
    }
    if (bias) sum += bias[c];
    y[idx] = sum;
}

// ============================================================================
// Attention Kernels
// ============================================================================

// Reshape: (batch*seq, num_heads*head_dim) -> (batch, num_heads, seq, head_dim)
__global__ void reshape_to_heads_kernel(
    const float* in, float* out,
    size_t batch, size_t seq_len, size_t num_heads, size_t head_dim) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch * num_heads * seq_len * head_dim;
    if (idx >= total) return;

    // Output index: [b, h, s, d]
    size_t d = idx % head_dim;
    size_t s = (idx / head_dim) % seq_len;
    size_t h = (idx / (head_dim * seq_len)) % num_heads;
    size_t b = idx / (head_dim * seq_len * num_heads);

    // Input index: [b*seq + s, h*head_dim + d]
    size_t in_idx = (b*seq_len + s) * (num_heads*head_dim) + h*head_dim + d;
    out[idx] = in[in_idx];
}

// Reshape: (batch, num_heads, seq, head_dim) -> (batch*seq, num_heads*head_dim)
__global__ void reshape_from_heads_kernel(
    const float* in, float* out,
    size_t batch, size_t seq_len, size_t num_heads, size_t head_dim) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch * seq_len * num_heads * head_dim;
    if (idx >= total) return;

    // Output index: [b*seq + s, h*head_dim + d]
    size_t d = idx % head_dim;
    size_t h = (idx / head_dim) % num_heads;
    size_t s = (idx / (head_dim * num_heads)) % seq_len;
    size_t b = idx / (head_dim * num_heads * seq_len);

    // Input index: [b, h, s, d]
    size_t in_idx = ((b * num_heads + h) * seq_len + s) * head_dim + d;
    out[idx] = in[in_idx];
}

// Reshape: (batch*seq, heads*dim) -> (batch, seq, heads, dim) for layernorm input
__global__ void reshape_for_layernorm_kernel(
    const float* in, float* out,
    size_t batch, size_t seq_len, size_t num_heads, size_t head_dim) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch * seq_len * num_heads * head_dim;
    if (idx >= total) return;

    // Output index: [b, s, h, d]
    size_t d = idx % head_dim;
    size_t h = (idx / head_dim) % num_heads;
    size_t s = (idx / (head_dim * num_heads)) % seq_len;
    size_t b = idx / (head_dim * num_heads * seq_len);

    // Input index: [b*seq + s, h*head_dim + d]
    size_t in_idx = (b*seq_len + s) * (num_heads*head_dim) + h*head_dim + d;
    out[idx] = in[in_idx];
}

// Fused operation combining transpose and split for ShortConv
// Transpose and split: (batch*seq, 3*hidden) -> B, C, x_gate each (batch, hidden, seq)
__global__ void transpose_split_BCx_kernel(
    const float* in, float* B, float* C, float* x_gate,
    size_t batch, size_t seq_len, size_t hidden_size) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch * hidden_size * seq_len;
    if (idx >= total) return;

    // Output index: [b, h, s] -> idx = (b * hidden_size + h) * seq_len + s
    size_t s = idx % seq_len;
    size_t h = (idx / seq_len) % hidden_size;
    size_t b = idx / (seq_len * hidden_size);

    // Input index: [(b * seq_len + s), c] (c = {h, h+hidden, h+2*hidden})
    size_t base_in_idx = (b * seq_len + s) * (3 * hidden_size);
    
    B[idx] = in[base_in_idx + h];
    C[idx] = in[base_in_idx + h + hidden_size];
    x_gate[idx] = in[base_in_idx + h + 2 * hidden_size];
}

// Transpose: (batch, hidden, seq) -> (batch, seq, hidden)
__global__ void transpose_hidden_seq_kernel(
    const float* in, float* out,
    size_t batch, size_t hidden_size, size_t seq_len) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch * seq_len * hidden_size;
    if (idx >= total) return;

    // Output index: [b, s, h] -> idx = (b * seq_len + s) * hidden_size + h
    size_t h = idx % hidden_size;
    size_t s = (idx / hidden_size) % seq_len;
    size_t b = idx / (hidden_size * seq_len);

    // Input index: [b, h, s] -> (b * hidden_size + h) * seq_len + s
    size_t in_idx = (b * hidden_size + h) * seq_len + s;
    out[idx] = in[in_idx];
}

// =============================================================================
// Fused ShortConv Kernel
// Combines: transpose_split_BCx + mul(B*gate) + conv1d + mul(C*conv) + transpose
// Input: in_proj_out (bs, 3*hidden)
// Output: y_out (batch, seq, hidden) - ready for out_proj
// =============================================================================
__global__ void shortconv_fused_kernel(
    const float* __restrict__ in_proj_out,  // (batch*seq, 3*hidden)
    const float* __restrict__ conv_w,       // (hidden, ksz)
    const float* __restrict__ conv_bias,    // (hidden) or nullptr
    float* __restrict__ out,                // (batch, seq, hidden)
    size_t batch, size_t seq_len, size_t hidden_size, size_t ksz) {

    // Output layout: (batch, seq, hidden)
    // Each thread computes one output element [b, s, h]
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch * seq_len * hidden_size;
    if (idx >= total) return;

    // Decode output position [b, s, h]
    size_t h = idx % hidden_size;
    size_t s = (idx / hidden_size) % seq_len;
    size_t b = idx / (hidden_size * seq_len);

    // Read C value from in_proj_out
    // in_proj_out layout: (batch*seq, 3*hidden) where [B, C, x_gate] are interleaved
    // For position (b, s): base = (b * seq_len + s) * 3 * hidden
    // B at offset h, C at offset h + hidden, x_gate at offset h + 2*hidden
    size_t in_base = (b * seq_len + s) * 3 * hidden_size;
    float C_val = in_proj_out[in_base + h + hidden_size];

    // Compute conv1d output for this position
    // conv_out[b, h, s] = sum_{k=0}^{ksz-1} Bx[b, h, s-(ksz-1)+k] * conv_w[h, k] + bias[h]
    // where Bx[b, h, pos] = B[b, h, pos] * x_gate[b, h, pos]
    float conv_sum = 0.0f;

    #pragma unroll
    for (size_t k = 0; k < ksz; ++k) {
        int input_pos = (int)s - ((int)ksz - 1) + (int)k;
        if (input_pos >= 0) {
            // Read B and x_gate for input_pos
            size_t in_pos_base = (b * seq_len + input_pos) * 3 * hidden_size;
            float B_val = in_proj_out[in_pos_base + h];
            float gate_val = in_proj_out[in_pos_base + h + 2 * hidden_size];
            float Bx_val = B_val * gate_val;
            conv_sum += Bx_val * conv_w[h * ksz + k];
        }
    }

    // Add bias if present
    if (conv_bias != nullptr) {
        conv_sum += conv_bias[h];
    }

    // Final output: y_pre = C * conv_out (already in correct layout for out_proj)
    out[idx] = C_val * conv_sum;
}

// Batched Scaled Dot-Product Attention with Causal Mask
__global__ void batched_attention_kernel(
    const float* Q, const float* K, const float* V, float* Out,
    size_t batch, size_t num_heads,
    size_t seq_len, size_t head_dim, float scale) {
    // Q, K, V: (batch, num_heads, seq_len, head_dim)
    // Output: (batch, num_heads, seq_len, head_dim)

    // Each block handles one (batch, head) pair
    size_t bh = blockIdx.x;
    size_t b = bh / num_heads;
    size_t h = bh % num_heads;

    if (b >= batch) return;

    // Pointers to this head's Q, K, V, Out
    const float* Q_head = Q + (b * num_heads + h) * seq_len * head_dim; // Q[b, h, :, :]
    const float* K_head = K + (b * num_heads + h) * seq_len * head_dim; // K[b, h, :, :]
    const float* V_head = V + (b * num_heads + h) * seq_len * head_dim; // V[b, h, :, :]
    float* Out_head = Out + (b * num_heads + h) * seq_len * head_dim; // Out[b, h, :, :]

    // Each thread handles one query position
    for (size_t i = threadIdx.x; i < seq_len; i += blockDim.x) {
        float max_score = -INFINITY;
        extern __shared__ float shared_mem[];
        float* scores = shared_mem + threadIdx.x * seq_len;

        // 1. Compute attention scores for row i (Q[i] @ K^T) & find max
        // only for j <= i (causal mask)
        for (size_t j = 0; j <= i; j++) {
            float score = 0.0f;
            for (size_t d = 0; d < head_dim; d++) {
                score += Q_head[i * head_dim + d] * K_head[j * head_dim + d];
            }
            score *= scale;
            scores[j] = score;
            max_score = fmaxf(max_score, score);
        }

        // 2. Compute softmax
        float sum_exp = 0.0f;
        for (size_t j = 0; j <= i; j++) {
            scores[j] = expf(scores[j] - max_score);
            sum_exp += scores[j];
        }

        float inv_sum = 1.0f / sum_exp;
        for (size_t j = 0; j <= i; j++) {
            scores[j] *= inv_sum;
        }

        // 3. Compute output: softmax(scores) @ V
        for (size_t d = 0; d < head_dim; d++) {
            float out_val = 0.0f;
            for (size_t j = 0; j <= i; j++) {
                out_val += scores[j] * V_head[j * head_dim + d];
            }
            Out_head[i * head_dim + d] = out_val;
        }
    }
}

// Matrix operations

void matmul(const Tensor& a, const Tensor& b, Tensor& c, cudaStream_t stream) {
    // a: (m, k), b: (k, n), c: (m, n)
    size_t m = a.size(0);
    size_t k = a.size(1);
    size_t n = b.size(1);
    if (c.size() == 0) c = Tensor({m, n});

    a.to_device(-1, stream); // Use current device
    b.to_device(-1, stream);
    c.to_device(-1, stream);

    dim3 block(BLOCK_MM, BLOCK_MM);
    dim3 grid = make_grid_2d(n, m, block.x, block.y);
    matmul_kernel<<<grid, block, 0, stream>>>(
        a.device_data(), b.device_data(), c.device_data(), m, k, n);

    c.mark_device_dirty();
}

void matmul_transposed(
    const Tensor& a, const Tensor& b, Tensor& c, cudaStream_t stream) {
    // a: (m, k), b: (n, k), c: (m, n)  [c = a @ b^T]
    size_t m = a.size(0);
    size_t k = a.size(1);
    size_t n = b.size(0);
    if (c.size() == 0) c = Tensor({m, n});

    a.to_device(-1, stream);
    b.to_device(-1, stream);
    c.to_device(-1, stream);

    // Optimized kernel launch config
    dim3 block(BLOCK_OPS);
    dim3 grid((n + 127) / 128, (m + 127) / 128);
    matmul_transpose_kernel<<<grid, block, 0, stream>>>(
        a.device_data(), b.device_data(), c.device_data(), m, k, n);

    c.mark_device_dirty();
}

// Element-wise operations
void add(const Tensor& a, const Tensor& b, Tensor& c, cudaStream_t stream) {
    size_t n = a.size();
    if (c.size() == 0) c = Tensor(a.shape());

    a.to_device(-1, stream);
    b.to_device(-1, stream);
    c.to_device(-1, stream);

    dim3 block(BLOCK_OPS);
    dim3 grid((n + block.x - 1) / block.x);
    add_kernel<<<grid, block, 0, stream>>>(a.device_data(), b.device_data(), c.device_data(), n);

    c.mark_device_dirty();
}

void add_scalar(const Tensor& a, float b, Tensor& c, cudaStream_t stream) {
    size_t n = a.size();
    if (c.size() == 0) c = Tensor(a.shape());

    a.to_device(-1, stream);
    c.to_device(-1, stream);

    dim3 block(BLOCK_OPS);
    dim3 grid((n + block.x - 1) / block.x);
    add_scalar_kernel<<<grid, block, 0, stream>>>(
        a.device_data(), b, c.device_data(), n);

    c.mark_device_dirty();
}

void add_bias(
    const Tensor& x, const Tensor& bias, Tensor& y, cudaStream_t stream) {
    size_t n = x.size();
    size_t cols = x.size(-1);
    if (y.size() == 0) y = Tensor(x.shape());

    x.to_device(-1, stream);
    bias.to_device(-1, stream);
    y.to_device(-1, stream);

    dim3 block(BLOCK_OPS);
    dim3 grid((n + block.x - 1) / block.x);
    add_bias_kernel<<<grid, block, 0, stream>>>(
        x.device_data(), bias.device_data(), y.device_data(), n, cols);

    y.mark_device_dirty();
}

void mul(const Tensor& a, const Tensor& b, Tensor& c, cudaStream_t stream) {
    size_t n = a.size();
    if (c.size() == 0) c = Tensor(a.shape());

    a.to_device(-1, stream);
    b.to_device(-1, stream);
    c.to_device(-1, stream);

    dim3 block(BLOCK_OPS);
    dim3 grid((n + block.x - 1) / block.x);
    mul_kernel<<<grid, block, 0, stream>>>(
        a.device_data(), b.device_data(), c.device_data(), n);

    c.mark_device_dirty();
}

void mul_scalar(const Tensor& a, float b, Tensor& c, cudaStream_t stream) {
    size_t n = a.size();
    if (c.size() == 0) c = Tensor(a.shape());

    a.to_device(-1, stream);
    c.to_device(-1, stream);

    dim3 block(BLOCK_OPS);
    dim3 grid((n + block.x - 1) / block.x);
    mul_scalar_kernel<<<grid, block, 0, stream>>>(
        a.device_data(), b, c.device_data(), n);

    c.mark_device_dirty();
}

// Activation functions
void sigmoid(const Tensor& x, Tensor& y, cudaStream_t stream) {
    size_t n = x.size();
    if (y.size() == 0) y = Tensor(x.shape());

    x.to_device(-1, stream);
    y.to_device(-1, stream);

    dim3 block(BLOCK_OPS);
    dim3 grid((n + block.x - 1) / block.x);
    sigmoid_kernel<<<grid, block, 0, stream>>>(
        x.device_data(), y.device_data(), n);

    y.mark_device_dirty();
}

void silu(const Tensor& x, Tensor& y, cudaStream_t stream) {
    size_t n = x.size();
    if (y.size() == 0) y = Tensor(x.shape());

    x.to_device(-1, stream);
    y.to_device(-1, stream);

    dim3 block(BLOCK_OPS);
    dim3 grid((n + block.x - 1) / block.x);
    silu_kernel<<<grid, block, 0, stream>>>(
        x.device_data(), y.device_data(), n);

    y.mark_device_dirty();
}

void softmax(const Tensor& x, Tensor& y, int dim, cudaStream_t stream) {
    // For simplicity, assume dim=-1 (last dimension)
    size_t outer_size = 1;
    for (size_t i = 0; i < x.ndim() - 1; i++) {
        outer_size *= x.size(i);
    }
    size_t inner_size = x.size(-1);
    if (y.size() == 0) y = Tensor(x.shape());

    x.to_device(-1, stream);
    y.to_device(-1, stream);

    dim3 block(inner_size >= BLOCK_SFTMX ? BLOCK_SFTMX : inner_size);
    dim3 grid(outer_size);
    softmax_kernel<<<grid, block, 0, stream>>>(
        x.device_data(), y.device_data(), inner_size, outer_size);

    y.mark_device_dirty();
}

// Normalization
void rms_norm(
    const Tensor& x, const Tensor& weight, float eps,
    Tensor& y, cudaStream_t stream) {

    size_t outer_size = 1;
    for (size_t i = 0; i < x.ndim() - 1; i++) {
        outer_size *= x.size(i);
    }
    size_t hidden_size = x.size(-1);
    if (y.size() == 0) y = Tensor(x.shape());

    x.to_device(-1, stream);
    weight.to_device(-1, stream);
    y.to_device(-1, stream);
    // New kernel: one block per row, 256 threads per block for parallel reduction
    dim3 block(BLOCK_OPS);
    dim3 grid(outer_size);
    rms_norm_kernel<<<grid, block, 0, stream>>>(
        x.device_data(), weight.device_data(), eps,
        y.device_data(), hidden_size, outer_size);

    y.mark_device_dirty();
}

// RoPE operations
void compute_rope_embeddings(
    size_t head_dim, size_t max_seq_len, float theta,
    Tensor& cos, Tensor& sin, cudaStream_t stream) {
    // cos, sin: (max_seq_len, head_dim)

    if (cos.size() == 0) cos = Tensor({max_seq_len, head_dim});
    if (sin.size() == 0) sin = Tensor({max_seq_len, head_dim});

    cos.to_device(-1, stream);
    sin.to_device(-1, stream);

    dim3 block(BLOCK_OPS);
    dim3 grid((max_seq_len + block.x - 1) / block.x);
    rope_compute_kernel<<<grid, block, 0, stream>>>(
        cos.device_data(), sin.device_data(), head_dim, max_seq_len, theta);

    cos.mark_device_dirty();
    sin.mark_device_dirty();
}

void apply_rotary_pos_emb(
    Tensor& q, Tensor& k,
    const Tensor& cos, const Tensor& sin, cudaStream_t stream) {
    // q: (batch, num_q_heads, seq_len, head_dim)
    // k: (batch, num_kv_heads, seq_len, head_dim)
    // cos, sin: (seq_len, head_dim)

    size_t batch = q.size(0);
    size_t num_q_heads = q.size(1);
    size_t num_kv_heads = k.size(1);
    size_t seq_len = q.size(2);
    size_t head_dim = q.size(3);

    q.to_device(-1, stream);
    k.to_device(-1, stream);
    cos.to_device(-1, stream);
    sin.to_device(-1, stream);

    // kernel space: (batch, max(num_q_heads, num_kv_heads)), block: (seq_len)
    size_t h_max = std::max(num_q_heads, num_kv_heads);
    dim3 grid(batch, h_max);
    dim3 block(seq_len);
    if (block.x > BLOCK_OPS) block.x = BLOCK_OPS;
    rope_apply_kernel<<<grid, block, 0, stream>>>(
        q.device_data(), k.device_data(), cos.device_data(), sin.device_data(),
        batch, num_q_heads, num_kv_heads, seq_len, head_dim);

    q.mark_device_dirty();
    k.mark_device_dirty();
}

// Grouped Query Attention operations
void repeat_kv(const Tensor& x, size_t n_rep, Tensor& y, cudaStream_t stream) {
    size_t batch = x.size(0);
    size_t num_kv_heads = x.size(1);
    size_t seq_len = x.size(2);
    size_t head_dim = x.size(3);
    if (y.size() == 0)
        y = Tensor({batch, num_kv_heads * n_rep, seq_len, head_dim});

    x.to_device(-1, stream);
    y.to_device(-1, stream);

    // kernel space: total elements
    size_t total = batch * num_kv_heads * n_rep * seq_len * head_dim;
    dim3 block(BLOCK_OPS);
    dim3 grid((total + block.x - 1) / block.x);
    repeat_kv_kernel<<<grid, block, 0, stream>>>(
        x.device_data(), y.device_data(),
        batch, num_kv_heads, n_rep, seq_len, head_dim);

    y.mark_device_dirty();
}

// Convolution operations
void causal_conv1d(
    const Tensor& x, const Tensor& weight, const Tensor* bias,
    Tensor& y, cudaStream_t stream) {
    // x: (batch, channels, seq_len) - Conv1d format
    // weight: (channels, 1, kernel_size) - grouped conv weights
    // bias: (channels) [optional]
    // y: (batch, channels, seq_len)

    size_t batch = x.size(0);
    size_t channels = x.size(1);
    size_t seq_len = x.size(2);
    size_t kernel_size = weight.size(2);
    if (y.size() == 0) y = Tensor({batch, channels, seq_len});

    x.to_device(-1, stream);
    weight.to_device(-1, stream);
    if (bias) bias->to_device(-1, stream);
    y.to_device(-1, stream);

    // kernel space: total elements
    size_t total = batch * channels * seq_len;
    dim3 block(BLOCK_OPS);
    dim3 grid((total + block.x - 1) / block.x);
    causal_conv1d_kernel<<<grid, block, 0, stream>>>(
        x.device_data(), weight.device_data(),
        bias ? bias->device_data() : nullptr,
        y.device_data(), batch, channels, seq_len, kernel_size);

    y.mark_device_dirty();
}

// ============================================================================
// Attention Operations
// ============================================================================

// Reshape from (batch*seq, num_heads*head_dim) to (batch, num_heads, seq, head_dim)
void reshape_to_heads(
    const Tensor& in, Tensor& out,
    size_t batch, size_t seq_len,
    size_t num_heads, size_t head_dim, cudaStream_t stream) {
    // in: (batch*seq, num_heads*head_dim)
    // out: (batch, num_heads, seq, head_dim)

    if (out.size() == 0) out = Tensor({batch, num_heads, seq_len, head_dim});

    in.to_device(-1, stream);
    out.to_device(-1, stream);

    size_t total = batch * num_heads * seq_len * head_dim;
    dim3 block(BLOCK_OPS);
    dim3 grid((total + block.x - 1) / block.x);
    reshape_to_heads_kernel<<<grid, block, 0, stream>>>(
        in.device_data(), out.device_data(),
        batch, seq_len, num_heads, head_dim);

    out.mark_device_dirty();
}

// Reshape from (batch, num_heads, seq, head_dim) to (batch*seq, num_heads*head_dim)
void reshape_from_heads(
    const Tensor& in, Tensor& out,
    size_t batch, size_t seq_len,
    size_t num_heads, size_t head_dim, cudaStream_t stream) {
    // in: (batch, num_heads, seq, head_dim)
    // out: (batch*seq, num_heads*head_dim)

    if (out.size() == 0) out = Tensor({batch * seq_len, num_heads * head_dim});

    in.to_device(-1, stream);
    out.to_device(-1, stream);

    size_t total = batch * seq_len * num_heads * head_dim;
    dim3 block(BLOCK_OPS);
    dim3 grid((total + block.x - 1) / block.x);
    reshape_from_heads_kernel<<<grid, block, 0, stream>>>(
        in.device_data(), out.device_data(),
        batch, seq_len, num_heads, head_dim);

    out.mark_device_dirty();
}

// Batched scaled dot-product attention with causal mask
void batched_attention(
    const Tensor& Q, const Tensor& K, const Tensor& V,
    Tensor& out, float scale, cudaStream_t stream) {
    // Q, K, V, out: (batch, num_heads, seq_len, head_dim)

    size_t batch = Q.size(0);
    size_t num_heads = Q.size(1);
    size_t seq_len = Q.size(2);
    size_t head_dim = Q.size(3);

    if (out.size() == 0) out = Tensor({batch, num_heads, seq_len, head_dim});

    Q.to_device(-1, stream);
    K.to_device(-1, stream);
    V.to_device(-1, stream);
    out.to_device(-1, stream);

    // Each block handles one (batch, head) pair
    // Each thread handles multiple query positions
    dim3 grid(batch * num_heads);
    dim3 block(std::min((size_t)BLOCK_OPS, seq_len));

    // Shared memory: seq_len for each thread
    size_t shared_mem_size = block.x * seq_len * sizeof(float);

    // Check if shared memory is sufficient (max ~48KB on most GPUs)
    if (shared_mem_size > 48 * 1024) {
        // Fallback: reduce threads or use different algorithm
        block.x = (48 * 1024) / (seq_len * sizeof(float));
        if (block.x < 1) block.x = 1;
        shared_mem_size = block.x * seq_len * sizeof(float);
    }

    batched_attention_kernel<<<grid, block, shared_mem_size, stream>>>(
        Q.device_data(), K.device_data(), V.device_data(), out.device_data(),
        batch, num_heads, seq_len, head_dim, scale);
    out.mark_device_dirty();
}

// Reshape for layernorm input
void reshape_for_layernorm(
    const Tensor& in, Tensor& out,
    size_t batch, size_t seq_len,
    size_t num_heads, size_t head_dim, cudaStream_t stream) {
    // in: (batch*seq, heads*dim)
    // out: (batch, seq, heads, dim)

    if (out.size() == 0) out = Tensor({batch, seq_len, num_heads, head_dim});

    in.to_device(-1, stream);
    out.to_device(-1, stream);

    size_t total = batch * seq_len * num_heads * head_dim;
    dim3 block(BLOCK_OPS);
    dim3 grid((total + block.x - 1) / block.x);
    reshape_for_layernorm_kernel<<<grid, block, 0, stream>>>(
        in.device_data(), out.device_data(),
        batch, seq_len, num_heads, head_dim);

    out.mark_device_dirty();
}

// Transpose and split for ShortConv input
void transpose_split_BCx(
    const Tensor& in_proj_out, Tensor& B, Tensor& C, Tensor& x_gate,
    size_t batch, size_t seq_len, size_t hidden_size, cudaStream_t stream) {
    // in_proj_out: (batch*seq, 3*hidden)
    // B, C, x_gate: (batch, hidden, seq)

    if (B.size() == 0) B = Tensor({batch, hidden_size, seq_len});
    if (C.size() == 0) C = Tensor({batch, hidden_size, seq_len});
    if (x_gate.size() == 0) x_gate = Tensor({batch, hidden_size, seq_len});

    in_proj_out.to_device(-1, stream);
    B.to_device(-1, stream);
    C.to_device(-1, stream);
    x_gate.to_device(-1, stream);

    size_t total = batch * hidden_size * seq_len;
    dim3 block(BLOCK_OPS);
    dim3 grid((total + block.x - 1) / block.x);
    transpose_split_BCx_kernel<<<grid, block, 0, stream>>>(
        in_proj_out.device_data(),
        B.device_data(), C.device_data(), x_gate.device_data(),
        batch, seq_len, hidden_size);

    B.mark_device_dirty();
    C.mark_device_dirty();
    x_gate.mark_device_dirty();
}

// Transpose for ShortConv output
void transpose_hidden_seq(
    const Tensor& in, Tensor& out,
    size_t batch, size_t hidden_size, size_t seq_len, cudaStream_t stream) {
    // in: (batch, hidden, seq)
    // out: (batch, seq, hidden)

    if (out.size() == 0) out = Tensor({batch, seq_len, hidden_size});

    in.to_device(-1, stream);
    out.to_device(-1, stream);

    size_t total = batch * seq_len * hidden_size;
    dim3 block(BLOCK_OPS);
    dim3 grid((total + block.x - 1) / block.x);
    transpose_hidden_seq_kernel<<<grid, block, 0, stream>>>(
        in.device_data(), out.device_data(),
        batch, hidden_size, seq_len);

    out.mark_device_dirty();
}

// Fused ShortConv operation
// Combines: transp_split_BCx + mul(B*gate) + conv1d + mul(C*conv) + transp
void shortconv_fused(
    const Tensor& in_proj_out, const Tensor& conv_weight,
    const Tensor* conv_bias, Tensor& out,
    size_t batch, size_t seq_len, size_t hidden_size, size_t kernel_size,
    cudaStream_t stream) {
    // in_proj_out: (batch*seq, 3*hidden)
    // conv_weight: (hidden, 1, ksz)
    // conv_bias: (hidden) or nullptr
    // out: (batch, seq, hidden)

    if (out.size() == 0) out = Tensor({batch, seq_len, hidden_size});

    in_proj_out.to_device(-1, stream);
    conv_weight.to_device(-1, stream);
    if (conv_bias) conv_bias->to_device(-1, stream);
    out.to_device(-1, stream);

    size_t total = batch * seq_len * hidden_size;
    dim3 block(BLOCK_OPS);
    dim3 grid((total + block.x - 1) / block.x);

    shortconv_fused_kernel<<<grid, block, 0, stream>>>(
        in_proj_out.device_data(),
        conv_weight.device_data(),
        conv_bias ? conv_bias->device_data() : nullptr,
        out.device_data(),
        batch, seq_len, hidden_size, kernel_size);

    out.mark_device_dirty();
}

} // namespace tensor_ops

// ============================================================================
// Layer Implementations - Small building blocks
// ============================================================================

// RMSNorm implementation
RMSNorm::RMSNorm(const std::string& weight_file) {
    weight_ = Tensor::load_from_file(weight_file);
}

void RMSNorm::forward(const Tensor& x, Tensor& y, cudaStream_t stream) {
    tensor_ops::rms_norm(x, weight_, RMS_NORM_EPS, y, stream);
}

// RotaryEmbedding implementation
RotaryEmbedding::RotaryEmbedding() : max_seq_len_(MAX_POSITION_EMBEDDINGS) {
    cos_cached_ = Tensor({max_seq_len_, HEAD_DIM});
    sin_cached_ = Tensor({max_seq_len_, HEAD_DIM});

    tensor_ops::compute_rope_embeddings(
        HEAD_DIM, max_seq_len_, ROPE_THETA, cos_cached_, sin_cached_);
}

void RotaryEmbedding::forward(size_t seq_len, Tensor& cos, Tensor& sin, cudaStream_t stream) {
    // Return cached values
    // cos, sin: (seq_len, head_dim)

    cos = cos_cached_.slice(0, 0, seq_len, stream);
    sin = sin_cached_.slice(0, 0, seq_len, stream);
}

