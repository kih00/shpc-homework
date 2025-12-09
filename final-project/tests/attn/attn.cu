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

// kernel launch parameters
constexpr int BLOCK_MM = 16;          // blockDim.x/y for matmul
constexpr int BLOCK_NORM = 256;       // threads for RMSNorm
constexpr int BLOCK_RESHAPE = 128;    // threads along seq_len for reshape
constexpr int BLOCK_FLAT = 256;       // threads for flatten

static float *x_gpu, *cos_gpu, *sin_gpu;
static float *q_proj_gpu, *k_proj_gpu, *v_proj_gpu, *o_proj_gpu;
static float *q_norm_gpu, *k_norm_gpu, *output_gpu;
static float *q_proj_out_gpu, *k_proj_out_gpu, *v_proj_out_gpu;
static float *q_normed_gpu, *k_normed_gpu;
static float *q_transposed_gpu, *k_transposed_gpu, *k_repeated_gpu, *v_transposed_gpu;
static float *attn_scores_gpu, *attn_out_gpu, *attn_out_transposed_gpu;

// ============================================================================
// CUDA kernels
// ============================================================================

// Matrix multiply: out[m, n] = x[m, k] @ w[n, k]^T
__global__ void matmul_transposed_kernel(const float* x, const float* w,
                     float* out, int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // 0..m-1
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // 0..n-1
    if (row >= m || col >= n) return;
    const float* x_row = x + row * k;
    const float* w_row = w + col * k;
    float sum = 0.0f;
    for (int i = 0; i < k; i++) {
        sum += x_row[i] * w_row[i];
    }
    out[row * n + col] = sum;
}

// RMSNorm over last dimension (vec_len)
__global__ void rmsnorm_kernel(const float* x, const float* weight, float* y,
                 int vec_len, int n_vec) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // vector index
    if (idx >= n_vec) return;
    const float* x_vec = x + idx * vec_len;
    float* y_vec = y + idx * vec_len;

    float sum_sq = 0.0f;
    for (int i = 0; i < vec_len; i++) sum_sq += x_vec[i] * x_vec[i];
    float rms = sqrtf(sum_sq / vec_len + 1e-5f);
    for (int i = 0; i < vec_len; i++) {
        y_vec[i] = (x_vec[i] / rms) * weight[i];
    }
}

// Reshape (B*S, H*D) -> (B, H, S, D)
__global__ void reshape_qkv_kernel(const float* in, float* out,
                   int batch, int seq_len, int heads, int head_dim) {
  int b = blockIdx.z;
  int h = blockIdx.y;
  int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= seq_len) return;
  const float* src = in + (b * seq_len + s) * (heads * head_dim) + h * head_dim;
  float* dst = out + ((b * heads + h) * seq_len + s) * head_dim;
  for (int d = 0; d < head_dim; d++) dst[d] = src[d];
}

// Apply RoPE in place for Q or K (expects (B, H, S, D))
__global__ void rope_kernel(float* x, const float* cos, const float* sin,
              int batch, int heads, int seq_len, int head_dim) {
  int b = blockIdx.z;
  int h = blockIdx.y;
  int s = blockIdx.x;
  int d = threadIdx.x;  // 0..head_dim/2 -1 processed twice below
  int half = head_dim / 2;
  if (d >= half) return;

  float* base = x + ((b * heads + h) * seq_len + s) * head_dim;
  float q1 = base[d];
  float q2 = base[d + half];
  float c1 = cos[s * head_dim + d];
  float c2 = cos[s * head_dim + d + half];
  float s1 = sin[s * head_dim + d];
  float s2 = sin[s * head_dim + d + half];
  base[d] = q1 * c1 - q2 * s1;
  base[d + half] = q2 * c2 + q1 * s2;
}

// Repeat KV heads to match Q heads (GQA)
__global__ void repeat_kv_kernel(const float* in, float* out,
                 int batch, int kv_heads, int heads,
                 int seq_len, int head_dim, int n_rep) {
  int b = blockIdx.z;
  int kvh = blockIdx.y;
  int s = blockIdx.x;
  int d = threadIdx.x;
  if (d >= head_dim) return;
  for (int n = 0; n < n_rep; n++) {
    int h = kvh * n_rep + n;  // target head
    const float* src = in + ((b * kv_heads + kvh) * seq_len + s) * head_dim;
    float* dst = out + ((b * heads + h) * seq_len + s) * head_dim;
    dst[d] = src[d];
  }
}

// Compute attention scores: (B,H,S,S) = Q @ K^T scaled, causal mask applied
__global__ void attn_scores_kernel(const float* q, const float* k, float* scores,
                   int batch, int heads, int seq_len, int head_dim, float scale) {
  int b = blockIdx.z;
  int h = blockIdx.y;
  int i = blockIdx.x;  // query position
  int j = threadIdx.x; // key position
  if (j >= seq_len) return;

  const float* q_vec = q + ((b * heads + h) * seq_len + i) * head_dim;
  const float* k_vec = k + ((b * heads + h) * seq_len + j) * head_dim;
  float sum = 0.0f;
  for (int d = 0; d < head_dim; d++) sum += q_vec[d] * k_vec[d];
  float val = (j > i) ? -1e9f : sum * scale;  // causal mask
  scores[((b * heads + h) * seq_len + i) * seq_len + j] = val;
}

// Softmax over last dimension of scores (in-place to attn_scores)
__global__ void softmax_kernel(float* scores, int batch, int heads, int seq_len) {
  int b = blockIdx.z;
  int h = blockIdx.y;
  int i = blockIdx.x;  // query position

  float* row = scores + ((b * heads + h) * seq_len + i) * seq_len;

  // max
  float m = row[0];
  for (int j = 0; j < seq_len; j++) m = fmaxf(m, row[j]);
  // exp and sum
  float sum = 0.0f;
  for (int j = 0; j < seq_len; j++) {
    row[j] = expf(row[j] - m);
    sum += row[j];
  }
  float inv = 1.0f / sum;
  for (int j = 0; j < seq_len; j++) row[j] *= inv;
}

// attn_out = softmax * V  => (B,H,S,D)
__global__ void attn_weighted_v_kernel(const float* attn, const float* v,
                     float* out, int batch, int heads,
                     int seq_len, int head_dim) {
  int b = blockIdx.z;
  int h = blockIdx.y;
  int s = blockIdx.x;  // query position
  int d = threadIdx.x;
  if (d >= head_dim) return;

  const float* attn_row = attn + ((b * heads + h) * seq_len + s) * seq_len;
  float sum = 0.0f;
  for (int j = 0; j < seq_len; j++) {
    float vval = v[((b * heads + h) * seq_len + j) * head_dim + d];
    sum += attn_row[j] * vval;
  }
  out[((b * heads + h) * seq_len + s) * head_dim + d] = sum;
}

// flatten (B,H,S,D) -> (B,S,H*D)
__global__ void flatten_kernel(const float* in, float* out,
                 int batch, int heads, int seq_len, int head_dim) {
  int b = blockIdx.z;
  int s = blockIdx.y;
  int idx = blockIdx.x * blockDim.x + threadIdx.x; // 0..heads*head_dim-1
  int hidden = heads * head_dim;
  if (idx >= hidden) return;
  int h = idx / head_dim;
  int d = idx % head_dim;
  const float* src = in + ((b * heads + h) * seq_len + s) * head_dim;
  out[(b * seq_len + s) * hidden + idx] = src[d];
}

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
    int kv_hidden = num_kv_heads * head_dim;
    int bs = batch * seq_len;
    int q_out_cols = hidden_size;
    int kv_out_cols = kv_hidden;
    int n_rep = num_heads / num_kv_heads;
    float scale = 1.0f / sqrtf((float)head_dim);

    // Copy input data to GPU
    CHECK_CUDA(cudaMemcpy(x_gpu, x, batch * seq_len * hidden_size * sizeof(float), cudaMemcpyHostToDevice));

    // 1) Q/K/V projections: (bs, hidden) @ (out_cols, hidden)^T
    dim3 block_mm(BLOCK_MM, BLOCK_MM);
    dim3 grid_q((q_out_cols + block_mm.x - 1) / block_mm.x,
                (bs + block_mm.y - 1) / block_mm.y);
    matmul_transposed_kernel<<<grid_q, block_mm>>>(x_gpu, q_proj_gpu, q_proj_out_gpu,
                                                   bs, hidden_size, q_out_cols);

    dim3 grid_k((kv_out_cols + block_mm.x - 1) / block_mm.x,
                (bs + block_mm.y - 1) / block_mm.y);
    matmul_transposed_kernel<<<grid_k, block_mm>>>(x_gpu, k_proj_gpu, k_proj_out_gpu,
                                                   bs, hidden_size, kv_out_cols);
    matmul_transposed_kernel<<<grid_k, block_mm>>>(x_gpu, v_proj_gpu, v_proj_out_gpu,
                                                   bs, hidden_size, kv_out_cols);

    // 2) RMSNorm on Q/K (vector length = head_dim)
    int q_vecs = bs * num_heads;
    int k_vecs = bs * num_kv_heads;
    int grid_norm_q = (q_vecs + BLOCK_NORM - 1) / BLOCK_NORM;
    int grid_norm_k = (k_vecs + BLOCK_NORM - 1) / BLOCK_NORM;
    rmsnorm_kernel<<<grid_norm_q, BLOCK_NORM>>>(q_proj_out_gpu, q_norm_gpu, q_normed_gpu,
                                                  head_dim, q_vecs);
    rmsnorm_kernel<<<grid_norm_k, BLOCK_NORM>>>(k_proj_out_gpu, k_norm_gpu, k_normed_gpu,
                                                  head_dim, k_vecs);

    // 3) Reshape to (B, H, S, D)
    dim3 grid_reshape((seq_len + BLOCK_RESHAPE - 1) / BLOCK_RESHAPE, num_heads, batch);
    reshape_qkv_kernel<<<grid_reshape, BLOCK_RESHAPE>>>(q_normed_gpu, q_transposed_gpu,
                                              batch, seq_len, num_heads, head_dim);

    dim3 grid_reshape_k((seq_len + BLOCK_RESHAPE - 1) / BLOCK_RESHAPE, num_kv_heads, batch);
    reshape_qkv_kernel<<<grid_reshape_k, BLOCK_RESHAPE>>>(k_normed_gpu, k_transposed_gpu,
                                                batch, seq_len, num_kv_heads, head_dim);
    reshape_qkv_kernel<<<grid_reshape_k, BLOCK_RESHAPE>>>(v_proj_out_gpu, v_transposed_gpu,
                                                batch, seq_len, num_kv_heads, head_dim);

    // 4) RoPE
    dim3 grid_rope_q(seq_len, num_heads, batch);
    rope_kernel<<<grid_rope_q, head_dim / 2>>>(q_transposed_gpu, cos_gpu, sin_gpu,
                                               batch, num_heads, seq_len, head_dim);
    dim3 grid_rope_k(seq_len, num_kv_heads, batch);
    rope_kernel<<<grid_rope_k, head_dim / 2>>>(k_transposed_gpu, cos_gpu, sin_gpu,
                                               batch, num_kv_heads, seq_len, head_dim);

    // 5) Repeat KV for GQA
    dim3 grid_rep(seq_len, num_kv_heads, batch);
    repeat_kv_kernel<<<grid_rep, head_dim>>>(k_transposed_gpu, k_repeated_gpu,
                                             batch, num_kv_heads, num_heads, seq_len, head_dim, n_rep);
    // Reuse attn_out_transposed_gpu as temporary buffer for repeated V
    repeat_kv_kernel<<<grid_rep, head_dim>>>(v_transposed_gpu, attn_out_transposed_gpu,
                                             batch, num_kv_heads, num_heads, seq_len, head_dim, n_rep);

    // 6) Attention scores and softmax
    dim3 grid_scores(seq_len, num_heads, batch);
    attn_scores_kernel<<<grid_scores, seq_len>>>(q_transposed_gpu, k_repeated_gpu, attn_scores_gpu,
                                                 batch, num_heads, seq_len, head_dim, scale);
    softmax_kernel<<<grid_scores, 1>>>(attn_scores_gpu, batch, num_heads, seq_len);

    // 7) Weighted sum with V
    attn_weighted_v_kernel<<<grid_scores, head_dim>>>(attn_scores_gpu, attn_out_transposed_gpu,
                                                      attn_out_gpu, batch, num_heads, seq_len, head_dim);

    // 8) Flatten to (B*S, hidden)
    dim3 grid_flat((hidden_size + BLOCK_FLAT - 1) / BLOCK_FLAT, seq_len, batch);
    flatten_kernel<<<grid_flat, BLOCK_FLAT>>>(attn_out_gpu, attn_out_transposed_gpu,
                                       batch, num_heads, seq_len, head_dim);

    // 9) Output projection
    int rows = bs;
    dim3 grid_out((hidden_size + block_mm.x - 1) / block_mm.x,
                  (rows + block_mm.y - 1) / block_mm.y);
    matmul_transposed_kernel<<<grid_out, block_mm>>>(attn_out_transposed_gpu, o_proj_gpu, output_gpu,
                                                     rows, hidden_size, hidden_size);
    
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
