#include "model.h"
#include "model_loader.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <cstring>
#include <cuda_runtime.h>

// Global model loader (definition)
std::unique_ptr<ModelLoader> g_model_loader;

// ============================================================================
// CUDA Kernels
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

__global__ void transpose_BSHD_BHSD_kernel(float *in, float *out, int B, int S, int H, int D) {
    // (B, S, H, D) -> (B, H, S, D)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * S * H * D;
    if (idx < total) {
        int d = idx % D;
        int rem = idx / D;
        int h = rem % H;
        rem /= H;
        int s = rem % S;
        int b = rem / S;
        
        // Out index: b, h, s, d
        int out_idx = ((b * H + h) * S + s) * D + d;
        out[out_idx] = in[idx];
    }
}

__global__ void transpose_BHSD_BSHD_kernel(float *in, float *out, int B, int H, int S, int D) {
    // (B, H, S, D) -> (B, S, H, D)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * S * D;
    if (idx < total) {
        int d = idx % D;
        int rem = idx / D;
        int s = rem % S;
        rem /= S;
        int h = rem % H;
        int b = rem / H;
        
        // Out index: b, s, h, d
        int out_idx = ((b * S + s) * H + h) * D + d;
        out[out_idx] = in[idx];
    }
}

__global__ void transpose_BSC_BCS_kernel(float *in, float *out, int B, int S, int C) {
    // (B, S, C) -> (B, C, S)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * S * C;
    if (idx < total) {
        int c = idx % C;
        int rem = idx / C;
        int s = rem % S;
        int b = rem / S;
        
        // Out index: b, c, s
        int out_idx = (b * C + c) * S + s;
        out[out_idx] = in[idx];
    }
}

__global__ void transpose_BCS_BSC_kernel(float *in, float *out, int B, int C, int S) {
    // (B, C, S) -> (B, S, C)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * S;
    if (idx < total) {
        int s = idx % S;
        int rem = idx / S;
        int c = rem % C;
        int b = rem / C;
        
        // Out index: b, s, c
        int out_idx = (b * S + s) * C + c;
        out[out_idx] = in[idx];
    }
}

// MoE Kernels
__global__ void router_kernel(float* router_logits, int* top_k_indices, float* top_k_weights, 
                              int num_tokens, int num_experts, int k, float expert_bias_val) {
    // Naive implementation for small k (k=2)
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < num_tokens) {
        // 1. Compute sigmoid and scores
        // Since k is small (2) and num_experts is small, we iterate.
        
        // Find top-k
        // For k=2, we can just find max, then second max.
        
        int best_idx = -1;
        float best_score = -1e20f;
        float best_weight = 0.0f;
        
        for (int e = 0; e < num_experts; ++e) {
            float logit = router_logits[t * num_experts + e];
            float weight = 1.0f / (1.0f + expf(-logit));
            float score = weight; // Assuming no bias for now or handled
            
            if (score > best_score) {
                best_score = score;
                best_idx = e;
                best_weight = weight;
            }
        }
        
        top_k_indices[t * k + 0] = best_idx;
        top_k_weights[t * k + 0] = best_weight;
        
        // Find second best
        int second_best_idx = -1;
        float second_best_score = -1e20f;
        float second_best_weight = 0.0f;
        
        for (int e = 0; e < num_experts; ++e) {
            if (e == best_idx) continue;
            
            float logit = router_logits[t * num_experts + e];
            float weight = 1.0f / (1.0f + expf(-logit));
            float score = weight;
            
            if (score > second_best_score) {
                second_best_score = score;
                second_best_idx = e;
                second_best_weight = weight;
            }
        }
        
        top_k_indices[t * k + 1] = second_best_idx;
        top_k_weights[t * k + 1] = second_best_weight;
        
        // Normalize
        float sum = top_k_weights[t * k + 0] + top_k_weights[t * k + 1];
        if (sum > 1e-6f) {
            top_k_weights[t * k + 0] /= sum;
            top_k_weights[t * k + 1] /= sum;
        }
        
        // Scale (assuming scaling factor is 1.0)
    }
}

__global__ void gather_kernel(float* x, int* top_k_indices, float* expert_input, 
                              int num_tokens, int hidden_size, int k, int expert_idx, int* count) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < num_tokens) {
        for (int i = 0; i < k; ++i) {
            if (top_k_indices[t * k + i] == expert_idx) {
                int pos = atomicAdd(count, 1);
                for (int h = 0; h < hidden_size; ++h) {
                    expert_input[pos * hidden_size + h] = x[t * hidden_size + h];
                }
                // Store expert input
                for (int h = 0; h < hidden_size; ++h) {
                    expert_input[pos * hidden_size + h] = x[t * hidden_size + h];
                }
            }
        }
    }
}

// ============================================================================
// Large Block Implementations - Complex layers and modules
// ============================================================================

// MLP (Feed-Forward Network) implementation
MLP::MLP(const std::string& w1_file, const std::string& w2_file, const std::string& w3_file) {
    w1_ = Tensor::load_from_file(w1_file);
    w2_ = Tensor::load_from_file(w2_file);
    w3_ = Tensor::load_from_file(w3_file);
}

void MLP::forward(const Tensor& x, Tensor& y) {
    // x: (batch, seq_len, hidden_size)
    // w1: (intermediate_size, hidden_size)
    // w3: (intermediate_size, hidden_size)
    // w2: (hidden_size, intermediate_size)
    
    size_t batch = x.size(0);
    size_t seq_len = x.size(1);
    size_t hidden_size = x.size(2);
    size_t intermediate_size = w1_.size(0);
    
    // Flatten batch and seq_len
    Tensor x_flat = x.view({batch * seq_len, hidden_size});
    
    // gate = silu(x @ w1.T)
    Tensor gate({batch * seq_len, intermediate_size});
    tensor_ops::matmul_transposed(x_flat, w1_, gate);
    Tensor gate_silu({batch * seq_len, intermediate_size});
    tensor_ops::silu(gate, gate_silu);
    
    // up = x @ w3.T
    Tensor up({batch * seq_len, intermediate_size});
    tensor_ops::matmul_transposed(x_flat, w3_, up);
    
    // hidden = gate_silu * up
    Tensor hidden({batch * seq_len, intermediate_size});
    tensor_ops::mul(gate_silu, up, hidden);
    
    // y = hidden @ w2.T
    Tensor y_flat({batch * seq_len, hidden_size});
    tensor_ops::matmul_transposed(hidden, w2_, y_flat);
    
    // Reshape and assign to output
    y_flat.reshape({batch, seq_len, hidden_size});
    
    // If y is not allocated, allocate it
    if (y.size() == 0) {
        y = Tensor({batch, seq_len, hidden_size});
    }
    std::memcpy(y.data(), y_flat.data(), y.size() * sizeof(float));
}

// Additional MoE Kernels
__global__ void gather_with_indices_kernel(float* x, int* top_k_indices, float* expert_input, int* indices_map,
                                           int num_tokens, int hidden_size, int k, int expert_idx, int* count) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < num_tokens) {
        for (int i = 0; i < k; ++i) {
            if (top_k_indices[t * k + i] == expert_idx) {
                int pos = atomicAdd(count, 1);
                indices_map[pos] = t; // Store original token index
                for (int h = 0; h < hidden_size; ++h) {
                    expert_input[pos * hidden_size + h] = x[t * hidden_size + h];
                }
            }
        }
    }
}

__global__ void scatter_add_kernel(float* expert_output, int* indices_map, float* top_k_weights, int* top_k_indices, 
                                   float* y, int count, int hidden_size, int k, int expert_idx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = count * hidden_size;
    
    if (idx < total) {
        int h = idx % hidden_size;
        int pos = idx / hidden_size;
        
        int t = indices_map[pos];
        float output_val = expert_output[pos * hidden_size + h];
        
        // Find weight for this expert
        float weight = 0.0f;
        for (int i = 0; i < k; ++i) {
            if (top_k_indices[t * k + i] == expert_idx) {
                weight = top_k_weights[t * k + i];
                break;
            }
        }
        
        atomicAdd(&y[t * hidden_size + h], output_val * weight);
    }
}

// SparseMoeBlock implementation
SparseMoeBlock::SparseMoeBlock(int layer_idx) {
    // Load gate weights (router)
    std::stringstream ss;
    ss << "layers." << layer_idx << ".feed_forward.gate.weight";
    gate_ = Tensor::load_from_file(ss.str());
    
    // Load expert weights
    experts_.reserve(NUM_EXPERTS);
    for (size_t i = 0; i < NUM_EXPERTS; i++) {
        std::stringstream ss_w1, ss_w2, ss_w3;
        ss_w1 << "layers." << layer_idx << ".feed_forward.experts." << i << ".w1.weight";
        ss_w2 << "layers." << layer_idx << ".feed_forward.experts." << i << ".w2.weight";
        ss_w3 << "layers." << layer_idx << ".feed_forward.experts." << i << ".w3.weight";
        
        experts_.emplace_back(ss_w1.str(), ss_w2.str(), ss_w3.str());
    }
    
    // Load expert bias if used
    if (USE_EXPERT_BIAS) {
        std::stringstream ss_bias;
        ss_bias << "layers." << layer_idx << ".feed_forward.expert_bias";
        expert_bias_ = Tensor::load_from_file(ss_bias.str());
    }
    
    // Initialize d_count_tensor_ (1 element)
    d_count_tensor_ = Tensor({1}); // TODO: undefined identifier
}

void SparseMoeBlock::route_tokens(const Tensor& router_logits, 
                                   Tensor& top_k_indices,
                                   Tensor& top_k_weights) { // TODO: different signature from header
    // router_logits: (batch * seq_len, num_experts)
    size_t num_tokens = router_logits.size(0);
    
    // Allocate outputs on GPU
    top_k_indices.resize({num_tokens, NUM_EXPERTS_PER_TOK});
    top_k_weights.resize({num_tokens, NUM_EXPERTS_PER_TOK});
    
    int threads = 256;
    int blocks = (num_tokens + threads - 1) / threads;
    
    // We need to cast float* to int* for indices
    router_kernel<<<blocks, threads>>>(
        (float*)router_logits.data(), 
        (int*)top_k_indices.data(), 
        top_k_weights.data(), 
        num_tokens, NUM_EXPERTS, NUM_EXPERTS_PER_TOK, 0.0f
    );
}

void SparseMoeBlock::forward(const Tensor& x, Tensor& y, Tensor& router_logits) {
    // x: (batch, seq_len, hidden_size)
    size_t batch = x.size(0);
    size_t seq_len = x.size(1);
    size_t hidden_size = x.size(2);
    size_t num_tokens = batch * seq_len;
    
    // Flatten
    Tensor x_flat = x.view({num_tokens, hidden_size});
    
    // Compute router logits
    router_logits_.resize({num_tokens, NUM_EXPERTS}); // TODO: undefined identifier
    tensor_ops::matmul_transposed(x_flat, gate_, router_logits_);
    
    // Route tokens
    route_tokens(router_logits_, top_k_indices_, top_k_weights_); // TODO: undefined identifier
    
    // Initialize output
    y = Tensor({batch, seq_len, hidden_size});
    y.zero();
    
    // Process each expert
    int* d_count = (int*)d_count_tensor_.data(); // TODO: undefined identifier
    
    // Temporary buffers for gather/scatter
    expert_input_.resize({num_tokens, hidden_size}); // TODO: undefined identifier
    expert_output_.resize({num_tokens, hidden_size}); // TODO: undefined identifier
    indices_map_.resize({num_tokens}); // Store original indices, treat as float* but use as int*
    
    for (size_t e = 0; e < NUM_EXPERTS; e++) {
        // Reset count
        cudaMemset(d_count, 0, sizeof(int));
        
        // Gather
        int threads = 256;
        int blocks = (num_tokens + threads - 1) / threads;
        gather_with_indices_kernel<<<blocks, threads>>>(
            (float*)x_flat.data(), 
            (int*)top_k_indices_.data(), 
            expert_input_.data(), 
            (int*)indices_map_.data(),
            num_tokens, hidden_size, NUM_EXPERTS_PER_TOK, e, d_count
        );
        
        // Get count to host
        int count = 0;
        cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
        
        if (count > 0) {
            // View into the buffers for the valid count
            Tensor curr_input = expert_input_.view({(size_t)count, hidden_size});
            Tensor curr_output = expert_output_.view({(size_t)count, hidden_size});
            
            // Forward pass through expert MLP
            experts_[e].forward(curr_input, curr_output);
            
            // Scatter add
            int total_elements = count * hidden_size;
            int scatter_blocks = (total_elements + threads - 1) / threads;
            scatter_add_kernel<<<scatter_blocks, threads>>>(
                curr_output.data(), 
                (int*)indices_map_.data(), 
                top_k_weights_.data(), 
                (int*)top_k_indices_.data(), 
                y.data(), 
                count, hidden_size, NUM_EXPERTS_PER_TOK, e
            );
        }
    }
    
    // Copy router_logits to output argument if needed
    if (router_logits.size() == 0) {
        router_logits = router_logits_.copy();
    } else {
        // Caller allocated, copy data.
        cudaMemcpy(router_logits.data(), router_logits_.data(), router_logits_.size() * sizeof(float), cudaMemcpyDeviceToDevice);
    }
}

// Attention implementation
Attention::Attention(int layer_idx) : layer_idx_(layer_idx) {
    std::stringstream ss_q, ss_k, ss_v, ss_o, ss_q_ln, ss_k_ln;
    ss_q << "layers." << layer_idx << ".self_attn.q_proj.weight";
    ss_k << "layers." << layer_idx << ".self_attn.k_proj.weight";
    ss_v << "layers." << layer_idx << ".self_attn.v_proj.weight";
    ss_o << "layers." << layer_idx << ".self_attn.out_proj.weight";
    ss_q_ln << "layers." << layer_idx << ".self_attn.q_layernorm.weight";
    ss_k_ln << "layers." << layer_idx << ".self_attn.k_layernorm.weight";
    
    q_proj_ = Tensor::load_from_file(ss_q.str());
    k_proj_ = Tensor::load_from_file(ss_k.str());
    v_proj_ = Tensor::load_from_file(ss_v.str());
    o_proj_ = Tensor::load_from_file(ss_o.str());
    
    q_layernorm_ = std::make_unique<RMSNorm>(ss_q_ln.str());
    k_layernorm_ = std::make_unique<RMSNorm>(ss_k_ln.str());
}

void Attention::forward(const Tensor& x, const Tensor& cos, const Tensor& sin,
                       const Tensor* attention_mask, Tensor& output) {
    // x: (batch, seq_len, hidden_size)
    size_t batch = x.size(0);
    size_t seq_len = x.size(1);
    size_t hidden_size = x.size(2);
    
    // Flatten
    Tensor x_flat = x.view({batch * seq_len, hidden_size});
    
    // Project Q, K, V
    q_proj_out_.resize({batch * seq_len, NUM_ATTENTION_HEADS * HEAD_DIM});
    k_proj_out_.resize({batch * seq_len, NUM_KEY_VALUE_HEADS * HEAD_DIM});
    v_proj_out_.resize({batch * seq_len, NUM_KEY_VALUE_HEADS * HEAD_DIM});
    
    tensor_ops::matmul_transposed(x_flat, q_proj_, q_proj_out_);
    tensor_ops::matmul_transposed(x_flat, k_proj_, k_proj_out_);
    tensor_ops::matmul_transposed(x_flat, v_proj_, v_proj_out_);
    
    // Reshape to (batch, seq_len, num_heads, head_dim) for layernorm
    // This is just a view if memory is contiguous (it is)
    Tensor q_reshaped = q_proj_out_.view({batch, seq_len, NUM_ATTENTION_HEADS, HEAD_DIM});
    Tensor k_reshaped = k_proj_out_.view({batch, seq_len, NUM_KEY_VALUE_HEADS, HEAD_DIM});
    Tensor v_reshaped = v_proj_out_.view({batch, seq_len, NUM_KEY_VALUE_HEADS, HEAD_DIM});
    
    // Apply layernorm to Q and K
    q_normed_.resize({batch, seq_len, NUM_ATTENTION_HEADS, HEAD_DIM}); // TODO: undefined identifier
    k_normed_.resize({batch, seq_len, NUM_KEY_VALUE_HEADS, HEAD_DIM});
    q_layernorm_->forward(q_reshaped, q_normed_);
    k_layernorm_->forward(k_reshaped, k_normed_);
    
    // Transpose to (batch, num_heads, seq_len, head_dim) for attention
    // Use kernel
    q_.resize({batch, NUM_ATTENTION_HEADS, seq_len, HEAD_DIM}); // TODO: undefined identifier
    k_.resize({batch, NUM_KEY_VALUE_HEADS, seq_len, HEAD_DIM});
    v_.resize({batch, NUM_KEY_VALUE_HEADS, seq_len, HEAD_DIM});
    
    int threads = 256;
    int blocks_q = (batch * NUM_ATTENTION_HEADS * seq_len * HEAD_DIM + threads - 1) / threads;
    int blocks_k = (batch * NUM_KEY_VALUE_HEADS * seq_len * HEAD_DIM + threads - 1) / threads;
    
    transpose_BSHD_BHSD_kernel<<<blocks_q, threads>>>(q_normed_.data(), q_.data(), batch, seq_len, NUM_ATTENTION_HEADS, HEAD_DIM);
    transpose_BSHD_BHSD_kernel<<<blocks_k, threads>>>(k_normed_.data(), k_.data(), batch, seq_len, NUM_KEY_VALUE_HEADS, HEAD_DIM);
    
    // For V, we need to transpose v_reshaped (B, S, H, D) -> (B, H, S, D)
    transpose_BSHD_BHSD_kernel<<<blocks_k, threads>>>(v_reshaped.data(), v_.data(), batch, seq_len, NUM_KEY_VALUE_HEADS, HEAD_DIM);
    
    // Apply RoPE
    tensor_ops::apply_rotary_pos_emb(q_, k_, cos, sin);
    
    // Repeat K, V for GQA
    k_repeated_.resize({batch, NUM_ATTENTION_HEADS, seq_len, HEAD_DIM}); // TODO: undefined identifier
    v_repeated_.resize({batch, NUM_ATTENTION_HEADS, seq_len, HEAD_DIM});
    tensor_ops::repeat_kv(k_, NUM_KEY_VALUE_GROUPS, k_repeated_);
    tensor_ops::repeat_kv(v_, NUM_KEY_VALUE_GROUPS, v_repeated_);
    
    // Compute attention: Q @ K^T
    float scale = 1.0f / std::sqrt((float)HEAD_DIM);
    scores_.resize({batch, NUM_ATTENTION_HEADS, seq_len, seq_len}); // TODO: undefined identifier
    
    int blocks_scores = (batch * NUM_ATTENTION_HEADS * seq_len * seq_len + threads - 1) / threads;
    batched_matmul_qk_kernel<<<blocks_scores, threads>>>(q_.data(), k_repeated_.data(), scores_.data(), batch, NUM_ATTENTION_HEADS, seq_len, HEAD_DIM, scale); // TODO: undefined identifier
    
    // Apply causal mask
    causal_mask_kernel<<<blocks_scores, threads>>>(scores_.data(), batch, NUM_ATTENTION_HEADS, seq_len); // TODO: undefined identifier
    
    // Softmax
    attn_weights_.resize({batch, NUM_ATTENTION_HEADS, seq_len, seq_len}); // TODO: undefined identifier
    tensor_ops::softmax(scores_, attn_weights_, -1);
    
    // Multiply by V: attn_weights @ V
    attn_output_.resize({batch, NUM_ATTENTION_HEADS, seq_len, HEAD_DIM}); // TODO: undefined identifier
    int blocks_out = (batch * NUM_ATTENTION_HEADS * seq_len * HEAD_DIM + threads - 1) / threads;
    batched_matmul_sv_kernel<<<blocks_out, threads>>>(attn_weights_.data(), v_repeated_.data(), attn_output_.data(), batch, NUM_ATTENTION_HEADS, seq_len, HEAD_DIM); // TODO: undefined identifier
    
    // Transpose back: (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, num_heads, head_dim)
    // And flatten to (batch * seq_len, hidden_size)
    // We can transpose directly to (batch, seq_len, hidden_size) if we treat H*D as contiguous
    attn_flat_.resize({batch * seq_len, hidden_size}); // TODO: undefined identifier
    transpose_BHSD_BSHD_kernel<<<blocks_out, threads>>>(attn_output_.data(), attn_flat_.data(), batch, NUM_ATTENTION_HEADS, seq_len, HEAD_DIM);
    
    output_flat_.resize({batch * seq_len, hidden_size}); // TODO: undefined identifier
    tensor_ops::matmul_transposed(attn_flat_, o_proj_, output_flat_);
    
    output_flat_.reshape({batch, seq_len, hidden_size});
    
    // Allocate output if needed
    if (output.size() == 0) {
        output = Tensor({batch, seq_len, hidden_size});
    }
    // Copy result
    cudaMemcpy(output.data(), output_flat_.data(), output.size() * sizeof(float), cudaMemcpyDeviceToDevice);
}

// ShortConv implementation
ShortConv::ShortConv(int layer_idx) : layer_idx_(layer_idx) {
    std::stringstream ss_conv, ss_in, ss_out;
    ss_conv << "layers." << layer_idx << ".conv.conv.weight";
    ss_in << "layers." << layer_idx << ".conv.in_proj.weight";
    ss_out << "layers." << layer_idx << ".conv.out_proj.weight";
    
    conv_weight_ = Tensor::load_from_file(ss_conv.str());
    in_proj_weight_ = Tensor::load_from_file(ss_in.str());
    out_proj_weight_ = Tensor::load_from_file(ss_out.str());
    
    // Load biases if they exist
    if (USE_CONV_BIAS) {
        std::stringstream ss_conv_bias, ss_in_bias, ss_out_bias;
        ss_conv_bias << "layers." << layer_idx << ".conv.conv.bias";
        ss_in_bias << "layers." << layer_idx << ".conv.in_proj.bias";
        ss_out_bias << "layers." << layer_idx << ".conv.out_proj.bias";
        
        if (g_model_loader->has_tensor(ss_conv_bias.str())) {
            conv_bias_ = Tensor::load_from_file(ss_conv_bias.str());
        }
        if (g_model_loader->has_tensor(ss_in_bias.str())) {
            in_proj_bias_ = Tensor::load_from_file(ss_in_bias.str());
        }
        if (g_model_loader->has_tensor(ss_out_bias.str())) {
            out_proj_bias_ = Tensor::load_from_file(ss_out_bias.str());
        }
    }
}

void ShortConv::forward(const Tensor& x, Tensor& y) {
    // x: (batch, seq_len, hidden_size)
    size_t batch = x.size(0);
    size_t seq_len = x.size(1);
    size_t hidden_size = x.size(2);
    
    // Flatten for matmul
    Tensor x_flat = x.view({batch * seq_len, hidden_size});
    
    // in_proj: (batch*seq_len, hidden_size) @ (3*hidden_size, hidden_size)^T -> (batch*seq_len, 3*hidden_size)
    in_proj_out_.resize({batch * seq_len, 3 * hidden_size}); // TODO: undefined identifier
    tensor_ops::matmul_transposed(x_flat, in_proj_weight_, in_proj_out_);
    
    // Add bias if present
    if (USE_CONV_BIAS && in_proj_bias_.size() > 0) {
        tensor_ops::add_bias(in_proj_out_, in_proj_bias_, in_proj_out_);
    }
    
    // Reshape to (batch, seq_len, 3*hidden_size)
    // Then Transpose to (batch, 3*hidden_size, seq_len) -> (B, C, S)
    // Use kernel
    BCx_.resize({batch, 3 * hidden_size, seq_len}); // TODO: undefined identifier
    int threads = 256;
    int blocks = (batch * 3 * hidden_size * seq_len + threads - 1) / threads;
    transpose_BSC_BCS_kernel<<<blocks, threads>>>(in_proj_out_.data(), BCx_.data(), batch, seq_len, 3 * hidden_size);
    
    // Split into B, C, x_gate
    // BCx_ is (B, 3H, S). Memory layout:
    // b=0: [3H lines of S]
    //   h=0..H-1: B
    //   h=H..2H-1: C
    //   h=2H..3H-1: x_gate
    // We copy these contiguous blocks to separate tensors.
    
    B_.resize({batch, hidden_size, seq_len}); // TODO: undefined identifier
    C_.resize({batch, hidden_size, seq_len});
    x_gate_.resize({batch, hidden_size, seq_len});
    
    // 3*Batch cudaMemcpys
    for (size_t b = 0; b < batch; b++) {
        size_t batch_offset = b * 3 * hidden_size * seq_len;
        size_t size_per_part = hidden_size * seq_len * sizeof(float);
        
        cudaMemcpy(B_.data() + b * hidden_size * seq_len, 
                   BCx_.data() + batch_offset, 
                   size_per_part, cudaMemcpyDeviceToDevice);
                   
        cudaMemcpy(C_.data() + b * hidden_size * seq_len, 
                   BCx_.data() + batch_offset + hidden_size * seq_len, 
                   size_per_part, cudaMemcpyDeviceToDevice);
                   
        cudaMemcpy(x_gate_.data() + b * hidden_size * seq_len, 
                   BCx_.data() + batch_offset + 2 * hidden_size * seq_len, 
                   size_per_part, cudaMemcpyDeviceToDevice);
    }
    
    // Bx = B * x_gate (element-wise)
    Bx_.resize({batch, hidden_size, seq_len}); // TODO: undefined identifier
    tensor_ops::mul(B_, x_gate_, Bx_);
    
    // Apply causal conv1d on Bx
    conv_out_.resize({batch, hidden_size, seq_len}); // TODO: undefined identifier
    tensor_ops::causal_conv1d(Bx_, conv_weight_, USE_CONV_BIAS ? &conv_bias_ : nullptr, conv_out_);
    
    // y_pre = C * conv_out (element-wise)
    y_pre_.resize({batch, hidden_size, seq_len}); // TODO: undefined identifier
    tensor_ops::mul(C_, conv_out_, y_pre_);
    
    // Transpose back: (batch, hidden_size, seq_len) -> (batch, seq_len, hidden_size)
    y_pre_transposed_.resize({batch, seq_len, hidden_size}); // TODO: undefined identifier
    transpose_BCS_BSC_kernel<<<blocks, threads>>>(y_pre_.data(), y_pre_transposed_.data(), batch, hidden_size, seq_len);
    
    // out_proj
    // Copy transposed data to flat buffer
    y_pre_flat_.resize({batch * seq_len, hidden_size}); // TODO: undefined identifier
    
    y_flat_.resize({batch * seq_len, hidden_size}); // TODO: undefined identifier
    tensor_ops::matmul_transposed(y_pre_transposed_.view({batch * seq_len, hidden_size}), out_proj_weight_, y_flat_);
    
    // Add bias if present
    if (USE_CONV_BIAS && out_proj_bias_.size() > 0) {
        tensor_ops::add_bias(y_flat_, out_proj_bias_, y_flat_);
    }
    
    // Reshape back
    y_flat_.reshape({batch, seq_len, hidden_size});
    
    if (y.size() == 0) {
        y = Tensor({batch, seq_len, hidden_size});
    }
    cudaMemcpy(y.data(), y_flat_.data(), y.size() * sizeof(float), cudaMemcpyDeviceToDevice);
}

// DecoderLayer implementation
DecoderLayer::DecoderLayer(int layer_idx, bool is_attention_layer)
    : layer_idx_(layer_idx), is_attention_layer_(is_attention_layer) {
    
    // Load normalization layers
    std::stringstream ss_norm1, ss_norm2;
    ss_norm1 << "layers." << layer_idx << ".operator_norm.weight";
    ss_norm2 << "layers." << layer_idx << ".ffn_norm.weight";
    
    input_layernorm_ = std::make_unique<RMSNorm>(ss_norm1.str());
    post_attention_layernorm_ = std::make_unique<RMSNorm>(ss_norm2.str());
    
    // Load attention or conv
    if (is_attention_layer) {
        self_attn_ = std::make_unique<Attention>(layer_idx);
    } else {
        short_conv_ = std::make_unique<ShortConv>(layer_idx);
    }
    
    // Load MoE block (only for layers >= num_dense_layers, first layers are dense)
    if (static_cast<size_t>(layer_idx) >= NUM_DENSE_LAYERS) {
        moe_block_ = std::make_unique<SparseMoeBlock>(layer_idx);
    } else {
        // Dense layer - load simple MLP
        std::stringstream ss_w1, ss_w2, ss_w3;
        ss_w1 << "layers." << layer_idx << ".feed_forward.w1.weight";
        ss_w2 << "layers." << layer_idx << ".feed_forward.w2.weight";
        ss_w3 << "layers." << layer_idx << ".feed_forward.w3.weight";
        dense_mlp_ = std::make_unique<MLP>(ss_w1.str(), ss_w2.str(), ss_w3.str());
    }
}

void DecoderLayer::forward(const Tensor& x, const Tensor& cos, const Tensor& sin,
                          const Tensor* attention_mask, Tensor& output) {
    // Input norm
    Tensor normed_input(x.shape());
    input_layernorm_->forward(x, normed_input);
    
    // Attention or Conv
    Tensor attn_output(x.shape());
    if (is_attention_layer_) {
        self_attn_->forward(normed_input, cos, sin, attention_mask, attn_output);
    } else {
        short_conv_->forward(normed_input, attn_output);
    }
    
    // Residual connection
    Tensor hidden_states(x.shape());
    tensor_ops::add(x, attn_output, hidden_states);
    
    // Post attention norm
    Tensor normed_hidden(x.shape());
    post_attention_layernorm_->forward(hidden_states, normed_hidden);
    
    // MoE block or dense MLP
    Tensor ffn_output;
    if (moe_block_) {
        // MoE layer (layers >= 2)
        Tensor router_logits;
        moe_block_->forward(normed_hidden, ffn_output, router_logits);
    } else {
        // Dense layer (layers 0-1)
        dense_mlp_->forward(normed_hidden, ffn_output);
    }
    
    // Residual connection
    tensor_ops::add(hidden_states, ffn_output, output);
}

// ============================================================================
// LFM2Model Implementation - Complete model
// ============================================================================

LFM2Model::LFM2Model(const std::string& model_file) {
    std::cout << "Loading LFM2-8B-A1B model from " << model_file << std::endl;
    
    // Initialize global model loader
    g_model_loader = std::make_unique<ModelLoader>(model_file);
    
    load_embeddings();
    load_layers();
    load_output_layers();
    
    // Initialize RoPE
    rotary_emb_ = std::make_unique<RotaryEmbedding>();
    
    std::cout << "Model loaded successfully!" << std::endl;
}

void LFM2Model::load_embeddings() {
    std::cout << "Loading embeddings..." << std::endl;
    embed_tokens_ = Tensor::load_from_file("embed_tokens.weight");
    std::cout << "  Embeddings shape: " << embed_tokens_.size(0) << " x " << embed_tokens_.size(1) << std::endl;
}

void LFM2Model::load_layers() {
    std::cout << "Loading " << NUM_HIDDEN_LAYERS << " decoder layers..." << std::endl;
    
    // Read layer types from config.h LAYER_TYPES array
    // 0 = full_attention, 1 = conv
    layers_.reserve(NUM_HIDDEN_LAYERS);
    for (size_t i = 0; i < NUM_HIDDEN_LAYERS; i++) {
        bool is_attention = (LAYER_TYPES[i] == 0);
        std::cout << "  Layer " << i << ": " << (is_attention ? "Attention" : "Conv") << std::endl;
        layers_.push_back(std::make_unique<DecoderLayer>(i, is_attention));
    }
}

// TODO: delete all below and re-implement from original baseline

void LFM2Model::load_output_layers() {
    // Load embedding
    embedding_ = Tensor::load_from_file("model.embed_tokens.weight");
    
    // Load layers
    layers_.reserve(NUM_LAYERS);
    for (int i = 0; i < NUM_LAYERS; i++) {
        layers_.push_back(std::make_unique<DecoderLayer>(i));
    }
    
    // Load norm
    norm_ = std::make_unique<RMSNorm>("model.norm.weight");
    
    // Load lm_head
    lm_head_ = Tensor::load_from_file("lm_head.weight");
}

void LFM2Model::forward(const std::vector<int>& tokens, Tensor& logits) {
    size_t seq_len = tokens.size();
    size_t batch = 1; // Currently only support batch size 1 for inference
    
    // Allocate input ids on GPU
    int* d_tokens;
    cudaMalloc(&d_tokens, seq_len * sizeof(int));
    cudaMemcpy(d_tokens, tokens.data(), seq_len * sizeof(int), cudaMemcpyHostToDevice);
    
    // Embedding lookup
    Tensor x({batch, seq_len, HIDDEN_SIZE});
    int threads = 256;
    int blocks = (batch * seq_len * HIDDEN_SIZE + threads - 1) / threads;
    embedding_kernel<<<blocks, threads>>>(d_tokens, embedding_.data(), x.data(), batch, seq_len, HIDDEN_SIZE);
    
    cudaFree(d_tokens);
    
    // Precompute RoPE cos/sin
    Tensor cos({seq_len, HEAD_DIM / 2});
    Tensor sin({seq_len, HEAD_DIM / 2});
    tensor_ops::compute_rope_embeddings(cos, sin, seq_len, HEAD_DIM, ROPE_THETA);
    
    // Forward through layers
    for (auto& layer : layers_) {
        Tensor layer_out; // Temporary output
        layer->forward(x, cos, sin, nullptr, layer_out); // Pass nullptr for mask for now (causal mask handled in attn)
        // Update x (residual connection is inside layer? No, DecoderLayer::forward does residual)
        // Wait, DecoderLayer::forward signature:
        // void forward(const Tensor& x, ..., Tensor& output);
        // It usually adds residual.
        // Let's check DecoderLayer::forward in src/model.cu (it's below).
        // It calls attn and moe/mlp.
        // We need to make sure x is updated.
        // Actually, usually we do x = layer(x).
        // If DecoderLayer writes to `output`, we should swap or copy.
        // Let's look at DecoderLayer::forward.
        // It takes `x` and writes to `output`.
        // So we should do:
        // layer->forward(x, ..., temp);
        // x = temp; (move or copy)
        Tensor temp_x = x; // Create a temporary copy of x for the layer input
        x = std::move(layer_out); // Move layer_out to x for the next iteration
    }
    
    // Final norm
    Tensor x_normed({batch, seq_len, HIDDEN_SIZE});
    norm_->forward(x, x_normed);
    
    // LM Head
    // x_normed: (batch, seq_len, hidden)
    // lm_head: (vocab, hidden)
    // logits: (batch, seq_len, vocab)
    // Flatten x_normed
    Tensor x_flat = x_normed.view({batch * seq_len, HIDDEN_SIZE});
    Tensor logits_flat({batch * seq_len, VOCAB_SIZE});
    
    tensor_ops::matmul_transposed(x_flat, lm_head_, logits_flat);
    
    // Reshape logits
    logits_flat.reshape({batch, seq_len, VOCAB_SIZE});
    
    if (logits.size() == 0) {
        logits = Tensor({batch, seq_len, VOCAB_SIZE});
    }
    cudaMemcpy(logits.data(), logits_flat.data(), logits.size() * sizeof(float), cudaMemcpyDeviceToDevice);
}