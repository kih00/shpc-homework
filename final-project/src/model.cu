#include "model.h"
#include "model_loader.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <cstring>
#include <utility>
#include <omp.h>

// Global model loader (definition)
std::unique_ptr<ModelLoader> g_model_loader;

// ============================================================================
// CUDA Kernels for SparseMoeBlock
// ============================================================================

// Optimized Top-K routing kernel using warp parallelism
// Each warp handles one token, threads cooperate to find top-k experts
__global__ void route_tokens_kernel(
    const float* router_logits, const float* expert_bias,
    int* topk_indices, float* topk_weights,
    size_t num_tokens, size_t num_experts, size_t num_experts_per_tok,
    bool norm_topk_prob, float routed_scaling_factor) {

    size_t t = blockIdx.x; // token index
    if (t >= num_tokens) return;

    const int tid = threadIdx.x;

    // Shared memory for scores and selection
    __shared__ float scores[32];  // num_experts = 32
    __shared__ float sigmoid_vals[32];
    __shared__ int sel_idx[8];  // max top-k
    __shared__ float sel_w[8];

    // Step 1: Compute sigmoid scores in parallel (32 threads for 32 experts)
    if (tid < num_experts) {
        float logit = router_logits[t * num_experts + tid];
        float sig = 1.0f / (1.0f + expf(-logit));
        sigmoid_vals[tid] = sig;
        scores[tid] = (expert_bias != nullptr) ? (sig + expert_bias[tid]) : sig;
    }
    __syncthreads();

    // Step 2: Greedy top-k selection (thread 0 does sequential selection with help)
    if (tid == 0) {
        for (size_t k = 0; k < num_experts_per_tok; k++) {
            float best_score = -1e30f;
            int best_idx = 0;

            for (size_t j = 0; j < num_experts; j++) {
                // Skip already selected
                bool used = false;
                for (size_t kk = 0; kk < k; kk++) {
                    if (sel_idx[kk] == (int)j) { used = true; break; }
                }
                if (used) continue;

                if (scores[j] > best_score) {
                    best_score = scores[j];
                    best_idx = j;
                }
            }
            sel_idx[k] = best_idx;
            sel_w[k] = sigmoid_vals[best_idx];
        }

        // Normalize if needed
        if (norm_topk_prob) {
            float sum = 0.0f;
            for (size_t k = 0; k < num_experts_per_tok; k++) sum += sel_w[k];
            float inv = (sum > 1e-6f) ? (1.0f / sum) : 0.0f;
            for (size_t k = 0; k < num_experts_per_tok; k++) sel_w[k] *= inv;
        }

        // Store results
        for (size_t k = 0; k < num_experts_per_tok; k++) {
            topk_indices[t * num_experts_per_tok + k] = sel_idx[k];
            topk_weights[t * num_experts_per_tok + k] = sel_w[k] * routed_scaling_factor;
        }
    }
}

// Kernel to gather tokens for a specific expert
// Gathers tokens assigned to expert_id from x_flat into expert_input
__global__ void gather_expert_input_kernel(
    const float* x_flat,           // (num_tokens, hidden_size)
    const int* topk_indices,       // (num_tokens, num_experts_per_tok)
    float* expert_input,           // (max_tokens_per_expert, hidden_size)
    int* token_ids_for_expert,     // which original token each gathered row came from
    int* num_tokens_for_expert,    // output: actual count
    size_t num_tokens, size_t hidden_size,
    size_t num_experts_per_tok, int expert_id) {

    // First pass: count tokens (done by thread 0)
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int count = 0;
        for (size_t t = 0; t < num_tokens; t++) {
            for (size_t k = 0; k < num_experts_per_tok; k++) {
                if (topk_indices[t * num_experts_per_tok + k] == expert_id) {
                    token_ids_for_expert[count] = t;
                    count++;
                    break;  // Each token counted once per expert
                }
            }
        }
        *num_tokens_for_expert = count;
    }
    __syncthreads();

    // Second pass: copy data
    int count = *num_tokens_for_expert;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = count * hidden_size;

    if (idx < total) {
        size_t row = idx / hidden_size;
        size_t col = idx % hidden_size;
        int src_token = token_ids_for_expert[row];
        expert_input[row * hidden_size + col] = x_flat[src_token * hidden_size + col];
    }
}

// Embedding lookup kernel: gather rows from embedding table
// input_ids: (num_tokens,) - token IDs
// embed_table: (vocab_size, hidden_size) - embedding table
// output: (num_tokens, hidden_size) - output embeddings
__global__ void embedding_lookup_kernel(
    const int* input_ids,
    const float* embed_table,
    float* output,
    size_t num_tokens, size_t hidden_size) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = num_tokens * hidden_size;

    if (idx < total) {
        size_t token_idx = idx / hidden_size;
        size_t dim = idx % hidden_size;
        int token_id = input_ids[token_idx];
        output[idx] = embed_table[token_id * hidden_size + dim];
    }
}

// Extract last token from each batch
// input: (batch, seq_len, hidden_size)
// output: (batch, hidden_size)
__global__ void extract_last_token_kernel(
    const float* input,
    float* output,
    size_t batch, size_t seq_len, size_t hidden_size) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch * hidden_size;

    if (idx < total) {
        size_t b = idx / hidden_size;
        size_t h = idx % hidden_size;
        // Copy last token (seq_len - 1) from each batch
        output[b * hidden_size + h] = input[(b * seq_len + (seq_len - 1)) * hidden_size + h];
    }
}

// Kernel to scatter and accumulate expert outputs with weights
// Simple version: adds w * src[idx*hidden:] to dst[token_id*hidden:]
__global__ void weighted_scatter_add_kernel(
    float* dst,                    // (num_tokens, hidden_size)
    const float* src,              // (num_expert_tokens, hidden_size)
    const int* token_ids,          // (num_expert_tokens,) - which token each row maps to
    const float* weights,          // (num_expert_tokens,) - weight for each row
    size_t num_expert_tokens, size_t hidden_size) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = num_expert_tokens * hidden_size;

    if (idx < total) {
        size_t row = idx / hidden_size;
        size_t col = idx % hidden_size;
        int token_id = token_ids[row];
        float w = weights[row];

        atomicAdd(&dst[token_id * hidden_size + col], w * src[row * hidden_size + col]);
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
    y = std::move(y_flat);
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

    // Assign experts to 4 GPUs in a round-robin fashion (expert parallelism)
    expert_devices_.resize(NUM_EXPERTS);
    for (size_t i = 0; i < NUM_EXPERTS; i++) {
        expert_devices_[i] = static_cast<int>(i % EXPERT_PARALLEL_GPUS);
    }
    
    // Load expert bias if used
    if (USE_EXPERT_BIAS) {
        std::stringstream ss_bias;
        ss_bias << "layers." << layer_idx << ".feed_forward.expert_bias";
        expert_bias_ = Tensor::load_from_file(ss_bias.str());
    }
}

void SparseMoeBlock::route_tokens(const Tensor& router_logits,
                                   std::vector<int>& top_k_indices,
                                   std::vector<float>& top_k_weights) {
    // router_logits: (batch * seq_len, num_experts) - should already be on GPU
    size_t num_tokens = router_logits.size(0);

    // Resize output vectors
    top_k_indices.resize(num_tokens * NUM_EXPERTS_PER_TOK);
    top_k_weights.resize(num_tokens * NUM_EXPERTS_PER_TOK);

    // Ensure router_logits is on device
    router_logits.to_device(-1);

    // Allocate device buffers for kernel outputs
    int* d_topk_indices;
    float* d_topk_weights;
    CHECK_CUDA(cudaMalloc(&d_topk_indices, num_tokens * NUM_EXPERTS_PER_TOK * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_topk_weights, num_tokens * NUM_EXPERTS_PER_TOK * sizeof(float)));

    // Get expert_bias device pointer (nullptr if not used)
    const float* d_expert_bias = nullptr;
    if (USE_EXPERT_BIAS && expert_bias_.size() > 0) {
        expert_bias_.to_device(-1);
        d_expert_bias = expert_bias_.device_data();
    }

    // Launch routing kernel - 32 threads per block for parallel sigmoid
    dim3 block(32);
    dim3 grid(num_tokens);

    route_tokens_kernel<<<grid, block>>>(
        router_logits.device_data(), d_expert_bias,
        d_topk_indices, d_topk_weights,
        num_tokens, NUM_EXPERTS, NUM_EXPERTS_PER_TOK,
        NORM_TOPK_PROB, ROUTED_SCALING_FACTOR);

    // Copy results back to host (small data: num_tokens * 4 * sizeof(int/float))
    CHECK_CUDA(cudaMemcpy(top_k_indices.data(), d_topk_indices,
                          num_tokens * NUM_EXPERTS_PER_TOK * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(top_k_weights.data(), d_topk_weights,
                          num_tokens * NUM_EXPERTS_PER_TOK * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Free temporary device buffers
    CHECK_CUDA(cudaFree(d_topk_indices));
    CHECK_CUDA(cudaFree(d_topk_weights));
}

void SparseMoeBlock::forward(const Tensor& x, Tensor& y, Tensor& router_logits) {
    // x: (batch, seq_len, hidden_size)
    size_t batch = x.size(0);
    size_t seq_len = x.size(1);
    size_t hidden_size = x.size(2);
    size_t num_tokens = batch * seq_len;
    const int base_device = 0;

    // Flatten - keep on GPU
    Tensor x_flat = x.view({num_tokens, hidden_size});
    x_flat.to_device(base_device);

    // Compute router logits on base device (GPU)
    CHECK_CUDA(cudaSetDevice(base_device));
    router_logits = Tensor({num_tokens, NUM_EXPERTS});
    tensor_ops::matmul_transposed(x_flat, gate_, router_logits);

    // Route tokens - routing indices/weights copied to host (small data)
    std::vector<int> top_k_indices;
    std::vector<float> top_k_weights;
    route_tokens(router_logits, top_k_indices, top_k_weights);

    // Bucket tokens per expert (host-side, but operates on small routing data)
    std::vector<std::vector<size_t>> tokens_per_expert(NUM_EXPERTS);
    std::vector<std::vector<float>> weights_per_expert(NUM_EXPERTS);
    for (size_t t = 0; t < num_tokens; t++) {
        for (size_t k = 0; k < NUM_EXPERTS_PER_TOK; k++) {
            int expert_idx = top_k_indices[t * NUM_EXPERTS_PER_TOK + k];
            float weight = top_k_weights[t * NUM_EXPERTS_PER_TOK + k];
            tokens_per_expert[expert_idx].push_back(t);
            weights_per_expert[expert_idx].push_back(weight);
        }
    }

    // Group experts by device
    std::vector<std::vector<size_t>> experts_on_device(EXPERT_PARALLEL_GPUS);
    for (size_t e = 0; e < NUM_EXPERTS; e++) {
        int dev = expert_devices_[e];
        if (tokens_per_expert[e].size() > 0) {
            experts_on_device[dev].push_back(e);
        }
    }

    // Copy x_flat to all GPUs that have active experts
    std::vector<Tensor> x_flat_per_gpu(EXPERT_PARALLEL_GPUS);
    for (int dev = 0; dev < EXPERT_PARALLEL_GPUS; dev++) {
        if (!experts_on_device[dev].empty()) {
            x_flat_per_gpu[dev] = x_flat.copy();
            x_flat_per_gpu[dev].copy_to_device(dev);
        }
    }

    // Storage for expert outputs and token indices per GPU
    std::vector<std::vector<Tensor>> expert_outputs(EXPERT_PARALLEL_GPUS);
    std::vector<std::vector<std::vector<size_t>>> expert_token_lists(EXPERT_PARALLEL_GPUS);
    std::vector<std::vector<std::vector<float>>> expert_weight_lists(EXPERT_PARALLEL_GPUS);

    for (int dev = 0; dev < EXPERT_PARALLEL_GPUS; dev++) {
        expert_outputs[dev].resize(experts_on_device[dev].size());
        expert_token_lists[dev].resize(experts_on_device[dev].size());
        expert_weight_lists[dev].resize(experts_on_device[dev].size());
    }

    // Parallel execution across 4 GPUs using OpenMP
    #pragma omp parallel num_threads(EXPERT_PARALLEL_GPUS)
    {
        int dev = omp_get_thread_num();
        CHECK_CUDA(cudaSetDevice(dev));

        for (size_t i = 0; i < experts_on_device[dev].size(); i++) {
            size_t expert_idx = experts_on_device[dev][i];
            const auto& token_list = tokens_per_expert[expert_idx];
            const auto& weight_list = weights_per_expert[expert_idx];
            size_t tok_count = token_list.size();

            if (tok_count == 0) continue;

            // Store token list and weights for later scatter
            expert_token_lists[dev][i] = token_list;
            expert_weight_lists[dev][i] = weight_list;

            // Create expert input on GPU - gather tokens
            Tensor expert_in({tok_count, 1, hidden_size});
            expert_in.to_device(dev);

            // GPU gather: copy selected rows from x_flat_per_gpu[dev] to expert_in
            const float* src = x_flat_per_gpu[dev].device_data();
            float* dst = expert_in.device_data();
            for (size_t idx = 0; idx < tok_count; idx++) {
                size_t t = token_list[idx];
                CHECK_CUDA(cudaMemcpyAsync(dst + idx * hidden_size,
                                           src + t * hidden_size,
                                           hidden_size * sizeof(float),
                                           cudaMemcpyDeviceToDevice));
            }
            CHECK_CUDA(cudaDeviceSynchronize());
            expert_in.mark_device_dirty();

            // Execute expert on its owning GPU (stays on GPU)
            Tensor expert_out;
            experts_[expert_idx].forward(expert_in, expert_out);

            // Keep expert_out on GPU, just store it
            expert_outputs[dev][i] = std::move(expert_out);
        }
    }

    // Initialize output tensor on base device
    y = Tensor({batch, seq_len, hidden_size});
    y.to_device(base_device);
    CHECK_CUDA(cudaSetDevice(base_device));
    CHECK_CUDA(cudaMemset(y.device_data(), 0, num_tokens * hidden_size * sizeof(float)));
    y.mark_device_dirty();

    // Accumulate results using GPU scatter_add kernel
    for (int dev = 0; dev < EXPERT_PARALLEL_GPUS; dev++) {
        for (size_t i = 0; i < experts_on_device[dev].size(); i++) {
            const auto& token_list = expert_token_lists[dev][i];
            const auto& weight_list = expert_weight_lists[dev][i];
            Tensor& expert_out = expert_outputs[dev][i];

            if (expert_out.size() == 0) continue;

            size_t tok_count = token_list.size();

            // Copy expert_out to base device if on different GPU
            if (dev != base_device) {
                expert_out.copy_to_device(base_device);
            }
            CHECK_CUDA(cudaSetDevice(base_device));

            // Prepare token_ids and weights arrays on GPU
            int* d_token_ids;
            float* d_weights;
            CHECK_CUDA(cudaMalloc(&d_token_ids, tok_count * sizeof(int)));
            CHECK_CUDA(cudaMalloc(&d_weights, tok_count * sizeof(float)));

            // Copy token_ids and weights to device
            std::vector<int> token_ids_vec(tok_count);
            for (size_t idx = 0; idx < tok_count; idx++) {
                token_ids_vec[idx] = static_cast<int>(token_list[idx]);
            }
            CHECK_CUDA(cudaMemcpy(d_token_ids, token_ids_vec.data(),
                                  tok_count * sizeof(int), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_weights, weight_list.data(),
                                  tok_count * sizeof(float), cudaMemcpyHostToDevice));

            // Launch scatter-add kernel
            size_t total = tok_count * hidden_size;
            dim3 block(256);
            dim3 grid((total + 255) / 256);

            weighted_scatter_add_kernel<<<grid, block>>>(
                y.device_data(), expert_out.device_data(),
                d_token_ids, d_weights,
                tok_count, hidden_size);

            CHECK_CUDA(cudaFree(d_token_ids));
            CHECK_CUDA(cudaFree(d_weights));
        }
    }

    // Synchronize and keep on base device
    CHECK_CUDA(cudaSetDevice(base_device));
    CHECK_CUDA(cudaDeviceSynchronize());
    y.reshape({batch, seq_len, hidden_size});
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

    // Project Q, K, V (GPU)
    Tensor q_proj_out({batch * seq_len, NUM_ATTENTION_HEADS * HEAD_DIM});
    Tensor k_proj_out({batch * seq_len, NUM_KEY_VALUE_HEADS * HEAD_DIM});
    Tensor v_proj_out({batch * seq_len, NUM_KEY_VALUE_HEADS * HEAD_DIM});

    tensor_ops::matmul_transposed(x_flat, q_proj_, q_proj_out);
    tensor_ops::matmul_transposed(x_flat, k_proj_, k_proj_out);
    tensor_ops::matmul_transposed(x_flat, v_proj_, v_proj_out);

    // Reshape Q to (batch, num_heads, seq, head_dim) for layernorm then attention
    // Note: layernorm expects (batch, seq, heads, head_dim), so we do intermediate reshape
    Tensor q_for_norm({batch, seq_len, NUM_ATTENTION_HEADS, HEAD_DIM});
    Tensor k_for_norm({batch, seq_len, NUM_KEY_VALUE_HEADS, HEAD_DIM});
    Tensor v_reshaped({batch, seq_len, NUM_KEY_VALUE_HEADS, HEAD_DIM});

    // GPU reshape: (batch*seq, heads*dim) -> (batch, seq, heads, dim) using CUDA kernel
    tensor_ops::reshape_for_layernorm(q_proj_out, q_for_norm, batch, seq_len, NUM_ATTENTION_HEADS, HEAD_DIM);
    tensor_ops::reshape_for_layernorm(k_proj_out, k_for_norm, batch, seq_len, NUM_KEY_VALUE_HEADS, HEAD_DIM);
    tensor_ops::reshape_for_layernorm(v_proj_out, v_reshaped, batch, seq_len, NUM_KEY_VALUE_HEADS, HEAD_DIM);

    // Apply layernorm to Q and K (normalizes over last dim = head_dim)
    Tensor q_normed({batch, seq_len, NUM_ATTENTION_HEADS, HEAD_DIM});
    Tensor k_normed({batch, seq_len, NUM_KEY_VALUE_HEADS, HEAD_DIM});
    q_layernorm_->forward(q_for_norm, q_normed);
    k_layernorm_->forward(k_for_norm, k_normed);

    // Transpose to (batch, num_heads, seq_len, head_dim) for attention (GPU)
    Tensor q({batch, NUM_ATTENTION_HEADS, seq_len, HEAD_DIM});
    Tensor k({batch, NUM_KEY_VALUE_HEADS, seq_len, HEAD_DIM});
    Tensor v({batch, NUM_KEY_VALUE_HEADS, seq_len, HEAD_DIM});

    // GPU transpose: (batch, seq, heads, dim) -> (batch, heads, seq, dim)
    tensor_ops::reshape_to_heads(q_normed.view({batch * seq_len, NUM_ATTENTION_HEADS * HEAD_DIM}),
                                  q, batch, seq_len, NUM_ATTENTION_HEADS, HEAD_DIM);
    tensor_ops::reshape_to_heads(k_normed.view({batch * seq_len, NUM_KEY_VALUE_HEADS * HEAD_DIM}),
                                  k, batch, seq_len, NUM_KEY_VALUE_HEADS, HEAD_DIM);
    tensor_ops::reshape_to_heads(v_reshaped.view({batch * seq_len, NUM_KEY_VALUE_HEADS * HEAD_DIM}),
                                  v, batch, seq_len, NUM_KEY_VALUE_HEADS, HEAD_DIM);

    // Apply RoPE (GPU)
    tensor_ops::apply_rotary_pos_emb(q, k, cos, sin);

    // Repeat K, V for GQA (GPU)
    Tensor k_repeated({batch, NUM_ATTENTION_HEADS, seq_len, HEAD_DIM});
    Tensor v_repeated({batch, NUM_ATTENTION_HEADS, seq_len, HEAD_DIM});
    tensor_ops::repeat_kv(k, NUM_KEY_VALUE_GROUPS, k_repeated);
    tensor_ops::repeat_kv(v, NUM_KEY_VALUE_GROUPS, v_repeated);

    // Compute attention (GPU) - Q @ K^T, causal mask, softmax, @ V
    float scale = 1.0f / std::sqrt((float)HEAD_DIM);
    Tensor attn_output({batch, NUM_ATTENTION_HEADS, seq_len, HEAD_DIM});
    tensor_ops::batched_attention(q, k_repeated, v_repeated, attn_output, scale);

    // Reshape output: (batch, heads, seq, dim) -> (batch*seq, heads*dim) (GPU)
    Tensor attn_flat({batch * seq_len, hidden_size});
    tensor_ops::reshape_from_heads(attn_output, attn_flat, batch, seq_len, NUM_ATTENTION_HEADS, HEAD_DIM);

    // Output projection (GPU)
    Tensor output_flat({batch * seq_len, hidden_size});
    tensor_ops::matmul_transposed(attn_flat, o_proj_, output_flat);

    output_flat.reshape({batch, seq_len, hidden_size});
    output = std::move(output_flat);
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
    // Python: BCx = self.in_proj(x).transpose(-1, -2)
    // Result: (batch, 3*hidden_size, seq_len) for Conv1d
    
    size_t batch = x.size(0);
    size_t seq_len = x.size(1);
    size_t hidden_size = x.size(2);
    
    // Flatten for matmul
    Tensor x_flat = x.view({batch * seq_len, hidden_size});
    
    // in_proj: (batch*seq_len, hidden_size) @ (3*hidden_size, hidden_size)^T -> (batch*seq_len, 3*hidden_size)
    Tensor in_proj_out({batch * seq_len, 3 * hidden_size});
    tensor_ops::matmul_transposed(x_flat, in_proj_weight_, in_proj_out);
    
    // Add bias if present
    if (USE_CONV_BIAS && in_proj_bias_.size() > 0) {
        for (size_t i = 0; i < batch * seq_len; i++) {
            for (size_t j = 0; j < 3 * hidden_size; j++) {
                in_proj_out.at(i, j) += in_proj_bias_[j];
            }
        }
    }
    
    // Fused transpose and split: (batch*seq, 3*hidden) -> B, C, x_gate each (batch, hidden, seq)
    // Using CUDA kernel for efficiency
    Tensor B({batch, hidden_size, seq_len});
    Tensor C({batch, hidden_size, seq_len});
    Tensor x_gate({batch, hidden_size, seq_len});
    tensor_ops::transpose_split_BCx(in_proj_out, B, C, x_gate, batch, seq_len, hidden_size);
    
    // Bx = B * x_gate (element-wise)
    Tensor Bx({batch, hidden_size, seq_len});
    tensor_ops::mul(B, x_gate, Bx);
    
    // Apply causal conv1d on Bx (expects: batch, channels, seq_len)
    Tensor conv_out({batch, hidden_size, seq_len});
    tensor_ops::causal_conv1d(Bx, conv_weight_, USE_CONV_BIAS ? &conv_bias_ : nullptr, conv_out);
    
    // y_pre = C * conv_out (element-wise)
    Tensor y_pre({batch, hidden_size, seq_len});
    tensor_ops::mul(C, conv_out, y_pre);
    
    // Transpose back: (batch, hidden_size, seq_len) -> (batch, seq_len, hidden_size)
    // Using CUDA kernel for efficiency
    Tensor y_pre_transposed({batch, seq_len, hidden_size});
    tensor_ops::transpose_hidden_seq(y_pre, y_pre_transposed, batch, hidden_size, seq_len);
    
    // out_proj: (batch*seq_len, hidden_size) @ (hidden_size, hidden_size)^T -> (batch*seq_len, hidden_size)
    Tensor y_pre_flat = y_pre_transposed.view({batch * seq_len, hidden_size});
    Tensor y_flat({batch * seq_len, hidden_size});
    tensor_ops::matmul_transposed(y_pre_flat, out_proj_weight_, y_flat);
    
    // Add bias if present
    if (USE_CONV_BIAS && out_proj_bias_.size() > 0) {
        for (size_t i = 0; i < batch * seq_len; i++) {
            for (size_t j = 0; j < hidden_size; j++) {
                y_flat.at(i, j) += out_proj_bias_[j];
            }
        }
    }
    
    // Reshape back to (batch, seq_len, hidden_size)
    y_flat.reshape({batch, seq_len, hidden_size});
    y = std::move(y_flat);
}

// DecoderLayer implementation
DecoderLayer::DecoderLayer(int layer_idx, bool is_attention_layer, int device_id)
    : layer_idx_(layer_idx), is_attention_layer_(is_attention_layer), device_id_(device_id) {
    
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
double get_time_layer() {
  struct timespec tv;
  clock_gettime(CLOCK_MONOTONIC, &tv);

  return tv.tv_sec + tv.tv_nsec * 1e-9;
}

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
        // Keep non-MoE work on a single base GPU (0); expert parallelism will spread work across GPUs inside MoE blocks.
        int device_id = 0;
        std::cout << "  Layer " << i << ": " << (is_attention ? "Attention" : "Conv") << std::endl;
        layers_.push_back(std::make_unique<DecoderLayer>(i, is_attention, device_id));
    }
}

void LFM2Model::load_output_layers() {
    std::cout << "Loading output layers..." << std::endl;
    
    norm_ = std::make_unique<RMSNorm>("embedding_norm.weight");
    
    // LM head might share weights with embeddings
    if (g_model_loader->has_tensor("lm_head.weight")) {
        lm_head_ = Tensor::load_from_file("lm_head.weight");
    } else {
        // Use tied weights (same as embeddings)
        lm_head_ = embed_tokens_;
        std::cout << "  Using tied weights for LM head" << std::endl;
    }
}

void LFM2Model::forward(const std::vector<int>& input_ids, size_t batch, size_t seq_len, Tensor& logits) {
    if (batch == 0 || seq_len == 0 || input_ids.size() != batch * seq_len) {
        throw std::runtime_error("Invalid batch/seq_len for forward");
    }

    // debug: print shape of input
    std::cout << "\nForward pass: batch=" << batch << ", seq_len=" << seq_len << std::endl;

    size_t num_tokens = batch * seq_len;

    // GPU Embedding lookup
    CHECK_CUDA(cudaSetDevice(0));

    // Copy input_ids to device
    int* d_input_ids;
    CHECK_CUDA(cudaMalloc(&d_input_ids, num_tokens * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_input_ids, input_ids.data(), num_tokens * sizeof(int), cudaMemcpyHostToDevice));

    // Ensure embedding table is on device
    embed_tokens_.to_device(0);

    // Allocate output tensor on device
    Tensor hidden_states({batch, seq_len, HIDDEN_SIZE});
    hidden_states.to_device(0);

    // Launch embedding lookup kernel
    size_t total = num_tokens * HIDDEN_SIZE;
    dim3 block(256);
    dim3 grid((total + 255) / 256);

    embedding_lookup_kernel<<<grid, block>>>(
        d_input_ids, embed_tokens_.device_data(),
        hidden_states.device_data(),
        num_tokens, HIDDEN_SIZE);

    CHECK_CUDA(cudaFree(d_input_ids));
    hidden_states.mark_device_dirty();
    
    // Compute RoPE embeddings
    Tensor cos({seq_len, HEAD_DIM});
    Tensor sin({seq_len, HEAD_DIM});
    cos.copy_to_device(0);
    sin.copy_to_device(0);

    rotary_emb_->forward(seq_len, cos, sin);
    
    // Create causal attention mask (not strictly needed for CPU impl)
    Tensor* attention_mask = nullptr;
    
    // Keep activations on base GPU (0) and let MoE dispatch handle expert parallelism internally.
    CHECK_CUDA(cudaSetDevice(0));
    hidden_states.copy_to_device(0);
    
    for (size_t i = 0; i < NUM_HIDDEN_LAYERS; i++) {
        double start_time = get_time_layer();

        Tensor output({batch, seq_len, HIDDEN_SIZE});
        layers_[i]->forward(hidden_states, cos, sin, attention_mask, output);
        hidden_states = std::move(output);

        // debug: print progress
        double end_time = get_time_layer();
        std::cout << "  Layer " << i << " forward time: " << (end_time - start_time) << " seconds" << std::endl;
    }
    
    // Final operations on GPU 0 (keep on same device)
    CHECK_CUDA(cudaSetDevice(0));

    // Final norm
    Tensor normed_output({batch, seq_len, HIDDEN_SIZE});
    norm_->forward(hidden_states, normed_output);

    // Extract last token using GPU kernel
    Tensor last_hidden({batch, HIDDEN_SIZE});
    last_hidden.to_device(0);

    size_t total_last = batch * HIDDEN_SIZE;
    dim3 block_last(256);
    dim3 grid_last((total_last + 255) / 256);

    extract_last_token_kernel<<<grid_last, block_last>>>(
        normed_output.device_data(), last_hidden.device_data(),
        batch, seq_len, HIDDEN_SIZE);
    last_hidden.mark_device_dirty();

    // LM head projection on GPU
    logits = Tensor({batch, VOCAB_SIZE});
    tensor_ops::matmul_transposed(last_hidden, lm_head_, logits);

    // Reset to device 0
    CHECK_CUDA(cudaSetDevice(0));
}