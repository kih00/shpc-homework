#include "model.h"
#include "model_loader.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

// Global model loader (definition)
std::unique_ptr<ModelLoader> g_model_loader;

// ============================================================================
// CUDA Kernels
// ============================================================================
// Kernels are imported from layer.h
// embedding_kernel is shared but declared here if not in layer.h
__global__ void embedding_kernel(int *input_ids, float *embedding_table, float *output, int batch, int seq_len, int hidden_size);

// ============================================================================
// CUDA Kernels
// ============================================================================
// Kernels are now imported from layer.h


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

// Additional MoE Kernels removed (using shared kernels)


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
    d_count_tensor_ = Tensor({1});
    d_count_tensor_.zero();
}

void SparseMoeBlock::route_tokens(const Tensor& router_logits, 
                                   Tensor& top_k_indices,
                                   Tensor& top_k_weights) {
    // router_logits: (batch * seq_len, num_experts)
    size_t num_tokens = router_logits.size(0);
    
    // Allocate outputs on GPU
    top_k_indices = Tensor({num_tokens, NUM_EXPERTS_PER_TOK});
    top_k_weights = Tensor({num_tokens, NUM_EXPERTS_PER_TOK});
    
    int threads = 256;
    int blocks = (num_tokens + threads - 1) / threads;
    
    // We need to cast float* to int* for indices
    // We need to cast float* to int* for indices
    router_kernel<<<blocks, threads>>>(
        (float*)router_logits.data(), 
        USE_EXPERT_BIAS ? expert_bias_.data() : nullptr,
        (int*)top_k_indices.data(), 
        top_k_weights.data(), 
        num_tokens, NUM_EXPERTS, NUM_EXPERTS_PER_TOK, USE_EXPERT_BIAS
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
    router_logits_ = Tensor({num_tokens, NUM_EXPERTS});
    tensor_ops::matmul_transposed(x_flat, gate_, router_logits_);
    
    // Route tokens
    route_tokens(router_logits_, top_k_indices_, top_k_weights_);
    
    // Initialize output
    y = Tensor({batch, seq_len, hidden_size});
    y.zero();
    
    // Host-side scheduling
    // 1. Copy indices and weights to host
    std::vector<int> host_indices(num_tokens * NUM_EXPERTS_PER_TOK);
    std::vector<float> host_weights(num_tokens * NUM_EXPERTS_PER_TOK);
    
    CHECK_CUDA(cudaMemcpy(host_indices.data(), top_k_indices_.data(), num_tokens * NUM_EXPERTS_PER_TOK * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(host_weights.data(), top_k_weights_.data(), num_tokens * NUM_EXPERTS_PER_TOK * sizeof(float), cudaMemcpyDeviceToHost));
    
    // 2. Sort tokens by expert
    std::vector<std::vector<int>> expert_token_indices(NUM_EXPERTS);
    std::vector<std::vector<float>> expert_token_weights(NUM_EXPERTS);
    
    for (size_t t = 0; t < num_tokens; ++t) {
        for (int k = 0; k < NUM_EXPERTS_PER_TOK; ++k) {
            int expert_idx = host_indices[t * NUM_EXPERTS_PER_TOK + k];
            float weight = host_weights[t * NUM_EXPERTS_PER_TOK + k];
            
            if (expert_idx >= 0 && expert_idx < NUM_EXPERTS) {
                expert_token_indices[expert_idx].push_back(t);
                expert_token_weights[expert_idx].push_back(weight);
            }
        }
    }
    
    // Temporary buffers for gather/scatter
    expert_input_ = Tensor({num_tokens, hidden_size});
    expert_output_ = Tensor({num_tokens, hidden_size});
    indices_map_ = Tensor({num_tokens}); // Reused for indices
    // We reuse top_k_weights_ for weights buffer as it is large enough (num_tokens * k >= num_tokens)
    
    for (size_t e = 0; e < NUM_EXPERTS; e++) {
        int count = expert_token_indices[e].size();
        if (count == 0) continue;
        
        // Copy indices and weights to device
        CHECK_CUDA(cudaMemcpy(indices_map_.data(), expert_token_indices[e].data(), count * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(top_k_weights_.data(), expert_token_weights[e].data(), count * sizeof(float), cudaMemcpyHostToDevice));
        
        int threads = 256;
        int blocks = (count + threads - 1) / threads;
        
        // Gather
        // gather_kernel(x, expert_in, indices, count, hidden)
        CHECK_CUDA(gather_kernel<<<blocks, threads>>>(
            x_flat.data(), 
            expert_input_.data(), 
            (int*)indices_map_.data(), 
            count, hidden_size
        ));
        
        // Expert Forward
        Tensor curr_input = expert_input_.view({(size_t)count, hidden_size});
        Tensor curr_output = expert_output_.view({(size_t)count, hidden_size});
        experts_[e].forward(curr_input, curr_output);
        
        // Scatter Add
        // scatter_add_kernel(expert_out, output, indices, weights, count, hidden)
        int total_elements = count * hidden_size;
        int scatter_blocks = (total_elements + threads - 1) / threads;
        CHECK_CUDA(scatter_add_kernel<<<scatter_blocks, threads>>>(
            curr_output.data(), 
            y.data(), 
            (int*)indices_map_.data(), 
            top_k_weights_.data(), 
            count, hidden_size
        ));
    }
    
    // Copy router_logits to output argument if needed
    if (router_logits.size() == 0) {
        router_logits = router_logits_.copy();
    } else {
        // Caller allocated, copy data.
        CHECK_CUDA(cudaMemcpy(router_logits.data(), router_logits_.data(), router_logits_.size() * sizeof(float), cudaMemcpyDeviceToDevice));
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
    q_proj_out_ = Tensor({batch * seq_len, NUM_ATTENTION_HEADS * HEAD_DIM});
    k_proj_out_ = Tensor({batch * seq_len, NUM_KEY_VALUE_HEADS * HEAD_DIM});
    v_proj_out_ = Tensor({batch * seq_len, NUM_KEY_VALUE_HEADS * HEAD_DIM});
    
    tensor_ops::matmul_transposed(x_flat, q_proj_, q_proj_out_);
    tensor_ops::matmul_transposed(x_flat, k_proj_, k_proj_out_);
    tensor_ops::matmul_transposed(x_flat, v_proj_, v_proj_out_);
    
    // Reshape to (batch, seq_len, num_heads, head_dim) for layernorm
    // This is just a view if memory is contiguous (it is)
    Tensor q_reshaped = q_proj_out_.view({batch, seq_len, NUM_ATTENTION_HEADS, HEAD_DIM});
    Tensor k_reshaped = k_proj_out_.view({batch, seq_len, NUM_KEY_VALUE_HEADS, HEAD_DIM});
    Tensor v_reshaped = v_proj_out_.view({batch, seq_len, NUM_KEY_VALUE_HEADS, HEAD_DIM});
    
    // Apply layernorm to Q and K
    q_normed_ = Tensor({batch, seq_len, NUM_ATTENTION_HEADS, HEAD_DIM});
    k_normed_ = Tensor({batch, seq_len, NUM_KEY_VALUE_HEADS, HEAD_DIM});
    q_layernorm_->forward(q_reshaped, q_normed_);
    k_layernorm_->forward(k_reshaped, k_normed_);
    
    // Transpose to (batch, num_heads, seq_len, head_dim) for attention
    // Use kernel
    q_ = Tensor({batch, NUM_ATTENTION_HEADS, seq_len, HEAD_DIM});
    k_ = Tensor({batch, NUM_KEY_VALUE_HEADS, seq_len, HEAD_DIM});
    v_ = Tensor({batch, NUM_KEY_VALUE_HEADS, seq_len, HEAD_DIM});
    
    int threads = 256;
    // Transpose to (batch, num_heads, seq_len, head_dim) for attention
    // Use kernel
    q_ = Tensor({batch, NUM_ATTENTION_HEADS, seq_len, HEAD_DIM});
    k_ = Tensor({batch, NUM_KEY_VALUE_HEADS, seq_len, HEAD_DIM});
    v_ = Tensor({batch, NUM_KEY_VALUE_HEADS, seq_len, HEAD_DIM});
    
    int threads = 256;
    int blocks_q = (batch * NUM_ATTENTION_HEADS * seq_len * HEAD_DIM + threads - 1) / threads;
    int blocks_k = (batch * NUM_KEY_VALUE_HEADS * seq_len * HEAD_DIM + threads - 1) / threads;
    
    CHECK_CUDA(transpose_kernel<<<blocks_q, threads>>>(q_normed_.data(), q_.data(), batch, seq_len, NUM_ATTENTION_HEADS, HEAD_DIM));
    CHECK_CUDA(transpose_kernel<<<blocks_k, threads>>>(k_normed_.data(), k_.data(), batch, seq_len, NUM_KEY_VALUE_HEADS, HEAD_DIM));
    
    // For V, we need to transpose v_reshaped (B, S, H, D) -> (B, H, S, D)
    CHECK_CUDA(transpose_kernel<<<blocks_k, threads>>>(v_reshaped.data(), v_.data(), batch, seq_len, NUM_KEY_VALUE_HEADS, HEAD_DIM));
    
    // Apply RoPE
    tensor_ops::apply_rotary_pos_emb(q_, k_, cos, sin);
    
    // Repeat K, V for GQA
    k_repeated_ = Tensor({batch, NUM_ATTENTION_HEADS, seq_len, HEAD_DIM});
    v_repeated_ = Tensor({batch, NUM_ATTENTION_HEADS, seq_len, HEAD_DIM});
    tensor_ops::repeat_kv(k_, NUM_KEY_VALUE_GROUPS, k_repeated_);
    tensor_ops::repeat_kv(v_, NUM_KEY_VALUE_GROUPS, v_repeated_);
    
    // Compute attention: Q @ K^T
    float scale = 1.0f / std::sqrt((float)HEAD_DIM);
    scores_ = Tensor({batch, NUM_ATTENTION_HEADS, seq_len, seq_len});
    
    int blocks_scores = (batch * NUM_ATTENTION_HEADS * seq_len * seq_len + threads - 1) / threads;
    dim3 grid_scores((seq_len + 15)/16, (seq_len + 15)/16, batch * NUM_ATTENTION_HEADS);
    dim3 block_scores(16, 16);
    // Note: batched_matmul_qk_kernel signature in layer.h: (Q, K, Scores, S, D, scale)
    // It uses 3D grid (s_k, s_q, b_h)
    CHECK_CUDA(batched_matmul_qk_kernel<<<grid_scores, block_scores>>>(q_.data(), k_repeated_.data(), scores_.data(), seq_len, HEAD_DIM, scale));
    
    // Softmax (masked)
    // Note: masked_softmax_kernel signature in layer.h: (scores, S)
    // It uses 2D grid (s_q, b_h)
    dim3 grid_softmax((seq_len + 255)/256, batch * NUM_ATTENTION_HEADS);
    CHECK_CUDA(masked_softmax_kernel<<<grid_softmax, 256>>>(scores_.data(), seq_len));
    
    // Multiply by V: attn_weights @ V
    attn_output_ = Tensor({batch, NUM_ATTENTION_HEADS, seq_len, HEAD_DIM});
    // Note: batched_matmul_sv_kernel signature in layer.h: (Scores, V, Out, S, D)
    // It uses 3D grid (d, s_q, b_h)
    dim3 grid_out((HEAD_DIM + 15)/16, (seq_len + 15)/16, batch * NUM_ATTENTION_HEADS);
    CHECK_CUDA(batched_matmul_sv_kernel<<<grid_out, block_scores>>>(scores_.data(), v_repeated_.data(), attn_output_.data(), seq_len, HEAD_DIM));
    
    // Transpose back: (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, num_heads, head_dim)
    // And flatten to (batch * seq_len, hidden_size)
    // We can transpose directly to (batch, seq_len, hidden_size) if we treat H*D as contiguous
    attn_flat_ = Tensor({batch * seq_len, hidden_size});
    int blocks_tb = (batch * NUM_ATTENTION_HEADS * seq_len * HEAD_DIM + threads - 1) / threads;
    CHECK_CUDA(transpose_back_kernel<<<blocks_tb, threads>>>(attn_output_.data(), attn_flat_.data(), batch, NUM_ATTENTION_HEADS, seq_len, HEAD_DIM));
    
    output_flat_ = Tensor({batch * seq_len, hidden_size});
    tensor_ops::matmul_transposed(attn_flat_, o_proj_, output_flat_);
    
    output_flat_.reshape({batch, seq_len, hidden_size});
    
    // Allocate output if needed
    if (output.size() == 0) {
        output = Tensor({batch, seq_len, hidden_size});
    }
    // Copy result
    CHECK_CUDA(cudaMemcpy(output.data(), output_flat_.data(), output.size() * sizeof(float), cudaMemcpyDeviceToDevice));
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
    in_proj_out_ = Tensor({batch * seq_len, 3 * hidden_size});
    tensor_ops::matmul_transposed(x_flat, in_proj_weight_, in_proj_out_);
    
    // Add bias if present
    if (USE_CONV_BIAS && in_proj_bias_.size() > 0) {
        tensor_ops::add_bias(in_proj_out_, in_proj_bias_, in_proj_out_);
    }
    

    // Bx = B * x_gate (element-wise)
    Bx_ = Tensor({batch, hidden_size, seq_len});
    C_ = Tensor({batch, hidden_size, seq_len}); // This will hold C in [B, H, S] layout
    
    int total_elements = batch * seq_len * hidden_size;
    int gating_blocks = (total_elements + threads - 1) / threads;
    CHECK_CUDA(pre_conv_gating_kernel<<<gating_blocks, threads>>>(in_proj_out_.data(), Bx_.data(), C_.data(), batch, seq_len, hidden_size));
    
    // Apply causal conv1d on Bx
    conv_out_ = Tensor({batch, hidden_size, seq_len});
    tensor_ops::causal_conv1d(Bx_, conv_weight_, USE_CONV_BIAS ? &conv_bias_ : nullptr, conv_out_);
    
    // y_pre = C * conv_out (element-wise)
    // post_conv_gating_kernel takes (ConvOut, C, Y_pre).
    y_pre_ = Tensor({batch, seq_len, hidden_size}); // [B, S, H]
    CHECK_CUDA(post_conv_gating_kernel<<<gating_blocks, threads>>>(conv_out_.data(), C_.data(), y_pre_.data(), batch, seq_len, hidden_size));
    
    // out_proj
    // y_pre_ is already [B, S, H] (flat).
    
    y_flat_ = Tensor({batch * seq_len, hidden_size});
    tensor_ops::matmul_transposed(y_pre_.view({batch * seq_len, hidden_size}), out_proj_weight_, y_flat_);
    
    // Add bias if present
    if (USE_CONV_BIAS && out_proj_bias_.size() > 0) {
        tensor_ops::add_bias(y_flat_, out_proj_bias_, y_flat_);
    }
    
    // Reshape back
    y_flat_.reshape({batch, seq_len, hidden_size});
    
    if (y.size() == 0) {
        y = Tensor({batch, seq_len, hidden_size});
    }
    CHECK_CUDA(cudaMemcpy(y.data(), y_flat_.data(), y.size() * sizeof(float), cudaMemcpyDeviceToDevice));

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

void LFM2Model::load_output_layers() {
    // Load norm
    norm_ = std::make_unique<RMSNorm>("norm.weight");
    
    // Load lm_head
    lm_head_ = Tensor::load_from_file("lm_head.weight");
}

void LFM2Model::forward(const std::vector<int>& tokens, Tensor& logits) {
    size_t seq_len = tokens.size();
    size_t batch = 1; // Currently only support batch size 1 for inference
    
    // Allocate input ids on GPU
    int* d_tokens;
    CHECK_CUDA(cudaMalloc(&d_tokens, seq_len * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_tokens, tokens.data(), seq_len * sizeof(int), cudaMemcpyHostToDevice));
    
    // Embedding lookup
    Tensor x({batch, seq_len, HIDDEN_SIZE});
    int threads = 256;
    int blocks = (batch * seq_len * HIDDEN_SIZE + threads - 1) / threads;
    embedding_kernel<<<blocks, threads>>>(d_tokens, embed_tokens_.data(), x.data(), batch, seq_len, HIDDEN_SIZE);
    
    CHECK_CUDA(cudaFree(d_tokens));
    
    // Precompute RoPE cos/sin
    Tensor cos({seq_len, HEAD_DIM / 2});
    Tensor sin({seq_len, HEAD_DIM / 2});
    // Correct signature: (head_dim, max_seq_len, theta, cos, sin)
    tensor_ops::compute_rope_embeddings(HEAD_DIM, seq_len, ROPE_THETA, cos, sin);
    
    // Forward through layers
    for (auto& layer : layers_) {
        Tensor layer_out; // Temporary output
        layer->forward(x, cos, sin, nullptr, layer_out); // Pass nullptr for mask for now (causal mask handled in attn)
        
        // Update x (DecoderLayer adds residual, so layer_out is the new hidden state)
        x = std::move(layer_out);
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
    CHECK_CUDA(cudaMemcpy(logits.data(), logits_flat.data(), logits.size() * sizeof(float), cudaMemcpyDeviceToDevice));
}