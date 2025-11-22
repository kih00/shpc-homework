#include "model.h"
#include "model_loader.h"
#include "layer.h"
#include "tensor.h"
#include <cmath>
#include <iostream>
#include <sstream>
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

// Global model loader (definition)
std::unique_ptr<ModelLoader> g_model_loader;

// ============================================================================
// MLP
// ============================================================================

MLP::MLP(const std::string& w1_file, const std::string& w2_file, const std::string& w3_file) {
    w1_ = Tensor::load_from_file(w1_file, nullptr);  // Uses global g_model_loader
    w2_ = Tensor::load_from_file(w2_file, nullptr);
    w3_ = Tensor::load_from_file(w3_file, nullptr);
}

void MLP::forward(const Tensor& x, Tensor& y) {
    // x: (batch, seq_len, hidden_size)
    size_t batch = x.size(0);
    size_t seq_len = x.size(1);
    size_t hidden_size = x.size(2);
    size_t intermediate_size = w1_.size(0);
    
    // Flatten
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
    
    y_flat.reshape({batch, seq_len, hidden_size});
    y = y_flat.copy();
}

// ============================================================================
// SparseMoeBlock
// ============================================================================

SparseMoeBlock::SparseMoeBlock(int layer_idx) {
    std::stringstream ss;
    ss << "layers." << layer_idx << ".feed_forward.gate.weight";
    gate_ = Tensor::load_from_file(ss.str(), nullptr);  // Uses global g_model_loader
    
    experts_.reserve(NUM_EXPERTS);
    for (size_t i = 0; i < NUM_EXPERTS; i++) {
        std::stringstream ss_w1, ss_w2, ss_w3;
        ss_w1 << "layers." << layer_idx << ".feed_forward.experts." << i << ".w1.weight";
        ss_w2 << "layers." << layer_idx << ".feed_forward.experts." << i << ".w2.weight";
        ss_w3 << "layers." << layer_idx << ".feed_forward.experts." << i << ".w3.weight";
        
        experts_.emplace_back(ss_w1.str(), ss_w2.str(), ss_w3.str());
    }
    
    if (USE_EXPERT_BIAS) {
        std::stringstream ss_bias;
        ss_bias << "layers." << layer_idx << ".feed_forward.expert_bias";
        expert_bias_ = Tensor::load_from_file(ss_bias.str(), nullptr);  // Uses global g_model_loader
    } else {
        expert_bias_ = Tensor({NUM_EXPERTS});
        expert_bias_.zero();
    }
    
    // Initialize GPU pointers
    std::vector<float*> w1_ptrs_host(NUM_EXPERTS);
    std::vector<float*> w2_ptrs_host(NUM_EXPERTS);
    std::vector<float*> w3_ptrs_host(NUM_EXPERTS);
    
    for (int i = 0; i < NUM_EXPERTS; i++) {
        w1_ptrs_host[i] = experts_[i].w1().data();
        w2_ptrs_host[i] = experts_[i].w2().data();
        w3_ptrs_host[i] = experts_[i].w3().data();
    }
    
    CHECK_CUDA(cudaMalloc(&w1_ptrs_gpu_, NUM_EXPERTS * sizeof(float*)));
    CHECK_CUDA(cudaMalloc(&w2_ptrs_gpu_, NUM_EXPERTS * sizeof(float*)));
    CHECK_CUDA(cudaMalloc(&w3_ptrs_gpu_, NUM_EXPERTS * sizeof(float*)));
    
    CHECK_CUDA(cudaMemcpy(w1_ptrs_gpu_, w1_ptrs_host.data(), NUM_EXPERTS * sizeof(float*), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(w2_ptrs_gpu_, w2_ptrs_host.data(), NUM_EXPERTS * sizeof(float*), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(w3_ptrs_gpu_, w3_ptrs_host.data(), NUM_EXPERTS * sizeof(float*), cudaMemcpyHostToDevice));
}

SparseMoeBlock::~SparseMoeBlock() {
    if (w1_ptrs_gpu_) cudaFree(w1_ptrs_gpu_);
    if (w2_ptrs_gpu_) cudaFree(w2_ptrs_gpu_);
    if (w3_ptrs_gpu_) cudaFree(w3_ptrs_gpu_);
}

void SparseMoeBlock::forward(const Tensor& x, Tensor& y, Tensor& router_logits) {
    size_t batch = x.size(0);
    size_t seq_len = x.size(1);
    size_t hidden_size = x.size(2);
    
    Tensor x_flat = x.view({batch * seq_len, hidden_size});
    
    router_logits = Tensor({batch * seq_len, NUM_EXPERTS});
    tensor_ops::matmul_transposed(x_flat, gate_, router_logits);
    
    Tensor top_k_indices_tensor({batch * seq_len, NUM_EXPERTS_PER_TOK});
    Tensor top_k_weights_tensor({batch * seq_len, NUM_EXPERTS_PER_TOK});
    
    tensor_ops::route_tokens(router_logits, expert_bias_, 
                            top_k_indices_tensor, top_k_weights_tensor,
                            batch * seq_len, NUM_EXPERTS, NUM_EXPERTS_PER_TOK);
    
    y = Tensor({batch, seq_len, hidden_size});
    y.zero();
    
    size_t intermediate_size = experts_[0].w1().size(0);
    tensor_ops::moe_expert_dispatch(x_flat, top_k_indices_tensor, top_k_weights_tensor,
                                    w1_ptrs_gpu_, w2_ptrs_gpu_, w3_ptrs_gpu_,
                                    y,
                                    batch * seq_len, hidden_size, intermediate_size,
                                    NUM_EXPERTS_PER_TOK, seq_len);
}

void SparseMoeBlock::route_tokens(const Tensor& router_logits, std::vector<int>& top_k_indices,
                                  std::vector<float>& top_k_weights) {
    // Deprecated - GPU version used
}

// ============================================================================
// Attention
// ============================================================================

Attention::Attention(int layer_idx) : layer_idx_(layer_idx) {
    std::stringstream ss_q, ss_k, ss_v, ss_o, ss_q_ln, ss_k_ln;
    ss_q << "layers." << layer_idx << ".self_attn.q_proj.weight";
    ss_k << "layers." << layer_idx << ".self_attn.k_proj.weight";
    ss_v << "layers." << layer_idx << ".self_attn.v_proj.weight";
    ss_o << "layers." << layer_idx << ".self_attn.out_proj.weight";
    ss_q_ln << "layers." << layer_idx << ".self_attn.q_layernorm.weight";
    ss_k_ln << "layers." << layer_idx << ".self_attn.k_layernorm.weight";
    
    q_proj_ = Tensor::load_from_file(ss_q.str(), nullptr);  // Uses global g_model_loader
    k_proj_ = Tensor::load_from_file(ss_k.str(), nullptr);
    v_proj_ = Tensor::load_from_file(ss_v.str(), nullptr);
    o_proj_ = Tensor::load_from_file(ss_o.str(), nullptr);
    
    q_layernorm_ = std::make_unique<RMSNorm>(ss_q_ln.str());
    k_layernorm_ = std::make_unique<RMSNorm>(ss_k_ln.str());
}

void Attention::forward(const Tensor& x, const Tensor& cos, const Tensor& sin,
                        const Tensor* attention_mask, Tensor& output) {
    int batch = x.size(0);
    int seq_len = x.size(1);
    
    tensor_ops::attention(x, cos, sin, 
                         q_proj_, k_proj_, v_proj_, o_proj_,
                         q_layernorm_->weight(), k_layernorm_->weight(),
                         output,
                         batch, seq_len, NUM_ATTENTION_HEADS, HEAD_DIM, NUM_KEY_VALUE_HEADS);
}

// ============================================================================
// ShortConv
// ============================================================================

ShortConv::ShortConv(int layer_idx) : layer_idx_(layer_idx) {
    std::stringstream ss_conv_w, ss_conv_b, ss_in_w, ss_in_b, ss_out_w, ss_out_b;
    ss_conv_w << "layers." << layer_idx << ".short_conv.conv_1d.weight";
    ss_conv_b << "layers." << layer_idx << ".short_conv.conv_1d.bias";
    ss_in_w << "layers." << layer_idx << ".short_conv.in_proj.weight";
    ss_in_b << "layers." << layer_idx << ".short_conv.in_proj.bias";
    ss_out_w << "layers." << layer_idx << ".short_conv.out_proj.weight";
    ss_out_b << "layers." << layer_idx << ".short_conv.out_proj.bias";
        
    conv_weight_ = Tensor::load_from_file(ss_conv_w.str(), nullptr);  // Uses global g_model_loader
    conv_bias_ = Tensor::load_from_file(ss_conv_b.str(), nullptr);
    in_proj_weight_ = Tensor::load_from_file(ss_in_w.str(), nullptr);
    in_proj_bias_ = Tensor::load_from_file(ss_in_b.str(), nullptr);
    out_proj_weight_ = Tensor::load_from_file(ss_out_w.str(), nullptr);
    out_proj_bias_ = Tensor::load_from_file(ss_out_b.str(), nullptr);
}

void ShortConv::forward(const Tensor& x, Tensor& y) {
    int batch = x.size(0);
    int seq_len = x.size(1);
    int hidden_size = x.size(2);
    int kernel_size = conv_weight_.size(1);
    
    tensor_ops::conv(x, conv_weight_, in_proj_weight_, out_proj_weight_,
                    &conv_bias_, &in_proj_bias_, &out_proj_bias_,
                    y,
                    batch, seq_len, hidden_size, kernel_size);
}

// ============================================================================
// DecoderLayer
// ============================================================================

DecoderLayer::DecoderLayer(int layer_idx, bool is_attention_layer) 
    : layer_idx_(layer_idx), is_attention_layer_(is_attention_layer) {
    
    std::stringstream ss_norm1, ss_norm2;
    ss_norm1 << "layers." << layer_idx << ".operator_norm.weight";
    ss_norm2 << "layers." << layer_idx << ".ffn_norm.weight";
    
    input_layernorm_ = std::make_unique<RMSNorm>(ss_norm1.str());
    post_attention_layernorm_ = std::make_unique<RMSNorm>(ss_norm2.str());
    
    if (is_attention_layer) {
        self_attn_ = std::make_unique<Attention>(layer_idx);
    } else {
        short_conv_ = std::make_unique<ShortConv>(layer_idx);
    }
    
    if (layer_idx >= 2) {
        moe_block_ = std::make_unique<SparseMoeBlock>(layer_idx);
    } else {
        std::stringstream ss_w1, ss_w2, ss_w3;
        ss_w1 << "layers." << layer_idx << ".feed_forward.gate_proj.weight";
        ss_w2 << "layers." << layer_idx << ".feed_forward.down_proj.weight";
        ss_w3 << "layers." << layer_idx << ".feed_forward.up_proj.weight";
        
        dense_mlp_ = std::make_unique<MLP>(ss_w1.str(), ss_w2.str(), ss_w3.str());
    }
}

void DecoderLayer::forward(const Tensor& x, const Tensor& cos, const Tensor& sin,
                           const Tensor* attention_mask, Tensor& output) {
    Tensor normed_input({x.size(0), x.size(1), x.size(2)});
    input_layernorm_->forward(x, normed_input);
    
    Tensor attn_output({x.size(0), x.size(1), x.size(2)});
    if (is_attention_layer_) {
        self_attn_->forward(normed_input, cos, sin, attention_mask, attn_output);
    } else {
        short_conv_->forward(normed_input, attn_output);
    }
    
    Tensor hidden_states({x.size(0), x.size(1), x.size(2)});
    tensor_ops::add(x, attn_output, hidden_states);
    
    Tensor normed_hidden({x.size(0), x.size(1), x.size(2)});
    post_attention_layernorm_->forward(hidden_states, normed_hidden);
    
    Tensor ffn_output({x.size(0), x.size(1), x.size(2)});
    if (moe_block_) {
        Tensor router_logits;
        moe_block_->forward(normed_hidden, ffn_output, router_logits);
    } else {
        dense_mlp_->forward(normed_hidden, ffn_output);
    }
    
    tensor_ops::add(hidden_states, ffn_output, output);
}

// ============================================================================
// LFM2Model
// ============================================================================

LFM2Model::LFM2Model(const std::string& model_file) {
    loader_ = std::make_unique<ModelLoader>(model_file);
    g_model_loader = loader_.get();
    
    load_embeddings();
    load_layers();
    load_output_layers();
}

void LFM2Model::load_embeddings() {
    embed_tokens_ = Tensor::load_from_file("embed_tokens.weight", loader_.get());  // Uses instance loader
}

void LFM2Model::load_layers() {
    for (int i = 0; i < NUM_HIDDEN_LAYERS; i++) {
        bool is_attn = (LAYER_TYPES[i] == 0);
        layers_.push_back(std::make_unique<DecoderLayer>(i, is_attn));
    }
}

void LFM2Model::load_output_layers() {
    norm_ = std::make_unique<RMSNorm>("norm.weight");
    lm_head_ = Tensor::load_from_file("lm_head.weight", loader_.get());  // Uses instance loader
    rotary_emb_ = std::make_unique<RotaryEmbedding>();
}

void LFM2Model::forward(const std::vector<int>& input_ids, Tensor& logits) {
    int batch = 1;
    int seq_len = input_ids.size();
    
    int* input_ids_gpu;
    CHECK_CUDA(cudaMalloc(&input_ids_gpu, seq_len * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(input_ids_gpu, input_ids.data(), seq_len * sizeof(int), cudaMemcpyHostToDevice));
    
    Tensor hidden_states({(size_t)batch, (size_t)seq_len, HIDDEN_SIZE});
    tensor_ops::embedding_lookup(input_ids_gpu, embed_tokens_, hidden_states, batch, seq_len, HIDDEN_SIZE);
    
    CHECK_CUDA(cudaFree(input_ids_gpu));
    
    Tensor cos({(size_t)seq_len, HEAD_DIM});
    Tensor sin({(size_t)seq_len, HEAD_DIM});
    rotary_emb_->forward(seq_len, cos, sin);
    
    for (auto& layer : layers_) {
        Tensor layer_out({(size_t)batch, (size_t)seq_len, HIDDEN_SIZE});
        layer->forward(hidden_states, cos, sin, nullptr, layer_out);
        hidden_states = std::move(layer_out);
    }
    
    Tensor norm_out({(size_t)batch, (size_t)seq_len, HIDDEN_SIZE});
    norm_->forward(hidden_states, norm_out);
    
    Tensor output({(size_t)batch * seq_len, VOCAB_SIZE});
    tensor_ops::matmul_transposed(norm_out, lm_head_, output);
    
    logits = std::move(output);
}