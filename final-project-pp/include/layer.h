#pragma once

#include "tensor.h"
#include "config.h"
#include <vector>
#include <memory>

constexpr int NUM_GPUS = 4;

// Buffer Pool for reducing memory allocation overhead
class BufferPool {
public:
    BufferPool();
    ~BufferPool();

    void init(size_t max_batch, size_t max_seq_len, int device_id = 0);
    void cleanup();

    // Attention buffers
    Tensor q_proj_out, k_proj_out, v_proj_out;
    Tensor q_reshaped, k_reshaped, v_reshaped;
    Tensor q_normed, k_normed;
    Tensor q_heads, k_heads, v_heads;
    Tensor k_repeated, v_repeated;
    Tensor attn_output;
    Tensor attn_flat;
    Tensor attn_proj_out;

    // ShortConv buffers
    Tensor conv_in_proj;
    Tensor conv_B, conv_C, conv_x_gate;
    Tensor conv_Bx, conv_out;
    Tensor conv_y_pre;
    Tensor conv_transposed;
    Tensor conv_proj_out;

    // MLP buffers (for dense layers)
    Tensor mlp_gate;
    Tensor mlp_gate_silu;
    Tensor mlp_up, mlp_hidden, mlp_out;

    // DecoderLayer buffers
    Tensor layer_normed_input;
    Tensor layer_attn_out;
    Tensor layer_hidden;
    Tensor layer_normed_hidden;

    bool is_initialized() const { return initialized_; }

private:
    bool initialized_;
    int device_id_;
    size_t num_samples_;
    size_t max_seq_len_;
};

// Active buffer pool/stream context for the current stage
extern thread_local BufferPool* g_bufpool;
extern thread_local cudaStream_t* g_streams;
extern thread_local cudaEvent_t* g_events;

void set_stage_resources(int device_id, size_t max_batch, size_t max_seq_len);
void cleanup_stage_resources();

// RMSNorm Layer
class RMSNorm {
public:
    RMSNorm(const std::string& weight_file);
    void forward(const Tensor& x, Tensor& y, cudaStream_t stream = 0);
    
private:
    Tensor weight_;
};

// Rotary Position Embedding
class RotaryEmbedding {
public:
    RotaryEmbedding();
    void forward(size_t seq_len, Tensor& cos, Tensor& sin, cudaStream_t stream = 0);
    
private:
    Tensor cos_cached_;
    Tensor sin_cached_;
    size_t max_seq_len_;
};

// MLP Layer (Feed-Forward Network)
class MLP {
public:
    struct Scratch {
        Tensor gate;
        Tensor gate_silu;
        Tensor up;
        Tensor hidden;
        Tensor y_flat;
    };

    MLP(const std::string& w1_file, const std::string& w2_file,
        const std::string& w3_file);
    ~MLP();
    void forward(const Tensor& x, Tensor& y, cudaStream_t stream = 0,
                 bool use_aux = true, Scratch* scratch = nullptr);
    
private:
    Tensor w1_;  // up projection
    Tensor w3_;  // gate projection
    Tensor w2_;  // down projection
};

// Sparse MoE Block
class SparseMoeBlock {
public:
    SparseMoeBlock(int layer_idx);
    void forward(const Tensor& x, Tensor& y, Tensor& router_logits,
                 cudaStream_t stream = 0);
    
private:
    Tensor gate_;  // router
    std::vector<std::unique_ptr<MLP>> experts_;
    Tensor expert_bias_;  // optional
    
    void route_tokens(const Tensor& router_logits, std::vector<int>& top_k_indices,
                      std::vector<float>& top_k_weights, cudaStream_t stream = 0);
};

// Multi-Head Attention
class Attention {
public:
    Attention(int layer_idx);
    ~Attention();
    void forward(const Tensor& x, const Tensor& cos, const Tensor& sin,
                 const Tensor* attention_mask, Tensor& output,
                 cudaStream_t stream = 0);
    
private:
    Tensor q_proj_;
    Tensor k_proj_;
    Tensor v_proj_;
    Tensor o_proj_;
    std::unique_ptr<RMSNorm> q_layernorm_;
    std::unique_ptr<RMSNorm> k_layernorm_;
    int layer_idx_;
};

// Short Convolution (Mamba-style)
class ShortConv {
public:
    ShortConv(int layer_idx);
    void forward(const Tensor& x, Tensor& y, cudaStream_t stream = 0);
    
private:
    Tensor conv_weight_;
    Tensor conv_bias_;
    Tensor in_proj_weight_;
    Tensor in_proj_bias_;
    Tensor out_proj_weight_;
    Tensor out_proj_bias_;
    int layer_idx_;
};

// DecoderLayer
class DecoderLayer {
public:
    DecoderLayer(int layer_idx, bool is_attention_layer, int device_id);
    void forward(const Tensor& x, const Tensor& cos, const Tensor& sin,
                 const Tensor* attention_mask, Tensor& output, cudaStream_t stream = 0);
    
    bool is_attention_layer() const { return is_attention_layer_; }
    int device_id() const { return device_id_; }
    
private:
    int layer_idx_;
    bool is_attention_layer_;
    int device_id_;  // GPU this layer runs on
    
    // Components
    std::unique_ptr<RMSNorm> input_layernorm_;
    std::unique_ptr<RMSNorm> post_attention_layernorm_;
    
    // Either attention or conv
    std::unique_ptr<Attention> self_attn_;
    std::unique_ptr<ShortConv> short_conv_;
    
    // Either MoE block (layers >= 2) or dense MLP (layers 0-1)
    std::unique_ptr<SparseMoeBlock> moe_block_;
    std::unique_ptr<MLP> dense_mlp_;
};
