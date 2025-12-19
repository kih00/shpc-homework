#pragma once

#include "tensor.h"
#include "layer.h"
#include "config.h"
#include "model_loader.h"
#include <vector>
#include <memory>
#include <string>

// Global model loader (defined in model.cu)
extern std::unique_ptr<ModelLoader> g_model_loader;

// Global MPI rank for conditional debug output (defined in model.cu)
extern int g_mpi_rank;

class LFM2Model {
public:
    LFM2Model(const std::string& model_file);
    ~LFM2Model();

    // Forward pass
    void forward(const std::vector<int>& input_ids, size_t batch,
                 size_t seq_len, Tensor& logits, cudaStream_t stream = 0);

private:
    std::unique_ptr<ModelLoader> loader_;

    // Embeddings
    Tensor embed_tokens_;

    // Decoder layers
    std::vector<std::unique_ptr<DecoderLayer>> layers_;

    // Final norm
    std::unique_ptr<RMSNorm> norm_;

    // LM head (output projection)
    Tensor lm_head_;

    // GPU management
    int layers_per_stage_[NUM_GPUS] = {7, 6, 6, 5}; // layer 0, 1 is lightweighted

    // RoPE
    std::unique_ptr<RotaryEmbedding> rotary_emb_;

    // Helper functions
    void load_embeddings();
    void load_layers();
    void load_output_layers();
};
