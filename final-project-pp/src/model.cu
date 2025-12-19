#include "model.h"
#include "model_loader.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <array>
#include <random>
#include <cstring>
#include <utility>
#include <omp.h>

// Global model loader (definition)
std::unique_ptr<ModelLoader> g_model_loader;

// Global MPI rank for conditional debug output (set from main.cpp)
int g_mpi_rank = 0;

// Active stage context pointers (defined later)
thread_local BufferPool* g_bufpool = nullptr;
thread_local cudaStream_t* g_streams = nullptr;
thread_local cudaEvent_t* g_events = nullptr;
thread_local int g_active_device = 0;

// Debug print macro - only rank 0 prints
#define DEBUG_PRINT(x) do { if (g_mpi_rank == 0) { std::cout << x; } } while(0)
#define DEBUG_PRINTLN(x) do { if (g_mpi_rank == 0) { std::cout << x << std::endl; } } while(0)

// Tuning knob: micro-batch size divisor (num_samples / NUM_BATCH)
constexpr size_t NUM_BATCH = NUM_GPUS * 4;
constexpr int NUM_EXPERT_STREAMS = 4;
constexpr int EXPERT_SLOTS = 2;

// ============================================================================
// CUDA Kernels for SparseMoeBlock
// ============================================================================

// Optimized Top-K routing kernel using warp parallelism
__global__ void route_tokens_kernel(
    const float* router_logits, const float* expert_bias,
    int* topk_indices, float* topk_weights, size_t num_tokens,
    bool norm_topk_prob, float routed_scaling_factor) {
    // router_logits: (num_tokens, NUM_EXPERTS)
    // topk_indices, topk_weights: (num_tokens, NUM_EXPERTS_PER_TOK)

    size_t t = blockIdx.x; // 1 block = 1 token
    if (t >= num_tokens) return;

    const int tid = threadIdx.x;

    // Shared memory for scores and selection
    __shared__ float scores[NUM_EXPERTS];
    __shared__ float sigmoid_vals[NUM_EXPERTS];
    __shared__ int sel_idx[NUM_EXPERTS_PER_TOK];
    __shared__ float sel_w[NUM_EXPERTS_PER_TOK];

    // Compute sigmoid scores (32 threads for 32 experts)
    if (tid < NUM_EXPERTS) {
        float logit = router_logits[t * NUM_EXPERTS + tid];
        float sig = 1.0f / (1.0f + expf(-logit));
        sigmoid_vals[tid] = sig;
        scores[tid] = (expert_bias != nullptr) ? (sig + expert_bias[tid]) : sig;
    }
    __syncthreads();

    // Greedy top-k selection (only computed by thread 0)
    if (tid == 0) {
        for (size_t k = 0; k < NUM_EXPERTS_PER_TOK; k++) {
            float best_score = -1e30f;
            int best_idx = 0;

            for (size_t j = 0; j < NUM_EXPERTS; j++) {
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

        // Normalize (if needed)
        if (norm_topk_prob) {
            float sum = 0.0f;
            for (size_t k = 0; k < NUM_EXPERTS_PER_TOK; k++) sum += sel_w[k];
            float inv = (sum > 1e-6f) ? (1.0f / sum) : 0.0f;
            for (size_t k = 0; k < NUM_EXPERTS_PER_TOK; k++) sel_w[k] *= inv;
        }

        // Store results
        for (size_t k = 0; k < NUM_EXPERTS_PER_TOK; k++) {
            topk_indices[t * NUM_EXPERTS_PER_TOK + k] = sel_idx[k];
            topk_weights[t * NUM_EXPERTS_PER_TOK + k] = sel_w[k] * routed_scaling_factor;
        }
    }
}

// Gathers rows from src specified by indices into dst
__global__ void gather_expert_input_kernel(
    const float* src, float* dst, const int* indices,
    size_t num_gathered, size_t hidden_size) {
    // src: (total_tokens, hidden_size)
    // dst: (num_gathered, hidden_size)
    // indices: (num_gathered,)

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = num_gathered * hidden_size;

    if (idx < total) {
        size_t row = idx / hidden_size;
        size_t col = idx % hidden_size;
        int src_row = indices[row];
        dst[row * hidden_size + col] = src[src_row * hidden_size + col];
    }
}

// Gather rows from embedding table
__global__ void embedding_lookup_kernel(
    const int* input_ids, const float* embed_table, float* output,
    size_t num_tokens, size_t hidden_size) {
    // input_ids: (num_tokens,) - token IDs
    // embed_table: (vocab_size, hidden_size) - embedding table
    // output: (num_tokens, hidden_size) - output embeddings

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = num_tokens * hidden_size;

    if (idx < total) {
        size_t token_idx = idx / hidden_size;
        size_t dim = idx % hidden_size;
        int token_id = input_ids[token_idx];
        output[idx] = embed_table[token_id * hidden_size + dim];
    }
}

// Scatter and accumulate expert outputs with weights
// dst[token_id, hidden_size] <- sum(w * src[idx, hidden_size])
__global__ void weighted_scatter_add_kernel(
    float* dst, const float* src, const int* token_ids, const float* weights,
    size_t num_expert_tokens, size_t hidden_size) {
    // dst: (num_tokens, hidden_size)
    // src: (num_expert_tokens, hidden_size)
    // token_ids, weights: (num_expert_tokens,)

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = num_expert_tokens * hidden_size;

    if (idx < total) {
        size_t row = idx / hidden_size;
        size_t col = idx % hidden_size;
        int token_id = token_ids[row];
        float w = weights[row];

        // atomic add to avoid race conditions
        atomicAdd(&dst[token_id * hidden_size + col], w * src[row * hidden_size + col]);
    }
}

// Extract last token from each batch
__global__ void extract_last_token_kernel(
    const float* input, float* output,
    size_t batch, size_t seq_len, size_t hidden_size) {
    // input: (batch, seq_len, hidden_size)
    // output: (batch, hidden_size)

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch * hidden_size;

    if (idx < total) {
        size_t b = idx / hidden_size;
        size_t h = idx % hidden_size;
        // Copy last token (seq_len - 1) from each batch
        output[b * hidden_size + h] =
            input[(b * seq_len + (seq_len - 1)) * hidden_size + h];
    }
}

// =============================================================================
// BufferPool and StageResources implementations
// =============================================================================

// buffer pool for intermediate tensors
BufferPool::BufferPool()
    : initialized_(false), device_id_(0), num_samples_(0), max_seq_len_(0) {}

BufferPool::~BufferPool() {
    cleanup();
}

void BufferPool::init(size_t num_samples, size_t max_seq_len, int device_id) {
    if (initialized_) return;

    device_id_ = device_id;
    num_samples_ = num_samples;
    max_seq_len_ = max_seq_len;

    CHECK_CUDA(cudaSetDevice(device_id_));

    size_t bs = num_samples * max_seq_len;
    size_t hidden = HIDDEN_SIZE;
    size_t num_heads = NUM_ATTENTION_HEADS;
    size_t num_kv_heads = NUM_KEY_VALUE_HEADS;
    size_t head_dim = HEAD_DIM;
    size_t intermediate = INTERMEDIATE_SIZE;

    // Attention buffers
    q_proj_out = Tensor({bs, num_heads * head_dim});
    k_proj_out = Tensor({bs, num_kv_heads * head_dim});
    v_proj_out = Tensor({bs, num_kv_heads * head_dim});
    q_reshaped = Tensor({num_samples, max_seq_len, num_heads, head_dim});
    k_reshaped = Tensor({num_samples, max_seq_len, num_kv_heads, head_dim});
    v_reshaped = Tensor({num_samples, max_seq_len, num_kv_heads, head_dim});
    q_normed = Tensor({num_samples, max_seq_len, num_heads, head_dim});
    k_normed = Tensor({num_samples, max_seq_len, num_kv_heads, head_dim});
    q_heads = Tensor({num_samples, num_heads, max_seq_len, head_dim});
    k_heads = Tensor({num_samples, num_kv_heads, max_seq_len, head_dim});
    v_heads = Tensor({num_samples, num_kv_heads, max_seq_len, head_dim});
    k_repeated = Tensor({num_samples, num_heads, max_seq_len, head_dim});
    v_repeated = Tensor({num_samples, num_heads, max_seq_len, head_dim});
    attn_output = Tensor({num_samples, num_heads, max_seq_len, head_dim});
    attn_flat = Tensor({bs, hidden});
    attn_proj_out = Tensor({bs, hidden});

    // ShortConv buffers
    conv_in_proj = Tensor({bs, 3 * hidden});
    conv_B = Tensor({num_samples, hidden, max_seq_len});
    conv_C = Tensor({num_samples, hidden, max_seq_len});
    conv_x_gate = Tensor({num_samples, hidden, max_seq_len});
    conv_Bx = Tensor({num_samples, hidden, max_seq_len});
    conv_out = Tensor({num_samples, hidden, max_seq_len});
    conv_y_pre = Tensor({num_samples, hidden, max_seq_len});
    conv_transposed = Tensor({num_samples, max_seq_len, hidden});
    conv_proj_out = Tensor({bs, hidden});

    // MLP buffers (for dense layers)
    mlp_gate = Tensor({bs, intermediate});
    mlp_gate_silu = Tensor({bs, intermediate});
    mlp_up = Tensor({bs, intermediate});
    mlp_hidden = Tensor({bs, intermediate});
    mlp_out = Tensor({bs, hidden});

    // DecoderLayer buffers
    layer_normed_input = Tensor({num_samples, max_seq_len, hidden});
    layer_attn_out = Tensor({num_samples, max_seq_len, hidden});
    layer_hidden = Tensor({num_samples, max_seq_len, hidden});
    layer_normed_hidden = Tensor({num_samples, max_seq_len, hidden});

    // Pre-allocate on device
    q_proj_out.to_device(device_id_);
    k_proj_out.to_device(device_id_);
    v_proj_out.to_device(device_id_);
    q_reshaped.to_device(device_id_);
    k_reshaped.to_device(device_id_);
    v_reshaped.to_device(device_id_);
    q_normed.to_device(device_id_);
    k_normed.to_device(device_id_);
    q_heads.to_device(device_id_);
    k_heads.to_device(device_id_);
    v_heads.to_device(device_id_);
    k_repeated.to_device(device_id_);
    v_repeated.to_device(device_id_);
    attn_output.to_device(device_id_);
    attn_flat.to_device(device_id_);
    attn_proj_out.to_device(device_id_);

    conv_in_proj.to_device(device_id_);
    conv_B.to_device(device_id_);
    conv_C.to_device(device_id_);
    conv_x_gate.to_device(device_id_);
    conv_Bx.to_device(device_id_);
    conv_out.to_device(device_id_);
    conv_y_pre.to_device(device_id_);
    conv_transposed.to_device(device_id_);
    conv_proj_out.to_device(device_id_);

    mlp_gate.to_device(device_id_);
    mlp_gate_silu.to_device(device_id_);
    mlp_up.to_device(device_id_);
    mlp_hidden.to_device(device_id_);
    mlp_out.to_device(device_id_);

    layer_normed_input.to_device(device_id_);
    layer_attn_out.to_device(device_id_);
    layer_hidden.to_device(device_id_);
    layer_normed_hidden.to_device(device_id_);

    initialized_ = true;
}

void BufferPool::cleanup() {
    if (!initialized_) return;
    // Tensors will be automatically cleaned up by their destructors
    initialized_ = false;
}

struct MoeResources {
    bool initialized = false;
    size_t max_expert_tokens = 0;
    std::array<std::array<cudaStream_t, 2>, NUM_EXPERT_STREAMS> streams{};
    std::array<std::array<cudaEvent_t, 2>, NUM_EXPERT_STREAMS> events{};
    std::array<std::array<cudaEvent_t, EXPERT_SLOTS>, NUM_EXPERT_STREAMS> slot_done_events{};
    std::array<std::array<int*, EXPERT_SLOTS>, NUM_EXPERT_STREAMS> h_indices{};
    std::array<std::array<float*, EXPERT_SLOTS>, NUM_EXPERT_STREAMS> h_weights{};
    std::array<int*, NUM_EXPERT_STREAMS> d_indices{};
    std::array<float*, NUM_EXPERT_STREAMS> d_weights{};
    std::array<Tensor, NUM_EXPERT_STREAMS> expert_in_buf{};
    std::array<MLP::Scratch, NUM_EXPERT_STREAMS> expert_scratch{};
    cudaEvent_t expert_ready = nullptr;
};

// StageResources structure to hold per-GPU resources
struct StageResources {
    BufferPool buffer_pool;
    bool streams_initialized;
    cudaStream_t shared_streams[2];
    cudaEvent_t shared_events[3];
    MoeResources moe;

    StageResources() : streams_initialized(false) {}
};

static std::array<StageResources, NUM_GPUS> g_stage_resources;

// Set active stage resources for the given device
void set_stage_resources(int device_id, size_t num_samples, size_t max_seq_len) {
    if (device_id < 0 || device_id >= NUM_GPUS) {
        throw std::runtime_error("Invalid device id for stage resources");
    }

    StageResources& res = g_stage_resources[device_id];

    if (!res.buffer_pool.is_initialized()) {
        res.buffer_pool.init(num_samples, max_seq_len, device_id);
    }

    if (!res.streams_initialized) {
        CHECK_CUDA(cudaSetDevice(device_id));
        for (int i = 0; i < 2; ++i) {
            CHECK_CUDA(cudaStreamCreateWithFlags(
                &res.shared_streams[i], cudaStreamNonBlocking));
        }
        for (int i = 0; i < 3; ++i) {
            CHECK_CUDA(cudaEventCreateWithFlags(
                &res.shared_events[i], cudaEventDisableTiming));
        }
        res.streams_initialized = true;
    }

    CHECK_CUDA(cudaSetDevice(device_id));
    g_active_device = device_id;
    g_bufpool = &res.buffer_pool;
    g_streams = res.shared_streams;
    g_events = res.shared_events;
}

static void free_moe_buffers(MoeResources& moe) {
    for (int s = 0; s < NUM_EXPERT_STREAMS; ++s) {
        if (moe.d_indices[s]) {
            CHECK_CUDA(cudaFree(moe.d_indices[s]));
            moe.d_indices[s] = nullptr;
        }
        if (moe.d_weights[s]) {
            CHECK_CUDA(cudaFree(moe.d_weights[s]));
            moe.d_weights[s] = nullptr;
        }
        for (int slot = 0; slot < EXPERT_SLOTS; ++slot) {
            if (moe.h_indices[s][slot]) {
                CHECK_CUDA(cudaFreeHost(moe.h_indices[s][slot]));
                moe.h_indices[s][slot] = nullptr;
            }
            if (moe.h_weights[s][slot]) {
                CHECK_CUDA(cudaFreeHost(moe.h_weights[s][slot]));
                moe.h_weights[s][slot] = nullptr;
            }
        }
        moe.expert_in_buf[s] = Tensor();
    }
    moe.max_expert_tokens = 0;
}

static void ensure_moe_resources(
    StageResources& res, int device_id, size_t max_expert_tokens) {
    MoeResources& moe = res.moe;

    CHECK_CUDA(cudaSetDevice(device_id));

    if (!moe.initialized) {
        for (int s = 0; s < NUM_EXPERT_STREAMS; ++s) {
            CHECK_CUDA(cudaStreamCreateWithFlags(
                &moe.streams[s][0], cudaStreamNonBlocking));
            CHECK_CUDA(cudaStreamCreateWithFlags(
                &moe.streams[s][1], cudaStreamNonBlocking));
            CHECK_CUDA(cudaEventCreateWithFlags(
                &moe.events[s][0], cudaEventDisableTiming));
            CHECK_CUDA(cudaEventCreateWithFlags(
                &moe.events[s][1], cudaEventDisableTiming));
            for (int slot = 0; slot < EXPERT_SLOTS; ++slot) {
                CHECK_CUDA(cudaEventCreateWithFlags(
                    &moe.slot_done_events[s][slot], cudaEventDisableTiming));
            }
        }
        CHECK_CUDA(cudaEventCreateWithFlags(
            &moe.expert_ready, cudaEventDisableTiming));
        moe.initialized = true;
    }

    if (moe.max_expert_tokens < max_expert_tokens) {
        free_moe_buffers(moe);
        for (int s = 0; s < NUM_EXPERT_STREAMS; ++s) {
            CHECK_CUDA(cudaMalloc(
                &moe.d_indices[s], max_expert_tokens * sizeof(int)));
            CHECK_CUDA(cudaMalloc(
                &moe.d_weights[s], max_expert_tokens * sizeof(float)));
            for (int slot = 0; slot < EXPERT_SLOTS; ++slot) {
                CHECK_CUDA(cudaMallocHost(
                    &moe.h_indices[s][slot], max_expert_tokens * sizeof(int)));
                CHECK_CUDA(cudaMallocHost(
                    &moe.h_weights[s][slot], max_expert_tokens * sizeof(float)));
            }
            moe.expert_in_buf[s] =
                Tensor({max_expert_tokens, 1, HIDDEN_SIZE}, device_id);
        }
        moe.max_expert_tokens = max_expert_tokens;
    }
}

// Cleanup all stage resources
void cleanup_stage_resources() {
    for (int device = 0; device < NUM_GPUS; ++device) {
        StageResources& res = g_stage_resources[device];
        if (res.streams_initialized) {
            CHECK_CUDA(cudaSetDevice(device));
            for (int i = 0; i < 2; ++i) cudaStreamDestroy(res.shared_streams[i]);
            for (int i = 0; i < 3; ++i) cudaEventDestroy(res.shared_events[i]);
            res.streams_initialized = false;
        }
        if (res.moe.initialized || res.moe.max_expert_tokens > 0) {
            CHECK_CUDA(cudaSetDevice(device));
            if (res.moe.initialized) {
                for (int s = 0; s < NUM_EXPERT_STREAMS; ++s) {
                    for (int slot = 0; slot < EXPERT_SLOTS; ++slot) {
                        cudaEventDestroy(res.moe.slot_done_events[s][slot]);
                    }
                    cudaEventDestroy(res.moe.events[s][0]);
                    cudaEventDestroy(res.moe.events[s][1]);
                    cudaStreamDestroy(res.moe.streams[s][0]);
                    cudaStreamDestroy(res.moe.streams[s][1]);
                }
                if (res.moe.expert_ready) {
                    cudaEventDestroy(res.moe.expert_ready);
                    res.moe.expert_ready = nullptr;
                }
                res.moe.initialized = false;
            }
            free_moe_buffers(res.moe);
            for (int s = 0; s < NUM_EXPERT_STREAMS; ++s) {
                res.moe.expert_scratch[s] = MLP::Scratch{};
            }
        }
        if (res.buffer_pool.is_initialized()) {
            res.buffer_pool.cleanup();
        }
    }
    g_bufpool = nullptr;
    g_streams = nullptr;
    g_events = nullptr;
}

// ============================================================================
// Large Block Implementations - Complex layers and modules
// ============================================================================

// MLP (Feed-Forward Network) implementation
MLP::MLP(
    const std::string& w1_file, const std::string& w2_file,
    const std::string& w3_file) {
    w1_ = Tensor::load_from_file(w1_file);
    w2_ = Tensor::load_from_file(w2_file);
    w3_ = Tensor::load_from_file(w3_file);
}

MLP::~MLP() {
}

void MLP::forward(
    const Tensor& x, Tensor& y, cudaStream_t stream, bool use_aux, Scratch* scratch) {
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

    bool do_aux = use_aux && (g_streams != nullptr) && (g_events != nullptr);
    cudaStream_t stream_aux = stream;
    cudaEvent_t event_main_ready = nullptr;
    cudaEvent_t event_aux_done = nullptr;
    if (do_aux) {
        // Use StageConfig-managed shared stream/event slots
        stream_aux = g_streams[0];
        event_main_ready = g_events[0];
        event_aux_done = g_events[1];
        // Record event to ensure input data is ready for aux stream
        CHECK_CUDA(cudaEventRecord(event_main_ready, stream));
        CHECK_CUDA(cudaStreamWaitEvent(stream_aux, event_main_ready, 0));
    }

    const size_t tokens = batch * seq_len;
    Tensor gate;
    Tensor gate_silu;
    Tensor up;
    Tensor hidden;
    Tensor y_flat;

    if (scratch) {
        const size_t gate_elems = tokens * intermediate_size;
        const size_t y_elems = tokens * hidden_size;
        if (scratch->gate.size() < gate_elems) {
            scratch->gate = Tensor({tokens, intermediate_size}, g_active_device);
            scratch->gate_silu = Tensor({tokens, intermediate_size}, g_active_device);
            scratch->up = Tensor({tokens, intermediate_size}, g_active_device);
            scratch->hidden = Tensor({tokens, intermediate_size}, g_active_device);
        }
        if (scratch->y_flat.size() < y_elems) {
            scratch->y_flat = Tensor({tokens, hidden_size}, g_active_device);
        }

        gate = scratch->gate.view({tokens, intermediate_size});
        gate_silu = scratch->gate_silu.view({tokens, intermediate_size});
        up = scratch->up.view({tokens, intermediate_size});
        hidden = scratch->hidden.view({tokens, intermediate_size});
        y_flat = scratch->y_flat.view({tokens, hidden_size});
    } else {
        gate = Tensor({tokens, intermediate_size});
        gate_silu = Tensor({tokens, intermediate_size});
        up = Tensor({tokens, intermediate_size});
        hidden = Tensor({tokens, intermediate_size});
        y_flat = Tensor({tokens, hidden_size});
    }

    // gate = silu(x @ w1.T)
    tensor_ops::matmul_transposed(x_flat, w1_, gate, stream);
    tensor_ops::silu(gate, gate_silu, stream);
    
    // up = x @ w3.T (Aux stream if enabled)
    tensor_ops::matmul_transposed(x_flat, w3_, up, stream_aux);
    
    if (do_aux) {
        // Record completion of aux stream
        CHECK_CUDA(cudaEventRecord(event_aux_done, stream_aux));

        // Wait for aux stream to finish before using up
        CHECK_CUDA(cudaStreamWaitEvent(stream, event_aux_done, 0));
    }

    // hidden = gate_silu * up
    tensor_ops::mul(gate_silu, up, hidden, stream);

    // y = hidden @ w2.T
    tensor_ops::matmul_transposed(hidden, w2_, y_flat, stream);

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
        ss_w1 << "layers." << layer_idx
            << ".feed_forward.experts." << i << ".w1.weight";
        ss_w2 << "layers." << layer_idx
            << ".feed_forward.experts." << i << ".w2.weight";
        ss_w3 << "layers." << layer_idx
            << ".feed_forward.experts." << i << ".w3.weight";

        experts_.push_back(std::make_unique<MLP>(
            ss_w1.str(), ss_w2.str(), ss_w3.str()));
    }

    // Load expert bias if used
    if (USE_EXPERT_BIAS) {
        std::stringstream ss_bias;
        ss_bias << "layers." << layer_idx << ".feed_forward.expert_bias";
        expert_bias_ = Tensor::load_from_file(ss_bias.str());
    }
}

void SparseMoeBlock::route_tokens(
    const Tensor& router_logits,
    std::vector<int>& top_k_indices, std::vector<float>& top_k_weights,
    cudaStream_t stream) {
    // router_logits: (batch * seq_len, num_experts) = (num_tokens, NUM_EXPERTS)
    size_t num_tokens = router_logits.size(0);

    top_k_indices.resize(num_tokens * NUM_EXPERTS_PER_TOK);
    top_k_weights.resize(num_tokens * NUM_EXPERTS_PER_TOK);

    router_logits.to_device(-1, stream);

    int* d_topk_indices;
    float* d_topk_weights;
    CHECK_CUDA(cudaMalloc(
        &d_topk_indices, num_tokens * NUM_EXPERTS_PER_TOK * sizeof(int)));
    CHECK_CUDA(cudaMalloc(
        &d_topk_weights, num_tokens * NUM_EXPERTS_PER_TOK * sizeof(float)));

    const float* d_expert_bias = nullptr;
    if (USE_EXPERT_BIAS && expert_bias_.size() > 0) {
        expert_bias_.to_device(-1, stream);
        d_expert_bias = expert_bias_.device_data();
    }

    // Launch routing kernel
    dim3 block(NUM_EXPERTS);
    dim3 grid(num_tokens);

    route_tokens_kernel<<<grid, block, 0, stream>>>(
        router_logits.device_data(), d_expert_bias,
        d_topk_indices, d_topk_weights, num_tokens,
        NORM_TOPK_PROB, ROUTED_SCALING_FACTOR);

    // Copy results back to host
    CHECK_CUDA(cudaMemcpyAsync(
        top_k_indices.data(), d_topk_indices,
        num_tokens * NUM_EXPERTS_PER_TOK * sizeof(int),
        cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaMemcpyAsync(
        top_k_weights.data(), d_topk_weights,
        num_tokens * NUM_EXPERTS_PER_TOK * sizeof(float),
        cudaMemcpyDeviceToHost, stream));

    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaFree(d_topk_indices));
    CHECK_CUDA(cudaFree(d_topk_weights));
}

void SparseMoeBlock::forward(
    const Tensor& x, Tensor& y, Tensor& router_logits, cudaStream_t stream) {
    // x: (batch, seq_len, hidden_size)
    size_t batch = x.size(0);
    size_t seq_len = x.size(1);
    size_t hidden_size = x.size(2);
    size_t num_tokens = batch * seq_len;
    int device = g_active_device;

    CHECK_CUDA(cudaSetDevice(device));

    size_t max_expert_tokens = num_tokens * NUM_EXPERTS_PER_TOK;
    StageResources& stage_res = g_stage_resources[device];
    ensure_moe_resources(stage_res, device, max_expert_tokens);
    MoeResources& moe = stage_res.moe;
    auto& expert_streams = moe.streams;
    auto& expert_events = moe.events;
    auto& slot_done_events = moe.slot_done_events;
    auto& h_indices = moe.h_indices;
    auto& h_weights = moe.h_weights;
    auto& d_indices = moe.d_indices;
    auto& d_weights = moe.d_weights;
    auto& expert_in_buf = moe.expert_in_buf;
    auto& expert_scratch = moe.expert_scratch;

    bool slot_busy[NUM_EXPERT_STREAMS][EXPERT_SLOTS] = {};
    int next_slot[NUM_EXPERT_STREAMS] = {0};

    // Flatten
    Tensor x_flat = x.view({num_tokens, hidden_size});
    x_flat.to_device(device, stream);

    // Compute router logits
    router_logits = Tensor({num_tokens, NUM_EXPERTS});
    tensor_ops::matmul_transposed(x_flat, gate_, router_logits, stream);

    // Route tokens
    std::vector<int> top_k_indices;
    std::vector<float> top_k_weights;
    route_tokens(router_logits, top_k_indices, top_k_weights, stream);

    // Bucket tokens per expert on host
    std::vector<std::vector<int>> tokens_per_expert(NUM_EXPERTS);
    std::vector<std::vector<float>> weights_per_expert(NUM_EXPERTS);
    for (size_t t = 0; t < num_tokens; ++t) {
        for (size_t k = 0; k < NUM_EXPERTS_PER_TOK; ++k) {
            int expert_idx = top_k_indices[t * NUM_EXPERTS_PER_TOK + k];
            float weight = top_k_weights[t * NUM_EXPERTS_PER_TOK + k];
            tokens_per_expert[expert_idx].push_back(static_cast<int>(t));
            weights_per_expert[expert_idx].push_back(weight);
        }
    }

    // Initialize output
    y = Tensor({batch, seq_len, hidden_size});
    y.to_device(device, stream);
    CHECK_CUDA(cudaMemsetAsync(
        y.device_data(), 0, num_tokens * hidden_size * sizeof(float), stream));
    y.mark_device_dirty();
    CHECK_CUDA(cudaEventRecord(moe.expert_ready, stream));

    cudaStream_t* stage_streams = g_streams;
    cudaEvent_t* stage_events = g_events;

    auto acquire_slot = [&](int stream_idx) -> int {
        for (int attempt = 0; attempt < EXPERT_SLOTS; ++attempt) {
            int slot = (next_slot[stream_idx] + attempt) % EXPERT_SLOTS;
            if (slot_busy[stream_idx][slot]) {
                cudaError_t err = cudaEventQuery(slot_done_events[stream_idx][slot]);
                if (err == cudaSuccess) {
                    slot_busy[stream_idx][slot] = false;
                } else if (err != cudaErrorNotReady) {
                    CHECK_CUDA(err);
                }
            }
            if (!slot_busy[stream_idx][slot]) {
                next_slot[stream_idx] = (slot + 1) % EXPERT_SLOTS;
                return slot;
            }
        }

        // Both slots busy: wait for the next slot to complete.
        int slot = next_slot[stream_idx];
        CHECK_CUDA(cudaEventSynchronize(slot_done_events[stream_idx][slot]));
        slot_busy[stream_idx][slot] = false;
        next_slot[stream_idx] = (slot + 1) % EXPERT_SLOTS;
        return slot;
    };

    // Process each token through selected experts on the active stage device
    for (size_t expert_idx = 0; expert_idx < NUM_EXPERTS; expert_idx++) {
        const auto& token_list = tokens_per_expert[expert_idx];
        const auto& weight_list = weights_per_expert[expert_idx];
        size_t tok_count = token_list.size();

        if (tok_count == 0) continue;

        int stream_idx = static_cast<int>(expert_idx % NUM_EXPERT_STREAMS);
        cudaStream_t stream_expert = expert_streams[stream_idx][1];
        int slot = acquire_slot(stream_idx);
        CHECK_CUDA(cudaStreamWaitEvent(stream_expert, moe.expert_ready, 0));

        // Gather expert input with kernel
        Tensor expert_in = expert_in_buf[stream_idx].view({tok_count, 1, hidden_size});
        const float* src = x_flat.device_data();
        float* dst = expert_in.device_data();

        std::memcpy(
            h_indices[stream_idx][slot], token_list.data(), tok_count * sizeof(int));
        CHECK_CUDA(cudaMemcpyAsync(
            d_indices[stream_idx], h_indices[stream_idx][slot], tok_count * sizeof(int),
            cudaMemcpyHostToDevice, stream_expert));

        size_t total_elements = tok_count * hidden_size;
        dim3 block(BLOCK_DEFAULT);
        dim3 grid((total_elements + BLOCK_DEFAULT - 1) / BLOCK_DEFAULT);

        gather_expert_input_kernel<<<grid, block, 0, stream_expert>>>(
            src, dst, d_indices[stream_idx], tok_count, hidden_size);

        expert_in.mark_device_dirty();

        // Execute expert (use per-expert aux stream/events)
        Tensor expert_out;
        g_streams = expert_streams[stream_idx].data();
        g_events = expert_events[stream_idx].data();
        experts_[expert_idx]->forward(
            expert_in, expert_out, stream_expert, true, &expert_scratch[stream_idx]);
        g_streams = stage_streams;
        g_events = stage_events;

        // Scatter-add results
        std::memcpy(
            h_indices[stream_idx][slot], token_list.data(), tok_count * sizeof(int));
        std::memcpy(
            h_weights[stream_idx][slot], weight_list.data(), tok_count * sizeof(float));

        CHECK_CUDA(cudaMemcpyAsync(
            d_indices[stream_idx], h_indices[stream_idx][slot],
            tok_count * sizeof(int), cudaMemcpyHostToDevice, stream_expert));
        CHECK_CUDA(cudaMemcpyAsync(
            d_weights[stream_idx], h_weights[stream_idx][slot],
            tok_count * sizeof(float), cudaMemcpyHostToDevice, stream_expert));

        size_t total = tok_count * hidden_size;
        grid = dim3((total + BLOCK_DEFAULT - 1) / BLOCK_DEFAULT);

        weighted_scatter_add_kernel<<<grid, block, 0, stream_expert>>>(
            y.device_data(), expert_out.device_data(),
            d_indices[stream_idx], d_weights[stream_idx],
            tok_count, hidden_size);

        CHECK_CUDA(cudaEventRecord(slot_done_events[stream_idx][slot], stream_expert));
        slot_busy[stream_idx][slot] = true;
    }

    for (int s = 0; s < NUM_EXPERT_STREAMS; ++s) {
        for (int slot = 0; slot < EXPERT_SLOTS; ++slot) {
            if (slot_busy[s][slot]) {
                CHECK_CUDA(cudaEventSynchronize(slot_done_events[s][slot]));
            }
        }
    }

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

Attention::~Attention() {
}

void Attention::forward(
    const Tensor& x, const Tensor& cos, const Tensor& sin,
    const Tensor* attention_mask, Tensor& output, cudaStream_t stream) {
    // x: (batch, seq_len, hidden_size)
    size_t batch = x.size(0);
    size_t seq_len = x.size(1);
    size_t hidden_size = x.size(2);

    // Flatten
    Tensor x_flat = x.view({batch * seq_len, hidden_size});

    // Use global stage shared streams and events
    cudaStream_t stream_k = g_streams[0];
    cudaStream_t stream_v = g_streams[1];
    cudaEvent_t event_main_ready = g_events[0];
    cudaEvent_t event_k_done = g_events[1];
    cudaEvent_t event_v_done = g_events[2];

    // Record event to ensure input data is ready for streams
    CHECK_CUDA(cudaEventRecord(event_main_ready, stream));
    CHECK_CUDA(cudaStreamWaitEvent(stream_k, event_main_ready, 0));
    CHECK_CUDA(cudaStreamWaitEvent(stream_v, event_main_ready, 0));

    // Use buffer pool for intermediate tensors
    size_t flat_tokens = batch * seq_len;
    Tensor q_proj_out = g_bufpool->q_proj_out
        .view({flat_tokens, NUM_ATTENTION_HEADS * HEAD_DIM});
    Tensor k_proj_out = g_bufpool->k_proj_out
        .view({flat_tokens, NUM_KEY_VALUE_HEADS * HEAD_DIM});
    Tensor v_proj_out = g_bufpool->v_proj_out
        .view({flat_tokens, NUM_KEY_VALUE_HEADS * HEAD_DIM});

    Tensor q_reshaped = g_bufpool->q_reshaped
        .view({batch, seq_len, NUM_ATTENTION_HEADS, HEAD_DIM});
    Tensor k_reshaped = g_bufpool->k_reshaped
        .view({batch, seq_len, NUM_KEY_VALUE_HEADS, HEAD_DIM});
    Tensor v_reshaped = g_bufpool->v_reshaped
        .view({batch, seq_len, NUM_KEY_VALUE_HEADS, HEAD_DIM});

    Tensor q_normed = g_bufpool->q_normed
        .view({batch, seq_len, NUM_ATTENTION_HEADS, HEAD_DIM});
    Tensor k_normed = g_bufpool->k_normed
        .view({batch, seq_len, NUM_KEY_VALUE_HEADS, HEAD_DIM});

    Tensor q = g_bufpool->q_heads
        .view({batch, NUM_ATTENTION_HEADS, seq_len, HEAD_DIM});
    Tensor k = g_bufpool->k_heads
        .view({batch, NUM_KEY_VALUE_HEADS, seq_len, HEAD_DIM});
    Tensor v = g_bufpool->v_heads
        .view({batch, NUM_KEY_VALUE_HEADS, seq_len, HEAD_DIM});

    Tensor k_repeated = g_bufpool->k_repeated
       .view({batch, NUM_ATTENTION_HEADS, seq_len, HEAD_DIM});
    Tensor v_repeated = g_bufpool->v_repeated
        .view({batch, NUM_ATTENTION_HEADS, seq_len, HEAD_DIM});

    Tensor attn_out = g_bufpool->attn_output
        .view({batch, NUM_ATTENTION_HEADS, seq_len, HEAD_DIM});
    Tensor attn_flat = g_bufpool->attn_flat
        .view({flat_tokens, HIDDEN_SIZE});
    Tensor output_flat = g_bufpool->attn_proj_out
        .view({flat_tokens, HIDDEN_SIZE});

    // Project Q, K, V (multi-stream)
    tensor_ops::matmul_transposed(x_flat, q_proj_, q_proj_out, stream);
    tensor_ops::matmul_transposed(x_flat, k_proj_, k_proj_out, stream_k);
    tensor_ops::matmul_transposed(x_flat, v_proj_, v_proj_out, stream_v);

    // Reshape to (batch, seq, num_heads, head_dim) for layernorm (multi-stream)
    tensor_ops::reshape_for_layernorm(
        q_proj_out, q_reshaped, batch, seq_len, NUM_ATTENTION_HEADS, HEAD_DIM, stream);
    tensor_ops::reshape_for_layernorm(
        k_proj_out, k_reshaped, batch, seq_len, NUM_KEY_VALUE_HEADS, HEAD_DIM, stream_k);
    tensor_ops::reshape_for_layernorm(
        v_proj_out, v_reshaped, batch, seq_len, NUM_KEY_VALUE_HEADS, HEAD_DIM, stream_v);

    // Apply layernorm to Q and K (normalizes over last dim = head_dim)
    q_layernorm_->forward(q_reshaped, q_normed, stream);
    k_layernorm_->forward(k_reshaped, k_normed, stream_k);

    // Transpose to (batch, num_heads, seq_len, head_dim) for attention
    tensor_ops::reshape_to_heads(
        q_normed.view({batch * seq_len, NUM_ATTENTION_HEADS * HEAD_DIM}),
        q, batch, seq_len, NUM_ATTENTION_HEADS, HEAD_DIM, stream);
    tensor_ops::reshape_to_heads(
        k_normed.view({batch * seq_len, NUM_KEY_VALUE_HEADS * HEAD_DIM}),
        k, batch, seq_len, NUM_KEY_VALUE_HEADS, HEAD_DIM, stream_k);
    tensor_ops::reshape_to_heads(
        v_reshaped.view({batch * seq_len, NUM_KEY_VALUE_HEADS * HEAD_DIM}),
        v, batch, seq_len, NUM_KEY_VALUE_HEADS, HEAD_DIM, stream_v);

    // Record completion of K and V streams
    CHECK_CUDA(cudaEventRecord(event_k_done, stream_k));
    CHECK_CUDA(cudaEventRecord(event_v_done, stream_v));

    // Wait for K to be ready (Q is already on main stream)
    CHECK_CUDA(cudaStreamWaitEvent(stream, event_k_done, 0));

    // Apply RoPE
    tensor_ops::apply_rotary_pos_emb(q, k, cos, sin, stream);

    // Repeat K, V for GQA
    tensor_ops::repeat_kv(k, NUM_KEY_VALUE_GROUPS, k_repeated, stream);

    // Wait for V to be ready
    CHECK_CUDA(cudaStreamWaitEvent(stream, event_v_done, 0));
    tensor_ops::repeat_kv(v, NUM_KEY_VALUE_GROUPS, v_repeated, stream);

    // Compute attention
    float scale = 1.0f / std::sqrt((float)HEAD_DIM);
    tensor_ops::batched_attention(q, k_repeated, v_repeated, attn_out, scale, stream);

    // Reshape and project output
    tensor_ops::reshape_from_heads(attn_out, attn_flat, batch, seq_len, NUM_ATTENTION_HEADS, HEAD_DIM, stream);
    tensor_ops::matmul_transposed(attn_flat, o_proj_, output_flat, stream);

    // Copy result to output buffer
    if (output.size() == 0) {
        output = Tensor({batch, seq_len, hidden_size});
        output.to_device(g_active_device, stream);
    }
    CHECK_CUDA(cudaMemcpyAsync(output.device_data(), output_flat.device_data(),
               batch * seq_len * hidden_size * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    output.mark_device_dirty();
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

void ShortConv::forward(const Tensor& x, Tensor& y, cudaStream_t stream) {
    // x: (batch, seq_len, hidden_size)
    // Python: BCx = self.in_prog(x).transpose(-1, -2)
    // Result: (batch, 3*hidden_size, seq_len) for Conv1d

    size_t batch = x.size(0);
    size_t seq_len = x.size(1);
    size_t hidden_size = x.size(2);
    size_t kernel_size = conv_weight_.size(2);

    // Flatten for matmul
    Tensor x_flat = x.view({batch * seq_len, hidden_size});

    // Use buffer pool for intermediate tensors
    size_t flat_tokens = batch * seq_len;
    Tensor in_proj_out = g_bufpool->conv_in_proj
        .view({flat_tokens, 3 * hidden_size});
    Tensor conv_out = g_bufpool->conv_transposed
        .view({batch, seq_len, hidden_size});
    Tensor y_flat = g_bufpool->conv_proj_out
        .view({flat_tokens, hidden_size});

    // in_proj: (b*s, hidden) @ (3*hidden, hidden)^T -> (b*s, 3*hidden)
    tensor_ops::matmul_transposed(x_flat, in_proj_weight_, in_proj_out, stream);

    // Add bias if present
    if (USE_CONV_BIAS && in_proj_bias_.size() > 0) {
        tensor_ops::add_bias(in_proj_out, in_proj_bias_, in_proj_out, stream);
    }

    // Fused: transp_split + B*gate + conv1d + C*conv + transp
    // Output: (batch, seq, hidden)
    tensor_ops::shortconv_fused(
        in_proj_out, conv_weight_,
        USE_CONV_BIAS ? &conv_bias_ : nullptr,
        conv_out, batch, seq_len, hidden_size, kernel_size, stream);

    // out_proj: (b*s, hidden) @ (hidden, hidden)^T -> (b*s, hidden)
    Tensor conv_flat = conv_out.view({batch * seq_len, hidden_size});
    tensor_ops::matmul_transposed(
        conv_flat, out_proj_weight_, y_flat, stream);

    // Add bias if present
    if (USE_CONV_BIAS && out_proj_bias_.size() > 0) {
        tensor_ops::add_bias(y_flat, out_proj_bias_, y_flat, stream);
    }

    // Copy result to output buffer
    if (y.size() == 0) {
        y = Tensor({batch, seq_len, hidden_size});
        y.to_device(g_active_device, stream);
    }
    CHECK_CUDA(cudaMemcpyAsync(y.device_data(), y_flat.device_data(),
               batch * seq_len * hidden_size * sizeof(float),
               cudaMemcpyDeviceToDevice, stream));
    y.mark_device_dirty();
}

// DecoderLayer implementation
DecoderLayer::DecoderLayer(
    int layer_idx, bool is_attention_layer, int device_id)
    : layer_idx_(layer_idx), is_attention_layer_(is_attention_layer),
      device_id_(device_id) {

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
    if (layer_idx >= static_cast<int>(NUM_DENSE_LAYERS)) {
        moe_block_ = std::make_unique<SparseMoeBlock>(layer_idx);
    } else {
        // Dense layer - load simple MLP
        std::stringstream ss_w1, ss_w2, ss_w3;
        ss_w1 << "layers." << layer_idx << ".feed_forward.w1.weight";
        ss_w2 << "layers." << layer_idx << ".feed_forward.w2.weight";
        ss_w3 << "layers." << layer_idx << ".feed_forward.w3.weight";
        dense_mlp_ = std::make_unique<MLP>(
            ss_w1.str(), ss_w2.str(), ss_w3.str());
    }
}

void DecoderLayer::forward(
    const Tensor& x, const Tensor& cos, const Tensor& sin,
    const Tensor* attention_mask, Tensor& output, cudaStream_t stream) {

    size_t batch = x.size(0);
    size_t seq_len = x.size(1);
    size_t hidden_size = x.size(2);

    // Use buffer pool for intermediate tensors
    Tensor normed_input = g_bufpool->layer_normed_input
        .view({batch, seq_len, hidden_size});
    Tensor attn_output = g_bufpool->layer_attn_out
        .view({batch, seq_len, hidden_size});
    Tensor hidden_states = g_bufpool->layer_hidden
        .view({batch, seq_len, hidden_size});
    Tensor normed_hidden = g_bufpool->layer_normed_hidden
        .view({batch, seq_len, hidden_size});

    // Input norm
    input_layernorm_->forward(x, normed_input, stream);

    // Attention or Conv
    if (is_attention_layer_) {
        self_attn_->forward(
            normed_input, cos, sin, attention_mask, attn_output, stream);
    } else {
        short_conv_->forward(
            normed_input, attn_output, stream);
    }

    // Residual connection
    tensor_ops::add(x, attn_output, hidden_states, stream);

    // Post attention norm
    post_attention_layernorm_->forward(hidden_states, normed_hidden, stream);

    // MoE block or dense MLP
    Tensor ffn_output;
    if (moe_block_) {
        // MoE layer (layers >= 2)
        Tensor router_logits;
        moe_block_->forward(normed_hidden, ffn_output, router_logits, stream);
    } else {
        // Dense layer (layers 0-1)
        dense_mlp_->forward(normed_hidden, ffn_output, stream, true);
    }

    // Residual connection - output needs to be a new tensor for caller
    output = Tensor({batch, seq_len, hidden_size});
    output.to_device(g_active_device, stream);
    tensor_ops::add(hidden_states, ffn_output, output, stream);
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
    DEBUG_PRINTLN("Loading LFM2-8B-A1B model from " << model_file);

    // Initialize global model loader
    g_model_loader = std::make_unique<ModelLoader>(model_file);

    load_embeddings();
    load_layers();
    load_output_layers();

    // Initialize RoPE
    rotary_emb_ = std::make_unique<RotaryEmbedding>();

    DEBUG_PRINTLN("Model loaded successfully!");
}

LFM2Model::~LFM2Model() {
    cleanup_stage_resources();
}

void LFM2Model::load_embeddings() {
    DEBUG_PRINTLN("Loading embeddings...");
    embed_tokens_ = Tensor::load_from_file("embed_tokens.weight");
    DEBUG_PRINTLN(
        "  Embeddings shape: " << embed_tokens_.size(0)
        << " x " << embed_tokens_.size(1));
}

void LFM2Model::load_layers() {
    DEBUG_PRINTLN("Loading " << NUM_HIDDEN_LAYERS << " decoder layers...");

    // Read layer types from config.h LAYER_TYPES array
    // 0 = full_attention, 1 = con
    layers_.reserve(NUM_HIDDEN_LAYERS);
    int device_id = 0;
    int num_layers_curr_stage = 0;

    for (size_t i = 0; i < NUM_HIDDEN_LAYERS; i++) {
        bool is_attention = (LAYER_TYPES[i] == 0);
        num_layers_curr_stage++;
        if (num_layers_curr_stage > layers_per_stage_[device_id]) {
            device_id++;
            num_layers_curr_stage = 1;
        }

        DEBUG_PRINTLN(
            "  Layer " << i << ": " << (is_attention ? "Attention" : "Conv"));
        layers_.push_back(
            std::make_unique<DecoderLayer>(i, is_attention, device_id));
    }
}

void LFM2Model::load_output_layers() {
    DEBUG_PRINTLN("Loading output layers...");

    norm_ = std::make_unique<RMSNorm>("embedding_norm.weight");

    // LM head might share weights with embeddings
    if (g_model_loader->has_tensor("lm_head.weight")) {
        lm_head_ = Tensor::load_from_file("lm_head.weight");
    } else {
        // Use tied weights (same as embeddings)
        lm_head_ = embed_tokens_;
        DEBUG_PRINTLN("  Using tied weights for LM head");
    }
}

void LFM2Model::forward(
    const std::vector<int>& input_ids, size_t num_samples,
    size_t seq_len, Tensor& logits, cudaStream_t) {

    DEBUG_PRINTLN("\nForward pass: num_samples=" << num_samples << ", seq_len=" << seq_len);

    // Define batch and stage structures
    struct BatchState {
        size_t id = 0;
        size_t offset = 0;
        size_t size = 0;
        double start_time = 0.0;
        Tensor hidden;
    };

    struct StageConfig {
        int device_id = 0;
        std::vector<DecoderLayer*> layers;
        cudaStream_t stream = nullptr;
        Tensor cos;
        Tensor sin;
    };

    size_t batch_size = std::max<size_t>(1, num_samples / NUM_BATCH);
    batch_size = std::min(batch_size, num_samples);
    size_t num_batch = (num_samples + batch_size - 1) / batch_size;

    // Prepare batches
    std::vector<BatchState> batches(num_batch);
    for (size_t b = 0; b < num_batch; ++b) {
        size_t start = b * batch_size;
        size_t size = std::min(batch_size, num_samples - start);
        batches[b].id = b;
        batches[b].offset = start;
        batches[b].size = size;
    }

    // Distribute layers to stages
    std::vector<std::vector<DecoderLayer*>> stage_layers(NUM_GPUS);
    for (auto& layer : layers_) {
        size_t stage_idx = static_cast<size_t>(layer->device_id());
        if (stage_idx >= NUM_GPUS) stage_idx = NUM_GPUS - 1;
        stage_layers[stage_idx].push_back(layer.get());
    }

    // Prepare stages
    std::vector<StageConfig> stages(NUM_GPUS);
    for (size_t idx = 0; idx < NUM_GPUS; ++idx) {
        stages[idx].device_id = static_cast<int>(idx);
        stages[idx].layers = stage_layers[idx];
        CHECK_CUDA(cudaSetDevice(stages[idx].device_id));
        CHECK_CUDA(cudaStreamCreateWithFlags(
            &stages[idx].stream, cudaStreamNonBlocking));
    }

    // Prepare RoPE for each stage
    set_stage_resources(stages.front().device_id, num_samples, seq_len);
    Tensor cos_ref({seq_len, HEAD_DIM}); // reference cos
    Tensor sin_ref({seq_len, HEAD_DIM}); // reference sin
    cos_ref.to_device(stages.front().device_id);
    sin_ref.to_device(stages.front().device_id);
    rotary_emb_->forward(seq_len, cos_ref, sin_ref, 0);
    cos_ref.to_host();
    sin_ref.to_host();

    for (StageConfig& stage : stages) {
        stage.cos = cos_ref;
        stage.sin = sin_ref;
        stage.cos.to_device(stage.device_id);
        stage.sin.to_device(stage.device_id);
    }

    // Input IDs buffer
    const int first_device = stages.front().device_id; // 0
    set_stage_resources(first_device, num_samples, seq_len);
    embed_tokens_.to_device(first_device);
    size_t max_tokens = batch_size * seq_len;
    int* d_input_ids = nullptr;
    CHECK_CUDA(cudaMalloc(&d_input_ids, max_tokens * sizeof(int)));

    // Output logits buffer
    logits = Tensor({num_samples, VOCAB_SIZE});
    logits.to_host();
    float* logits_host = logits.data();

    // Stage synchronization gates
    std::vector<int> stage_gate(num_batch, 0);

    // Pipeline execution: each stage in its own thread (using OpenMP)
    #pragma omp parallel num_threads(NUM_GPUS)
    {
        int stage_idx = omp_get_thread_num();
        StageConfig& stage = stages[stage_idx];
        set_stage_resources(stage.device_id, num_samples, seq_len);

        for (size_t b = 0; b < num_batch; ++b) {
            while (true) {
                int gate_val = 0;
                #pragma omp atomic read
                gate_val = stage_gate[b];
                if (gate_val == stage_idx) break; // busy wait
            }

            BatchState& batch = batches[b];
            cudaStream_t stream = stage.stream;

            // Preprocessing: embedding lookup or receive from previous stage
            if (stage_idx == 0) {
                // First stage: embedding lookup
                batch.start_time = get_time_layer();
                size_t tokens = batch.size * seq_len;

                // Copy input IDs to device
                const int* host_ptr = input_ids.data() + batch.offset * seq_len;
                CHECK_CUDA(cudaMemcpyAsync(
                    d_input_ids, host_ptr, tokens * sizeof(int),
                    cudaMemcpyHostToDevice, stream));

                Tensor output_emb({batch.size, seq_len, HIDDEN_SIZE});
                output_emb.to_device(first_device, stream);

                // Launch embedding lookup kernel
                size_t total = tokens * HIDDEN_SIZE;
                dim3 block(BLOCK_DEFAULT);
                dim3 grid((total + BLOCK_DEFAULT - 1) / BLOCK_DEFAULT);
                embedding_lookup_kernel<<<grid, block, 0, stream>>>(
                    d_input_ids, embed_tokens_.device_data(),
                    output_emb.device_data(), tokens, HIDDEN_SIZE);
                output_emb.mark_device_dirty();
                batch.hidden = std::move(output_emb);
            } else {
                // Receive hidden(output) from previous stage
                batch.hidden.to_device(stage.device_id, stream);
            }

            // Process all layers in this stage
            Tensor hidden = std::move(batch.hidden);
            for (DecoderLayer* layer : stage.layers) {
                Tensor output({batch.size, seq_len, HIDDEN_SIZE});
                layer->forward(
                    hidden, stage.cos, stage.sin,
                    nullptr, output, stream);
                hidden = std::move(output);
            }
            batch.hidden = std::move(hidden);

            // Poseprocessing: send to next stage or compute logits
            if (stage_idx + 1 < static_cast<int>(NUM_GPUS)) {
                // Send hidden to next stage
                batch.hidden.to_device(stages[stage_idx + 1].device_id, stream);
                CHECK_CUDA(cudaStreamSynchronize(stream));
            } else {
                // Compute logits if last stage
                // Apply final layer norm
                Tensor normed({batch.size, seq_len, HIDDEN_SIZE});
                norm_->forward(batch.hidden, normed, stream);

                // Extract last token hidden states
                Tensor last_hidden({batch.size, HIDDEN_SIZE});
                last_hidden.to_device(stage.device_id, stream);
                size_t total = batch.size * HIDDEN_SIZE;
                dim3 block_last(BLOCK_DEFAULT);
                dim3 grid_last((total + BLOCK_DEFAULT - 1) / BLOCK_DEFAULT);
                extract_last_token_kernel<<<grid_last, block_last, 0, stream>>>(
                    normed.device_data(), last_hidden.device_data(),
                    batch.size, seq_len, HIDDEN_SIZE);
                last_hidden.mark_device_dirty();

                // Compute logits: last_hidden @ lm_head_^T
                Tensor logits_batch({batch.size, VOCAB_SIZE});
                tensor_ops::matmul_transposed(
                    last_hidden, lm_head_, logits_batch, stream);
                logits_batch.to_host(stream);
                CHECK_CUDA(cudaStreamSynchronize(stream));

                size_t offset = batch.offset * VOCAB_SIZE;
                std::memcpy(
                    logits_host + offset, logits_batch.data(),
                    batch.size * VOCAB_SIZE * sizeof(float));
                double elapsed = get_time_layer() - batch.start_time;
                #pragma omp critical
                {
                DEBUG_PRINTLN(
                    "[PP] batch " << batch.id << ": " << elapsed
                    << "s (offset=" << batch.offset << ")");
                }
                batch.hidden = Tensor();
            }

            // Signal next stage
            #pragma omp atomic write
            stage_gate[b] = stage_idx + 1;
        }
    }

    // Cleanup
    for (auto& stage : stages) {
        if (stage.stream) {
            CHECK_CUDA(cudaSetDevice(stage.device_id));
            cudaStreamDestroy(stage.stream);
        }
    }
    if (d_input_ids) {
        CHECK_CUDA(cudaSetDevice(first_device));
        cudaFree(d_input_ids);
    }
}
