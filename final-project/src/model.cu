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

// Global MPI rank for conditional debug output (set from main.cpp)
int g_mpi_rank = 0;

// Debug print macro - only rank 0 prints
#define DEBUG_PRINT(x) do { if (g_mpi_rank == 0) { std::cout << x; } } while(0)
#define DEBUG_PRINTLN(x) do { if (g_mpi_rank == 0) { std::cout << x << std::endl; } } while(0)

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

// ============================================================================
// Large Block Implementations - Complex layers and modules
// ============================================================================

// Global resources for multi-stream (to avoid creating thousands of streams)
static cudaStream_t g_shared_streams[NUM_GPUS][2]; // 0: aux1 (MLP/Attn-K), 1: aux2 (Attn-V)
static cudaEvent_t g_shared_events[NUM_GPUS][3]; // 0: main_ready, 1: aux1_done, 2: aux2_done
static bool g_shared_resources_init = false;

void init_shared_resources() {
    if (g_shared_resources_init) return;

    int prev_device;
    CHECK_CUDA(cudaGetDevice(&prev_device));

    // make global streams per GPUs
    for (int i = 0; i < NUM_GPUS; ++i) {
        CHECK_CUDA(cudaSetDevice(i));
        for (int j = 0; j < 2; ++j) {
            CHECK_CUDA(cudaStreamCreateWithFlags(
                &g_shared_streams[i][j], cudaStreamNonBlocking));
        }
        for (int j = 0; j < 3; ++j) {
            CHECK_CUDA(cudaEventCreateWithFlags(
                &g_shared_events[i][j], cudaEventDisableTiming));
        }
    }

    CHECK_CUDA(cudaSetDevice(prev_device));
    g_shared_resources_init = true;
}

void cleanup_shared_resources() {
    if (!g_shared_resources_init) return;

    for (int i = 0; i < NUM_GPUS; ++i) {
        // No need to set device for destroy usually, but safer
        for (int j = 0; j < 2; ++j) cudaStreamDestroy(g_shared_streams[i][j]);
        for (int j = 0; j < 3; ++j) cudaEventDestroy(g_shared_events[i][j]);
    }
    g_shared_resources_init = false;
}

// MLP (Feed-Forward Network) implementation
MLP::MLP(
    const std::string& w1_file, const std::string& w2_file,
    const std::string& w3_file, int device_id)
    : device_id_(device_id) {
    w1_ = Tensor::load_from_file(w1_file);
    w2_ = Tensor::load_from_file(w2_file);
    w3_ = Tensor::load_from_file(w3_file);
}

MLP::~MLP() {
}

void MLP::forward(const Tensor& x, Tensor& y, cudaStream_t stream) {
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

    // shared resources for current device
    int dev = device_id_;
    cudaStream_t stream_aux = g_shared_streams[dev][0];
    cudaEvent_t event_main_ready = g_shared_events[dev][0];
    cudaEvent_t event_aux_done = g_shared_events[dev][1];
    
    // Record event to ensure input data is ready for aux stream
    CHECK_CUDA(cudaEventRecord(event_main_ready, stream));
    CHECK_CUDA(cudaStreamWaitEvent(stream_aux, event_main_ready, 0));

    // gate = silu(x @ w1.T)
    Tensor gate({batch * seq_len, intermediate_size});
    tensor_ops::matmul_transposed(x_flat, w1_, gate, stream);
    Tensor gate_silu({batch * seq_len, intermediate_size});
    tensor_ops::silu(gate, gate_silu, stream);
    
    // up = x @ w3.T (Aux stream)
    Tensor up({batch * seq_len, intermediate_size});
    tensor_ops::matmul_transposed(x_flat, w3_, up, stream_aux);
    
    // Record completion of aux stream
    CHECK_CUDA(cudaEventRecord(event_aux_done, stream_aux));
    
    // Wait for aux stream to finish before using up
    CHECK_CUDA(cudaStreamWaitEvent(stream, event_aux_done, 0));

    // hidden = gate_silu * up
    Tensor hidden({batch * seq_len, intermediate_size});
    tensor_ops::mul(gate_silu, up, hidden, stream);
    
    // y = hidden @ w2.T
    Tensor y_flat({batch * seq_len, hidden_size});
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

        // allocate experts for expert parallelism
        int expert_device = i % NUM_GPUS;
        experts_.push_back(std::make_unique<MLP>(
            ss_w1.str(), ss_w2.str(), ss_w3.str(), expert_device));
    }

    // Assign experts to 4 GPUs in a round-robin fashion (expert parallelism)
    expert_devices_.resize(NUM_EXPERTS);
    for (size_t i = 0; i < NUM_EXPERTS; i++) {
        expert_devices_[i] = static_cast<int>(i % NUM_GPUS);
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
    // router_logits: (batch * seq_len, num_experts)
    size_t num_tokens = router_logits.size(0);

    top_k_indices.resize(num_tokens * NUM_EXPERTS_PER_TOK);
    top_k_weights.resize(num_tokens * NUM_EXPERTS_PER_TOK);

    // Ensure router_logits is on device
    router_logits.to_device(-1, stream);

    // Allocate device buffers & pointer
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

    // Synchronize for bucketing in host
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // Free temporary device buffers
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
    const int base_device = 0;

    // Flatten
    Tensor x_flat = x.view({num_tokens, hidden_size});
    x_flat.to_device(base_device, stream);

    // Compute router logits (base device)
    CHECK_CUDA(cudaSetDevice(base_device));
    router_logits = Tensor({num_tokens, NUM_EXPERTS});
    tensor_ops::matmul_transposed(x_flat, gate_, router_logits, stream);

    // Route tokens (base device)
    std::vector<int> top_k_indices;
    std::vector<float> top_k_weights;
    route_tokens(router_logits, top_k_indices, top_k_weights, stream);

    // Bucket tokens per expert (host)
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
    std::vector<std::vector<size_t>> experts_on_device(NUM_GPUS);
    for (size_t e = 0; e < NUM_EXPERTS; e++) {
        int dev = expert_devices_[e];
        if (tokens_per_expert[e].size() > 0) {
            experts_on_device[dev].push_back(e);
        }
    }

    // Copy x_flat to GPUs
    std::vector<Tensor> x_flat_per_gpu(NUM_GPUS);
    for (int dev = 0; dev < NUM_GPUS; dev++) {
        if (!experts_on_device[dev].empty()) {
            x_flat_per_gpu[dev] = x_flat.copy();
            x_flat_per_gpu[dev].to_device(dev, stream);
        }
    }

    // Expert outputs and token indices per GPU
    std::vector<std::vector<Tensor>> expert_outputs(NUM_GPUS);
    std::vector<std::vector<std::vector<size_t>>> expert_token_lists(NUM_GPUS);
    std::vector<std::vector<std::vector<float>>> expert_weight_lists(NUM_GPUS);

    for (int dev = 0; dev < NUM_GPUS; dev++) {
        expert_outputs[dev].resize(experts_on_device[dev].size());
        expert_token_lists[dev].resize(experts_on_device[dev].size());
        expert_weight_lists[dev].resize(experts_on_device[dev].size());
    }

    // Parallel execution across 4 GPUs using OpenMP
    #pragma omp parallel num_threads(NUM_GPUS)
    {
        int dev = omp_get_thread_num();
        CHECK_CUDA(cudaSetDevice(dev));

        // Create a local stream for current GPU
        cudaStream_t local_stream;
        CHECK_CUDA(cudaStreamCreateWithFlags(&local_stream, cudaStreamNonBlocking));

        std::vector<int*> ptrs_to_free;

        // Process each expert on current GPU
        for (size_t i = 0; i < experts_on_device[dev].size(); i++) {
            size_t expert_idx = experts_on_device[dev][i];
            const auto& token_list = tokens_per_expert[expert_idx];
            const auto& weight_list = weights_per_expert[expert_idx];
            size_t tok_count = token_list.size();

            if (tok_count == 0) continue;

            // Store token list and weights (for scatter)
            expert_token_lists[dev][i] = token_list;
            expert_weight_lists[dev][i] = weight_list;

            // Expert input on GPU
            Tensor expert_in({tok_count, 1, hidden_size});
            expert_in.to_device(dev, local_stream);

            // Gather expert input with kernel
            const float* src = x_flat_per_gpu[dev].device_data();
            float* dst = expert_in.device_data();
            int* d_indices;
            CHECK_CUDA(cudaMalloc(&d_indices, tok_count * sizeof(int)));

            std::vector<int> indices_vec(tok_count);
            for(size_t k=0; k<tok_count; ++k) indices_vec[k] = (int)token_list[k];

            CHECK_CUDA(cudaMemcpyAsync(
                d_indices, indices_vec.data(), tok_count * sizeof(int),
                cudaMemcpyHostToDevice, local_stream));

            size_t total_elements = tok_count * hidden_size;
            dim3 block(256);
            dim3 grid((total_elements + 255) / 256);

            gather_expert_input_kernel<<<grid, block, 0, local_stream>>>(
                src, dst, d_indices, tok_count, hidden_size
            );

            ptrs_to_free.push_back(d_indices);
            expert_in.mark_device_dirty();

            // Execute expert
            Tensor expert_out;
            experts_[expert_idx]->forward(expert_in, expert_out, local_stream);

            // Keep output on current GPU
            expert_outputs[dev][i] = std::move(expert_out);
        }
        
        CHECK_CUDA(cudaStreamSynchronize(local_stream));
        for(int* ptr : ptrs_to_free) CHECK_CUDA(cudaFree(ptr));
        CHECK_CUDA(cudaStreamDestroy(local_stream));
    }

    // Initialize output tensor (base device)
    y = Tensor({batch, seq_len, hidden_size});
    y.to_device(base_device);
    CHECK_CUDA(cudaSetDevice(base_device));
    CHECK_CUDA(cudaMemset(
        y.device_data(), 0, num_tokens * hidden_size * sizeof(float)));
    y.mark_device_dirty();

    // Accumulate results using kernel
    for (int dev = 0; dev < NUM_GPUS; dev++) {
        for (size_t i = 0; i < experts_on_device[dev].size(); i++) {
            const auto& token_list = expert_token_lists[dev][i];
            const auto& weight_list = expert_weight_lists[dev][i];
            Tensor& expert_out = expert_outputs[dev][i];

            if (expert_out.size() == 0) continue;

            size_t tok_count = token_list.size();

            // Copy expert_out to base device if on different GPU
            if (dev != base_device) {
                expert_out.to_device(base_device, stream);
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
            CHECK_CUDA(cudaMemcpyAsync(
                d_token_ids, token_ids_vec.data(),
                tok_count * sizeof(int), cudaMemcpyHostToDevice, stream));
            CHECK_CUDA(cudaMemcpyAsync(
                d_weights, weight_list.data(),
                tok_count * sizeof(float), cudaMemcpyHostToDevice, stream));

            // Launch scatter-add kernel
            size_t total = tok_count * hidden_size;
            dim3 block(256);
            dim3 grid((total + 255) / 256);

            weighted_scatter_add_kernel<<<grid, block, 0, stream>>>(
                y.device_data(), expert_out.device_data(),
                d_token_ids, d_weights,
                tok_count, hidden_size);

            // Wait for kernel to finish -> free memory
            CHECK_CUDA(cudaStreamSynchronize(stream));
            CHECK_CUDA(cudaFree(d_token_ids));
            CHECK_CUDA(cudaFree(d_weights));
        }
    }

    // Keep on base device
    CHECK_CUDA(cudaSetDevice(base_device));
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

void Attention::forward(const Tensor& x, const Tensor& cos, const Tensor& sin,
                       const Tensor* attention_mask, Tensor& output, cudaStream_t stream) {
    // x: (batch, seq_len, hidden_size)
    size_t batch = x.size(0);
    size_t seq_len = x.size(1);
    size_t hidden_size = x.size(2);

    // Flatten
    Tensor x_flat = x.view({batch * seq_len, hidden_size});

    // Get shared resources for this device
    int dev = q_proj_.device_id();
    cudaStream_t stream_k = g_shared_streams[dev][0];
    cudaStream_t stream_v = g_shared_streams[dev][1];
    cudaEvent_t event_main_ready = g_shared_events[dev][0];
    cudaEvent_t event_k_done = g_shared_events[dev][1];
    cudaEvent_t event_v_done = g_shared_events[dev][2];

    // Record event to ensure input data is ready for aux streams
    CHECK_CUDA(cudaEventRecord(event_main_ready, stream));
    CHECK_CUDA(cudaStreamWaitEvent(stream_k, event_main_ready, 0));
    CHECK_CUDA(cudaStreamWaitEvent(stream_v, event_main_ready, 0));

    // Project Q, K, V (multi-stream)
    Tensor q_proj_out({batch * seq_len, NUM_ATTENTION_HEADS * HEAD_DIM});
    Tensor k_proj_out({batch * seq_len, NUM_KEY_VALUE_HEADS * HEAD_DIM});
    Tensor v_proj_out({batch * seq_len, NUM_KEY_VALUE_HEADS * HEAD_DIM});

    tensor_ops::matmul_transposed(x_flat, q_proj_, q_proj_out, stream);
    tensor_ops::matmul_transposed(x_flat, k_proj_, k_proj_out, stream_k);
    tensor_ops::matmul_transposed(x_flat, v_proj_, v_proj_out, stream_v);

    // Reshape to (batch, num_heads, seq, head_dim) for layernorm (multi-stream)
    Tensor q_reshaped({batch, seq_len, NUM_ATTENTION_HEADS, HEAD_DIM});
    Tensor k_reshaped({batch, seq_len, NUM_KEY_VALUE_HEADS, HEAD_DIM});
    Tensor v_reshaped({batch, seq_len, NUM_KEY_VALUE_HEADS, HEAD_DIM});

    tensor_ops::reshape_for_layernorm(q_proj_out, q_reshaped, batch, seq_len, NUM_ATTENTION_HEADS, HEAD_DIM, stream);
    tensor_ops::reshape_for_layernorm(k_proj_out, k_reshaped, batch, seq_len, NUM_KEY_VALUE_HEADS, HEAD_DIM, stream_k);
    tensor_ops::reshape_for_layernorm(v_proj_out, v_reshaped, batch, seq_len, NUM_KEY_VALUE_HEADS, HEAD_DIM, stream_v);

    // Apply layernorm to Q and K (normalizes over last dim = head_dim)
    Tensor q_normed({batch, seq_len, NUM_ATTENTION_HEADS, HEAD_DIM});
    Tensor k_normed({batch, seq_len, NUM_KEY_VALUE_HEADS, HEAD_DIM});
    q_layernorm_->forward(q_reshaped, q_normed, stream);
    k_layernorm_->forward(k_reshaped, k_normed, stream_k);

    // Transpose to (batch, num_heads, seq_len, head_dim) for attention
    Tensor q({batch, NUM_ATTENTION_HEADS, seq_len, HEAD_DIM});
    Tensor k({batch, NUM_KEY_VALUE_HEADS, seq_len, HEAD_DIM});
    Tensor v({batch, NUM_KEY_VALUE_HEADS, seq_len, HEAD_DIM});

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
    Tensor k_repeated({batch, NUM_ATTENTION_HEADS, seq_len, HEAD_DIM});
    Tensor v_repeated({batch, NUM_ATTENTION_HEADS, seq_len, HEAD_DIM});
    tensor_ops::repeat_kv(k, NUM_KEY_VALUE_GROUPS, k_repeated, stream);
    
    // Wait for V to be ready
    CHECK_CUDA(cudaStreamWaitEvent(stream, event_v_done, 0));
    tensor_ops::repeat_kv(v, NUM_KEY_VALUE_GROUPS, v_repeated, stream);

    // Compute attention
    float scale = 1.0f / std::sqrt((float)HEAD_DIM);
    Tensor attn_output({batch, NUM_ATTENTION_HEADS, seq_len, HEAD_DIM});
    tensor_ops::batched_attention(q, k_repeated, v_repeated, attn_output, scale, stream);

    // Reshape and project output
    Tensor attn_flat({batch * seq_len, hidden_size});
    tensor_ops::reshape_from_heads(attn_output, attn_flat, batch, seq_len, NUM_ATTENTION_HEADS, HEAD_DIM, stream);

    Tensor output_flat({batch * seq_len, hidden_size});
    tensor_ops::matmul_transposed(attn_flat, o_proj_, output_flat, stream);

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

void ShortConv::forward(const Tensor& x, Tensor& y, cudaStream_t stream) {
    // x: (batch, seq_len, hidden_size)
    // Python: BCx = self.in_proj(x).transpose(-1, -2)
    // Result: (batch, 3*hidden_size, seq_len) for Conv1d

    size_t batch = x.size(0);
    size_t seq_len = x.size(1);
    size_t hidden_size = x.size(2);

    // Flatten for matmul
    Tensor x_flat = x.view({batch * seq_len, hidden_size});

    // in_proj: (b*s, hidden) @ (3*hidden, hidden)^T -> (b*s, 3*hidden)
    Tensor in_proj_out({batch * seq_len, 3 * hidden_size});
    tensor_ops::matmul_transposed(x_flat, in_proj_weight_, in_proj_out, stream);

    // Add bias if present
    if (USE_CONV_BIAS && in_proj_bias_.size() > 0) {
        tensor_ops::add_bias(in_proj_out, in_proj_bias_, in_proj_out, stream);
    }

    // Reshape, transpose & split: (b*s, 3*hidden) -> B, C, x_gate (b, hidden, s)
    Tensor B({batch, hidden_size, seq_len});
    Tensor C({batch, hidden_size, seq_len});
    Tensor x_gate({batch, hidden_size, seq_len});
    tensor_ops::transpose_split_BCx(
        in_proj_out, B, C, x_gate, batch, seq_len, hidden_size, stream);

    // Bx = B * x_gate (element-wise)
    Tensor Bx({batch, hidden_size, seq_len});
    tensor_ops::mul(B, x_gate, Bx, stream);

    // Apply causal conv1d on Bx (expects: batch, channels, seq_len)
    Tensor conv_out({batch, hidden_size, seq_len});
    tensor_ops::causal_conv1d(
        Bx, conv_weight_,
        USE_CONV_BIAS ? &conv_bias_ : nullptr, conv_out, stream);

    // y_pre = C * conv_out (element-wise)
    Tensor y_pre({batch, hidden_size, seq_len});
    tensor_ops::mul(C, conv_out, y_pre, stream);

    // Transpose back: (b, hidden, s) -> (b, s, hidden)
    Tensor y_pre_transposed({batch, seq_len, hidden_size});
    tensor_ops::transpose_hidden_seq(
        y_pre, y_pre_transposed, batch, hidden_size, seq_len, stream);

    // out_proj: (b*s, hidden) @ (hidden, hidden)^T -> (b*s, hidden)
    Tensor y_pre_flat = y_pre_transposed.view({batch * seq_len, hidden_size});
    Tensor y_flat({batch * seq_len, hidden_size});
    tensor_ops::matmul_transposed(y_pre_flat, out_proj_weight_, y_flat, stream);

    // Add bias if present
    if (USE_CONV_BIAS && out_proj_bias_.size() > 0) {
        tensor_ops::add_bias(y_flat, out_proj_bias_, y_flat, stream);
    }

    // Reshape back to (batch, seq_len, hidden_size)
    y_flat.reshape({batch, seq_len, hidden_size});
    y = std::move(y_flat);
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
    if (static_cast<size_t>(layer_idx) >= NUM_DENSE_LAYERS) {
        moe_block_ = std::make_unique<SparseMoeBlock>(layer_idx);
    } else {
        // Dense layer - load simple MLP
        std::stringstream ss_w1, ss_w2, ss_w3;
        ss_w1 << "layers." << layer_idx << ".feed_forward.w1.weight";
        ss_w2 << "layers." << layer_idx << ".feed_forward.w2.weight";
        ss_w3 << "layers." << layer_idx << ".feed_forward.w3.weight";
        dense_mlp_ = std::make_unique<MLP>(
            ss_w1.str(), ss_w2.str(), ss_w3.str(), device_id_);
    }
}

void DecoderLayer::forward(
    const Tensor& x, const Tensor& cos, const Tensor& sin,
    const Tensor* attention_mask, Tensor& output, cudaStream_t stream) {
    // Input norm
    Tensor normed_input(x.shape());
    input_layernorm_->forward(x, normed_input, stream);
    
    // Attention or Conv
    Tensor attn_output(x.shape());
    if (is_attention_layer_) {
        self_attn_->forward(
            normed_input, cos, sin, attention_mask, attn_output, stream);
    } else {
        short_conv_->forward(
            normed_input, attn_output, stream);
    }
    
    // Residual connection
    Tensor hidden_states(x.shape());
    tensor_ops::add(x, attn_output, hidden_states, stream);
    
    // Post attention norm
    Tensor normed_hidden(x.shape());
    post_attention_layernorm_->forward(hidden_states, normed_hidden, stream);
    
    // MoE block or dense MLP
    Tensor ffn_output;
    if (moe_block_) {
        // MoE layer (layers >= 2) -> expert parallelism
        Tensor router_logits;
        moe_block_->forward(normed_hidden, ffn_output, router_logits, stream);
    } else {
        // Dense layer (layers 0-1)
        dense_mlp_->forward(normed_hidden, ffn_output, stream);
    }
    
    // Residual connection
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

    // Initialize global shared resources for multi-stream execution
    init_shared_resources();

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
    cleanup_shared_resources();
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
    // 0 = full_attention, 1 = conv
    layers_.reserve(NUM_HIDDEN_LAYERS);
    for (size_t i = 0; i < NUM_HIDDEN_LAYERS; i++) {
        bool is_attention = (LAYER_TYPES[i] == 0);
        // Keep non-MoE work on a base device (0)
        // Expert parallelism spreads work across devices inside MoE blocks.
        int device_id = 0;
        DEBUG_PRINTLN(
            "  Layer " << i << ": " << (is_attention ? "Attention" : "Conv"));
        layers_.push_back(std::make_unique<DecoderLayer>(
            i, is_attention, device_id));
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

void LFM2Model::forward(const std::vector<int>& input_ids, size_t batch, size_t seq_len, Tensor& logits, cudaStream_t stream) {
    if (batch == 0 || seq_len == 0 || input_ids.size() != batch * seq_len) {
        throw std::runtime_error("Invalid batch/seq_len for forward");
    }

    DEBUG_PRINTLN("\nForward pass: batch=" << batch << ", seq_len=" << seq_len);

    size_t num_tokens = batch * seq_len;

    // Embedding lookup (base device)
    CHECK_CUDA(cudaSetDevice(0));

    // Copy input_ids to device
    int* d_input_ids;
    CHECK_CUDA(cudaMalloc(&d_input_ids, num_tokens * sizeof(int)));
    CHECK_CUDA(cudaMemcpyAsync(
        d_input_ids, input_ids.data(), num_tokens * sizeof(int),
        cudaMemcpyHostToDevice, stream));

    embed_tokens_.to_device(0, stream);
    Tensor hidden_states({batch, seq_len, HIDDEN_SIZE});
    hidden_states.to_device(0, stream);

    // Launch kernel
    size_t total = num_tokens * HIDDEN_SIZE;
    dim3 block(256);
    dim3 grid((total + 255) / 256);

    embedding_lookup_kernel<<<grid, block, 0, stream>>>(
        d_input_ids, embed_tokens_.device_data(), hidden_states.device_data(),
        num_tokens, HIDDEN_SIZE);

    CHECK_CUDA(cudaFree(d_input_ids));
    hidden_states.mark_device_dirty();

    // Compute RoPE embeddings
    Tensor cos({seq_len, HEAD_DIM});
    Tensor sin({seq_len, HEAD_DIM});
    cos.to_device(0, stream);
    sin.to_device(0, stream);
    rotary_emb_->forward(seq_len, cos, sin, stream);

    // Create causal attention mask (not strictly needed for CPU impl)
    Tensor* attention_mask = nullptr;

    // Keep activations on base device
    CHECK_CUDA(cudaSetDevice(0));
    hidden_states.to_device(0, stream);

    for (size_t i = 0; i < NUM_HIDDEN_LAYERS; i++) {
        double start_time = get_time_layer();

        Tensor output({batch, seq_len, HIDDEN_SIZE});
        layers_[i]->forward(hidden_states, cos, sin, attention_mask, output, stream);
        hidden_states = std::move(output);

        double end_time = get_time_layer();
        DEBUG_PRINTLN(
            "  Layer " << i << ": " << (end_time - start_time) << " seconds");
    }

    CHECK_CUDA(cudaSetDevice(0));

    // Final norm (base device)
    Tensor normed_output({batch, seq_len, HIDDEN_SIZE});
    norm_->forward(hidden_states, normed_output, stream);

    // Extract last token
    Tensor last_hidden({batch, HIDDEN_SIZE});
    last_hidden.to_device(0, stream);

    size_t total_last = batch * HIDDEN_SIZE;
    dim3 block_last(256);
    dim3 grid_last((total_last + 255) / 256);

    extract_last_token_kernel<<<grid_last, block_last, 0, stream>>>(
        normed_output.device_data(), last_hidden.device_data(),
        batch, seq_len, HIDDEN_SIZE);
    last_hidden.mark_device_dirty();

    // LM head projection
    logits = Tensor({batch, VOCAB_SIZE});
    tensor_ops::matmul_transposed(last_hidden, lm_head_, logits, stream);

    // Reset to device 0
    CHECK_CUDA(cudaSetDevice(0));
}