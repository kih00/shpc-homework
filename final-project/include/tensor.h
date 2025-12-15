#pragma once

#include <cstddef>
#include <vector>
#include <string>
#include <memory>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

/* Macro for checking CUDA errors */
#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)


// Forward declaration
class ModelLoader;

class Tensor {
public:
    // Constructors
    Tensor();
    Tensor(const std::vector<size_t>& shape);
    Tensor(const std::vector<size_t>& shape, float* data, bool copy = true);
    ~Tensor();
    
    // Copy constructor and assignment
    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);
    
    // Move constructor and assignment
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    // Shape operations
    size_t ndim() const { return shape_.size(); }
    const std::vector<size_t>& shape() const { return shape_; }
    size_t size() const { return size_; }
    size_t size(int dim) const;
    
    // Data access
    float* data() { ensure_host_data(); mark_host_dirty(); return host_data_; }
    const float* data() const { ensure_host_data(); return host_data_; }
    float& operator[](size_t idx) { ensure_host_data(); mark_host_dirty(); return host_data_[idx]; }
    const float& operator[](size_t idx) const { ensure_host_data(); return host_data_[idx]; }
    
    // Element access
    float& at(size_t i);
    float& at(size_t i, size_t j);
    float& at(size_t i, size_t j, size_t k);
    float& at(size_t i, size_t j, size_t k, size_t l);
    
    const float& at(size_t i) const;
    const float& at(size_t i, size_t j) const;
    const float& at(size_t i, size_t j, size_t k) const;
    const float& at(size_t i, size_t j, size_t k, size_t l) const;
    
    // Reshape
    void reshape(const std::vector<size_t>& new_shape);
    Tensor view(const std::vector<size_t>& new_shape) const;
    
    // Device management
    bool is_cuda() const { return device_data_ != nullptr; }
    int device_id() const { return device_id_; }
    float* device_data() { return device_data_; }
    const float* device_data() const { return device_data_; }
    void to_device(int device_id = 0, cudaStream_t stream = 0) const;
    void to_host(cudaStream_t stream = 0) const;
    void sync_device_from_host(cudaStream_t stream = 0) const;
    void sync_host_from_device(cudaStream_t stream = 0) const;
    
    // Multi-GPU support: copy tensor to another device
    void copy_to_device(int target_device, cudaStream_t stream = 0) const;
    
    // IO operations
    static Tensor load_from_file(const std::string& filename, ModelLoader* loader = nullptr);
    void save_to_file(const std::string& filename) const;
    
    // Tensor operations
    Tensor transpose(int dim0, int dim1) const;
    Tensor slice(int dim, size_t start, size_t end) const;
    Tensor copy() const;
    
    // Fill operations
    void fill(float value);
    void zero();
    void ones();

    // Dirty-state helpers for device/host freshness
    void mark_device_dirty() const { device_dirty_ = true; host_dirty_ = false; }
    void mark_host_dirty() const { host_dirty_ = true; device_dirty_ = false; }

private:
    std::vector<size_t> shape_;
    size_t size_;
  mutable float* host_data_;
  mutable float* device_data_;
  mutable bool owns_host_;
  mutable bool owns_device_;
  mutable int device_id_;
  mutable bool host_dirty_;
  mutable bool device_dirty_;
    
  void allocate_host() const;
  void deallocate();
    size_t compute_size() const;
    size_t compute_stride(int dim) const;
  void ensure_host_data() const;
};

// Tensor operations
namespace tensor_ops {
    // Matrix operations
    void matmul(const Tensor& a, const Tensor& b, Tensor& c);
    void matmul_transposed(const Tensor& a, const Tensor& b, Tensor& c); // c = a @ b^T
    
    // Element-wise operations
    void add(const Tensor& a, const Tensor& b, Tensor& c);
    void add_scalar(const Tensor& a, float b, Tensor& c);
    void mul(const Tensor& a, const Tensor& b, Tensor& c);
    void mul_scalar(const Tensor& a, float b, Tensor& c);
    
    // Activation functions
    void silu(const Tensor& x, Tensor& y); // SiLU(x) = x * sigmoid(x)
    void sigmoid(const Tensor& x, Tensor& y);
    void softmax(const Tensor& x, Tensor& y, int dim);
    
    // Normalization
    void rms_norm(const Tensor& x, const Tensor& weight, float eps, Tensor& y);
    
    // RoPE (Rotary Position Embedding)
    void apply_rotary_pos_emb(Tensor& q, Tensor& k, const Tensor& cos, const Tensor& sin);
    void compute_rope_embeddings(size_t head_dim, size_t max_seq_len, float theta, 
                                 Tensor& cos, Tensor& sin);
    
    // Repeat KV for GQA (Grouped Query Attention)
    void repeat_kv(const Tensor& x, size_t n_rep, Tensor& y);
    
    // Convolution
    void causal_conv1d(const Tensor& x, const Tensor& weight, const Tensor* bias,
                       Tensor& y);

    // Attention operations
    void reshape_to_heads(const Tensor& in, Tensor& out,
                          size_t batch, size_t seq_len, size_t num_heads, size_t head_dim);
    void reshape_from_heads(const Tensor& in, Tensor& out,
                            size_t batch, size_t seq_len, size_t num_heads, size_t head_dim);
    void batched_attention(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& out, float scale);
}
