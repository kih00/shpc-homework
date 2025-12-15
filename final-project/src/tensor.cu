#include "tensor.h"
#include <cstring>
#include <stdexcept>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <numeric>
#include "model_loader.h"

// Global model loader is declared in model.h
extern std::unique_ptr<ModelLoader> g_model_loader;

// Tensor class implementation - structure and data management only
// All tensor operations are implemented in layer.cu

// Tensor constructors and destructors
Tensor::Tensor() : size_(0), host_data_(nullptr), device_data_(nullptr),
                   owns_host_(false), owns_device_(false), device_id_(0),
                   host_dirty_(false), device_dirty_(false) {}

Tensor::Tensor(const std::vector<size_t>& shape) 
        : shape_(shape), host_data_(nullptr), device_data_(nullptr),
            owns_host_(true), owns_device_(false), device_id_(0),
            host_dirty_(false), device_dirty_(false) {
    size_ = compute_size();
    allocate_host();
}

Tensor::Tensor(const std::vector<size_t>& shape, float* data, bool copy)
        : shape_(shape), host_data_(nullptr), device_data_(nullptr),
            owns_host_(copy), owns_device_(false), device_id_(0),
            host_dirty_(false), device_dirty_(false) {
    size_ = compute_size();
    if (copy) {
        allocate_host();
        std::memcpy(host_data_, data, size_ * sizeof(float));
    } else {
        host_data_ = data;
    }
}

Tensor::~Tensor() {
    deallocate();
}

// Copy constructor
Tensor::Tensor(const Tensor& other)
        : shape_(other.shape_), size_(other.size_), host_data_(nullptr), device_data_(nullptr),
            owns_host_(true), owns_device_(false), device_id_(other.device_id_),
            host_dirty_(false), device_dirty_(false) {
    if (other.size_ > 0) {
        other.to_host();
        allocate_host();
        std::memcpy(host_data_, other.host_data_, size_ * sizeof(float));
    }
}

// Copy assignment
Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        deallocate();
        shape_ = other.shape_;
        size_ = other.size_;
        device_id_ = other.device_id_;
        owns_host_ = true;
        owns_device_ = false;
        host_dirty_ = false;
        device_dirty_ = false;
        if (other.size_ > 0) {
            other.to_host();
            allocate_host();
            std::memcpy(host_data_, other.host_data_, size_ * sizeof(float));
        }
    }
    return *this;
}

// Move constructor
Tensor::Tensor(Tensor&& other) noexcept
        : shape_(std::move(other.shape_)), size_(other.size_),
            host_data_(other.host_data_), device_data_(other.device_data_),
            owns_host_(other.owns_host_), owns_device_(other.owns_device_),
            device_id_(other.device_id_),
            host_dirty_(other.host_dirty_), device_dirty_(other.device_dirty_) {
        other.host_data_ = nullptr;
        other.device_data_ = nullptr;
        other.size_ = 0;
        other.owns_host_ = false;
        other.owns_device_ = false;
        other.host_dirty_ = false;
        other.device_dirty_ = false;
}

// Move assignment
Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        deallocate();
        shape_ = std::move(other.shape_);
        size_ = other.size_;
        host_data_ = other.host_data_;
        device_data_ = other.device_data_;
        owns_host_ = other.owns_host_;
        owns_device_ = other.owns_device_;
        device_id_ = other.device_id_;
        host_dirty_ = other.host_dirty_;
        device_dirty_ = other.device_dirty_;
        
        other.host_data_ = nullptr;
        other.device_data_ = nullptr;
        other.size_ = 0;
        other.owns_host_ = false;
        other.owns_device_ = false;
        other.host_dirty_ = false;
        other.device_dirty_ = false;
    }
    return *this;
}

void Tensor::allocate_host() const {
    if (size_ > 0 && host_data_ == nullptr) {
        host_data_ = new float[size_];
    }
}

void Tensor::deallocate() {
    if (owns_host_ && host_data_ != nullptr) {
        delete[] host_data_;
        host_data_ = nullptr;
    }
    if (owns_device_ && device_data_ != nullptr) {
        cudaFree(device_data_);
        device_data_ = nullptr;
    }
}

size_t Tensor::compute_size() const {
    if (shape_.empty()) return 0;
    return std::accumulate(shape_.begin(), shape_.end(), 1ULL, std::multiplies<size_t>());
}

size_t Tensor::size(int dim) const {
    if (dim < 0) dim += shape_.size();
    if (dim < 0 || static_cast<size_t>(dim) >= shape_.size()) {
        throw std::out_of_range("Dimension out of range");
    }
    return shape_[dim];
}

size_t Tensor::compute_stride(int dim) const {
    size_t stride = 1;
    for (size_t i = dim + 1; i < shape_.size(); i++) {
        stride *= shape_[i];
    }
    return stride;
}

// Element access
float& Tensor::at(size_t i) {
    ensure_host_data();
    return host_data_[i];
}

float& Tensor::at(size_t i, size_t j) {
    ensure_host_data();
    return host_data_[i * shape_[1] + j];
}

float& Tensor::at(size_t i, size_t j, size_t k) {
    ensure_host_data();
    return host_data_[(i * shape_[1] + j) * shape_[2] + k];
}

float& Tensor::at(size_t i, size_t j, size_t k, size_t l) {
    ensure_host_data();
    return host_data_[((i * shape_[1] + j) * shape_[2] + k) * shape_[3] + l];
}

const float& Tensor::at(size_t i) const {
    ensure_host_data();
    return host_data_[i];
}

const float& Tensor::at(size_t i, size_t j) const {
    ensure_host_data();
    return host_data_[i * shape_[1] + j];
}

const float& Tensor::at(size_t i, size_t j, size_t k) const {
    ensure_host_data();
    return host_data_[(i * shape_[1] + j) * shape_[2] + k];
}

const float& Tensor::at(size_t i, size_t j, size_t k, size_t l) const {
    ensure_host_data();
    return host_data_[((i * shape_[1] + j) * shape_[2] + k) * shape_[3] + l];
}

// Reshape
void Tensor::reshape(const std::vector<size_t>& new_shape) {
    size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(), 1ULL, std::multiplies<size_t>());
    if (new_size != size_) {
        throw std::invalid_argument("New shape must have same number of elements");
    }
    shape_ = new_shape;
}

Tensor Tensor::view(const std::vector<size_t>& new_shape) const {
    // Verify new shape has same number of elements
    size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(), 1ULL, std::multiplies<size_t>());
    if (new_size != size_) {
        throw std::invalid_argument("New shape must have same number of elements");
    }
    
    // Create a view that shares data with this tensor (no copy)
    ensure_host_data();
    Tensor result(new_shape, host_data_, false);  // false means don't copy data
    result.device_data_ = device_data_;
    result.device_id_ = device_id_;
    result.owns_device_ = false;
    result.device_dirty_ = device_dirty_;
    result.host_dirty_ = host_dirty_;
    return result;
}

// IO operations
Tensor Tensor::load_from_file(const std::string& filename, ModelLoader* loader) {
    // If a specific loader is provided, use it
    if (loader) {
        return loader->load_tensor(filename);
    }
    
    // Otherwise, if global model loader is available, use it
    if (g_model_loader) {
        // The filename is the tensor name (e.g., "embed_tokens.weight")
        // No need to strip anything if properly passed
        return g_model_loader->load_tensor(filename);
    }
    
    // Fallback to individual file loading (if model.bin not used)
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    // Read number of dimensions
    uint32_t ndim;
    file.read(reinterpret_cast<char*>(&ndim), sizeof(uint32_t));
    
    // Read shape
    std::vector<size_t> shape(ndim);
    for (uint32_t i = 0; i < ndim; i++) {
        uint32_t dim;
        file.read(reinterpret_cast<char*>(&dim), sizeof(uint32_t));
        shape[i] = dim;
    }
    
    // Create tensor
    Tensor tensor(shape);
    
    // Read data
    file.read(reinterpret_cast<char*>(tensor.data()), tensor.size() * sizeof(float));
    
    file.close();
    return tensor;
}

void Tensor::save_to_file(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    // Write number of dimensions
    uint32_t ndim = shape_.size();
    file.write(reinterpret_cast<const char*>(&ndim), sizeof(uint32_t));
    
    // Write shape
    for (size_t dim : shape_) {
        uint32_t dim32 = dim;
        file.write(reinterpret_cast<const char*>(&dim32), sizeof(uint32_t));
    }
    
    // Write data
    ensure_host_data();
    file.write(reinterpret_cast<const char*>(host_data_), size_ * sizeof(float));
    
    file.close();
}

// Tensor operations
Tensor Tensor::copy() const {
    ensure_host_data();
    return Tensor(shape_, host_data_, true);
}

void Tensor::fill(float value) {
    ensure_host_data();
    std::fill(host_data_, host_data_ + size_, value);
    if (device_data_) sync_device_from_host();
}

void Tensor::zero() {
    ensure_host_data();
    if (host_data_ != nullptr && size_ > 0) {
        std::memset(host_data_, 0, size_ * sizeof(float));
    }
    if (device_data_) {
        CHECK_CUDA(cudaMemset(device_data_, 0, size_ * sizeof(float)));
    }
}

void Tensor::ones() {
    ensure_host_data();
    if (host_data_ != nullptr && size_ > 0) {
        std::fill(host_data_, host_data_ + size_, 1.0f);
    }
    if (device_data_) sync_device_from_host();
}

void Tensor::ensure_host_data() const {
    if (size_ == 0) return;
    // Allocate host buffer if missing
    if (host_data_ == nullptr) {
        host_data_ = new float[size_];
        owns_host_ = true;
    }
    // If device has fresher data, sync it back
    if (device_data_ != nullptr && device_dirty_) {
        CHECK_CUDA(cudaMemcpy(host_data_, device_data_, size_ * sizeof(float), cudaMemcpyDeviceToHost));
        device_dirty_ = false;
        host_dirty_ = false;
    }
}

void Tensor::to_device(int device_id, cudaStream_t stream) const {
    if (size_ == 0) return;
    
    // If device_id is -1, use current device
    if (device_id == -1) {
        CHECK_CUDA(cudaGetDevice(&device_id));
    }
    
    // If already on target device, do nothing
    if (device_data_ != nullptr && device_id_ == device_id) return;
    
    // If on a different device, free and reallocate
    if (device_data_ != nullptr && owns_device_ && device_id_ != device_id) {
        int old_device;
        CHECK_CUDA(cudaGetDevice(&old_device));
        CHECK_CUDA(cudaSetDevice(device_id_));
        CHECK_CUDA(cudaFree(device_data_));
        CHECK_CUDA(cudaSetDevice(old_device));
        device_data_ = nullptr;
        owns_device_ = false;
    }
    
    device_id_ = device_id;
    CHECK_CUDA(cudaSetDevice(device_id_));
    if (device_data_ == nullptr) {
        CHECK_CUDA(cudaMalloc(&device_data_, size_ * sizeof(float)));
        owns_device_ = true;
    }
    ensure_host_data();
    CHECK_CUDA(cudaMemcpyAsync(device_data_, host_data_, size_ * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    host_dirty_ = false;
    device_dirty_ = false;
}

void Tensor::to_host(cudaStream_t stream) const {
    if (size_ == 0) return;
    ensure_host_data();
    if (device_data_ != nullptr) {
        CHECK_CUDA(cudaMemcpyAsync(host_data_, device_data_, size_ * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        device_dirty_ = false;
        host_dirty_ = false;
    }
}

void Tensor::sync_device_from_host(cudaStream_t stream) const {
    if (size_ == 0 || device_data_ == nullptr) return;
    ensure_host_data();
    CHECK_CUDA(cudaMemcpyAsync(device_data_, host_data_, size_ * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    host_dirty_ = false;
    device_dirty_ = false;
}

void Tensor::sync_host_from_device(cudaStream_t stream) const {
    if (size_ == 0 || device_data_ == nullptr) return;
    ensure_host_data();
    CHECK_CUDA(cudaMemcpyAsync(host_data_, device_data_, size_ * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    device_dirty_ = false;
    host_dirty_ = false;
}

void Tensor::copy_to_device(int target_device, cudaStream_t stream) const {
    if (size_ == 0) return;
    
    // If already on target device, do nothing
    if (device_data_ != nullptr && device_id_ == target_device) return;
    
    // Save old device data pointer and device id for device-to-device copy
    float* old_device_data = device_data_;
    int old_device_id = device_id_;
    bool had_device_data = (device_data_ != nullptr);
    
    // Allocate on target device
    device_id_ = target_device;
    CHECK_CUDA(cudaSetDevice(target_device));
    CHECK_CUDA(cudaMalloc(&device_data_, size_ * sizeof(float)));
    
    if (had_device_data) {
        // Device-to-device copy (peer copy across GPUs)
        CHECK_CUDA(cudaMemcpyAsync(device_data_, old_device_data, size_ * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        
        // Free old device data
        if (owns_device_) {
            CHECK_CUDA(cudaSetDevice(old_device_id));
            CHECK_CUDA(cudaFree(old_device_data));
            CHECK_CUDA(cudaSetDevice(target_device));
        }
    } else {
        // Host to device copy
        ensure_host_data();
        CHECK_CUDA(cudaMemcpyAsync(device_data_, host_data_, size_ * sizeof(float), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        host_dirty_ = false;
        device_dirty_ = false;
    }
    // After a device-to-device copy, device has the freshest data and host is stale.
    if (had_device_data) {
        host_dirty_ = false;
        device_dirty_ = true;
    }
    owns_device_ = true;
}

