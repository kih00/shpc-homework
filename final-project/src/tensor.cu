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
Tensor::Tensor() : size_(0), data_(nullptr), owns_data_(false) {}

Tensor::Tensor(const std::vector<size_t>& shape) 
    : shape_(shape), owns_data_(true) {
    size_ = compute_size();
    allocate();
}

Tensor::Tensor(const std::vector<size_t>& shape, float* data, bool copy)
    : shape_(shape), owns_data_(copy) {
    size_ = compute_size();
    if (copy) {
        allocate();
        // Assume data is host pointer when using this constructor with copy=true
        CHECK_CUDA(cudaMemcpy(data_, data, size_ * sizeof(float), cudaMemcpyHostToDevice));
    } else {
        data_ = data;
    }
}

Tensor::~Tensor() {
    deallocate();
}

// Copy constructor
Tensor::Tensor(const Tensor& other)
    : shape_(other.shape_), size_(other.size_), owns_data_(true) {
    if (other.size_ > 0) {
        allocate();
        CHECK_CUDA(cudaMemcpy(data_, other.data_, size_ * sizeof(float), cudaMemcpyDeviceToDevice));
    }
}

// Copy assignment
Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        deallocate();
        shape_ = other.shape_;
        size_ = other.size_;
        owns_data_ = true;
        if (other.size_ > 0) {
            allocate();
            CHECK_CUDA(cudaMemcpy(data_, other.data_, size_ * sizeof(float), cudaMemcpyDeviceToDevice));
        }
    }
    return *this;
}

// Move constructor
Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)), size_(other.size_),
      data_(other.data_), owns_data_(other.owns_data_) {
    other.data_ = nullptr;
    other.size_ = 0;
    other.owns_data_ = false;
}

// Move assignment
Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        deallocate();
        shape_ = std::move(other.shape_);
        size_ = other.size_;
        data_ = other.data_;
        owns_data_ = other.owns_data_;
        
        other.data_ = nullptr;
        other.size_ = 0;
        other.owns_data_ = false;
    }
    return *this;
}

void Tensor::allocate() {
    if (size_ > 0) {
        CHECK_CUDA(cudaMalloc(&data_, size_ * sizeof(float)));
    }
}

void Tensor::deallocate() {
    if (owns_data_ && data_ != nullptr) {
        CHECK_CUDA(cudaFree(data_));
        data_ = nullptr;
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
// WARNING: These will crash if called on host since data_ is device pointer.
// We keep them for compilation but they shouldn't be used in host code.
float& Tensor::at(size_t i) {
    return data_[i];
}

float& Tensor::at(size_t i, size_t j) {
    return data_[i * shape_[1] + j];
}

float& Tensor::at(size_t i, size_t j, size_t k) {
    return data_[(i * shape_[1] + j) * shape_[2] + k];
}

float& Tensor::at(size_t i, size_t j, size_t k, size_t l) {
    return data_[((i * shape_[1] + j) * shape_[2] + k) * shape_[3] + l];
}

const float& Tensor::at(size_t i) const {
    return data_[i];
}

const float& Tensor::at(size_t i, size_t j) const {
    return data_[i * shape_[1] + j];
}

const float& Tensor::at(size_t i, size_t j, size_t k) const {
    return data_[(i * shape_[1] + j) * shape_[2] + k];
}

const float& Tensor::at(size_t i, size_t j, size_t k, size_t l) const {
    return data_[((i * shape_[1] + j) * shape_[2] + k) * shape_[3] + l];
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
    Tensor result(new_shape, data_, false);  // false means don't copy data
    return result;
}

// IO operations
// Static method to load tensor from file
Tensor Tensor::load_from_file(const std::string& filename, ModelLoader* loader) {
    // If a specific loader is provided, use it
    if (loader) {
        return loader->load_tensor(filename);
    }
    
    // Otherwise, if global model loader is available, use it
    if (g_model_loader) {
        return g_model_loader->load_tensor(filename);
    }
    
    // Fallback to individual file loading
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
    
    // Create tensor (allocates on GPU)
    Tensor tensor(shape);
    
    // Read data to temp host buffer
    std::vector<float> temp_data(tensor.size());
    file.read(reinterpret_cast<char*>(temp_data.data()), tensor.size() * sizeof(float));
    
    // Copy to GPU
    CHECK_CUDA(cudaMemcpy(tensor.data(), temp_data.data(), tensor.size() * sizeof(float), cudaMemcpyHostToDevice));
    
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
    
    // Copy data to temp host buffer
    std::vector<float> temp_data(size_);
    CHECK_CUDA(cudaMemcpy(temp_data.data(), data_, size_ * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Write data
    file.write(reinterpret_cast<const char*>(temp_data.data()), size_ * sizeof(float));
    
    file.close();
}

// Tensor operations
Tensor Tensor::copy() const {
    Tensor result(shape_);
    CHECK_CUDA(cudaMemcpy(result.data(), data_, size_ * sizeof(float), cudaMemcpyDeviceToDevice));
    return result;
}

// Helper kernel for fill
__global__ void fill_kernel(float* data, float value, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = value;
    }
}

void Tensor::fill(float value) {
    if (data_ != nullptr && size_ > 0) {
        fill_kernel<<<(size_ + 255) / 256, 256>>>(data_, value, size_);
    }
}

void Tensor::zero() {
    if (data_ != nullptr && size_ > 0) {
        CHECK_CUDA(cudaMemset(data_, 0, size_ * sizeof(float)));
    }
}

void Tensor::ones() {
    fill(1.0f);
}
