#include <torch/extension.h>
#include "nexus_gpu.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string>
#include <stdexcept>
#include <iostream>
#include <mutex>

class NexusWeightHotSwapper {
private:
    std::string shm_name_;
    size_t tensor_size_bytes_;
    int shm_fd_;
    void* host_shm_ptr_;
    void* d_active_weights_;
    void* d_shadow_weights_;
    bool active_slot_is_a_;
    std::mutex swap_mutex_;

public:
    NexusWeightHotSwapper(const std::string& shm_name, size_t tensor_size_bytes, int device_id)
        : shm_name_(shm_name), tensor_size_bytes_(tensor_size_bytes), active_slot_is_a_(true) {
        
        nexusError_t err = nexusSetDevice(device_id);
        if (err != nexusSuccess) {
            throw std::runtime_error("[NexusShm] Failed to set device for hot-swapper.");
        }

        shm_fd_ = shm_open(shm_name_.c_str(), O_CREAT | O_RDWR, 0666);
        if (shm_fd_ == -1) {
            throw std::runtime_error("[NexusShm] Failed to open POSIX shared memory descriptor.");
        }

        if (ftruncate(shm_fd_, tensor_size_bytes_ * 2) == -1) {
            throw std::runtime_error("[NexusShm] Failed to truncate POSIX shared memory size.");
        }

        host_shm_ptr_ = mmap(NULL, tensor_size_bytes_ * 2, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_, 0);
        if (host_shm_ptr_ == MAP_FAILED) {
            throw std::runtime_error("[NexusShm] mmap failed on shared memory region.");
        }

        nexusMalloc(&d_active_weights_, tensor_size_bytes_);
        nexusMalloc(&d_shadow_weights_, tensor_size_bytes_);

        std::cout << "[NexusWeightHotSwapper] Initialized dual-buffered VRAM weight slots (" 
                  << (tensor_size_bytes_ / (1024 * 1024)) << " MB each) linked to shm: " << shm_name_ << std::endl;
    }

    ~NexusWeightHotSwapper() {
        if (host_shm_ptr_ != MAP_FAILED) {
            munmap(host_shm_ptr_, tensor_size_bytes_ * 2);
        }
        if (shm_fd_ != -1) {
            close(shm_fd_);
            shm_unlink(shm_name_.c_str());
        }
        if (d_active_weights_) nexusFree(d_active_weights_);
        if (d_shadow_weights_) nexusFree(d_shadow_weights_);
    }

    void stage_shadow_weights(int64_t source_offset_bytes) {
        std::lock_guard<std::mutex> lock(swap_mutex_);
        void* target_shadow = active_slot_is_a_ ? d_shadow_weights_ : d_active_weights_;
        char* source_host = static_cast<char*>(host_shm_ptr_) + source_offset_bytes;

        nexusError_t err = nexusMemcpyAsync(target_shadow, source_host, tensor_size_bytes_, nexusMemcpyHostToDevice, 0);
        if (err != nexusSuccess) {
            throw std::runtime_error("[NexusShm] Async weight staging failed.");
        }
    }

    void commit_hot_swap() {
        std::lock_guard<std::mutex> lock(swap_mutex_);
        void* temp = d_active_weights_;
        d_active_weights_ = d_shadow_weights_;
        d_shadow_weights_ = temp;
        active_slot_is_a_ = !active_slot_is_a_;
        std::cout << "[NexusWeightHotSwapper] Atomic hot-swap committed successfully." << std::endl;
    }

    int64_t get_active_weights_ptr() {
        std::lock_guard<std::mutex> lock(swap_mutex_);
        return reinterpret_cast<int64_t>(d_active_weights_);
    }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<NexusWeightHotSwapper>(m, "NexusWeightHotSwapper")
        .def(py::init<const std::string&, size_t, int>())
        .def("stage_shadow_weights", &NexusWeightHotSwapper::stage_shadow_weights)
        .def("commit_hot_swap", &NexusWeightHotSwapper::commit_hot_swap)
        .def("get_active_weights_ptr", &NexusWeightHotSwapper::get_active_weights_ptr);
}