#include <torch/extension.h>
#include "nexus_gpu.h"
#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a GPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define NCCL_CHECK(cmd) do { \
    nexusResult_t res = cmd; \
    if (res != ncclSuccess) { \
        throw std::runtime_error(std::string("[NexusNCCL] Collective communication error occurred.")); \
    } \
} while(0)

class NexusNCCLCoordinator {
private:
    nexusComm_t comm_;
    int rank_;
    int world_size_;
    int device_id_;

public:
    NexusNCCLCoordinator(int rank, int world_size, int device_id, const std::vector<char>& unique_id_bytes)
        : rank_(rank), world_size_(world_size), device_id_(device_id) {
        
        nexusError_t err = nexusSetDevice(device_id_);
        if (err != nexusSuccess) {
            throw std::runtime_error("[NexusNCCL] Failed to set device context.");
        }

        ncclUniqueId nccl_id;
        if (unique_id_bytes.size() != sizeof(ncclUniqueId)) {
            throw std::runtime_error("[NexusNCCL] Invalid Unique ID size.");
        }
        std::memcpy(&nccl_id, unique_id_bytes.data(), sizeof(ncclUniqueId));

        NCCL_CHECK(ncclCommInitRank(&comm_, world_size_, nccl_id, rank_));
    }

    ~NexusNCCLCoordinator() {
        if (comm_) {
            ncclCommDestroy(comm_);
        }
    }

    static std::vector<char> get_unique_id() {
        ncclUniqueId nccl_id;
        NCCL_CHECK(ncclGetUniqueId(&nccl_id));
        std::vector<char> id_bytes(sizeof(ncclUniqueId));
        std::memcpy(id_bytes.data(), &nccl_id, sizeof(ncclUniqueId));
        return id_bytes;
    }

    void all_reduce(torch::Tensor tensor, int64_t cuda_stream_ptr) {
        CHECK_INPUT(tensor);
        void* d_src = tensor.data_ptr();
        size_t count = tensor.numel();
        nexusStream_t stream = reinterpret_cast<nexusStream_t>(cuda_stream_ptr);

        ncclDataType_t dtype;
        if (tensor.scalar_type() == torch::kFloat32) dtype = ncclFloat32;
        else if (tensor.scalar_type() == torch::kFloat16) dtype = ncclFloat16;
        else if (tensor.scalar_type() == torch::kBFloat16) dtype = ncclBfloat16;
        else throw std::runtime_error("[NexusNCCL] Unsupported dtype for AllReduce.");

        NCCL_CHECK(ncclAllReduce(d_src, d_src, count, dtype, ncclSum, comm_, stream));
    }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<NexusNCCLCoordinator>(m, "NexusNCCLCoordinator")
        .def(py::init<int, int, int, const std::vector<char>&>())
        .def("all_reduce", &NexusNCCLCoordinator::all_reduce)
        .def_static("get_unique_id", &NexusNCCLCoordinator::get_unique_id);
}