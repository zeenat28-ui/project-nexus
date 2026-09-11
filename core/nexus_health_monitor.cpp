#include <torch/extension.h>
#include "nexus_gpu.h"
#include <iostream>
#include <stdexcept>
#include <chrono>
#include <atomic>

class NexusHealthMonitor {
private:
    int device_id_;
    std::atomic<int> circuit_state_;
    std::atomic<uint64_t> consecutive_failures_;
    uint64_t failure_threshold_;
    uint64_t cooldown_ms_;
#if defined(NEXUS_CUDA)
    nvmlDevice_t nvml_device_;
#elif defined(NEXUS_ROCM)
    int rocm_smi_initialized_;
#endif

public:
    NexusHealthMonitor(int device_id, uint64_t failure_threshold = 3, uint64_t cooldown_ms = 5000)
        : device_id_(device_id), failure_threshold_(failure_threshold), cooldown_ms_(cooldown_ms),
          circuit_state_(0), consecutive_failures_(0) {
        
#if defined(NEXUS_CUDA)
        if (nvmlInit() != NVML_SUCCESS) {
            throw std::runtime_error("[NexusHealth] NVML Init failed.");
        }
        char pci_bus_id[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
        if (nexusDeviceGetPCIBusId(pci_bus_id, NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE, device_id_) != nexusSuccess) {
            throw std::runtime_error("[NexusHealth] Failed to get PCI bus ID.");
        }
        if (nvmlDeviceGetHandleByPciBusId(pci_bus_id, &nvml_device_) != NVML_SUCCESS) {
            throw std::runtime_error("[NexusHealth] Failed to get NVML device handle.");
        }
#elif defined(NEXUS_ROCM)
        rsmi_status_t status = rsmi_init(0);
        if (status != RSMI_STATUS_SUCCESS) {
            throw std::runtime_error("[NexusHealth] RSMI Init failed.");
        }
        rocm_smi_initialized_ = 1;
#endif
    }

    ~NexusHealthMonitor() {
#if defined(NEXUS_CUDA)
        nvmlShutdown();
#elif defined(NEXUS_ROCM)
        if (rocm_smi_initialized_) {
            rsmi_shut_down();
        }
#endif
    }

    bool check_hardware_health() {
#if defined(NEXUS_CUDA)
        unsigned long long uncorrectable_ecc = 0;
        if (nvmlDeviceGetMemoryErrorCounter(nvml_device_, NVML_MEMORY_ERROR_TYPE_UNCORRECTED, 
                                            NVML_ECC_COUNTER_TYPE_AGGREGATE, &uncorrectable_ecc) == NVML_SUCCESS) {
            if (uncorrectable_ecc > 0) {
                trip_circuit();
                return false;
            }
        }
#elif defined(NEXUS_ROCM)
        uint64_t uc_err = 0;
        rsmi_status_t status = rsmi_dev_ecc_count_get(device_id_, RSMI_LED_INDICATION_VRC_CORRECTED, &uc_err);
        if (status == RSMI_STATUS_SUCCESS && uc_err > 100) {
            trip_circuit();
            return false;
        }
#endif
        return true;
    }

    void record_success() {
        consecutive_failures_ = 0;
        if (circuit_state_ == 2) circuit_state_ = 0;
    }

    void record_failure() {
        consecutive_failures_++;
        if (consecutive_failures_ >= failure_threshold_) {
            trip_circuit();
        }
    }

    void trip_circuit() {
        circuit_state_ = 1;
        std::cerr << "[NexusHealth] CRITICAL: Circuit breaker tripped to OPEN state." << std::endl;
    }

    std::string get_circuit_state() {
        int state = circuit_state_.load();
        if (state == 0) return "CLOSED";
        if (state == 1) return "OPEN";
        return "HALF_OPEN";
    }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<NexusHealthMonitor>(m, "NexusHealthMonitor")
        .def(py::init<int, uint64_t, uint64_t>())
        .def("check_hardware_health", &NexusHealthMonitor::check_hardware_health)
        .def("record_success", &NexusHealthMonitor::record_success)
        .def("record_failure", &NexusHealthMonitor::record_failure)
        .def("get_circuit_state", &NexusHealthMonitor::get_circuit_state);
}