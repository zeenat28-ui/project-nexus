#include <torch/extension.h>
#include "nexus_gpu.h"
#include <vector>
#include <stdexcept>
#include <iostream>

struct BlockMetadata {
    size_t offset;
    size_t size;
    bool is_free;
    BlockMetadata* next;
    BlockMetadata* prev;
};

class NexusVRAMArena {
private:
    void* d_arena_base_;
    size_t total_capacity_;
    BlockMetadata* head_block_;
    int device_id_;

public:
    NexusVRAMArena(size_t capacity_bytes, int device_id) 
        : total_capacity_(capacity_bytes), device_id_(device_id) {
        
        nexusError_t err = nexusSetDevice(device_id_);
        if (err != nexusSuccess) {
            throw std::runtime_error(std::string("[NexusArena] Failed to set GPU device: ") + nexusGetErrorString(err));
        }

        err = nexusMalloc(&d_arena_base_, total_capacity_);
        if (err != nexusSuccess) {
            throw std::runtime_error(std::string("[NexusArena] VRAM Arena allocation failed: ") + nexusGetErrorString(err));
        }

        head_block_ = new BlockMetadata{
            .offset = 0,
            .size = total_capacity_,
            .is_free = true,
            .next = nullptr,
            .prev = nullptr
        };
        
        std::cout << "[NexusVRAMArena] Initialized fixed VRAM pool of " 
                  << (total_capacity_ / (1024 * 1024)) << " MB on Device " << device_id_ << std::endl;
    }

    ~NexusVRAMArena() {
        BlockMetadata* current = head_block_;
        while (current != nullptr) {
            BlockMetadata* next = current->next;
            delete current;
            current = next;
        }
        if (d_arena_base_) {
            nexusFree(d_arena_base_);
        }
        std::cout << "[NexusVRAMArena] Arena destroyed and VRAM released." << std::endl;
    }

    int64_t allocate(size_t size) {
        size_t aligned_size = (size + 255) & ~255;
        BlockMetadata* current = head_block_;
        while (current != nullptr) {
            if (current->is_free && current->size >= aligned_size) {
                if (current->size > aligned_size) {
                    BlockMetadata* new_block = new BlockMetadata{
                        .offset = current->offset + aligned_size,
                        .size = current->size - aligned_size,
                        .is_free = true,
                        .next = current->next,
                        .prev = current
                    };
                    if (current->next != nullptr) {
                        current->next->prev = new_block;
                    }
                    current->next = new_block;
                    current->size = aligned_size;
                }
                current->is_free = false;
                return static_cast<int64_t>(current->offset);
            }
            current = current->next;
        }
        throw std::runtime_error("[NexusVRAMArena] OOM: Arena exhausted during allocation request.");
    }

    void deallocate(int64_t offset) {
        BlockMetadata* current = head_block_;
        while (current != nullptr) {
            if (current->offset == static_cast<size_t>(offset)) {
                current->is_free = true;

                if (current->next != nullptr && current->next->is_free) {
                    BlockMetadata* next_block = current->next;
                    current->size += next_block->size;
                    current->next = next_block->next;
                    if (current->next != nullptr) {
                        current->next->prev = current;
                    }
                    delete next_block;
                }

                if (current->prev != nullptr && current->prev->is_free) {
                    BlockMetadata* prev_block = current->prev;
                    prev_block->size += current->size;
                    prev_block->next = current->next;
                    if (current->next != nullptr) {
                        current->next->prev = prev_block;
                    }
                    delete current;
                }
                return;
            }
            current = current->next;
        }
        throw std::runtime_error("[NexusVRAMArena] Invalid offset passed for deallocation.");
    }

    int64_t get_base_address() {
        return reinterpret_cast<int64_t>(d_arena_base_);
    }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<NexusVRAMArena>(m, "NexusVRAMArena")
        .def(py::init<size_t, int>())
        .def("allocate", &NexusVRAMArena::allocate)
        .def("deallocate", &NexusVRAMArena::deallocate)
        .def("get_base_address", &NexusVRAMArena::get_base_address);
}