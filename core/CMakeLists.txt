cmake_minimum_required(VERSION 3.18)
project(nexus_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development)
find_package(Torch REQUIRED)

include_directories(${TORCH_INCLUDE_DIRS})
link_directories(${TORCH_LIBRARIES})

if(DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH $ENV{ROCM_PATH})
else()
    set(ROCM_PATH /opt/rocm)
endif()

if(EXISTS ${ROCM_PATH})
    message(STATUS "ROCm/HIP environment detected. Compiling for AMD architecture.")
    add_definitions(-D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1)
    include_directories(${ROCM_PATH}/include ${ROCM_PATH}/include/rccl ${ROCM_PATH}/include/rocm_smi)
    link_directories(${ROCM_PATH}/lib)
    set(COMM_LIBRARY rccl)
    set(SMI_LIBRARY amd_smi)
else()
    message(STATUS "NVIDIA CUDA environment detected. Compiling for CUDA architecture.")
    find_package(CUDAToolkit REQUIRED)
    include_directories(${CUDAToolkit_INCLUDE_DIRS})
    link_directories(${CUDAToolkit_LIBRARY_DIR})
    set(COMM_LIBRARY nccl)
    set(SMI_LIBRARY nvidia-ml)
endif()

add_library(nexus_arena MODULE nexus_arena.cpp)
target_link_libraries(nexus_arena PRIVATE ${TORCH_LIBRARIES})

add_library(nexus_nccl_fabric MODULE nexus_nccl_fabric.cpp)
target_link_libraries(nexus_nccl_fabric PRIVATE ${TORCH_LIBRARIES} ${COMM_LIBRARY})

add_library(nexus_health_monitor MODULE nexus_health_monitor.cpp)
target_link_libraries(nexus_health_monitor PRIVATE ${TORCH_LIBRARIES} ${SMI_LIBRARY})

add_library(nexus_weights_shm MODULE nexus_weights_shm.cpp)
target_link_libraries(nexus_weights_shm PRIVATE ${TORCH_LIBRARIES})