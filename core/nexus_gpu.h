#pragma once

#if defined(__HIP_PLATFORM_AMD__) || defined(USE_ROCM)
#include <hip/hip_runtime.h>
#include <rccl/rccl.h>
#include <rocm_smi/rocm_smi.h>
#define NEXUS_ROCM 1

typedef hipStream_t nexusStream_t;
typedef hipError_t nexusError_t;
typedef rcclComm_t nexusComm_t;
typedef ncclResult_t nexusResult_t;

#define nexusMalloc hipMalloc
#define nexusFree hipFree
#define nexusMemcpyAsync hipMemcpyAsync
#define nexusSetDevice hipSetDevice
#define nexusGetErrorString hipGetErrorString
#define nexusSuccess hipSuccess
#define nexusDeviceGetPCIBusId hipDeviceGetPCIBusId

#else
#include <cuda_runtime.h>
#include <nccl.h>
#include <nvml.h>
#define NEXUS_CUDA 1

typedef cudaStream_t nexusStream_t;
typedef cudaError_t nexusError_t;
typedef ncclComm_t nexusComm_t;
typedef ncclResult_t nexusResult_t;

#define nexusMalloc cudaMalloc
#define nexusFree cudaFree
#define nexusMemcpyAsync cudaMemcpyAsync
#define nexusSetDevice cudaSetDevice
#define nexusGetErrorString cudaGetErrorString
#define nexusSuccess cudaSuccess
#define nexusDeviceGetPCIBusId cudaDeviceGetPCIBusId
#endif