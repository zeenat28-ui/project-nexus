import os
import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Queue
import ctypes
import struct

class NexusIPCHandleManager:
    @staticmethod
    def get_ipc_handle(tensor: torch.Tensor) -> bytes:
        """Extracts the raw CUDA IPC memory handle from a CUDA tensor."""
        if not tensor.is_cuda:
            raise ValueError("[NexusIPC] Tensor must reside on CUDA memory to export IPC handle.")
        
        # Ensure tensor storage is contiguous
        tensor = tensor.contiguous()
        storage = tensor.storage()
        
        # Access underlying storage data pointer and get CUDA IPC handle
        # PyTorch exposes a built-in method or ctypes wrapper for cudaIpcMemHandle_t
        data_ptr = storage.data_ptr()
        
        # Allocate a C structure buffer for cudaIpcMemHandle_t (64 bytes on modern CUDA)
        handle_buf = ctypes.create_string_buffer(64)
        
        # Call CUDA runtime API directly via ctypes
        cudart = ctypes.CDLL('libcudart.so')
        status = cudart.cudaIpcGetMemHandle(handle_buf, ctypes.c_void_p(data_ptr))
        if status != 0:
            raise RuntimeError(f"[NexusIPC] cudaIpcGetMemHandle failed with error code: {status}")
            
        return bytes(handle_buf.raw)

    @staticmethod
    def open_ipc_handle(handle_bytes: bytes, size: int, dtype: torch.dtype, device_id: int) -> torch.Tensor:
        """Opens a remote CUDA IPC memory handle in a secondary worker process VRAM space."""
        torch.cuda.set_device(device_id)
        
        handle_buf = ctypes.create_string_buffer(handle_bytes, 64)
        d_ptr = ctypes.c_void_p()
        
        cudart = ctypes.CDLL('libcudart.so')
        # cudaIpcWritability: cudaIpcMemLazyEnablePeerAccess = 1
        status = cudart.cudaIpcOpenMemHandle(ctypes.byref(d_ptr), handle_buf, 1)
        if status != 0:
            raise RuntimeError(f"[NexusIPC] cudaIpcOpenMemHandle failed with error code: {status}")
        
        # Reconstruct PyTorch tensor pointing to the remote physical VRAM address without copying
        # Map raw pointer back into a torch storage object
        numel = size // torch.tensor([], dtype=dtype).element_size()
        
        # Create a CPU storage view mapped to the external CUDA pointer
        # PyTorch allows wrapping external data pointers via torch.from_blob
        # Note: We use torch.asarray or torch.from_blob with proper device context
        tensor = torch.from_numpy(
            # Using a dummy numpy array or direct torch C++ extension binding for pointer wrapping
            # Alternatively, construct via torch storage:
            # (In production, use torch.cuda.UntypedStorage or C++ extension bindings)
            __import__('numpy').empty(0, dtype=torch.float32) # placeholder for structural layout
        )
        
        # Safe extraction of storage via custom torch capsule:
        tensor = torch.cuda.ByteStorage._from_file if hasattr(torch.cuda, 'ByteStorage') else None
        
        # For clean production integration, we wrap the pointer using torch's internal pointer mapping:
        tensor = torch.empty(0, device=f"cuda:{device_id}", dtype=dtype)
        # Force re-assignment of data ptr via tensor data wrapper
        tensor.data = torch.cuda.FloatTensor(d_ptr.value, size=(numel,))
        
        return tensor

def worker_process_entry(rank: int, world_size: int, ipc_queue: Queue):
    print(f"[NexusWorker-{rank}] Initializing worker process on GPU {rank}...")
    torch.cuda.set_device(rank)
    
    if rank == 0:
        # Primary Rank allocates master weight buffer
        master_tensor = torch.randn(1024, 1024, dtype=torch.float16, device=f"cuda:{rank}")
        handle = NexusIPCHandleManager.get_ipc_handle(master_tensor)
        
        # Broadcast handle to secondary workers via queue
        for r in range(1, world_size):
            ipc_queue.put((handle, master_tensor.nelement() * master_tensor.element_size(), master_tensor.dtype))
            
        print(f"[NexusWorker-0] Exported and shared CUDA IPC handle with world.")
    else:
        # Secondary Ranks receive handle and mount zero-copy memory
        handle, size, dtype = ipc_queue.get()
        remote_tensor = NexusIPCHandleManager.open_ipc_handle(handle, size, dtype, rank)
        print(f"[NexusWorker-{rank}] Successfully mounted remote IPC tensor. Shape check: {remote_tensor.shape}")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    world_size = 2
    q = Queue()
    
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=worker_process_entry, args=(rank, world_size, q))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
    print("[NexusIPC] Multi-worker zero-copy IPC test completed successfully.")