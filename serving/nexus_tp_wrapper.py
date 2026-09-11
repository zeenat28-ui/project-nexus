import torch
import torch.multiprocessing as mp
from typing import List, Optional

try:
    import nexus_nccl_fabric
except ImportError:
    nexus_nccl_fabric = None

class NexusTensorParallelWrapper:
    def __init__(self, rank: int, world_size: int, device_id: int, unique_id: List[int]):
        self.rank = rank
        self.world_size = world_size
        self.device_id = device_id
        
        torch.cuda.set_device(self.device_id)
        
        if nexus_nccl_fabric is not None:
            # Convert python list/bytes to character vector for C++ extension constructor
            id_bytes = bytes(unique_id)
            self.coordinator = nexus_nccl_fabric.NexusNCCLCoordinator(
                rank=self.rank,
                world_size=self.world_size,
                device_id=self.device_id,
                unique_id_bytes=list(id_bytes)
            )
            print(f"[NexusTP] NCCL coordinator successfully bound for Rank {self.rank}/{self.world_size}")
        else:
            self.coordinator = None
            print(f"[NexusTP] Warning: nexus_nccl_fabric extension unavailable. Running in simulation mode.")

    def all_reduce_tensor(self, tensor: torch.Tensor, stream: Optional.cuda.Stream = None) -> torch.Tensor:
        """Performs an in-place all-reduce operation across tensor-parallel ranks."""
        if not tensor.is_cuda:
            raise ValueError("[NexusTP] Tensor must reside on GPU for NCCL communication.")
        
        if self.coordinator is not None:
            cuda_stream_ptr = stream.cuda_stream if stream else torch.cuda.current_stream().cuda_stream
            self.coordinator.all_reduce(tensor, cuda_stream_ptr)
        else:
            # Fallback mock synchronization for single-node development
            torch.cuda.synchronize(self.device_id)
            
        return tensor

    @staticmethod
    def generate_unique_id() -> List[int]:
        """Generates a raw NCCL unique ID byte array from Rank 0."""
        if nexus_nccl_fabric is not None:
            id_bytes = nexus_nccl_fabric.NexusNCCLCoordinator.get_unique_id()
            return list(id_bytes)
        else:
            return [0] * 128 # Mock ID size