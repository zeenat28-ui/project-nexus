import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension

is_rocm = os.environ.get("ROCM_PATH") is not None or os.path.exists("/opt/rocm")

extra_compile_args = {"cxx": ["-O3", "-std=c++17"]}
if is_rocm:
    extra_compile_args["cxx"].extend(["-D__HIP_PLATFORM_AMD__=1", "-DUSE_ROCM=1"])

ext_modules = [
    CppExtension(
        name="nexus_arena",
        sources=["core/nexus_arena.cpp"],
        extra_compile_args=extra_compile_args
    ),
    CppExtension(
        name="nexus_nccl_fabric",
        sources=["core/nexus_nccl_fabric.cpp"],
        libraries=["rccl" if is_rocm else "nccl"],
        extra_compile_args=extra_compile_args
    ),
    CppExtension(
        name="nexus_health_monitor",
        sources=["core/nexus_health_monitor.cpp"],
        libraries=["amd_smi" if is_rocm else "nvidia-ml"],
        extra_compile_args=extra_compile_args
    ),
    CppExtension(
        name="nexus_weights_shm",
        sources=["core/nexus_weights_shm.cpp"],
        extra_compile_args=extra_compile_args
    ),
]

setup(
    name="project_nexus",
    version="1.0.0",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch>=2.0.0", "fastapi", "uvicorn", "pydantic"],
)