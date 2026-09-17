#!/usr/bin/env python3
"""
TGCF-IDS: Environment & Hardware Diagnostic Tool.
Checks Python, PyTorch, CUDA, GPU acceleration, and PyTorch Geometric availability.
"""

import sys
import platform
from pathlib import Path


def run_diagnostics() -> dict:
    print("=" * 65)
    print("      TGCF-IDS : Research Environment Diagnostic Report       ")
    print("=" * 65)

    info = {}

    # 1. Python Environment & OS Details
    info["os"] = f"{platform.system()} {platform.release()} ({platform.machine()})"
    info["python_version"] = sys.version.split()[0]
    info["python_executable"] = sys.executable

    print(f"[*] Operating System    : {info['os']}")
    print(f"[*] Python Version      : {info['python_version']}")
    print(f"[*] Python Executable   : {info['python_executable']}")
    print("-" * 65)

    # 2. PyTorch & CUDA Diagnostics
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        print(f"[+] PyTorch Version     : {info['torch_version']}")
        print(f"[+] CUDA Available      : {info['cuda_available']}")

        if info["cuda_available"]:
            info["cuda_version"] = torch.version.cuda
            info["device_count"] = torch.cuda.device_count()
            info["device_name"] = torch.cuda.get_device_name(0)
            print(f"[+] CUDA Runtime Version: {info['cuda_version']}")
            print(f"[+] GPU Count           : {info['device_count']}")
            print(f"[+] GPU Device Name     : {info['device_name']}")
            
            # Memory check
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"[+] GPU VRAM Total      : {gpu_mem:.2f} GB")
        else:
            info["cuda_version"] = "N/A"
            info["device_name"] = "N/A"
            print("[-] CUDA is NOT available. PyTorch is running in CPU-only mode.")
            print("    If you have an NVIDIA GPU, ensure NVIDIA Drivers and CUDA-enabled PyTorch are installed.")
    except ImportError:
        info["torch_version"] = "Not Installed"
        info["cuda_available"] = False
        info["cuda_version"] = "N/A"
        info["device_name"] = "N/A"
        print("[-] PyTorch is NOT installed.")

    print("-" * 65)

    # 3. PyTorch Geometric (PyG) Diagnostics
    try:
        import torch_geometric
        info["pyg_version"] = torch_geometric.__version__
        print(f"[+] PyTorch Geometric   : {info['pyg_version']}")
    except ImportError:
        info["pyg_version"] = "Not Installed"
        print("[-] PyTorch Geometric (torch_geometric) is NOT installed.")

    # 4. Optional / Core Supporting Packages
    supporting_packages = ["numpy", "scipy", "pandas", "sklearn", "yaml", "networkx", "pytest"]
    print("-" * 65)
    print("[*] Core Research Dependencies:")
    for pkg_name in supporting_packages:
        try:
            mod = __import__(pkg_name)
            ver = getattr(mod, "__version__", "Installed")
            print(f"    - {pkg_name:<15}: {ver}")
        except ImportError:
            print(f"    - {pkg_name:<15}: Not Installed")

    print("=" * 65)
    return info


if __name__ == "__main__":
    run_diagnostics()
