#!/usr/bin/env python3
"""
Setup Verification Script
Checks if all dependencies are installed and system is ready for benchmarking
"""

import sys
import subprocess
from typing import Tuple, List

def check_python_version() -> Tuple[bool, str]:
    """Check Python version"""
    version = sys.version_info
    if version >= (3, 10):
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    else:
        return False, f"Python {version.major}.{version.minor}.{version.micro} (requires 3.10+)"

def check_package(package_name: str, import_name: str = None) -> Tuple[bool, str]:
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        return True, f"{package_name} {version}"
    except ImportError:
        return False, f"{package_name} not installed"

def check_cuda() -> Tuple[bool, str]:
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            return True, f"CUDA {cuda_version}, {device_count} GPU(s), {device_name}"
        else:
            return False, "CUDA not available in PyTorch"
    except ImportError:
        return False, "PyTorch not installed"

def check_nvidia_smi() -> Tuple[bool, str]:
    """Check nvidia-smi"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            driver_line = [l for l in lines if 'Driver Version' in l]
            if driver_line:
                return True, "nvidia-smi available, " + driver_line[0].strip()
            return True, "nvidia-smi available"
        else:
            return False, "nvidia-smi command failed"
    except FileNotFoundError:
        return False, "nvidia-smi not found"

def main():
    print("="*80)
    print("GPU Benchmark Suite - Setup Verification")
    print("="*80)
    print()
    
    checks: List[Tuple[str, Tuple[bool, str]]] = []
    
    # Core checks
    checks.append(("Python Version", check_python_version()))
    checks.append(("NVIDIA Driver", check_nvidia_smi()))
    checks.append(("CUDA", check_cuda()))
    
    # Essential packages
    essential_packages = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("datasets", "datasets"),
        ("accelerate", "accelerate"),
        ("peft", "peft"),
        ("bitsandbytes", "bitsandbytes"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("pyyaml", "yaml"),
        ("pynvml", "pynvml"),
    ]
    
    for pkg_name, import_name in essential_packages:
        checks.append((pkg_name, check_package(pkg_name, import_name)))
    
    # Optional packages
    optional_packages = [
        ("vllm", "vllm"),
        ("cudf", "cudf"),
        ("cupy", "cupy"),
        ("flash-attn", "flash_attn"),
    ]
    
    # Print results
    print("Essential Components:")
    print("-"*80)
    
    all_essential_ok = True
    for name, (status, message) in checks:
        symbol = "✓" if status else "✗"
        color = "\033[92m" if status else "\033[91m"
        reset = "\033[0m"
        print(f"{color}{symbol}{reset} {name:20s}: {message}")
        if not status:
            all_essential_ok = False
    
    print()
    print("Optional Components:")
    print("-"*80)
    
    for pkg_name, import_name in optional_packages:
        status, message = check_package(pkg_name, import_name)
        symbol = "✓" if status else "○"
        color = "\033[92m" if status else "\033[93m"
        reset = "\033[0m"
        print(f"{color}{symbol}{reset} {pkg_name:20s}: {message}")
    
    print()
    print("="*80)
    
    if all_essential_ok:
        print("\033[92m✓ All essential components are installed!\033[0m")
        print()
        print("Next steps:")
        print("  1. Review configuration: config/benchmark_config.yaml")
        print("  2. Run a test benchmark:")
        print("     python inference/text_inference.py --platform dgx_spark")
        print("  3. Run full suite:")
        print("     BENCHMARK_PLATFORM=dgx_spark bash run_all_benchmarks.sh")
        print()
        return 0
    else:
        print("\033[91m✗ Some essential components are missing!\033[0m")
        print()
        print("Installation instructions:")
        print("  pip install -r requirements.txt")
        print()
        print("For RAPIDS (optional, for memory benchmarks):")
        print("  pip install cudf-cu12 cupy-cuda12x --extra-index-url=https://pypi.nvidia.com")
        print()
        print("For vLLM (optional, for optimized inference):")
        print("  pip install vllm")
        print()
        return 1

if __name__ == "__main__":
    sys.exit(main())