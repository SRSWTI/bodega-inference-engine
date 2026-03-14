#!/usr/bin/env python3
"""
Hardware detection for Apple Silicon (same logic as llm_context_benchmarks/benchmark_common.py).

Uses psutil + system_profiler for CPU cores (P/E breakdown), GPU cores, and RAM.
"""

import platform
import re
import subprocess
from typing import Any


def get_hardware_info() -> dict[str, Any]:
    """Get hardware information for the system."""
    info: dict[str, Any] = {}

    info["platform"] = platform.platform()
    info["processor"] = platform.processor()
    info["machine"] = platform.machine()

    try:
        import psutil
        info["cpu_count"] = psutil.cpu_count(logical=False)
        info["cpu_count_logical"] = psutil.cpu_count(logical=True)
        mem = psutil.virtual_memory()
        info["memory_gb"] = round(mem.total / (1024**3), 1)
    except ImportError:
        info["cpu_count"] = 0
        info["cpu_count_logical"] = 0
        info["memory_gb"] = 0

    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["system_profiler", "SPHardwareDataType"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                output = result.stdout
                chip_match = re.search(r"Chip:\s+(.+)", output)
                if chip_match:
                    info["chip"] = chip_match.group(1).strip()
                model_match = re.search(r"Model Name:\s+(.+)", output)
                if model_match:
                    info["model"] = model_match.group(1).strip()
                cores_match = re.search(
                    r"Total Number of Cores:\s+(\d+)(?:\s*\((\d+)\s+performance\s+and\s+(\d+)\s+efficiency\))?",
                    output,
                )
                if cores_match:
                    info["total_cores"] = int(cores_match.group(1))
                    if cores_match.group(2) and cores_match.group(3):
                        info["performance_cores"] = int(cores_match.group(2))
                        info["efficiency_cores"] = int(cores_match.group(3))

            gpu_result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True, text=True, timeout=5,
            )
            if gpu_result.returncode == 0:
                gpu_cores_match = re.search(r"Total Number of Cores:\s+(\d+)", gpu_result.stdout)
                if gpu_cores_match:
                    info["gpu_cores"] = int(gpu_cores_match.group(1))
        except Exception:
            pass

    return info


def format_hardware_string(hw_info: dict[str, Any]) -> str:
    """Format hardware info into a readable string."""
    parts = []
    if "chip" in hw_info:
        parts.append(hw_info["chip"])
    elif "processor" in hw_info and hw_info["processor"]:
        parts.append(str(hw_info["processor"])[:50])
    if "memory_gb" in hw_info:
        parts.append(f"{hw_info['memory_gb']}GB RAM")
    if "performance_cores" in hw_info and "efficiency_cores" in hw_info:
        parts.append(f"{hw_info['total_cores']} CPU cores ({hw_info['performance_cores']}P+{hw_info['efficiency_cores']}E)")
    elif "total_cores" in hw_info:
        parts.append(f"{hw_info['total_cores']} CPU cores")
    elif "cpu_count" in hw_info:
        parts.append(f"{hw_info['cpu_count']} CPU cores")
    if "gpu_cores" in hw_info:
        parts.append(f"{hw_info['gpu_cores']} GPU cores")
    return ", ".join(parts) if parts else "Unknown hardware"


if __name__ == "__main__":
    hw = get_hardware_info()
    print(format_hardware_string(hw))
