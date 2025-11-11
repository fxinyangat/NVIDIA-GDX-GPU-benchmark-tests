"""
GPU Monitoring Utility
Tracks GPU metrics (utilization, memory, power, temperature) in background thread
Includes fallback to nvidia-smi for new GPUs with limited NVML support
"""

import time
import threading
import subprocess
import re
from typing import List, Dict, Optional
import numpy as np

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("Warning: pynvml not available. GPU monitoring will use nvidia-smi fallback.")


class GPUMonitor:
    """Monitor GPU metrics in background thread during benchmark execution"""
    
    def __init__(self, device_id: int = 0, sample_interval: float = 0.5):
        """
        Initialize GPU monitor
        
        Args:
            device_id: CUDA device ID to monitor
            sample_interval: Seconds between samples
        """
        self.device_id = device_id
        self.sample_interval = sample_interval
        self.metrics: List[Dict] = []
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.use_nvml = False
        self.use_smi = False
        
        # Try to initialize NVML
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                self.gpu_name = pynvml.nvmlDeviceGetName(self.handle)
                self.use_nvml = True
                print(f"GPU Monitor initialized for: {self.gpu_name}")
            except Exception as e:
                print(f"Warning: NVML initialization failed: {e}")
                self.handle = None
                self.use_nvml = False
        
        # Fallback to nvidia-smi
        if not self.use_nvml:
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader', f'--id={device_id}'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    self.gpu_name = result.stdout.strip()
                    self.use_smi = True
                    print(f"GPU Monitor using nvidia-smi for: {self.gpu_name}")
                else:
                    print("Warning: GPU monitoring unavailable")
                    self.gpu_name = "Unknown"
            except Exception as e:
                print(f"Warning: nvidia-smi not available: {e}")
                self.gpu_name = "Unknown"
    
    def _get_metrics_from_smi(self) -> Dict:
        """Get GPU metrics using nvidia-smi command"""
        try:
            result = subprocess.run([
                'nvidia-smi',
                f'--id={self.device_id}',
                '--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,power.limit',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=2)
            
            if result.returncode == 0:
                values = result.stdout.strip().split(', ')
                if len(values) >= 7:
                    return {
                        'gpu_utilization': float(values[0]) if values[0] != '[N/A]' else 0,
                        'memory_utilization': float(values[1]) if values[1] != '[N/A]' else 0,
                        'memory_used_mb': float(values[2]) if values[2] != '[N/A]' else 0,
                        'memory_total_mb': float(values[3]) if values[3] != '[N/A]' else 0,
                        'temperature_c': float(values[4]) if values[4] != '[N/A]' else 0,
                        'power_draw_w': float(values[5]) if values[5] != '[N/A]' else 0,
                        'power_limit_w': float(values[6]) if values[6] != '[N/A]' else 0,
                    }
        except Exception as e:
            pass
        
        return {
            'gpu_utilization': 0,
            'memory_utilization': 0,
            'memory_used_mb': 0,
            'memory_total_mb': 0,
            'temperature_c': 0,
            'power_draw_w': 0,
            'power_limit_w': 0,
        }
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        while self.monitoring:
            try:
                timestamp = time.time()
                
                # Try nvidia-smi first for new GPUs
                if self.use_smi:
                    smi_metrics = self._get_metrics_from_smi()
                    metric = {
                        'timestamp': timestamp,
                        'gpu_utilization': smi_metrics['gpu_utilization'],
                        'memory_utilization': smi_metrics['memory_utilization'],
                        'memory_used_mb': smi_metrics['memory_used_mb'],
                        'memory_total_mb': smi_metrics['memory_total_mb'],
                        'memory_percent': (smi_metrics['memory_used_mb'] / smi_metrics['memory_total_mb'] * 100) if smi_metrics['memory_total_mb'] > 0 else 0,
                        'temperature_c': smi_metrics['temperature_c'],
                        'power_draw_w': smi_metrics['power_draw_w'],
                        'power_limit_w': smi_metrics['power_limit_w'],
                        'sm_clock_mhz': 0,
                        'mem_clock_mhz': 0,
                    }
                    self.metrics.append(metric)
                    consecutive_errors = 0
                
                # Try NVML for older GPUs
                elif self.use_nvml:
                    # GPU utilization
                    try:
                        util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                        gpu_util = util.gpu
                        mem_util = util.memory
                    except:
                        gpu_util = 0
                        mem_util = 0
                    
                    # Memory info
                    try:
                        mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                        mem_used_mb = mem_info.used / (1024 ** 2)
                        mem_total_mb = mem_info.total / (1024 ** 2)
                        mem_percent = (mem_info.used / mem_info.total) * 100
                    except:
                        mem_used_mb = 0
                        mem_total_mb = 0
                        mem_percent = 0
                    
                    # Temperature
                    try:
                        temp = pynvml.nvmlDeviceGetTemperature(
                            self.handle, 
                            pynvml.NVML_TEMPERATURE_GPU
                        )
                    except:
                        temp = 0
                    
                    # Power draw
                    try:
                        power_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
                        power_w = power_mw / 1000.0
                        
                        # Power limit
                        power_limit_mw = pynvml.nvmlDeviceGetPowerManagementLimit(self.handle)
                        power_limit_w = power_limit_mw / 1000.0
                    except:
                        power_w = 0
                        power_limit_w = 0
                    
                    # Clock speeds
                    try:
                        sm_clock = pynvml.nvmlDeviceGetClockInfo(
                            self.handle,
                            pynvml.NVML_CLOCK_SM
                        )
                        mem_clock = pynvml.nvmlDeviceGetClockInfo(
                            self.handle,
                            pynvml.NVML_CLOCK_MEM
                        )
                    except:
                        sm_clock = 0
                        mem_clock = 0
                    
                    # Record metrics
                    metric = {
                        'timestamp': timestamp,
                        'gpu_utilization': gpu_util,
                        'memory_utilization': mem_util,
                        'memory_used_mb': mem_used_mb,
                        'memory_total_mb': mem_total_mb,
                        'memory_percent': mem_percent,
                        'temperature_c': temp,
                        'power_draw_w': power_w,
                        'power_limit_w': power_limit_w,
                        'sm_clock_mhz': sm_clock,
                        'mem_clock_mhz': mem_clock,
                    }
                    
                    self.metrics.append(metric)
                    consecutive_errors = 0  # Reset on success
                
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors <= 3:  # Only print first few errors
                    print(f"Warning: Error collecting metrics: {e}")
                
                # If too many consecutive errors, stop trying
                if consecutive_errors >= max_consecutive_errors:
                    print(f"Warning: Too many metric collection errors. Disabling monitoring.")
                    self.monitoring = False
                    break
            
            time.sleep(self.sample_interval)
    
    def start(self):
        """Start monitoring in background thread"""
        if not self.use_nvml and not self.use_smi:
            print("Warning: GPU monitoring unavailable, skipping metrics collection")
            return
        
        self.metrics = []
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        monitor_type = "nvidia-smi" if self.use_smi else "NVML"
        print(f"GPU monitoring started using {monitor_type} (sampling every {self.sample_interval}s)")
    
    def stop(self):
        """Stop monitoring and return collected metrics"""
        if self.monitoring:
            self.monitoring = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=2.0)
            print(f"GPU monitoring stopped ({len(self.metrics)} samples collected)")
        return self.metrics
    
    def get_summary(self) -> Dict:
        """Calculate summary statistics from collected metrics"""
        if not self.metrics:
            return {}
        
        # Extract arrays for each metric
        gpu_util = [m['gpu_utilization'] for m in self.metrics]
        mem_used = [m['memory_used_mb'] for m in self.metrics]
        mem_percent = [m['memory_percent'] for m in self.metrics]
        temp = [m['temperature_c'] for m in self.metrics]
        power = [m['power_draw_w'] for m in self.metrics]
        
        # Calculate time duration
        start_time = self.metrics[0]['timestamp']
        end_time = self.metrics[-1]['timestamp']
        duration_s = end_time - start_time
        
        # Calculate energy (integrate power over time)
        # Energy (Wh) = average_power (W) * duration (h)
        avg_power_w = np.mean(power)
        energy_wh = avg_power_w * (duration_s / 3600.0)
        
        summary = {
            'duration_s': duration_s,
            'num_samples': len(self.metrics),
            'gpu_name': self.gpu_name,
            
            # GPU utilization stats
            'gpu_util_mean': np.mean(gpu_util),
            'gpu_util_std': np.std(gpu_util),
            'gpu_util_min': np.min(gpu_util),
            'gpu_util_max': np.max(gpu_util),
            
            # Memory stats
            'memory_used_mb_mean': np.mean(mem_used),
            'memory_used_mb_peak': np.max(mem_used),
            'memory_percent_mean': np.mean(mem_percent),
            'memory_percent_peak': np.max(mem_percent),
            'memory_total_mb': self.metrics[0]['memory_total_mb'],
            
            # Temperature stats
            'temperature_c_mean': np.mean(temp),
            'temperature_c_max': np.max(temp),
            
            # Power stats
            'power_draw_w_mean': avg_power_w,
            'power_draw_w_max': np.max(power),
            'power_limit_w': self.metrics[0]['power_limit_w'],
            'energy_consumed_wh': energy_wh,
        }
        
        return summary
    
    def __del__(self):
        """Cleanup NVML"""
        try:
            if hasattr(self, 'monitoring') and self.monitoring:
                self.stop()
            pynvml.nvmlShutdown()
        except:
            pass


def get_gpu_info(device_id: int = 0) -> Dict:
    """Get static GPU information"""
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        
        info = {
            'name': pynvml.nvmlDeviceGetName(handle),
            'compute_capability': pynvml.nvmlDeviceGetCudaComputeCapability(handle),
            'memory_total_mb': pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024 ** 2),
            'pci_bus_id': pynvml.nvmlDeviceGetPciInfo(handle).busId,
        }
        
        pynvml.nvmlShutdown()
        return info
    except Exception as e:
        print(f"Warning: Could not get GPU info: {e}")
        return {}


if __name__ == "__main__":
    # Test GPU monitoring
    print("Testing GPU Monitor...")
    
    info = get_gpu_info()
    print(f"\nGPU Info:")
    for k, v in info.items():
        print(f"  {k}: {v}")
    
    print("\nStarting 5-second monitoring test...")
    monitor = GPUMonitor(device_id=0, sample_interval=0.5)
    monitor.start()
    
    # Simulate some GPU work
    import torch
    device = torch.device("cuda")
    x = torch.randn(10000, 10000, device=device)
    for _ in range(10):
        y = torch.matmul(x, x)
        time.sleep(0.5)
    
    metrics = monitor.stop()
    summary = monitor.get_summary()
    
    print(f"\nMonitoring Summary:")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")