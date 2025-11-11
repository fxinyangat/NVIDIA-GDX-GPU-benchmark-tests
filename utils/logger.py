"""
Logging Utilities
Functions for saving benchmark results and metrics to CSV files
"""

import os
import csv
import json
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd


class BenchmarkLogger:
    """Logger for benchmark results and metrics"""
    
    def __init__(self, log_dir: str, platform: str, benchmark_type: str):
        """
        Initialize logger
        
        Args:
            log_dir: Directory to save logs
            platform: Platform name (e.g., 'dgx_spark', 'lambda_h100')
            benchmark_type: Type of benchmark (e.g., 'text_inference', 'lora_finetune')
        """
        self.log_dir = log_dir
        self.platform = platform
        self.benchmark_type = benchmark_type
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Generate timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Define log file paths
        self.results_file = os.path.join(
            log_dir, 
            f"{platform}_{benchmark_type}_{self.timestamp}.csv"
        )
        self.metrics_file = os.path.join(
            log_dir,
            f"{platform}_{benchmark_type}_{self.timestamp}_metrics.csv"
        )
        self.config_file = os.path.join(
            log_dir,
            f"{platform}_{benchmark_type}_{self.timestamp}_config.json"
        )
        
        print(f"Logger initialized:")
        print(f"  Results: {self.results_file}")
        print(f"  Metrics: {self.metrics_file}")
        print(f"  Config: {self.config_file}")
    
    def log_config(self, config: Dict):
        """Save benchmark configuration"""
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Configuration saved to {self.config_file}")
    
    def log_results(self, results: List[Dict]):
        """
        Save benchmark results to CSV
        
        Args:
            results: List of result dictionaries
        """
        if not results:
            print("Warning: No results to log")
            return
        
        # Convert to DataFrame for easier handling
        df = pd.DataFrame(results)
        
        # Save to CSV
        df.to_csv(self.results_file, index=False)
        print(f"Results saved to {self.results_file}")
        
        # Print summary
        print("\nResults Summary:")
        print(df.to_string(index=False))
    
    def log_gpu_metrics(self, metrics: List[Dict]):
        """
        Save GPU monitoring metrics to CSV
        
        Args:
            metrics: List of metric dictionaries from GPUMonitor
        """
        if not metrics:
            print("Warning: No GPU metrics to log")
            return
        
        df = pd.DataFrame(metrics)
        df.to_csv(self.metrics_file, index=False)
        print(f"GPU metrics saved to {self.metrics_file} ({len(metrics)} samples)")
    
    def log_summary(self, summary: Dict):
        """
        Append summary statistics to main results file
        
        Args:
            summary: Summary dictionary (e.g., from GPUMonitor.get_summary())
        """
        # Add metadata
        summary['platform'] = self.platform
        summary['benchmark_type'] = self.benchmark_type
        summary['timestamp'] = self.timestamp
        
        # Append to summary file
        summary_file = os.path.join(self.log_dir, "benchmark_summary.csv")
        
        # Check if file exists to determine if we need headers
        file_exists = os.path.exists(summary_file)
        
        with open(summary_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=summary.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(summary)
        
        print(f"Summary appended to {summary_file}")


def create_result_dict(
    model_name: str,
    batch_size: int,
    num_samples: int,
    runtime_s: float,
    throughput: float,
    gpu_summary: Dict,
    platform: str,
    additional_metrics: Dict = None
) -> Dict:
    """
    Create standardized result dictionary
    
    Args:
        model_name: Name of the model
        batch_size: Batch size used
        num_samples: Number of samples processed
        runtime_s: Total runtime in seconds
        throughput: Throughput (samples/sec or tokens/sec)
        gpu_summary: GPU metrics summary from GPUMonitor
        platform: Platform name
        additional_metrics: Additional custom metrics
    
    Returns:
        Dictionary with all metrics
    """
    result = {
        'model': model_name,
        'platform': platform,
        'batch_size': batch_size,
        'num_samples': num_samples,
        'runtime_s': runtime_s,
        'throughput': throughput,
        
        # GPU metrics
        'gpu_name': gpu_summary.get('gpu_name', 'Unknown'),
        'gpu_util_mean': gpu_summary.get('gpu_util_mean', 0),
        'gpu_util_max': gpu_summary.get('gpu_util_max', 0),
        'memory_used_mb_peak': gpu_summary.get('memory_used_mb_peak', 0),
        'memory_percent_peak': gpu_summary.get('memory_percent_peak', 0),
        'memory_total_mb': gpu_summary.get('memory_total_mb', 0),
        'temperature_c_mean': gpu_summary.get('temperature_c_mean', 0),
        'temperature_c_max': gpu_summary.get('temperature_c_max', 0),
        'power_draw_w_mean': gpu_summary.get('power_draw_w_mean', 0),
        'power_draw_w_max': gpu_summary.get('power_draw_w_max', 0),
        'energy_consumed_wh': gpu_summary.get('energy_consumed_wh', 0),
        
        'timestamp': datetime.now().isoformat(),
    }
    
    # Add any additional metrics
    if additional_metrics:
        result.update(additional_metrics)
    
    return result


def calculate_cost(
    runtime_s: float,
    cost_per_hour: float
) -> float:
    """
    Calculate cost for a benchmark run
    
    Args:
        runtime_s: Runtime in seconds
        cost_per_hour: Cost per hour in USD
    
    Returns:
        Total cost in USD
    """
    runtime_hours = runtime_s / 3600.0
    return runtime_hours * cost_per_hour


def print_result_table(results: List[Dict]):
    """Print results in a formatted table"""
    if not results:
        print("No results to display")
        return
    
    df = pd.DataFrame(results)
    
    # Select key columns for display
    display_cols = [
        'model', 'batch_size', 'runtime_s', 'throughput',
        'gpu_util_mean', 'memory_used_mb_peak', 'power_draw_w_mean',
        'energy_consumed_wh'
    ]
    
    # Only include columns that exist
    display_cols = [col for col in display_cols if col in df.columns]
    
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    print(df[display_cols].to_string(index=False))
    print("="*80 + "\n")


def merge_log_files(log_dir: str, output_file: str = "all_results.csv"):
    """
    Merge all CSV log files in directory into single file
    
    Args:
        log_dir: Directory containing log files
        output_file: Output filename
    """
    import glob
    
    # Find all CSV files (excluding the output file itself)
    csv_files = glob.glob(os.path.join(log_dir, "*.csv"))
    csv_files = [f for f in csv_files if not f.endswith(output_file)]
    
    if not csv_files:
        print(f"No CSV files found in {log_dir}")
        return
    
    # Read and concatenate all files
    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            # Add source file column
            df['source_file'] = os.path.basename(csv_file)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not read {csv_file}: {e}")
    
    if dfs:
        merged_df = pd.concat(dfs, ignore_index=True, sort=False)
        output_path = os.path.join(log_dir, output_file)
        merged_df.to_csv(output_path, index=False)
        print(f"Merged {len(dfs)} files into {output_path}")
        return merged_df
    else:
        print("No valid CSV files to merge")
        return None


if __name__ == "__main__":
    # Test logger
    print("Testing BenchmarkLogger...")
    
    logger = BenchmarkLogger(
        log_dir="./test_logs",
        platform="test_platform",
        benchmark_type="test_benchmark"
    )
    
    # Test config logging
    test_config = {
        'model': 'test-model',
        'batch_size': 8,
        'num_samples': 100
    }
    logger.log_config(test_config)
    
    # Test results logging
    test_results = [
        create_result_dict(
            model_name='test-model',
            batch_size=8,
            num_samples=100,
            runtime_s=60.5,
            throughput=1.65,
            gpu_summary={
                'gpu_name': 'Test GPU',
                'gpu_util_mean': 85.5,
                'memory_used_mb_peak': 12000,
                'power_draw_w_mean': 300,
                'energy_consumed_wh': 5.0
            },
            platform='test_platform'
        )
    ]
    logger.log_results(test_results)
    print_result_table(test_results)
    
    print("\nTest completed successfully!")