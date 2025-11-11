"""
RAPIDS Memory-Intensive Benchmark
Large-scale DataFrame operations and GPU computing using RAPIDS (cuDF, cuPy)
"""

import os
import sys
import yaml
import argparse
import time
from typing import Dict, List
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import GPUMonitor, BenchmarkLogger, create_result_dict, print_result_table


def load_config(config_path: str = "config/benchmark_config.yaml") -> Dict:
    """Load benchmark configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def check_rapids_installation():
    """Check if RAPIDS libraries are installed"""
    try:
        import cudf
        import cupy
        print("RAPIDS libraries found:")
        print(f"  cuDF version: {cudf.__version__}")
        print(f"  CuPy version: {cupy.__version__}")
        return True
    except ImportError as e:
        print(f"Error: RAPIDS not installed properly: {e}")
        print("\nInstall RAPIDS with:")
        print("  pip install cudf-cu12 cupy-cuda12x --extra-index-url=https://pypi.nvidia.com")
        return False


def benchmark_dataframe_join(
    num_rows: int,
    join_size: int,
    gpu_monitor: GPUMonitor
) -> Dict:
    """
    Benchmark large-scale DataFrame join operation
    
    Args:
        num_rows: Number of rows in main DataFrame
        join_size: Number of rows in join DataFrame
        gpu_monitor: GPU monitor instance
    
    Returns:
        Dictionary with results
    """
    import cudf
    
    print(f"\n{'='*80}")
    print(f"DataFrame Join Benchmark")
    print(f"Main table: {num_rows:,} rows")
    print(f"Join table: {join_size:,} rows")
    print(f"{'='*80}\n")
    
    # Create synthetic data
    print("Generating synthetic data...")
    
    # Main DataFrame
    df1 = cudf.DataFrame({
        'id': cudf.Series(range(num_rows)),
        'value1': cudf.Series(np.random.randn(num_rows)),
        'value2': cudf.Series(np.random.randn(num_rows)),
        'category': cudf.Series(np.random.randint(0, 100, num_rows)),
        'text': cudf.Series(['text_' + str(i % 1000) for i in range(num_rows)])
    })
    
    # Join DataFrame
    df2 = cudf.DataFrame({
        'id': cudf.Series(np.random.randint(0, num_rows, join_size)),
        'join_value': cudf.Series(np.random.randn(join_size)),
        'join_category': cudf.Series(np.random.randint(0, 50, join_size))
    })
    
    print(f"Data generated:")
    print(f"  df1: {df1.shape}")
    print(f"  df2: {df2.shape}")
    print(f"  Total memory: ~{(df1.memory_usage(deep=True).sum() + df2.memory_usage(deep=True).sum()) / 1e9:.2f} GB")
    
    # Start monitoring
    gpu_monitor.start()
    start_time = time.time()
    
    # Perform join
    print("\nPerforming join operation...")
    result = df1.merge(df2, on='id', how='inner')
    
    # Force computation by accessing result
    result_len = len(result)
    
    # Stop monitoring
    end_time = time.time()
    metrics = gpu_monitor.stop()
    
    runtime = end_time - start_time
    throughput = (num_rows + join_size) / runtime
    
    print(f"\nJoin completed:")
    print(f"  Runtime: {runtime:.2f}s")
    print(f"  Result size: {result_len:,} rows")
    print(f"  Throughput: {throughput:.2f} rows/s")
    
    return {
        'operation': 'join',
        'runtime': runtime,
        'throughput': throughput,
        'input_rows': num_rows + join_size,
        'output_rows': result_len,
        'metrics': metrics
    }


def benchmark_groupby_aggregate(
    num_rows: int,
    num_groups: int,
    gpu_monitor: GPUMonitor
) -> Dict:
    """
    Benchmark GroupBy and aggregation operations
    
    Args:
        num_rows: Number of rows
        num_groups: Number of groups
        gpu_monitor: GPU monitor instance
    
    Returns:
        Dictionary with results
    """
    import cudf
    
    print(f"\n{'='*80}")
    print(f"GroupBy + Aggregate Benchmark")
    print(f"Rows: {num_rows:,}")
    print(f"Groups: {num_groups:,}")
    print(f"{'='*80}\n")
    
    # Create synthetic data
    print("Generating synthetic data...")
    df = cudf.DataFrame({
        'group_id': cudf.Series(np.random.randint(0, num_groups, num_rows)),
        'value1': cudf.Series(np.random.randn(num_rows)),
        'value2': cudf.Series(np.random.randn(num_rows)),
        'value3': cudf.Series(np.random.randn(num_rows)),
        'count': cudf.Series(np.random.randint(1, 100, num_rows))
    })
    
    print(f"Data generated: {df.shape}")
    print(f"  Memory: ~{df.memory_usage(deep=True).sum() / 1e9:.2f} GB")
    
    # Start monitoring
    gpu_monitor.start()
    start_time = time.time()
    
    # Perform groupby with multiple aggregations
    print("\nPerforming GroupBy + Aggregate...")
    result = df.groupby('group_id').agg({
        'value1': ['mean', 'std', 'min', 'max'],
        'value2': ['sum', 'mean'],
        'value3': 'mean',
        'count': 'sum'
    })
    
    # Force computation
    result_len = len(result)
    
    # Stop monitoring
    end_time = time.time()
    metrics = gpu_monitor.stop()
    
    runtime = end_time - start_time
    throughput = num_rows / runtime
    
    print(f"\nGroupBy completed:")
    print(f"  Runtime: {runtime:.2f}s")
    print(f"  Result groups: {result_len:,}")
    print(f"  Throughput: {throughput:.2f} rows/s")
    
    return {
        'operation': 'groupby_aggregate',
        'runtime': runtime,
        'throughput': throughput,
        'input_rows': num_rows,
        'output_rows': result_len,
        'metrics': metrics
    }


def benchmark_sort_filter(
    num_rows: int,
    gpu_monitor: GPUMonitor
) -> Dict:
    """
    Benchmark sort and filter operations
    
    Args:
        num_rows: Number of rows
        gpu_monitor: GPU monitor instance
    
    Returns:
        Dictionary with results
    """
    import cudf
    
    print(f"\n{'='*80}")
    print(f"Sort + Filter Benchmark")
    print(f"Rows: {num_rows:,}")
    print(f"{'='*80}\n")
    
    # Create synthetic data
    print("Generating synthetic data...")
    df = cudf.DataFrame({
        'id': cudf.Series(np.random.permutation(num_rows)),
        'value': cudf.Series(np.random.randn(num_rows)),
        'category': cudf.Series(np.random.randint(0, 100, num_rows)),
        'timestamp': cudf.Series(range(num_rows))
    })
    
    print(f"Data generated: {df.shape}")
    print(f"  Memory: ~{df.memory_usage(deep=True).sum() / 1e9:.2f} GB")
    
    # Start monitoring
    gpu_monitor.start()
    start_time = time.time()
    
    # Sort by multiple columns
    print("\nPerforming sort...")
    df_sorted = df.sort_values(['category', 'value'])
    
    # Filter operation
    print("Performing filter...")
    df_filtered = df_sorted[
        (df_sorted['value'] > 0) &
        (df_sorted['category'] > 25) &
        (df_sorted['category'] < 75)
    ]
    
    
    # Force computation
    result_len = len(df_filtered)
    
    # Stop monitoring
    end_time = time.time()
    metrics = gpu_monitor.stop()
    
    runtime = end_time - start_time
    throughput = num_rows / runtime
    
    print(f"\nSort + Filter completed:")
    print(f"  Runtime: {runtime:.2f}s")
    print(f"  Filtered rows: {result_len:,} ({100*result_len/num_rows:.1f}%)")
    print(f"  Throughput: {throughput:.2f} rows/s")
    
    return {
        'operation': 'sort_filter',
        'runtime': runtime,
        'throughput': throughput,
        'input_rows': num_rows,
        'output_rows': result_len,
        'metrics': metrics
    }


def benchmark_matrix_operations(
    matrix_size: List[int],
    num_iterations: int,
    gpu_monitor: GPUMonitor
) -> Dict:
    """
    Benchmark large matrix operations using CuPy
    
    Args:
        matrix_size: Matrix dimensions [rows, cols]
        num_iterations: Number of iterations
        gpu_monitor: GPU monitor instance
    
    Returns:
        Dictionary with results
    """
    import cupy as cp
    
    print(f"\n{'='*80}")
    print(f"Matrix Operations Benchmark")
    print(f"Matrix size: {matrix_size[0]:,} x {matrix_size[1]:,}")
    print(f"Iterations: {num_iterations}")
    print(f"{'='*80}\n")
    
    # Create large matrices
    print("Generating matrices...")
    rows, cols = matrix_size
    
    A = cp.random.randn(rows, cols, dtype=cp.float32)
    B = cp.random.randn(cols, rows, dtype=cp.float32)
    
    matrix_memory = (A.nbytes + B.nbytes) / 1e9
    print(f"Matrix memory: {matrix_memory:.2f} GB")
    
    # Start monitoring
    gpu_monitor.start()
    start_time = time.time()
    
    # Perform matrix operations
    print("\nPerforming matrix operations...")
    for i in range(num_iterations):
        # Matrix multiplication
        C = cp.matmul(A, B)
        
        # Element-wise operations
        D = cp.sqrt(cp.abs(C)) + cp.sin(C)
        
        # Reduction operations
        mean_val = cp.mean(D)
        std_val = cp.std(D)
        
        # Force synchronization
        cp.cuda.Stream.null.synchronize()
        
        if (i + 1) % max(1, num_iterations // 5) == 0:
            print(f"  Iteration {i + 1}/{num_iterations} completed")
    
    # Stop monitoring
    end_time = time.time()
    metrics = gpu_monitor.stop()
    
    runtime = end_time - start_time
    throughput = num_iterations / runtime
    
    # Calculate FLOPs
    matmul_flops = 2 * rows * cols * rows  # per iteration
    total_flops = matmul_flops * num_iterations
    tflops = total_flops / runtime / 1e12
    
    print(f"\nMatrix operations completed:")
    print(f"  Runtime: {runtime:.2f}s")
    print(f"  Throughput: {throughput:.2f} iterations/s")
    print(f"  Performance: {tflops:.2f} TFLOPS")
    
    return {
        'operation': 'matrix_ops',
        'runtime': runtime,
        'throughput': throughput,
        'iterations': num_iterations,
        'tflops': tflops,
        'matrix_size': matrix_size,
        'metrics': metrics
    }


def run_benchmark(
    config: Dict,
    platform: str,
    log_dir: str
):
    """
    Run complete RAPIDS benchmark suite
    
    Args:
        config: RAPIDS configuration
        platform: Platform name
        log_dir: Directory for logs
    """
    if not check_rapids_installation():
        return
    
    num_rows = config['num_rows']
    join_size = config['join_size']
    operations = config['operations']
    matrix_size = config['matrix_size']
    num_iterations = config['num_iterations']
    
    # Initialize logger
    logger = BenchmarkLogger(
        log_dir=log_dir,
        platform=platform,
        benchmark_type='rapids_memory'
    )
    
    # Log configuration
    logger.log_config({
        'num_rows': num_rows,
        'join_size': join_size,
        'operations': operations,
        'matrix_size': matrix_size,
        'num_iterations': num_iterations,
        'platform': platform
    })
    
    results = []
    
    # Run DataFrame operations
    if 'join' in operations:
        monitor = GPUMonitor(device_id=0, sample_interval=0.5)
        result = benchmark_dataframe_join(num_rows, join_size, monitor)
        
        gpu_summary = monitor.get_summary()
        result_dict = create_result_dict(
            model_name='cuDF_join',
            batch_size=1,
            num_samples=result['input_rows'],
            runtime_s=result['runtime'],
            throughput=result['throughput'],
            gpu_summary=gpu_summary,
            platform=platform,
            additional_metrics={
                'operation': result['operation'],
                'output_rows': result['output_rows']
            }
        )
        results.append(result_dict)
        logger.log_gpu_metrics(result['metrics'])
    
    if 'groupby' in operations:
        monitor = GPUMonitor(device_id=0, sample_interval=0.5)
        result = benchmark_groupby_aggregate(num_rows, 10000, monitor)
        
        gpu_summary = monitor.get_summary()
        result_dict = create_result_dict(
            model_name='cuDF_groupby',
            batch_size=1,
            num_samples=result['input_rows'],
            runtime_s=result['runtime'],
            throughput=result['throughput'],
            gpu_summary=gpu_summary,
            platform=platform,
            additional_metrics={
                'operation': result['operation'],
                'output_rows': result['output_rows']
            }
        )
        results.append(result_dict)
        logger.log_gpu_metrics(result['metrics'])
    
    if 'sort' in operations or 'filter' in operations:
        monitor = GPUMonitor(device_id=0, sample_interval=0.5)
        result = benchmark_sort_filter(num_rows, monitor)
        
        gpu_summary = monitor.get_summary()
        result_dict = create_result_dict(
            model_name='cuDF_sort_filter',
            batch_size=1,
            num_samples=result['input_rows'],
            runtime_s=result['runtime'],
            throughput=result['throughput'],
            gpu_summary=gpu_summary,
            platform=platform,
            additional_metrics={
                'operation': result['operation'],
                'output_rows': result['output_rows']
            }
        )
        results.append(result_dict)
        logger.log_gpu_metrics(result['metrics'])
    
    # Run matrix operations
    monitor = GPUMonitor(device_id=0, sample_interval=0.5)
    result = benchmark_matrix_operations(matrix_size, num_iterations, monitor)
    
    gpu_summary = monitor.get_summary()
    result_dict = create_result_dict(
        model_name='cuPy_matrix',
        batch_size=1,
        num_samples=result['iterations'],
        runtime_s=result['runtime'],
        throughput=result['throughput'],
        gpu_summary=gpu_summary,
        platform=platform,
        additional_metrics={
            'operation': result['operation'],
            'tflops': result['tflops'],
            'matrix_size': str(result['matrix_size'])
        }
    )
    results.append(result_dict)
    logger.log_gpu_metrics(result['metrics'])
    
    # Save results
    logger.log_results(results)
    print_result_table(results)
    
    print(f"\n{'='*80}")
    print("RAPIDS Benchmark completed successfully!")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="RAPIDS Memory-Intensive Benchmark")
    parser.add_argument(
        '--platform',
        type=str,
        required=True,
        help='Platform name (e.g., dgx_spark, lambda_h100)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/benchmark_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default='./logs',
        help='Directory for logs'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    rapids_config = config['memory_tasks']['rapids']
    
    # Run benchmark
    try:
        run_benchmark(
            config=rapids_config,
            platform=args.platform,
            log_dir=args.log_dir
        )
    except Exception as e:
        print(f"Error running benchmark: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()