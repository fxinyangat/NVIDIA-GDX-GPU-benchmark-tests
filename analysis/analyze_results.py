"""
Analysis Script
Compare benchmark results between DGX Spark and Lambda GPUs
Generate comparison reports and visualizations
"""

import os
import sys
import glob
import yaml
import argparse
from typing import Dict, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def load_config(config_path: str = "config/benchmark_config.yaml") -> Dict:
    """Load benchmark configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_all_results(log_dir: str) -> pd.DataFrame:
    """
    Load all CSV result files from log directory
    
    Args:
        log_dir: Directory containing log files
    
    Returns:
        Combined DataFrame with all results
    """
    print(f"Loading results from: {log_dir}")
    
    # Find all CSV files
    csv_files = glob.glob(os.path.join(log_dir, "*.csv"))
    csv_files = [f for f in csv_files if not f.endswith('_metrics.csv')]
    
    if not csv_files:
        print(f"No result files found in {log_dir}")
        return pd.DataFrame()
    
    print(f"Found {len(csv_files)} result files")
    
    # Load and combine
    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            df['source_file'] = os.path.basename(csv_file)
            dfs.append(df)
            print(f"  Loaded: {os.path.basename(csv_file)} ({len(df)} rows)")
        except Exception as e:
            print(f"  Warning: Could not load {csv_file}: {e}")
    
    if not dfs:
        return pd.DataFrame()
    
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal results: {len(combined_df)} rows")
    print(f"Platforms: {combined_df['platform'].unique().tolist()}")
    
    return combined_df


def calculate_cost_efficiency(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Calculate cost per job and cost efficiency metrics
    
    Args:
        df: Results DataFrame
        config: Benchmark configuration with platform costs
    
    Returns:
        DataFrame with cost metrics added
    """
    print("\nCalculating cost efficiency...")
    
    # Add cost per hour for each platform
    platform_costs = {
        platform: info['cost_per_hour']
        for platform, info in config['platforms'].items()
    }
    
    df['cost_per_hour'] = df['platform'].map(platform_costs)
    
    # Calculate cost per job
    df['runtime_hours'] = df['runtime_s'] / 3600.0
    df['cost_per_job'] = df['runtime_hours'] * df['cost_per_hour']
    
    # Calculate cost per sample/token
    if 'num_samples' in df.columns:
        df['cost_per_sample'] = df['cost_per_job'] / df['num_samples']
    
    if 'tokens_per_second' in df.columns:
        df['cost_per_1k_tokens'] = df['cost_per_job'] / (df['tokens_per_second'] * df['runtime_s'] / 1000)
    
    # Efficiency score (throughput per dollar)
    df['throughput_per_dollar'] = df['throughput'] / df['cost_per_hour']
    
    return df


def create_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create comparison table between platforms
    
    Args:
        df: Results DataFrame
    
    Returns:
        Comparison table
    """
    print("\nCreating comparison table...")
    
    # Group by model and platform
    comparison_cols = [
        'model', 'platform', 'runtime_s', 'throughput',
        'gpu_util_mean', 'memory_used_mb_peak', 'power_draw_w_mean',
        'energy_consumed_wh', 'cost_per_job', 'throughput_per_dollar'
    ]
    
    # Filter columns that exist
    comparison_cols = [col for col in comparison_cols if col in df.columns]
    
    comparison = df[comparison_cols].copy()
    
    # Round numeric columns
    numeric_cols = comparison.select_dtypes(include=[np.number]).columns
    comparison[numeric_cols] = comparison[numeric_cols].round(2)
    
    return comparison


def plot_performance_comparison(df: pd.DataFrame, output_dir: str):
    """
    Plot performance comparison charts
    
    Args:
        df: Results DataFrame
        output_dir: Directory to save plots
    """
    print("\nGenerating performance comparison plots...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Runtime comparison
    if 'model' in df.columns and len(df['platform'].unique()) > 1:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        df_pivot = df.pivot_table(
            values='runtime_s',
            index='model',
            columns='platform',
            aggfunc='mean'
        )
        
        df_pivot.plot(kind='bar', ax=ax)
        ax.set_title('Runtime Comparison by Model', fontsize=14, fontweight='bold')
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Runtime (seconds)', fontsize=12)
        ax.legend(title='Platform')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'runtime_comparison.png'), dpi=300)
        plt.close()
        print(f"  Saved: runtime_comparison.png")
    
    # 2. Throughput comparison
    if 'throughput' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        df_pivot = df.pivot_table(
            values='throughput',
            index='model',
            columns='platform',
            aggfunc='mean'
        )
        
        df_pivot.plot(kind='bar', ax=ax)
        ax.set_title('Throughput Comparison by Model', fontsize=14, fontweight='bold')
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Throughput (samples/s)', fontsize=12)
        ax.legend(title='Platform')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'throughput_comparison.png'), dpi=300)
        plt.close()
        print(f"  Saved: throughput_comparison.png")
    
    # 3. Energy efficiency
    if 'energy_consumed_wh' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        df_pivot = df.pivot_table(
            values='energy_consumed_wh',
            index='model',
            columns='platform',
            aggfunc='mean'
        )
        
        df_pivot.plot(kind='bar', ax=ax)
        ax.set_title('Energy Consumption by Model', fontsize=14, fontweight='bold')
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Energy (Wh)', fontsize=12)
        ax.legend(title='Platform')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'energy_comparison.png'), dpi=300)
        plt.close()
        print(f"  Saved: energy_comparison.png")
    
    # 4. Cost efficiency
    if 'throughput_per_dollar' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        df_pivot = df.pivot_table(
            values='throughput_per_dollar',
            index='model',
            columns='platform',
            aggfunc='mean'
        )
        
        df_pivot.plot(kind='bar', ax=ax)
        ax.set_title('Cost Efficiency (Throughput per Dollar)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Samples/s per $', fontsize=12)
        ax.legend(title='Platform')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cost_efficiency_comparison.png'), dpi=300)
        plt.close()
        print(f"  Saved: cost_efficiency_comparison.png")
    
    # 5. GPU utilization
    if 'gpu_util_mean' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        df_pivot = df.pivot_table(
            values='gpu_util_mean',
            index='model',
            columns='platform',
            aggfunc='mean'
        )
        
        df_pivot.plot(kind='bar', ax=ax)
        ax.set_title('GPU Utilization by Model', fontsize=14, fontweight='bold')
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('GPU Utilization (%)', fontsize=12)
        ax.legend(title='Platform')
        ax.set_ylim([0, 100])
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'gpu_utilization_comparison.png'), dpi=300)
        plt.close()
        print(f"  Saved: gpu_utilization_comparison.png")
    
    # 6. Memory usage
    if 'memory_used_mb_peak' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        df_pivot = df.pivot_table(
            values='memory_used_mb_peak',
            index='model',
            columns='platform',
            aggfunc='mean'
        )
        
        # Convert to GB
        df_pivot = df_pivot / 1024
        
        df_pivot.plot(kind='bar', ax=ax)
        ax.set_title('Peak Memory Usage by Model', fontsize=14, fontweight='bold')
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Memory (GB)', fontsize=12)
        ax.legend(title='Platform')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'memory_usage_comparison.png'), dpi=300)
        plt.close()
        print(f"  Saved: memory_usage_comparison.png")


def calculate_speedup(df: pd.DataFrame, baseline_platform: str) -> pd.DataFrame:
    """
    Calculate speedup compared to baseline platform
    
    Args:
        df: Results DataFrame
        baseline_platform: Baseline platform name
    
    Returns:
        DataFrame with speedup metrics
    """
    print(f"\nCalculating speedup (baseline: {baseline_platform})...")
    
    speedup_results = []
    
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        
        baseline = model_df[model_df['platform'] == baseline_platform]
        if len(baseline) == 0:
            continue
        
        baseline_runtime = baseline['runtime_s'].values[0]
        baseline_throughput = baseline['throughput'].values[0]
        
        for platform in model_df['platform'].unique():
            if platform == baseline_platform:
                continue
            
            platform_df = model_df[model_df['platform'] == platform]
            if len(platform_df) == 0:
                continue
            
            platform_runtime = platform_df['runtime_s'].values[0]
            platform_throughput = platform_df['throughput'].values[0]
            
            speedup_results.append({
                'model': model,
                'platform': platform,
                'baseline_platform': baseline_platform,
                'speedup': baseline_runtime / platform_runtime,
                'throughput_improvement': platform_throughput / baseline_throughput,
            })
    
    if speedup_results:
        speedup_df = pd.DataFrame(speedup_results)
        print("\nSpeedup Summary:")
        print(speedup_df.to_string(index=False))
        return speedup_df
    else:
        return pd.DataFrame()


def generate_report(df: pd.DataFrame, config: Dict, output_dir: str):
    """
    Generate comprehensive comparison report
    
    Args:
        df: Results DataFrame
        config: Benchmark configuration
        output_dir: Directory to save report
    """
    print("\n" + "="*80)
    print("GENERATING COMPARISON REPORT")
    print("="*80)
    
    report_lines = []
    
    # Header
    report_lines.append("GPU BENCHMARK COMPARISON REPORT")
    report_lines.append("="*80)
    report_lines.append("")
    
    # Platform information
    report_lines.append("PLATFORMS:")
    for platform, info in config['platforms'].items():
        if platform in df['platform'].unique():
            report_lines.append(f"  {platform}:")
            report_lines.append(f"    Name: {info['name']}")
            report_lines.append(f"    GPU: {info['gpu_type']}")
            report_lines.append(f"    Cost: ${info['cost_per_hour']:.2f}/hour")
    report_lines.append("")
    
    # Summary statistics by platform
    report_lines.append("SUMMARY BY PLATFORM:")
    report_lines.append("-"*80)
    
    for platform in df['platform'].unique():
        platform_df = df[df['platform'] == platform]
        
        report_lines.append(f"\n{platform}:")
        report_lines.append(f"  Number of benchmarks: {len(platform_df)}")
        report_lines.append(f"  Average runtime: {platform_df['runtime_s'].mean():.2f}s")
        report_lines.append(f"  Average throughput: {platform_df['throughput'].mean():.2f} samples/s")
        
        if 'gpu_util_mean' in platform_df.columns:
            report_lines.append(f"  Average GPU utilization: {platform_df['gpu_util_mean'].mean():.1f}%")
        
        if 'memory_used_mb_peak' in platform_df.columns:
            report_lines.append(f"  Average peak memory: {platform_df['memory_used_mb_peak'].mean()/1024:.2f} GB")
        
        if 'power_draw_w_mean' in platform_df.columns:
            report_lines.append(f"  Average power draw: {platform_df['power_draw_w_mean'].mean():.2f} W")
        
        if 'energy_consumed_wh' in platform_df.columns:
            report_lines.append(f"  Total energy consumed: {platform_df['energy_consumed_wh'].sum():.2f} Wh")
        
        if 'cost_per_job' in platform_df.columns:
            report_lines.append(f"  Average cost per job: ${platform_df['cost_per_job'].mean():.4f}")
            report_lines.append(f"  Total cost: ${platform_df['cost_per_job'].sum():.2f}")
    
    report_lines.append("")
    report_lines.append("="*80)
    
    # Save report
    report_path = os.path.join(output_dir, 'comparison_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\nReport saved to: {report_path}")
    
    # Print report
    print("\n" + '\n'.join(report_lines))


def main():
    parser = argparse.ArgumentParser(description="Analyze Benchmark Results")
    parser.add_argument(
        '--log-dir',
        type=str,
        default='./logs',
        help='Directory containing log files'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/benchmark_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./analysis/results',
        help='Directory for analysis outputs'
    )
    parser.add_argument(
        '--baseline',
        type=str,
        default=None,
        help='Baseline platform for speedup calculations'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    config = load_config(args.config)
    
    # Load results
    df = load_all_results(args.log_dir)
    
    if df.empty:
        print("No results found. Run benchmarks first.")
        return
    
    # Calculate cost efficiency
    df = calculate_cost_efficiency(df, config)
    
    # Create comparison table
    comparison = create_comparison_table(df)
    comparison_path = os.path.join(args.output_dir, 'comparison_table.csv')
    comparison.to_csv(comparison_path, index=False)
    print(f"\nComparison table saved to: {comparison_path}")
    
    # Calculate speedup if baseline specified
    if args.baseline and args.baseline in df['platform'].unique():
        speedup_df = calculate_speedup(df, args.baseline)
        if not speedup_df.empty:
            speedup_path = os.path.join(args.output_dir, 'speedup_analysis.csv')
            speedup_df.to_csv(speedup_path, index=False)
            print(f"Speedup analysis saved to: {speedup_path}")
    
    # Generate plots
    plot_performance_comparison(df, args.output_dir)
    
    # Generate report
    generate_report(df, config, args.output_dir)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print(f"Results saved to: {args.output_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()