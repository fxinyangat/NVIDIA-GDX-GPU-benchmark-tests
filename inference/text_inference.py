# """
# Text Inference Benchmark
# Benchmark LLM inference performance using vLLM and Hugging Face transformers
# """

# import os
# import sys
# import yaml
# import argparse
# import time
# from typing import List, Dict
# import torch
# from datasets import load_dataset
# from tqdm import tqdm

# # Add parent directory to path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from utils import GPUMonitor, BenchmarkLogger, create_result_dict, print_result_table


# torch.cuda.empty_cache()



# def load_config(config_path: str = "config/benchmark_config.yaml") -> Dict:
#     """Load benchmark configuration"""
#     with open(config_path, 'r') as f:
#         return yaml.safe_load(f)


# def prepare_dataset(config: Dict, num_samples: int = 1000) -> List[str]:
#     """
#     Load and prepare dataset for inference
    
#     Args:
#         config: Dataset configuration
#         num_samples: Number of samples to use
    
#     Returns:
#         List of input prompts
#     """
#     print(f"Loading dataset: {config['dataset']}")
    
#     # Handle datasets that require config name (like GSM8K)
#     dataset_name = config['dataset']
#     dataset_config = None
    
#     if dataset_name == 'openai/gsm8k':
#         dataset_config = 'main'
    
#     if dataset_config:
#         dataset = load_dataset(
#             dataset_name,
#             dataset_config,
#             split=config['dataset_split'],
#             trust_remote_code=True
#         )
#     else:
#         dataset = load_dataset(
#             dataset_name,
#             split=config['dataset_split'],
#             trust_remote_code=True
#         )
    
#     # Take subset
#     dataset = dataset.select(range(min(num_samples, len(dataset))))
    
#     # Format prompts (for GSM8K: math reasoning)
#     prompts = []
#     for example in dataset:
#         if 'question' in example:
#             prompt = f"Question: {example['question']}\nAnswer:"
#         elif 'text' in example:
#             prompt = example['text']
#         else:
#             # Use first text field
#             prompt = str(list(example.values())[0])
        
#         prompts.append(prompt)
    
#     print(f"Prepared {len(prompts)} prompts")
#     return prompts


# def run_inference_vllm(
#     model_name: str,
#     prompts: List[str],
#     batch_size: int,
#     max_new_tokens: int,
#     gpu_monitor: GPUMonitor
# ) -> Dict:
#     """
#     Run inference using vLLM (optimized for high throughput)
    
#     Args:
#         model_name: Model identifier
#         prompts: List of input prompts
#         batch_size: Batch size (note: vLLM manages batching internally)
#         max_new_tokens: Maximum tokens to generate
#         gpu_monitor: GPU monitor instance
    
#     Returns:
#         Dictionary with results
#     """
#     try:
#         from vllm import LLM, SamplingParams
#     except ImportError:
#         print("Error: vLLM not installed. Install with: pip install vllm")
#         sys.exit(1)
    
#     print(f"\n{'='*80}")
#     print(f"Running vLLM Inference: {model_name}")
#     print(f"Batch size: {batch_size}, Max tokens: {max_new_tokens}")
#     print(f"Number of prompts: {len(prompts)}")
#     print(f"{'='*80}\n")
    
#     # Initialize vLLM
#     sampling_params = SamplingParams(
#         temperature=0.8,
#         top_p=0.95,
#         max_tokens=max_new_tokens
#     )
    
#     llm = LLM(
#         model=model_name,
#         tensor_parallel_size=1,  # Set based on GPU count
#         gpu_memory_utilization=0.8,
#         trust_remote_code=True
#     )
    
#     # Start monitoring
#     gpu_monitor.start()
#     start_time = time.time()
    
#     # Run inference
#     outputs = llm.generate(prompts, sampling_params)
    
#     # Stop monitoring
#     end_time = time.time()
#     metrics = gpu_monitor.stop()
    
#     # Calculate statistics
#     runtime = end_time - start_time
#     throughput = len(prompts) / runtime
    
#     # Count total tokens generated
#     total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
#     tokens_per_second = total_tokens / runtime
    
#     print(f"\nInference completed:")
#     print(f"  Runtime: {runtime:.2f}s")
#     print(f"  Throughput: {throughput:.2f} samples/s")
#     print(f"  Token throughput: {tokens_per_second:.2f} tokens/s")
#     print(f"  Total tokens generated: {total_tokens}")
    
#     return {
#         'runtime': runtime,
#         'throughput': throughput,
#         'tokens_per_second': tokens_per_second,
#         'total_tokens': total_tokens,
#         'metrics': metrics,
#         'num_prompts': len(prompts)
#     }


# def run_inference_transformers(
#     model_name: str,
#     prompts: List[str],
#     batch_size: int,
#     max_new_tokens: int,
#     gpu_monitor: GPUMonitor
# ) -> Dict:
#     """
#     Run inference using Hugging Face transformers
    
#     Args:
#         model_name: Model identifier
#         prompts: List of input prompts
#         batch_size: Batch size
#         max_new_tokens: Maximum tokens to generate
#         gpu_monitor: GPU monitor instance
    
#     Returns:
#         Dictionary with results
#     """
#     from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    
#     print(f"\n{'='*80}")
#     print(f"Running Transformers Inference: {model_name}")
#     print(f"Batch size: {batch_size}, Max tokens: {max_new_tokens}")
#     print(f"Number of prompts: {len(prompts)}")
#     print(f"{'='*80}\n")
    
#     # Load model and tokenizer
#     print("Loading model...")
#     tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side='left')
    
#     # Set pad_token if not set (required for batching)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#         tokenizer.pad_token_id = tokenizer.eos_token_id
    
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         torch_dtype=torch.float16,
#         device_map="auto",
#         trust_remote_code=True
#     )
    
#     # Create pipeline
#     pipe = pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#         max_new_tokens=max_new_tokens,
#         do_sample=True,
#         temperature=0.8,
#         top_p=0.95,
#         batch_size=batch_size,
#         pad_token_id=tokenizer.pad_token_id
#     )
    
#     # Start monitoring
#     gpu_monitor.start()
#     start_time = time.time()
    
#     # Run inference in batches
#     all_outputs = []
#     total_tokens = 0
    
#     for i in tqdm(range(0, len(prompts), batch_size), desc="Processing batches"):
#         batch = prompts[i:i + batch_size]
#         outputs = pipe(batch)
#         all_outputs.extend(outputs)
        
#         # Count tokens
#         for output in outputs:
#             total_tokens += len(tokenizer.encode(output[0]['generated_text']))
    
#     # Stop monitoring
#     end_time = time.time()
#     metrics = gpu_monitor.stop()
    
#     # Calculate statistics
#     runtime = end_time - start_time
#     throughput = len(prompts) / runtime
#     tokens_per_second = total_tokens / runtime
    
#     print(f"\nInference completed:")
#     print(f"  Runtime: {runtime:.2f}s")
#     print(f"  Throughput: {throughput:.2f} samples/s")
#     print(f"  Token throughput: {tokens_per_second:.2f} tokens/s")
#     print(f"  Total tokens generated: {total_tokens}")
    
#     return {
#         'runtime': runtime,
#         'throughput': throughput,
#         'tokens_per_second': tokens_per_second,
#         'total_tokens': total_tokens,
#         'metrics': metrics,
#         'num_prompts': len(prompts)
#     }


# def run_benchmark(
#     model_config: Dict,
#     dataset_config: Dict,
#     platform: str,
#     use_vllm: bool,
#     log_dir: str
# ):
#     """
#     Run complete inference benchmark for a model
    
#     Args:
#         model_config: Model configuration
#         dataset_config: Dataset configuration
#         platform: Platform name
#         use_vllm: Whether to use vLLM
#         log_dir: Directory for logs
#     """
#     model_name = model_config['name']
#     batch_sizes = model_config['batch_sizes']
#     max_new_tokens = model_config['max_new_tokens']
#     num_samples = model_config['num_samples']
    
#     # Initialize logger
#     logger = BenchmarkLogger(
#         log_dir=log_dir,
#         platform=platform,
#         benchmark_type='text_inference'
#     )
    
#     # Log configuration
#     logger.log_config({
#         'model': model_name,
#         'batch_sizes': batch_sizes,
#         'max_new_tokens': max_new_tokens,
#         'num_samples': num_samples,
#         'use_vllm': use_vllm,
#         'platform': platform
#     })
    
#     # Prepare dataset
#     prompts = prepare_dataset(dataset_config, num_samples)
    
#     # Run inference for each batch size
#     results = []
    
#     for batch_size in batch_sizes:
#         print(f"\n{'#'*80}")
#         print(f"# Running with batch size: {batch_size}")
#         print(f"{'#'*80}\n")
        
#         # Initialize GPU monitor
#         gpu_monitor = GPUMonitor(device_id=0, sample_interval=0.5)
        
#         try:
#             # Run inference
#             if use_vllm and batch_size == batch_sizes[0]:
#                 # vLLM manages batching internally, run once
#                 result = run_inference_vllm(
#                     model_name, prompts, batch_size, 
#                     max_new_tokens, gpu_monitor
#                 )
#             else:
#                 result = run_inference_transformers(
#                     model_name, prompts, batch_size,
#                     max_new_tokens, gpu_monitor
#                 )
            
#             # Get GPU summary
#             gpu_summary = gpu_monitor.get_summary()
            
#             # Create result entry
#             result_dict = create_result_dict(
#                 model_name=model_name,
#                 batch_size=batch_size,
#                 num_samples=num_samples,
#                 runtime_s=result['runtime'],
#                 throughput=result['throughput'],
#                 gpu_summary=gpu_summary,
#                 platform=platform,
#                 additional_metrics={
#                     'tokens_per_second': result['tokens_per_second'],
#                     'total_tokens': result['total_tokens'],
#                     'max_new_tokens': max_new_tokens,
#                     'inference_engine': 'vllm' if use_vllm else 'transformers'
#                 }
#             )
            
#             results.append(result_dict)
            
#             # Log GPU metrics
#             logger.log_gpu_metrics(result['metrics'])
            
#             # For vLLM, only run once (it handles batching)
#             if use_vllm:
#                 print("\nNote: vLLM manages batching internally. Skipping other batch sizes.")
#                 break
            
#         except Exception as e:
#             print(f"Error with batch size {batch_size}: {e}")
#             import traceback
#             traceback.print_exc()
#             continue
    
#     # Save results
#     logger.log_results(results)
#     print_result_table(results)
    
#     print(f"\n{'='*80}")
#     print("Benchmark completed successfully!")
#     print(f"{'='*80}\n")


# def main():
#     parser = argparse.ArgumentParser(description="Text Inference Benchmark")
#     parser.add_argument(
#         '--platform',
#         type=str,
#         required=True,
#         help='Platform name (e.g., dgx_spark, lambda_h100)'
#     )
#     parser.add_argument(
#         '--config',
#         type=str,
#         default='config/benchmark_config.yaml',
#         help='Path to configuration file'
#     )
#     parser.add_argument(
#         '--model',
#         type=str,
#         default=None,
#         help='Specific model to benchmark (overrides config)'
#     )
#     parser.add_argument(
#         '--log-dir',
#         type=str,
#         default='./logs',
#         help='Directory for logs'
#     )
#     parser.add_argument(
#         '--no-vllm',
#         action='store_true',
#         help='Disable vLLM and use transformers instead'
#     )
    
#     args = parser.parse_args()
    
#     # Load configuration
#     config = load_config(args.config)
#     inference_config = config['inference']['text']
#     dataset_config = {
#         'dataset': inference_config['dataset'],
#         'dataset_split': inference_config['dataset_split']
#     }
    
#     use_vllm = inference_config.get('use_vllm', True) and not args.no_vllm
    
#     # Determine which models to benchmark
#     if args.model:
#         # Find model config
#         model_configs = [m for m in inference_config['models'] if m['name'] == args.model]
#         if not model_configs:
#             print(f"Error: Model {args.model} not found in config")
#             sys.exit(1)
#     else:
#         model_configs = inference_config['models']
    
#     # Run benchmark for each model
#     for model_config in model_configs:
#         try:
#             run_benchmark(
#                 model_config=model_config,
#                 dataset_config=dataset_config,
#                 platform=args.platform,
#                 use_vllm=use_vllm,
#                 log_dir=args.log_dir
#             )
#         except Exception as e:
#             print(f"Error benchmarking {model_config['name']}: {e}")
#             import traceback
#             traceback.print_exc()
#             continue


# if __name__ == "__main__":
#     main()

"""
Text Inference Benchmark - CORRECTED VERSION
Properly measures sequential token generation with CUDA verification and sanity checks
"""

import os
import sys
import yaml
import argparse
import time
from typing import List, Dict
import torch
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import GPUMonitor, BenchmarkLogger, create_result_dict, print_result_table

torch.cuda.empty_cache()


def verify_cuda_setup():
    """Verify CUDA is available and working"""
    print("=" * 80)
    print("CUDA VERIFICATION")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        raise RuntimeError("ERROR: CUDA is not available! Inference will run on CPU.")
    
    print(f"✓ CUDA Available: {torch.cuda.is_available()}")
    print(f"✓ CUDA Device Count: {torch.cuda.device_count()}")
    print(f"✓ Current CUDA Device: {torch.cuda.current_device()}")
    print(f"✓ CUDA Device Name: {torch.cuda.get_device_name(0)}")
    
    # Get GPU memory info
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"✓ Total GPU Memory: {total_memory:.2f} GB")
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    initial_memory = torch.cuda.memory_allocated() / 1024**3
    print(f"✓ Initial GPU Memory Used: {initial_memory:.2f} GB")
    print()
    
    return total_memory


def get_gpu_memory_bandwidth():
    """Get GPU memory bandwidth in GB/s based on detected GPU"""
    gpu_name = torch.cuda.get_device_name(0).lower()
    
    # Common GPU memory bandwidths
    bandwidth_map = {
        'gb10': 273,  # DGX Spark GB10 - LPDDR5x unified memory
        'nvidia gb10': 273,
        'rtx 6000 ada': 960,
        'a6000': 768,
        'h100': 3352,
        'a100': 1935,
        'v100': 900,
        'rtx 4090': 1008,
        'rtx 3090': 936,
    }
    
    for key, bandwidth in bandwidth_map.items():
        if key in gpu_name:
            print(f"✓ Detected GPU: {torch.cuda.get_device_name(0)}")
            print(f"✓ Memory Bandwidth: {bandwidth} GB/s")
            return bandwidth
    
    # Default conservative estimate
    print(f"⚠️  Warning: Unknown GPU '{torch.cuda.get_device_name(0)}', using conservative bandwidth estimate")
    return 600  # Conservative default


def calculate_theoretical_max(model_size_gb, memory_bandwidth_gbs):
    """Calculate theoretical maximum tokens/sec based on memory bandwidth"""
    max_tokens_per_sec = memory_bandwidth_gbs / model_size_gb
    realistic_max = max_tokens_per_sec * 0.8  # 80% efficiency is realistic
    
    print(f"\nTheoretical Performance Limits (Memory Bound):")
    print(f"  Memory Bandwidth: {memory_bandwidth_gbs} GB/s")
    print(f"  Model Size: {model_size_gb:.2f} GB")
    print(f"  Theoretical Max: {max_tokens_per_sec:.2f} tok/s")
    print(f"  Realistic Max (80%): {realistic_max:.2f} tok/s")
    print()
    
    return max_tokens_per_sec, realistic_max


def load_config(config_path: str = "config/benchmark_config.yaml") -> Dict:
    """Load benchmark configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def prepare_dataset(config: Dict, num_samples: int = 1000) -> List[str]:
    """Load and prepare dataset for inference"""
    print(f"Loading dataset: {config['dataset']}")
    
    dataset_name = config['dataset']
    dataset_config = None
    
    if dataset_name == 'openai/gsm8k':
        dataset_config = 'main'
    
    if dataset_config:
        dataset = load_dataset(
            dataset_name,
            dataset_config,
            split=config['dataset_split'],
            trust_remote_code=True
        )
    else:
        dataset = load_dataset(
            dataset_name,
            split=config['dataset_split'],
            trust_remote_code=True
        )
    
    # Take subset
    dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    # Format prompts
    prompts = []
    for example in dataset:
        if 'question' in example:
            prompt = f"Question: {example['question']}\nAnswer:"
        elif 'text' in example:
            prompt = example['text']
        else:
            prompt = str(list(example.values())[0])
        
        prompts.append(prompt)
    
    print(f"Prepared {len(prompts)} prompts")
    return prompts


def run_inference_transformers_sequential(
    model_name: str,
    prompts: List[str],
    max_new_tokens: int,
    gpu_monitor: GPUMonitor,
    memory_bandwidth_gbs: float
) -> Dict:
    """
    Run SEQUENTIAL inference using Hugging Face transformers
    This is the CORRECT way to measure per-request token generation latency
    
    Args:
        model_name: Model identifier
        prompts: List of input prompts
        max_new_tokens: Maximum tokens to generate
        gpu_monitor: GPU monitor instance
        memory_bandwidth_gbs: GPU memory bandwidth for sanity checks
    
    Returns:
        Dictionary with results
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    print(f"\n{'='*80}")
    print(f"Running SEQUENTIAL Transformers Inference: {model_name}")
    print(f"Max tokens: {max_new_tokens}")
    print(f"Number of prompts: {len(prompts)}")
    print(f"Method: Sequential (one-at-a-time) - CORRECT FOR LATENCY")
    print(f"{'='*80}\n")
    
    # Load model and tokenizer
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    # VERIFY MODEL IS ON GPU
    model_device = next(model.parameters()).device
    print(f"✓ Model device: {model_device}")
    
    if model_device.type != 'cuda':
        raise RuntimeError(f"ERROR: Model loaded to {model_device}, not CUDA!")
    
    # Check actual GPU memory usage
    model_memory_gb = torch.cuda.memory_allocated() / 1024**3
    peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3
    
    print(f"✓ Model GPU Memory: {model_memory_gb:.2f} GB")
    print(f"✓ Peak GPU Memory: {peak_memory_gb:.2f} GB")
    
    # Calculate model size from parameters
    param_count = sum(p.numel() for p in model.parameters())
    model_size_gb = param_count * 2 / 1024**3  # FP16 = 2 bytes per param
    print(f"✓ Model Parameters: {param_count/1e9:.2f}B")
    print(f"✓ Calculated Model Size (FP16): {model_size_gb:.2f} GB")
    
    # Calculate theoretical maximum
    theoretical_max, realistic_max = calculate_theoretical_max(
        model_size_gb, 
        memory_bandwidth_gbs
    )
    
    # Warmup
    print("Running warmup...")
    warmup_input = tokenizer(prompts[0], return_tensors="pt").to(model.device)
    with torch.no_grad():
        _ = model.generate(
            **warmup_input,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
    torch.cuda.synchronize()
    print("✓ Warmup complete\n")
    
    # Start monitoring
    gpu_monitor.start()
    
    # SEQUENTIAL PROCESSING - ONE REQUEST AT A TIME
    print("Running SEQUENTIAL inference (correct method)...")
    print("Processing one prompt at a time to measure true per-request performance\n")
    
    total_tokens_generated = 0
    total_inference_time = 0
    per_sample_times = []
    per_sample_tokens = []
    per_sample_tps = []
    
    start_time = time.time()
    
    for i, prompt in enumerate(tqdm(prompts, desc="Processing prompts")):
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_length = inputs.input_ids.shape[1]
        
        # Verify input on GPU
        if inputs.input_ids.device.type != 'cuda':
            raise RuntimeError(f"Input tensor on {inputs.input_ids.device}, not CUDA!")
        
        # Time this single generation
        torch.cuda.synchronize()
        sample_start = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Deterministic for consistent benchmarking
                pad_token_id=tokenizer.pad_token_id
            )
        
        torch.cuda.synchronize()
        sample_end = time.time()
        
        sample_time = sample_end - sample_start
        tokens_generated = outputs.shape[1] - input_length
        sample_tps = tokens_generated / sample_time
        
        total_tokens_generated += tokens_generated
        total_inference_time += sample_time
        per_sample_times.append(sample_time)
        per_sample_tokens.append(tokens_generated)
        per_sample_tps.append(sample_tps)
        
        # Print progress every 50 samples
        if (i + 1) % 50 == 0:
            avg_tps = np.mean(per_sample_tps[-50:])
            print(f"  Samples {i+1}/{len(prompts)}: Avg last 50 = {avg_tps:.2f} tok/s")
    
    total_time = time.time() - start_time
    
    # Stop monitoring
    metrics = gpu_monitor.stop()
    
    # Calculate statistics
    avg_tokens_per_sec = total_tokens_generated / total_inference_time
    median_tokens_per_sec = np.median(per_sample_tps)
    p95_tokens_per_sec = np.percentile(per_sample_tps, 95)
    p99_tokens_per_sec = np.percentile(per_sample_tps, 99)
    
    throughput_samples_per_sec = len(prompts) / total_time
    avg_tokens_per_sample = np.mean(per_sample_tokens)
    
    # SANITY CHECK
    print(f"\n{'='*80}")
    print("SANITY CHECK: Comparing Against Theoretical Maximum")
    print(f"{'='*80}")
    print(f"Measured Average: {avg_tokens_per_sec:.2f} tok/s")
    print(f"Theoretical Max: {theoretical_max:.2f} tok/s")
    print(f"Realistic Max (80%): {realistic_max:.2f} tok/s")
    print(f"Efficiency: {(avg_tokens_per_sec/theoretical_max)*100:.1f}%")
    
    if avg_tokens_per_sec > theoretical_max * 1.1:
        print(f"\n⚠️  WARNING: Measured throughput EXCEEDS theoretical maximum!")
        print(f"⚠️  This indicates a MEASUREMENT ERROR.")
        print(f"⚠️  Results are INVALID and should not be reported.")
        raise ValueError("Benchmark produced physically impossible results!")
    else:
        print(f"\n✓ Results are within theoretical bounds (VALID)")
    
    print(f"\n{'='*80}")
    print("Inference Results")
    print(f"{'='*80}")
    print(f"Total samples: {len(prompts)}")
    print(f"Total tokens generated: {total_tokens_generated}")
    print(f"Total inference time: {total_inference_time:.2f}s")
    print(f"Total wall-clock time: {total_time:.2f}s")
    print(f"\nPer-Request Token Generation (PRIMARY METRIC):")
    print(f"  Average: {avg_tokens_per_sec:.2f} tok/s")
    print(f"  Median:  {median_tokens_per_sec:.2f} tok/s")
    print(f"  P95:     {p95_tokens_per_sec:.2f} tok/s")
    print(f"  P99:     {p99_tokens_per_sec:.2f} tok/s")
    print(f"\nThroughput Metrics:")
    print(f"  Samples/sec: {throughput_samples_per_sec:.2f}")
    print(f"  Avg tokens/sample: {avg_tokens_per_sample:.1f}")
    print()
    
    return {
        'runtime': total_time,
        'inference_time': total_inference_time,
        'throughput': throughput_samples_per_sec,
        'tokens_per_second': avg_tokens_per_sec,
        'median_tokens_per_second': median_tokens_per_sec,
        'p95_tokens_per_second': p95_tokens_per_sec,
        'p99_tokens_per_second': p99_tokens_per_sec,
        'total_tokens': total_tokens_generated,
        'avg_tokens_per_sample': avg_tokens_per_sample,
        'metrics': metrics,
        'num_prompts': len(prompts),
        'theoretical_max': theoretical_max,
        'realistic_max': realistic_max,
        'efficiency': (avg_tokens_per_sec/theoretical_max)*100,
        'model_size_gb': model_size_gb,
        'peak_memory_gb': peak_memory_gb
    }


def run_benchmark(
    model_config: Dict,
    dataset_config: Dict,
    platform: str,
    log_dir: str
):
    """
    Run complete inference benchmark for a model
    
    Args:
        model_config: Model configuration
        dataset_config: Dataset configuration
        platform: Platform name
        log_dir: Directory for logs
    """
    model_name = model_config['name']
    max_new_tokens = model_config['max_new_tokens']
    num_samples = model_config['num_samples']
    
    print(f"\n{'#'*80}")
    print(f"# Benchmarking Model: {model_name}")
    print(f"# Platform: {platform}")
    print(f"{'#'*80}\n")
    
    # Verify CUDA setup
    total_vram = verify_cuda_setup()
    
    # Get GPU memory bandwidth
    memory_bandwidth = get_gpu_memory_bandwidth()
    
    # Initialize logger
    logger = BenchmarkLogger(
        log_dir=log_dir,
        platform=platform,
        benchmark_type='text_inference'
    )
    
    # Log configuration
    logger.log_config({
        'model': model_name,
        'max_new_tokens': max_new_tokens,
        'num_samples': num_samples,
        'platform': platform,
        'gpu_name': torch.cuda.get_device_name(0),
        'total_vram_gb': total_vram,
        'memory_bandwidth_gbs': memory_bandwidth,
        'method': 'sequential_transformers',
        'precision': 'fp16'
    })
    
    # Prepare dataset
    prompts = prepare_dataset(dataset_config, num_samples)
    
    # Initialize GPU monitor
    gpu_monitor = GPUMonitor(device_id=0, sample_interval=0.5)
    
    try:
        # Run sequential inference (CORRECT METHOD)
        result = run_inference_transformers_sequential(
            model_name=model_name,
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            gpu_monitor=gpu_monitor,
            memory_bandwidth_gbs=memory_bandwidth
        )
        
        # Get GPU summary
        gpu_summary = gpu_monitor.get_summary()
        
        # Create result entry
        result_dict = create_result_dict(
            model_name=model_name,
            batch_size=1,  # Sequential = batch size 1
            num_samples=num_samples,
            runtime_s=result['runtime'],
            throughput=result['throughput'],
            gpu_summary=gpu_summary,
            platform=platform,
            additional_metrics={
                'tokens_per_second': result['tokens_per_second'],
                'median_tokens_per_second': result['median_tokens_per_second'],
                'p95_tokens_per_second': result['p95_tokens_per_second'],
                'p99_tokens_per_second': result['p99_tokens_per_second'],
                'total_tokens': result['total_tokens'],
                'avg_tokens_per_sample': result['avg_tokens_per_sample'],
                'max_new_tokens': max_new_tokens,
                'inference_engine': 'transformers_sequential',
                'theoretical_max_tps': result['theoretical_max'],
                'realistic_max_tps': result['realistic_max'],
                'efficiency_percent': result['efficiency'],
                'model_size_gb': result['model_size_gb'],
                'peak_memory_gb': result['peak_memory_gb']
            }
        )
        
        # Save results
        logger.log_results([result_dict])
        logger.log_gpu_metrics(result['metrics'])
        print_result_table([result_dict])
        
        print(f"\n{'='*80}")
        print("Benchmark completed successfully!")
        print(f"Results saved to: {log_dir}")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"\nError during benchmark: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    parser = argparse.ArgumentParser(description="Text Inference Benchmark - CORRECTED VERSION")
    parser.add_argument(
        '--platform',
        type=str,
        required=True,
        help='Platform name (e.g., rtx_6000_ada, dgx_spark_gb10)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/benchmark_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Specific model to benchmark (overrides config)'
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default='./logs',
        help='Directory for logs'
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("TEXT INFERENCE BENCHMARK - CORRECTED VERSION")
    print("Measures SEQUENTIAL token generation (correct for latency)")
    print("Includes CUDA verification and sanity checks")
    print(f"{'='*80}\n")
    
    # Load configuration
    config = load_config(args.config)
    inference_config = config['inference']['text']
    dataset_config = {
        'dataset': inference_config['dataset'],
        'dataset_split': inference_config['dataset_split']
    }
    
    # Determine which models to benchmark
    if args.model:
        model_configs = [m for m in inference_config['models'] if m['name'] == args.model]
        if not model_configs:
            print(f"Error: Model {args.model} not found in config")
            sys.exit(1)
    else:
        model_configs = inference_config['models']
    
    # Run benchmark for each model
    for model_config in model_configs:
        try:
            run_benchmark(
                model_config=model_config,
                dataset_config=dataset_config,
                platform=args.platform,
                log_dir=args.log_dir
            )
        except Exception as e:
            print(f"Error benchmarking {model_config['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()