"""
Text Inference Benchmark
Benchmark LLM inference performance using vLLM and Hugging Face transformers
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

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import GPUMonitor, BenchmarkLogger, create_result_dict, print_result_table


torch.cuda.empty_cache()



def load_config(config_path: str = "config/benchmark_config.yaml") -> Dict:
    """Load benchmark configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def prepare_dataset(config: Dict, num_samples: int = 1000) -> List[str]:
    """
    Load and prepare dataset for inference
    
    Args:
        config: Dataset configuration
        num_samples: Number of samples to use
    
    Returns:
        List of input prompts
    """
    print(f"Loading dataset: {config['dataset']}")
    
    # Handle datasets that require config name (like GSM8K)
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
    
    # Format prompts (for GSM8K: math reasoning)
    prompts = []
    for example in dataset:
        if 'question' in example:
            prompt = f"Question: {example['question']}\nAnswer:"
        elif 'text' in example:
            prompt = example['text']
        else:
            # Use first text field
            prompt = str(list(example.values())[0])
        
        prompts.append(prompt)
    
    print(f"Prepared {len(prompts)} prompts")
    return prompts


def run_inference_vllm(
    model_name: str,
    prompts: List[str],
    batch_size: int,
    max_new_tokens: int,
    gpu_monitor: GPUMonitor
) -> Dict:
    """
    Run inference using vLLM (optimized for high throughput)
    
    Args:
        model_name: Model identifier
        prompts: List of input prompts
        batch_size: Batch size (note: vLLM manages batching internally)
        max_new_tokens: Maximum tokens to generate
        gpu_monitor: GPU monitor instance
    
    Returns:
        Dictionary with results
    """
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("Error: vLLM not installed. Install with: pip install vllm")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"Running vLLM Inference: {model_name}")
    print(f"Batch size: {batch_size}, Max tokens: {max_new_tokens}")
    print(f"Number of prompts: {len(prompts)}")
    print(f"{'='*80}\n")
    
    # Initialize vLLM
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=max_new_tokens
    )
    
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,  # Set based on GPU count
        gpu_memory_utilization=0.8,
        trust_remote_code=True
    )
    
    # Start monitoring
    gpu_monitor.start()
    start_time = time.time()
    
    # Run inference
    outputs = llm.generate(prompts, sampling_params)
    
    # Stop monitoring
    end_time = time.time()
    metrics = gpu_monitor.stop()
    
    # Calculate statistics
    runtime = end_time - start_time
    throughput = len(prompts) / runtime
    
    # Count total tokens generated
    total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    tokens_per_second = total_tokens / runtime
    
    print(f"\nInference completed:")
    print(f"  Runtime: {runtime:.2f}s")
    print(f"  Throughput: {throughput:.2f} samples/s")
    print(f"  Token throughput: {tokens_per_second:.2f} tokens/s")
    print(f"  Total tokens generated: {total_tokens}")
    
    return {
        'runtime': runtime,
        'throughput': throughput,
        'tokens_per_second': tokens_per_second,
        'total_tokens': total_tokens,
        'metrics': metrics,
        'num_prompts': len(prompts)
    }


def run_inference_transformers(
    model_name: str,
    prompts: List[str],
    batch_size: int,
    max_new_tokens: int,
    gpu_monitor: GPUMonitor
) -> Dict:
    """
    Run inference using Hugging Face transformers
    
    Args:
        model_name: Model identifier
        prompts: List of input prompts
        batch_size: Batch size
        max_new_tokens: Maximum tokens to generate
        gpu_monitor: GPU monitor instance
    
    Returns:
        Dictionary with results
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    
    print(f"\n{'='*80}")
    print(f"Running Transformers Inference: {model_name}")
    print(f"Batch size: {batch_size}, Max tokens: {max_new_tokens}")
    print(f"Number of prompts: {len(prompts)}")
    print(f"{'='*80}\n")
    
    # Load model and tokenizer
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side='left')
    
    # Set pad_token if not set (required for batching)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Create pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        batch_size=batch_size,
        pad_token_id=tokenizer.pad_token_id
    )
    
    # Start monitoring
    gpu_monitor.start()
    start_time = time.time()
    
    # Run inference in batches
    all_outputs = []
    total_tokens = 0
    
    for i in tqdm(range(0, len(prompts), batch_size), desc="Processing batches"):
        batch = prompts[i:i + batch_size]
        outputs = pipe(batch)
        all_outputs.extend(outputs)
        
        # Count tokens
        for output in outputs:
            total_tokens += len(tokenizer.encode(output[0]['generated_text']))
    
    # Stop monitoring
    end_time = time.time()
    metrics = gpu_monitor.stop()
    
    # Calculate statistics
    runtime = end_time - start_time
    throughput = len(prompts) / runtime
    tokens_per_second = total_tokens / runtime
    
    print(f"\nInference completed:")
    print(f"  Runtime: {runtime:.2f}s")
    print(f"  Throughput: {throughput:.2f} samples/s")
    print(f"  Token throughput: {tokens_per_second:.2f} tokens/s")
    print(f"  Total tokens generated: {total_tokens}")
    
    return {
        'runtime': runtime,
        'throughput': throughput,
        'tokens_per_second': tokens_per_second,
        'total_tokens': total_tokens,
        'metrics': metrics,
        'num_prompts': len(prompts)
    }


def run_benchmark(
    model_config: Dict,
    dataset_config: Dict,
    platform: str,
    use_vllm: bool,
    log_dir: str
):
    """
    Run complete inference benchmark for a model
    
    Args:
        model_config: Model configuration
        dataset_config: Dataset configuration
        platform: Platform name
        use_vllm: Whether to use vLLM
        log_dir: Directory for logs
    """
    model_name = model_config['name']
    batch_sizes = model_config['batch_sizes']
    max_new_tokens = model_config['max_new_tokens']
    num_samples = model_config['num_samples']
    
    # Initialize logger
    logger = BenchmarkLogger(
        log_dir=log_dir,
        platform=platform,
        benchmark_type='text_inference'
    )
    
    # Log configuration
    logger.log_config({
        'model': model_name,
        'batch_sizes': batch_sizes,
        'max_new_tokens': max_new_tokens,
        'num_samples': num_samples,
        'use_vllm': use_vllm,
        'platform': platform
    })
    
    # Prepare dataset
    prompts = prepare_dataset(dataset_config, num_samples)
    
    # Run inference for each batch size
    results = []
    
    for batch_size in batch_sizes:
        print(f"\n{'#'*80}")
        print(f"# Running with batch size: {batch_size}")
        print(f"{'#'*80}\n")
        
        # Initialize GPU monitor
        gpu_monitor = GPUMonitor(device_id=0, sample_interval=0.5)
        
        try:
            # Run inference
            if use_vllm and batch_size == batch_sizes[0]:
                # vLLM manages batching internally, run once
                result = run_inference_vllm(
                    model_name, prompts, batch_size, 
                    max_new_tokens, gpu_monitor
                )
            else:
                result = run_inference_transformers(
                    model_name, prompts, batch_size,
                    max_new_tokens, gpu_monitor
                )
            
            # Get GPU summary
            gpu_summary = gpu_monitor.get_summary()
            
            # Create result entry
            result_dict = create_result_dict(
                model_name=model_name,
                batch_size=batch_size,
                num_samples=num_samples,
                runtime_s=result['runtime'],
                throughput=result['throughput'],
                gpu_summary=gpu_summary,
                platform=platform,
                additional_metrics={
                    'tokens_per_second': result['tokens_per_second'],
                    'total_tokens': result['total_tokens'],
                    'max_new_tokens': max_new_tokens,
                    'inference_engine': 'vllm' if use_vllm else 'transformers'
                }
            )
            
            results.append(result_dict)
            
            # Log GPU metrics
            logger.log_gpu_metrics(result['metrics'])
            
            # For vLLM, only run once (it handles batching)
            if use_vllm:
                print("\nNote: vLLM manages batching internally. Skipping other batch sizes.")
                break
            
        except Exception as e:
            print(f"Error with batch size {batch_size}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    logger.log_results(results)
    print_result_table(results)
    
    print(f"\n{'='*80}")
    print("Benchmark completed successfully!")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Text Inference Benchmark")
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
    parser.add_argument(
        '--no-vllm',
        action='store_true',
        help='Disable vLLM and use transformers instead'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    inference_config = config['inference']['text']
    dataset_config = {
        'dataset': inference_config['dataset'],
        'dataset_split': inference_config['dataset_split']
    }
    
    use_vllm = inference_config.get('use_vllm', True) and not args.no_vllm
    
    # Determine which models to benchmark
    if args.model:
        # Find model config
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
                use_vllm=use_vllm,
                log_dir=args.log_dir
            )
        except Exception as e:
            print(f"Error benchmarking {model_config['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()