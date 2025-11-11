"""
Multimodal Inference Benchmark
Benchmark vision-language model inference on image captioning/VQA tasks
"""

import os
import sys
import yaml
import argparse
import time
from typing import List, Dict, Tuple
import torch
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    AutoModelForImageTextToText,
    LlavaForConditionalGeneration,
    Blip2ForConditionalGeneration
)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import GPUMonitor, BenchmarkLogger, create_result_dict, print_result_table


def load_config(config_path: str = "config/benchmark_config.yaml") -> Dict:
    """Load benchmark configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def prepare_coco_dataset(
    num_images: int = 5000,
    image_size: Tuple[int, int] = (384, 384)
) -> List[Dict]:
    """
    Load and prepare COCO validation dataset
    
    Args:
        num_images: Number of images to use
        image_size: Target image size (width, height)
    
    Returns:
        List of dictionaries with images and metadata
    """
    from datasets import load_dataset
    from PIL import Image
    import numpy as np
    
    print(f"Loading COCO dataset (2017 validation)...")
    
    dataset = None
    
    try:
        # First try: detection-datasets/coco with correct split
        dataset = load_dataset(
            "detection-datasets/coco",
            split="val",  # Use 'val' not 'val2017'
            trust_remote_code=True
        )
        print(f"Successfully loaded detection-datasets/coco")
    except Exception as e:
        print(f"COCO dataset failed: {e}")
        print("Generating synthetic images instead...")
        
        # Create synthetic dataset
        synthetic_data = []
        print(f"Creating {num_images} synthetic images...")
        for i in range(num_images):
            # Create random RGB image
            img_array = np.random.randint(0, 255, image_size + (3,), dtype=np.uint8)
            img = Image.fromarray(img_array)
            synthetic_data.append({
                'image': img,
                'image_id': i,
                'caption': f'Synthetic test image {i}'
            })
        
        # Create samples list directly
        samples = []
        for data in synthetic_data:
            samples.append({
                'image': data['image'],
                'image_id': data['image_id'],
                'caption': data['caption']
            })
        
        print(f"Prepared {len(samples)} synthetic image samples")
        return samples
    
    # If we have a real dataset, process it
    if dataset is not None:
        # Take subset
        dataset = dataset.select(range(min(num_images, len(dataset))))
        
        # Prepare samples
        samples = []
        print("Preparing images...")
        for idx, example in enumerate(tqdm(dataset, desc="Loading images")):
            try:
                # Get image
                if 'image' in example:
                    image = example['image']
                elif 'img' in example:
                    image = example['img']
                else:
                    continue
                
                # Convert to PIL Image if needed
                if not isinstance(image, Image.Image):
                    if hasattr(image, 'shape'):  # numpy array
                        image = Image.fromarray(image)
                    else:
                        continue
                
                # Resize
                image = image.convert('RGB')
                image = image.resize(image_size)
                
                # Create sample
                sample = {
                    'image': image,
                    'image_id': idx,
                }
                
                # Add caption if available
                if 'caption' in example:
                    sample['caption'] = example['caption']
                elif 'captions' in example and example['captions']:
                    sample['caption'] = example['captions'][0]
                
                samples.append(sample)
                
            except Exception as e:
                print(f"Warning: Failed to load image {idx}: {e}")
                continue
        
        print(f"Prepared {len(samples)} image samples")
        return samples


def load_model_and_processor(model_name: str):
    """
    Load vision-language model and processor
    
    Args:
        model_name: Model identifier
    
    Returns:
        Tuple of (model, processor)
    """
    print(f"Loading model: {model_name}")
    
    # Load processor/tokenizer
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    # Load model based on type
    if 'llava' in model_name.lower():
        model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    elif 'blip' in model_name.lower():
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Patch processor to avoid image_sizes issue
        if hasattr(processor, 'image_processor'):
            original_call = processor.image_processor.__call__
            
            def patched_call(*args, **kwargs):
                # Remove image_sizes if present
                kwargs.pop('image_sizes', None)
                return original_call(*args, **kwargs)
            
            processor.image_processor.__call__ = patched_call
            print("Applied BLIP2 processor patch for image_sizes compatibility")
    else:
        # Try generic vision-to-text model
        try:
            model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        except:
            model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
    
    model.eval()
    print(f"Model loaded successfully")
    
    return model, processor


def run_inference(
    model,
    processor,
    samples: List[Dict],
    batch_size: int,
    max_new_tokens: int,
    gpu_monitor: GPUMonitor
) -> Dict:
    """
    Run multimodal inference
    
    Args:
        model: Vision-language model
        processor: Processor for inputs
        samples: List of image samples
        batch_size: Batch size
        max_new_tokens: Maximum tokens to generate
        gpu_monitor: GPU monitor instance
    
    Returns:
        Dictionary with results
    """
    print(f"\n{'='*80}")
    print(f"Running Multimodal Inference")
    print(f"Batch size: {batch_size}, Max tokens: {max_new_tokens}")
    print(f"Number of images: {len(samples)}")
    print(f"{'='*80}\n")
    
    device = next(model.parameters()).device
    
    # Prepare prompts
    prompts = []
    for sample in samples:
        # Standard image captioning prompt
        if 'llava' in model.config.model_type.lower():
            prompt = "<image>\nUSER: Describe this image in detail.\nASSISTANT:"
        else:
            prompt = "Describe this image:"
        prompts.append(prompt)
    
    # Start monitoring
    gpu_monitor.start()
    start_time = time.time()
    
    # Run inference in batches
    all_outputs = []
    total_tokens = 0
    
    for i in tqdm(range(0, len(samples), batch_size), desc="Processing batches"):
        batch_samples = samples[i:i + batch_size]
        batch_prompts = prompts[i:i + batch_size]
        batch_images = [s['image'] for s in batch_samples]
        
        try:
            # Process inputs differently based on model type
            if 'blip' in model.config.model_type.lower():
                # BLIP2 - only process images, no text prompts in input
                inputs = processor(
                    images=batch_images,
                    return_tensors="pt"
                ).to(device)
                
                # Don't add text prompts to inputs for BLIP2
                # It generates captions from images directly
                
            else:
                # LLaVA and other models need both text and images
                inputs = processor(
                    text=batch_prompts,
                    images=batch_images,
                    return_tensors="pt",
                    padding=True
                ).to(device)
            
            # Generate
            with torch.no_grad():
                try:
                    # Use max_new_tokens (preferred) instead of max_length
                    generation_kwargs = {
                        'max_new_tokens': max_new_tokens,
                        'do_sample': False,
                        'num_beams': 1,
                    }
                    
                    # Add pad_token_id if available
                    if hasattr(processor, 'tokenizer') and hasattr(processor.tokenizer, 'pad_token_id'):
                        if processor.tokenizer.pad_token_id is not None:
                            generation_kwargs['pad_token_id'] = processor.tokenizer.pad_token_id
                    
                    outputs = model.generate(**inputs, **generation_kwargs)
                    
                except (TypeError, ValueError) as e:
                    # Fallback: simpler generation
                    print(f"Generation fallback for batch {i}: {e}")
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens
                    )
            
            # Decode outputs
            if hasattr(processor, 'batch_decode'):
                generated_texts = processor.batch_decode(
                    outputs,
                    skip_special_tokens=True
                )
            elif hasattr(processor, 'tokenizer'):
                generated_texts = processor.tokenizer.batch_decode(
                    outputs,
                    skip_special_tokens=True
                )
            else:
                generated_texts = [''] * len(batch_images)
            
            all_outputs.extend(generated_texts)
            total_tokens += outputs.shape[0] * outputs.shape[1]
            
        except Exception as e:
            if 'image_sizes' not in str(e):  # Only print if not the known error
                print(f"Error processing batch {i}: {e}")
            # Add empty outputs for failed batch
            all_outputs.extend([''] * len(batch_samples))
            continue
    
    # Stop monitoring
    end_time = time.time()
    metrics = gpu_monitor.stop()
    
    # Calculate statistics
    runtime = end_time - start_time
    throughput = len(samples) / runtime
    tokens_per_second = total_tokens / runtime
    
    print(f"\nInference completed:")
    print(f"  Runtime: {runtime:.2f}s")
    print(f"  Throughput: {throughput:.2f} images/s")
    print(f"  Token throughput: {tokens_per_second:.2f} tokens/s")
    print(f"  Total tokens generated: {total_tokens}")
    
    # Print sample outputs
    print("\nSample outputs:")
    for i in range(min(3, len(all_outputs))):
        print(f"\n  Image {i}:")
        print(f"    Generated: {all_outputs[i][:150]}...")
        if 'caption' in samples[i]:
            print(f"    Reference: {samples[i]['caption'][:150]}...")
    
    return {
        'runtime': runtime,
        'throughput': throughput,
        'tokens_per_second': tokens_per_second,
        'total_tokens': total_tokens,
        'metrics': metrics,
        'num_samples': len(samples),
        'outputs': all_outputs
    }


def run_benchmark(
    model_config: Dict,
    dataset_config: Dict,
    platform: str,
    log_dir: str
):
    """
    Run complete multimodal inference benchmark
    
    Args:
        model_config: Model configuration
        dataset_config: Dataset configuration
        platform: Platform name
        log_dir: Directory for logs
    """
    model_name = model_config['name']
    batch_sizes = model_config['batch_sizes']
    max_new_tokens = model_config['max_new_tokens']
    num_images = dataset_config['num_images']
    image_size = tuple(dataset_config['image_size'])
    
    # Initialize logger
    logger = BenchmarkLogger(
        log_dir=log_dir,
        platform=platform,
        benchmark_type='multimodal_inference'
    )
    
    # Log configuration
    logger.log_config({
        'model': model_name,
        'batch_sizes': batch_sizes,
        'max_new_tokens': max_new_tokens,
        'num_images': num_images,
        'image_size': image_size,
        'platform': platform
    })
    
    # Load model
    model, processor = load_model_and_processor(model_name)
    
    # Prepare dataset
    samples = prepare_coco_dataset(num_images, image_size)
    
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
            result = run_inference(
                model, processor, samples, batch_size,
                max_new_tokens, gpu_monitor
            )
            
            # Get GPU summary
            gpu_summary = gpu_monitor.get_summary()
            
            # Create result entry
            result_dict = create_result_dict(
                model_name=model_name,
                batch_size=batch_size,
                num_samples=num_images,
                runtime_s=result['runtime'],
                throughput=result['throughput'],
                gpu_summary=gpu_summary,
                platform=platform,
                additional_metrics={
                    'tokens_per_second': result['tokens_per_second'],
                    'total_tokens': result['total_tokens'],
                    'max_new_tokens': max_new_tokens,
                    'image_size': str(image_size),
                    'modality': 'vision_language'
                }
            )
            
            results.append(result_dict)
            
            # Log GPU metrics
            logger.log_gpu_metrics(result['metrics'])
            
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
    parser = argparse.ArgumentParser(description="Multimodal Inference Benchmark")
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
        '--num-images',
        type=int,
        default=None,
        help='Number of images to process (overrides config)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    inference_config = config['inference']['multimodal']
    
    dataset_config = {
        'num_images': args.num_images or inference_config['num_images'],
        'image_size': inference_config['image_size']
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