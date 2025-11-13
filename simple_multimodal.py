#!/usr/bin/env python3
"""
Simple Multimodal Inference Test
Uses synthetic images to avoid dataset download issues
"""

import os
import sys
import time
import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import GPUMonitor, BenchmarkLogger, create_result_dict, print_result_table


def create_synthetic_images(num_images: int = 5000):
    """Create synthetic test images"""
    print(f"Creating {num_images} synthetic test images...")
    images = []
    
    for i in range(num_images):
        # Create a random RGB image
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        images.append(img)
    
    print(f"Created {len(images)} images")
    return images


def run_multimodal_test(platform: str = "dgx_spark", num_images: int = 5000):
    """Run simple multimodal inference test"""
    
    print("="*80)
    print("Simple Multimodal Inference Test")
    print("="*80)
    print(f"Platform: {platform}")
    print(f"Model: BLIP2-FlanT5-XL")
    print(f"Images: {num_images} synthetic")
    print("="*80)
    
    # Create synthetic images
    images = create_synthetic_images(num_images)
    
    # Load model
    print("\nLoading model...")
    model_name = "Salesforce/blip2-flan-t5-xl"
    processor = AutoProcessor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("Model loaded")
    
    # Initialize logger
    logger = BenchmarkLogger(
        log_dir="./logs",
        platform=platform,
        benchmark_type="multimodal_simple"
    )
    
    # Test with different batch sizes
    batch_sizes = [1, 2, 4]
    results = []
    
    for batch_size in batch_sizes:
        print(f"\n{'='*80}")
        print(f"Testing batch size: {batch_size}")
        print(f"{'='*80}")
        
        # Initialize GPU monitor
        gpu_monitor = GPUMonitor(device_id=0, sample_interval=0.5)
        gpu_monitor.start()
        
        start_time = time.time()
        
        # Process in batches
        all_outputs = []
        total_tokens = 0
        prompt = "Describe this image:"
        
        for i in tqdm(range(0, len(images), batch_size), desc="Processing"):
            batch_images = images[i:i + batch_size]
            
            # Process inputs
            inputs = processor(
                images=batch_images,
                text=[prompt] * len(batch_images),
                return_tensors="pt",
                padding=True
            ).to(model.device)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50
                )
            
            # Decode
            texts = processor.batch_decode(outputs, skip_special_tokens=True)
            all_outputs.extend(texts)
            
            # Count tokens generated
            total_tokens += outputs.shape[0] * outputs.shape[1]
        
        # Stop monitoring
        end_time = time.time()
        metrics = gpu_monitor.stop()
        gpu_summary = gpu_monitor.get_summary()
        
        runtime = end_time - start_time
        throughput = len(images) / runtime
        tokens_per_second = total_tokens / runtime if runtime > 0 else 0
        
        print(f"\nCompleted:")
        print(f"  Runtime: {runtime:.2f}s")
        print(f"  Throughput: {throughput:.2f} images/s")
        print(f"  Token throughput: {tokens_per_second:.2f} tokens/s")
        print(f"  Total tokens: {total_tokens}")
        
        # Show sample outputs
        print(f"\nSample outputs:")
        for i in range(min(3, len(all_outputs))):
            print(f"  Image {i}: {all_outputs[i][:100]}...")
        
        # Create result
        result_dict = create_result_dict(
            model_name=model_name,
            batch_size=batch_size,
            num_samples=num_images,
            runtime_s=runtime,
            throughput=throughput,
            gpu_summary=gpu_summary,
            platform=platform,
            additional_metrics={
                'modality': 'vision_language',
                'image_type': 'synthetic',
                'tokens_per_second': tokens_per_second,
                'total_tokens': total_tokens,
                'max_new_tokens': 50
            }
        )
        results.append(result_dict)
        
        # Log metrics
        logger.log_gpu_metrics(metrics)
    
    # Save results
    logger.log_results(results)
    print_result_table(results)
    
    print("\n" + "="*80)
    print("Test Complete!")
    print("="*80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", default="H100")
    parser.add_argument("--num-images", type=int, default=5000)
    args = parser.parse_args()
    
    run_multimodal_test(args.platform, args.num_images)