# #!/usr/bin/env python3
# """
# Simple Multimodal Inference Test
# Uses synthetic images to avoid dataset download issues
# """

# import os
# import sys
# import time
# import numpy as np
# from PIL import Image
# import torch
# from transformers import AutoProcessor, Blip2ForConditionalGeneration
# from tqdm import tqdm

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from utils import GPUMonitor, BenchmarkLogger, create_result_dict, print_result_table


# def create_synthetic_images(num_images: int = 5000):
#     """Create synthetic test images"""
#     print(f"Creating {num_images} synthetic test images...")
#     images = []
    
#     for i in range(num_images):
#         # Create a random RGB image
#         img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
#         img = Image.fromarray(img_array)
#         images.append(img)
    
#     print(f"Created {len(images)} images")
#     return images


# def run_multimodal_test(platform: str = "dgx_spark", num_images: int = 5000):
#     """Run simple multimodal inference test"""
    
#     print("="*80)
#     print("Simple Multimodal Inference Test")
#     print("="*80)
#     print(f"Platform: {platform}")
#     print(f"Model: BLIP2-FlanT5-XL")
#     print(f"Images: {num_images} synthetic")
#     print("="*80)
    
#     # Create synthetic images
#     images = create_synthetic_images(num_images)
    
#     # Load model
#     print("\nLoading model...")
#     model_name = "Salesforce/blip2-flan-t5-xl"
#     processor = AutoProcessor.from_pretrained(model_name)
#     model = Blip2ForConditionalGeneration.from_pretrained(
#         model_name,
#         torch_dtype=torch.float16,
#         device_map="auto"
#     )
#     print("Model loaded")
    
#     # Initialize logger
#     logger = BenchmarkLogger(
#         log_dir="./logs",
#         platform=platform,
#         benchmark_type="multimodal_simple"
#     )
    
#     # Test with different batch sizes
#     batch_sizes = [1, 2, 4]
#     results = []
    
#     for batch_size in batch_sizes:
#         print(f"\n{'='*80}")
#         print(f"Testing batch size: {batch_size}")
#         print(f"{'='*80}")
        
#         # Initialize GPU monitor
#         gpu_monitor = GPUMonitor(device_id=0, sample_interval=0.5)
#         gpu_monitor.start()
        
#         start_time = time.time()
        
#         # Process in batches
#         all_outputs = []
#         total_tokens = 0
#         prompt = "Describe this image:"
        
#         for i in tqdm(range(0, len(images), batch_size), desc="Processing"):
#             batch_images = images[i:i + batch_size]
            
#             # Process inputs
#             inputs = processor(
#                 images=batch_images,
#                 text=[prompt] * len(batch_images),
#                 return_tensors="pt",
#                 padding=True
#             ).to(model.device)
            
#             # Generate
#             with torch.no_grad():
#                 outputs = model.generate(
#                     **inputs,
#                     max_new_tokens=50
#                 )
            
#             # Decode
#             texts = processor.batch_decode(outputs, skip_special_tokens=True)
#             all_outputs.extend(texts)
            
#             # Count tokens generated
#             total_tokens += outputs.shape[0] * outputs.shape[1]
        
#         # Stop monitoring
#         end_time = time.time()
#         metrics = gpu_monitor.stop()
#         gpu_summary = gpu_monitor.get_summary()
        
#         runtime = end_time - start_time
#         throughput = len(images) / runtime
#         tokens_per_second = total_tokens / runtime if runtime > 0 else 0
        
#         print(f"\nCompleted:")
#         print(f"  Runtime: {runtime:.2f}s")
#         print(f"  Throughput: {throughput:.2f} images/s")
#         print(f"  Token throughput: {tokens_per_second:.2f} tokens/s")
#         print(f"  Total tokens: {total_tokens}")
        
#         # Show sample outputs
#         print(f"\nSample outputs:")
#         for i in range(min(3, len(all_outputs))):
#             print(f"  Image {i}: {all_outputs[i][:100]}...")
        
#         # Create result
#         result_dict = create_result_dict(
#             model_name=model_name,
#             batch_size=batch_size,
#             num_samples=num_images,
#             runtime_s=runtime,
#             throughput=throughput,
#             gpu_summary=gpu_summary,
#             platform=platform,
#             additional_metrics={
#                 'modality': 'vision_language',
#                 'image_type': 'synthetic',
#                 'tokens_per_second': tokens_per_second,
#                 'total_tokens': total_tokens,
#                 'max_new_tokens': 50
#             }
#         )
#         results.append(result_dict)
        
#         # Log metrics
#         logger.log_gpu_metrics(metrics)
    
#     # Save results
#     logger.log_results(results)
#     print_result_table(results)
    
#     print("\n" + "="*80)
#     print("Test Complete!")
#     print("="*80)


# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--platform", default="H100")
#     parser.add_argument("--num-images", type=int, default=5000)
#     args = parser.parse_args()
    
#     run_multimodal_test(args.platform, args.num_images)


#!/usr/bin/env python3
"""
Corrected Multimodal Inference Benchmark
Includes explicit CUDA verification and GPU monitoring
"""

import torch
import time
import json
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import numpy as np

def verify_cuda_setup():
    """Verify CUDA is available"""
    print("=" * 80)
    print("CUDA VERIFICATION")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available! Will run on CPU.")
    
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Initial Memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    print()

def load_model_with_verification(model_name):
    """Load model and explicitly verify GPU placement"""
    print(f"Loading model: {model_name}")
    
    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # EXPLICIT DEVICE VERIFICATION
    model_device = next(model.parameters()).device
    print(f"Model device: {model_device}")
    
    if model_device.type != 'cuda':
        raise RuntimeError(f"ERROR: Model on {model_device}, not CUDA!")
    
    # Check GPU memory
    memory_gb = torch.cuda.memory_allocated() / 1024**3
    print(f"Model Memory: {memory_gb:.2f} GB")
    
    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params/1e9:.2f}B")
    print()
    
    return processor, model

def generate_synthetic_images(num_images, size=(224, 224)):
    """Generate synthetic RGB images"""
    images = []
    for i in range(num_images):
        # Random RGB image
        arr = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
        images.append(Image.fromarray(arr))
    return images

def run_multimodal_benchmark(model, processor, batch_size=4, num_images=1000):
    """Run multimodal benchmark with GPU monitoring"""
    print("=" * 80)
    print("MULTIMODAL BENCHMARK")
    print("=" * 80)
    print(f"Batch size: {batch_size}")
    print(f"Total images: {num_images}")
    print()
    
    # Generate images
    print("Generating synthetic images...")
    images = generate_synthetic_images(num_images)
    print(f"Generated {len(images)} images\n")
    
    # Warmup
    print("Warmup...")
    inputs = processor(images=images[0], return_tensors="pt").to(model.device)
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=10)
    torch.cuda.synchronize()
    print("Complete\n")
    
    # Record GPU utilization at start
    start_memory = torch.cuda.memory_allocated() / 1024**3
    print(f"Starting GPU memory: {start_memory:.2f} GB")
    
    # Run inference
    print("Running inference...")
    total_tokens = 0
    total_time = 0
    num_batches = (num_images + batch_size - 1) // batch_size
    
    for i in range(0, num_images, batch_size):
        batch_images = images[i:i+batch_size]
        
        inputs = processor(
            images=batch_images,
            return_tensors="pt",
            padding=True
        ).to(model.device)
        
        # VERIFY INPUT ON GPU
        if inputs.pixel_values.device.type != 'cuda':
            raise RuntimeError(f"Inputs on {inputs.pixel_values.device}, not GPU!")
        
        torch.cuda.synchronize()
        start = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                num_beams=1
            )
        
        torch.cuda.synchronize()
        end = time.time()
        
        batch_time = end - start
        batch_tokens = outputs.numel() - inputs.get('input_ids', torch.tensor([[]])).numel()
        
        total_tokens += batch_tokens
        total_time += batch_time
        
        if (i // batch_size + 1) % 50 == 0:
            batch_num = i // batch_size + 1
            print(f"  Batch {batch_num}/{num_batches}: {batch_time:.2f}s")
    
    # Check peak memory
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\nPeak GPU memory: {peak_memory:.2f} GB")
    print()
    
    # Calculate metrics
    img_throughput = num_images / total_time
    token_throughput = total_tokens / total_time
    
    results = {
        "batch_size": batch_size,
        "num_images": num_images,
        "total_time": total_time,
        "image_throughput": img_throughput,
        "token_throughput": token_throughput,
        "total_tokens": total_tokens,
        "peak_memory_gb": peak_memory
    }
    
    return results

def main():
    MODEL_NAME = "Salesforce/blip2-flan-t5-xl"
    BATCH_SIZES = [1, 2, 4]
    NUM_IMAGES = 1000
    
    print("MULTIMODAL INFERENCE BENCHMARK - CORRECTED\n")
    
    verify_cuda_setup()
    processor, model = load_model_with_verification(MODEL_NAME)
    
    all_results = {}
    
    for bs in BATCH_SIZES:
        print(f"\n{'='*80}")
        print(f"BATCH SIZE {bs}")
        print('='*80)
        
        results = run_multimodal_benchmark(model, processor, bs, NUM_IMAGES)
        all_results[f"batch_size_{bs}"] = results
        
        print("RESULTS:")
        print(f"  Runtime: {results['total_time']:.2f}s")
        print(f"  Image throughput: {results['image_throughput']:.2f} img/s")
        print(f"  Token throughput: {results['token_throughput']:.2f} tok/s")
    
    # Save results
    with open('multimodal_results_corrected.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\nResults saved to: multimodal_results_corrected.json")

if __name__ == "__main__":
    main()