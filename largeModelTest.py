#!/usr/bin/env python3
"""
Large Model Benchmark - Tests models that exceed RTX 6000 Ada capacity
This clearly differentiates GPUs based on VRAM capacity
"""

import torch
import time
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import sys

def verify_cuda_setup():
    """Verify CUDA and report GPU specs"""
    print("=" * 80)
    print("GPU SPECIFICATIONS")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        sys.exit(1)
    
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    
    # Get total VRAM
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"Total VRAM: {total_memory:.2f} GB")
    print()
    
    return total_memory

def estimate_model_size(model_name, quantization=None):
    """Estimate model memory requirements"""
    # Parameter counts for common models
    model_params = {
        "meta-llama/Llama-3.1-70B-Instruct": 70,
        "meta-llama/Llama-3.1-8B-Instruct": 8,
        "meta-llama/Llama-3.2-3B-Instruct": 3,
    }
    
    params_b = model_params.get(model_name, 70)  # default to 70B
    
    # Calculate size based on quantization
    if quantization == "8bit":
        size_gb = params_b * 1.0  # 1 byte per parameter
    elif quantization == "4bit":
        size_gb = params_b * 0.5  # 0.5 bytes per parameter
    else:
        size_gb = params_b * 2.0  # FP16: 2 bytes per parameter
    
    # Add overhead for KV cache and activations (rough estimate)
    size_gb = size_gb * 1.2
    
    return size_gb

def try_load_model(model_name, quantization="8bit", max_memory_gb=None):
    """Attempt to load model with given quantization"""
    print(f"Attempting to load: {model_name}")
    print(f"Quantization: {quantization}")
    
    estimated_size = estimate_model_size(model_name, quantization)
    print(f"Estimated size: {estimated_size:.1f} GB")
    
    if max_memory_gb and estimated_size > max_memory_gb:
        print(f"Model likely exceeds {max_memory_gb:.1f} GB VRAM capacity")
        print("Attempting to load anyway (will OOM if insufficient)...")
    
    print()
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Configure quantization
        if quantization == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        elif quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
        else:
            quantization_config = None
        
        print("Loading model (this may take several minutes)...")
        start_time = time.time()
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16 if quantization is None else None,
            low_cpu_mem_usage=True
        )
        
        load_time = time.time() - start_time
        
        # Check actual memory usage
        memory_used = torch.cuda.memory_allocated() / 1024**3
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        
        print(f"✓ Model loaded successfully in {load_time:.1f}s")
        print(f"✓ Actual memory used: {memory_used:.2f} GB")
        print(f"✓ Peak memory: {peak_memory:.2f} GB")
        
        # Verify on GPU
        device = next(model.parameters()).device
        print(f"✓ Model device: {device}")
        print()
        
        return tokenizer, model, True, peak_memory
        
    except torch.cuda.OutOfMemoryError:
        print("✗ FAILED: Out of Memory (OOM)")
        print(f"   Model requires more than {max_memory_gb:.1f} GB VRAM")
        print()
        return None, None, False, 0
    except Exception as e:
        print(f"✗ FAILED: {str(e)}")
        print()
        return None, None, False, 0

def run_inference_test(model, tokenizer, model_name):
    """Run basic inference test if model loaded successfully"""
    print("Running inference test...")
    
    prompt = "Explain quantum computing in simple terms:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False
        )
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    gen_time = end_time - start_time
    tokens_generated = outputs.shape[1] - inputs.input_ids.shape[1]
    tps = tokens_generated / gen_time
    
    print(f"✓ Generated {tokens_generated} tokens in {gen_time:.2f}s")
    print(f"✓ Throughput: {tps:.2f} tokens/sec")
    
    # Calculate theoretical max
    memory_bandwidth = 960  # GB/s for RTX 6000 Ada (update for your GPU)
    model_size = torch.cuda.memory_allocated() / 1024**3
    theoretical_max = memory_bandwidth / model_size
    
    print(f"  Theoretical max: {theoretical_max:.2f} tok/s")
    print(f"  Efficiency: {(tps/theoretical_max)*100:.1f}%")
    print()
    
    return tps

def main():
    print("=" * 80)
    print("LARGE MODEL CAPACITY TEST")
    print("=" * 80)
    print()
    
    total_vram = verify_cuda_setup()
    
    # Test configurations (model, quantization)
    test_configs = [
        ("meta-llama/Llama-3.2-3B-Instruct", None, "Baseline: 3B FP16"),
        ("meta-llama/Llama-3.1-8B-Instruct", None, "Baseline: 8B FP16"),
        ("meta-llama/Llama-3.1-70B-Instruct", "8bit", "Large: 70B INT8"),
        ("meta-llama/Llama-3.1-70B-Instruct", "4bit", "Large: 70B INT4"),
    ]
    
    results = {}
    
    for model_name, quant, description in test_configs:
        print("=" * 80)
        print(f"TEST: {description}")
        print("=" * 80)
        print()
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        tokenizer, model, success, peak_mem = try_load_model(
            model_name,
            quant,
            total_vram
        )
        
        if success:
            tps = run_inference_test(model, tokenizer, model_name)
            results[description] = {
                "success": True,
                "peak_memory_gb": peak_mem,
                "tokens_per_sec": tps
            }
            
            # Clean up
            del model
            del tokenizer
            torch.cuda.empty_cache()
        else:
            results[description] = {
                "success": False,
                "reason": "Out of memory"
            }
        
        print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"GPU VRAM: {total_vram:.1f} GB\n")
    
    for test_name, result in results.items():
        if result["success"]:
            print(f"✓ {test_name}")
            print(f"    Memory: {result['peak_memory_gb']:.1f} GB")
            print(f"    Speed: {result['tokens_per_sec']:.2f} tok/s")
        else:
            print(f"✗ {test_name}")
            print(f"    {result['reason']}")
        print()
    
    # Save results
    with open('large_model_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("✓ Results saved to: large_model_test_results.json")
    print()
    print("INTERPRETATION:")
    print("- Models that load show GPU can handle that capacity")
    print("- Models that OOM exceed GPU VRAM limits")
    print("- This clearly differentiates GPUs by memory capacity")

if __name__ == "__main__":
    main()