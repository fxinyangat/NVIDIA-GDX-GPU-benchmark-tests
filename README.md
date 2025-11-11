# GPU Benchmark Suite - Complete Documentation

## Project Overview

Comprehensive GPU benchmarking suite comparing NVIDIA GB200 (DGX Spark) against NVIDIA RTX A6000 across inference, fine-tuning, and memory-intensive workloads. Designed to measure real-world AI/ML performance and raw computational capabilities.

**Platform Comparison:**
- **DGX Spark:** NVIDIA GB200 Blackwell architecture
- **NVIDIA RTX A6000:** NVIDIA Ampere architecture

**Goal:** Quantify performance differences, energy efficiency, and cost-effectiveness between GPU generations.

---

## Benchmark Categories

### 1. Text Inference
### 2. Multimodal Inference (Vision-Language)
### 3. Fine-Tuning (LoRA)
### 4. Memory Operations (RAPIDS)

---

## 1. Text Inference Benchmark

### Purpose
Measures language model inference performance - how fast GPUs generate text from prompts.

### Models Tested
- **Phi-2** (2.7B parameters) - Fast, efficient small model
- **Llama-3-8B** (8B parameters) - Mid-size flagship model
- **Mixtral-8x7B** (47B parameters) - Large mixture-of-experts model

### What It Measures
- **Tokens per second:** Text generation speed
- **Throughput:** Samples processed per second
- **Latency:** Time to first token, time per token
- **GPU utilization:** Hardware efficiency
- **Memory usage:** Peak VRAM consumption
- **Energy consumption:** Power draw and total energy

### Batch Sizes Tested
- 1, 4, 8, 16, 32 (varying parallelism)

### Dataset
- **GSM8K:** Grade school math word problems
- **Size:** 100 samples (test), 1000 samples (production)

### Key Metrics
```
Throughput = samples_per_second
Token Rate = tokens_generated / runtime_seconds  
GPU Efficiency = gpu_utilization_percentage
Energy Efficiency = tokens_per_watt_hour
```

### Execution
```bash
# Quick test (Phi-2, 100 samples, ~20 min)
bash run_safe_inference.sh

# Production (All models, 1000 samples, ~2 hours)
bash run_production_inference.sh
```

### Expected Results
- **Phi-2:** 50-100 tokens/s
- **Llama-3-8B:** 30-60 tokens/s  
- **Mixtral-8x7B:** 15-30 tokens/s
- **GB200 vs a6000:** 

---

## 2. Multimodal Inference Benchmark

### Purpose
Measures vision-language model performance - generating text descriptions from images.

### Model Tested
- **BLIP2-FlanT5-XL** - Image captioning model (Vision encoder + Text decoder)

### What It Measures
- **Images per second:** Visual processing throughput
- **Tokens per second:** Caption generation speed
- **GPU utilization:** Hardware efficiency on multimodal workload
- **Memory usage:** VRAM for vision+language model
- **Total tokens generated:** Volume of text produced

### Batch Sizes Tested
- 1, 2, 4 (multimodal models are memory-intensive)

### Dataset
- **Synthetic RGB images:** 224×224×3 randomly generated
- **Size:** 50 images (test), 5000 images (production)
- **Why synthetic?** Eliminates download delays, tests same GPU operations as real images

### Key Metrics
```
Image Throughput = images_processed / runtime_seconds
Token Throughput = total_tokens / runtime_seconds
GPU Efficiency = gpu_utilization_percentage
```

### Execution
```bash
# Simple synthetic test (~10 min)
python inference/simple_multimodal_test.py --platform dgx_spark --num-images 50

# Full benchmark with dataset loading
python inference/multimodal_inference.py --platform dgx_spark --config config/benchmark_config.yaml
```

### Architecture Tested
```
Image → Vision Encoder (CLIP) → Cross-Attention → Text Decoder (T5) → Caption
         [GPU-intensive]          [Memory-bound]    [Sequential]
```

---

## 3. Fine-Tuning Benchmarks

### Purpose
Measures training performance - how fast GPUs can update model weights during learning.

### 3A. LoRA Fine-Tuning (Parameter-Efficient)

**What is LoRA?**
- Low-Rank Adaptation - trains small adapter layers instead of full model
- Uses QLoRA (4-bit quantization) for memory efficiency
- Typical use: Fine-tuning large models on consumer GPUs

**Model:** Phi-2 (2.7B parameters)  
**Trainable Parameters:** ~1% of total (highly efficient)

**What It Measures:**
- Training samples per second
- Gradient update speed
- Memory efficiency (4-bit vs 16-bit)
- GPU utilization during backpropagation

**Configuration:**
```yaml
epochs: 1
batch_size: 4
learning_rate: 2e-4
lora_r: 16
lora_alpha: 32
quantization: 4-bit
```

**Execution:**
```bash
# Test config (~15 min)
python finetuning/lora_finetune.py --platform dgx_spark --config config/test_config.yaml

# Production config (~45 min)
python finetuning/lora_finetune.py --platform dgx_spark
```

**Expected Results:**
- **Training speed:** 5-15 samples/s
- **Memory usage:** 15-20 GB (vs 40+ GB for full fine-tuning)
- **GB200 vs H100:** 

---

### 3B. Full Fine-Tuning (All Parameters)

**What is Full Fine-Tuning?**
- Updates all model parameters (traditional training)
- Higher memory requirements
- Better final performance but more expensive

**Model:** OPT-1.3B (1.3B parameters)  
**Trainable Parameters:** 100% (all weights updated)

**What It Measures:**
- Training throughput (samples/s)
- Convergence speed (loss reduction)
- Memory scaling with batch size
- Gradient computation efficiency

**Configuration:**
```yaml
epochs: 1
batch_size: 2
learning_rate: 5e-5
gradient_accumulation: 4
mixed_precision: fp16
```

**Execution:**
```bash
python finetuning/full_finetune.py --platform dgx_spark --config config/test_config.yaml
```


---

### Dataset for Fine-Tuning
**OpenAssistant Guanaco:**
- Instruction-following conversations
- Format: User question → Assistant response
- Size: 1000 samples (test), 10000 samples (production)

---

## 4. RAPIDS Memory Benchmark

### Purpose
Measures raw GPU computational power and memory bandwidth on data science operations, independent of ML frameworks.

### What is RAPIDS?
NVIDIA's GPU-accelerated pandas/numpy alternative. Runs data operations on GPU instead of CPU.

### Four Test Categories

#### 4A. DataFrame Join
**Operation:** Inner join of two massive tables  
**Workload:** 100M row table + 50M row table  
**Tests:** Memory bandwidth, hash operations, parallel matching  
**Metric:** Rows processed per second

**Real-world analog:** Merging customer database with transaction history

#### 4B. GroupBy Aggregation  
**Operation:** Group by category, compute statistics  
**Workload:** 200M rows, 1M unique groups  
**Tests:** Parallel reductions, statistical computations  
**Metric:** Aggregations per second

**Real-world analog:** Calculate average sales per product from millions of transactions

#### 4C. Sort Operations
**Operation:** Multi-column sort  
**Workload:** 100M rows × 4 columns  
**Tests:** Parallel sorting, data rearrangement  
**Metric:** Rows sorted per second

**Real-world analog:** Sorting billions of log entries by timestamp and severity

#### 4D. Matrix Operations
**Operation:** Matrix multiplication, SVD, eigenvalues  
**Workload:** 20K × 20K matrices (400M values)  
**Tests:** FLOPS, tensor core utilization  
**Metric:** TFLOPS (trillion floating-point operations per second)

**Real-world analog:** Neural network training math operations

### What Each Test Stresses

| Test | Primary | Secondary |
|------|---------|-----------|
| **Join** | Memory bandwidth | Hash computations |
| **GroupBy** | Reduction ops | Memory access patterns |
| **Sort** | Data movement | Comparison ops |
| **Matrix** | Compute (FLOPS) | Tensor cores |

### Installation
```bash
pip install cudf-cu12 cupy-cuda12x --extra-index-url=https://pypi.nvidia.com
```

### Execution
```bash
# Test config (10M-20M rows, ~15 min)
bash run_rapids_test.sh

# Production config (100M-200M rows, ~45 min)
python memory_tasks/rapids_benchmark.py --platform dgx_spark
```

### Why RAPIDS Matters
**ML benchmarks measure:** Model inference (indirect hardware test)  
**RAPIDS measures:** Pure GPU capability (direct hardware test)

Result: **Hardware baseline** that shows raw computational power before ML framework overhead.

---

## Project Structure
```
gpu_benchmark_suite/
├── inference/
│   ├── text_inference.py           # LLM inference benchmark
│   ├── multimodal_inference.py     # Vision-language benchmark
│   └── simple_multimodal_test.py   # Synthetic image test
├── finetuning/
│   ├── lora_finetune.py           # LoRA/QLoRA training
│   └── full_finetune.py           # Full parameter training
├── memory_tasks/
│   └── rapids_benchmark.py         # cuDF/CuPy operations
├── utils/
│   ├── gpu_monitor.py             # GPU metrics collection
│   ├── benchmark_logger.py        # Results logging
│   └── common.py                  # Shared utilities
├── config/
│   ├── benchmark_config.yaml      # Production settings
│   └── test_config.yaml           # Quick test settings
├── logs/                          # Benchmark results (CSV)
└── analysis/                      # Jupyter notebooks for visualization
```

---

## Quick Start Guide

### Prerequisites
```bash
# Python environment
python >= 3.10
pip install -r requirements.txt

# For RAPIDS
pip install cudf-cu12 cupy-cuda12x --extra-index-url=https://pypi.nvidia.com
```

### Running All Benchmarks
```bash
cd gpu_benchmark_suite

# 1. Multimodal (fastest, 10 min)
python inference/simple_multimodal_test.py --platform dgx_spark --num-images 50

# 2. Text Inference Test (20 min)
bash run_safe_inference.sh

# 3. RAPIDS (15 min)
bash run_rapids_test.sh

# 4. LoRA Fine-tuning (15 min)
python finetuning/lora_finetune.py --platform dgx_spark --config config/test_config.yaml

# 5. Text Inference Production (2 hours)
bash run_production_inference.sh
```

---

## Metrics Collected

### Performance Metrics
- **Throughput:** Samples/images/tokens per second
- **Latency:** Time to first token, end-to-end time
- **TFLOPS:** Floating-point operations per second

### Hardware Metrics
- **GPU Utilization:** Percentage of GPU cores active
- **Memory Usage:** Peak VRAM consumption (MB/GB)
- **Temperature:** GPU temperature (°C)
- **Power Draw:** Instantaneous power consumption (W)
- **Energy:** Total energy consumed (Wh)
- **Clock Speeds:** SM and memory clock frequencies (MHz)

### Efficiency Metrics
```
Performance per Watt = throughput / average_power_draw
Cost Efficiency = (performance × spot_price) / hour
Memory Efficiency = throughput / peak_memory_used
```

---

## Output Files

All benchmarks save results to `logs/` directory:
```
logs/
├── {platform}_{benchmark}_{timestamp}.csv           # Results summary
├── {platform}_{benchmark}_{timestamp}_metrics.csv   # GPU metrics (time-series)
└── {platform}_{benchmark}_{timestamp}_config.json   # Configuration used
```

### Results CSV Contains
- Model/operation name
- Platform (dgx_spark / lambda_h100)
- Batch size
- Runtime
- Throughput
- GPU metrics (utilization, memory, power)
- Timestamp

---

## Configuration Files

### Test Config (`config/test_config.yaml`)
**Purpose:** Quick validation, smaller workloads  
**Time:** 15-30 minutes per benchmark  
**Use when:** Testing setup, debugging, rapid iteration

### Production Config (`config/benchmark_config.yaml`)
**Purpose:** Comprehensive benchmarking, production workloads  
**Time:** 30 minutes - 2 hours per benchmark  
**Use when:** Final comparison, publication, detailed analysis

---

## Analysis & Visualization

After collecting data from both platforms:
```bash
cd analysis

# Generate comparison report
python compare_platforms.py --baseline lambda_h100 --comparison dgx_spark

# Launch Jupyter notebook for visualization
jupyter notebook analysis_notebook.ipynb
```

### Comparison Metrics
- Speedup ratio (GB200 / H100)
- Energy efficiency gains
- Cost-performance analysis
- Memory bandwidth utilization
- Scaling efficiency

---

## Known Issues & Limitations

### GB200 (Blackwell) Monitoring Limitations
**Issue:** Some NVML metrics return zero on GB200:
- `memory_used_mb_peak`: 0
- `power_draw_w_mean`: 0.0
- `energy_consumed_wh`: 0.0

**Cause:** GB200 too new for current NVML API support

**Workaround:** 
- Use `nvidia-smi` directly for manual monitoring
- Compare metrics available on both platforms only
- Run diagnostic: `python diagnose_gpu_metrics.py`

**Status:** Expected to be resolved in future NVIDIA driver updates

### BLIP2 Processor Compatibility
**Issue:** `image_sizes` parameter error with certain transformers versions

**Solution:** Script includes runtime patch for compatibility

---


---


## Citation & Research Use

If you use this benchmark suite in research, please cite:
```bibtex
@software{gpu_benchmark_suite_2025,
  title = {Comprehensive GPU Benchmark Suite: GB200 vs A6000 Comparison},
  author = {Francis Xavier Inyangat},
  year = {2025},
  institution = {Arizona State University},
  note = {AI Acceleration Lab}
}
```

---

## License

MIT License - See LICENSE file for details

---

## Contact

**Francis Xavier Inyangat**  
AI Research & Data Analyst  
Arizona State University  
Email: [xavierfranzings@gmail.com]  
GitHub: [https://github.com/fxinyangat]

---

## Acknowledgments

- NVIDIA for GPU hardware access (DGX Spark)
- Arizona State University AI Research Lab
- HuggingFace for model hosting
- RAPIDS team for cuDF/CuPy libraries

---

**Last Updated:** November 2025  
**Version:** 1.0  
**Status:** Active Development
