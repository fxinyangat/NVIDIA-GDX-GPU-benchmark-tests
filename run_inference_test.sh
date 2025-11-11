#!/bin/bash
# Quick Text Inference Test on DGX Spark

echo "=========================================="
echo "Text Inference Benchmark - DGX Spark"
echo "=========================================="
echo ""

# Set platform
export BENCHMARK_PLATFORM="dgx_spark"
export CUDA_VISIBLE_DEVICES=0

# Create logs directory
mkdir -p logs

echo "Starting text inference benchmark..."
echo "Platform: DGX Spark"
echo "Using test configuration for faster results"
echo ""

# Run with test config (smaller workload)
python inference/text_inference.py \
    --platform dgx_spark \
    --config config/test_config.yaml \
    --log-dir logs

echo ""
echo "=========================================="
echo "Benchmark Complete!"
echo "=========================================="
echo ""
echo "Results saved to: logs/"
echo ""
echo "To view results:"
echo "  ls -lh logs/"
echo "  cat logs/dgx_spark_text_inference_*.csv"