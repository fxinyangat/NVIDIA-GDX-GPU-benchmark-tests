"""
LoRA Fine-tuning Benchmark
Parameter-efficient fine-tuning using LoRA (Low-Rank Adaptation)
"""

import os
import sys
import yaml
import argparse
import time
from typing import Dict
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
import bitsandbytes as bnb

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import GPUMonitor, BenchmarkLogger, create_result_dict, print_result_table


def load_config(config_path: str = "config/benchmark_config.yaml") -> Dict:
    """Load benchmark configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def prepare_dataset(
    dataset_name: str,
    dataset_split: str,
    tokenizer,
    max_seq_length: int = 512,
    max_samples: int = None
):
    """
    Load and prepare dataset for fine-tuning
    
    Args:
        dataset_name: Dataset identifier
        dataset_split: Dataset split to use
        tokenizer: Tokenizer
        max_seq_length: Maximum sequence length
        max_samples: Maximum number of samples (None for all)
    
    Returns:
        Processed dataset
    """
    print(f"Loading dataset: {dataset_name} ({dataset_split})")
    
    dataset = load_dataset(dataset_name, split=dataset_split, trust_remote_code=True)
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    print(f"Dataset size: {len(dataset)}")
    
    # Tokenization function
    def tokenize_function(examples):
        # Format for instruction tuning (OpenAssistant format)
        if 'text' in examples:
            texts = examples['text']
        elif 'conversations' in examples:
            # Format conversations
            texts = []
            for conv in examples['conversations']:
                if isinstance(conv, list):
                    text = "\n".join([f"{turn['from']}: {turn['value']}" for turn in conv])
                else:
                    text = str(conv)
                texts.append(text)
        else:
            # Use first text field
            texts = [str(v) for v in list(examples.values())[0]]
        
        # Tokenize
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_seq_length,
            padding='max_length',
            return_tensors=None
        )
        
        # Labels are input_ids for causal LM
        tokenized['labels'] = tokenized['input_ids'].copy()
        
        return tokenized
    
    # Process dataset
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    print(f"Tokenized dataset size: {len(tokenized_dataset)}")
    return tokenized_dataset


def create_lora_model(
    model_name: str,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    use_4bit: bool = True
):
    """
    Create model with LoRA adapters
    
    Args:
        model_name: Model identifier
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        use_4bit: Use 4-bit quantization (QLoRA)
    
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"\nLoading model: {model_name}")
    print(f"LoRA config: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    print(f"4-bit quantization: {use_4bit}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with optional quantization
    if use_4bit:
        # QLoRA: 4-bit quantization
        from transformers import BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Prepare for k-bit training
        model = prepare_model_for_kbit_training(model)
    else:
        # Standard 16-bit
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Common for transformers
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Add LoRA adapters
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"Total parameters: {total_params:,}")
    
    return model, tokenizer


def run_training(
    model,
    tokenizer,
    train_dataset,
    training_args: TrainingArguments,
    gpu_monitor: GPUMonitor
) -> Dict:
    """
    Run LoRA fine-tuning
    
    Args:
        model: PEFT model with LoRA
        tokenizer: Tokenizer
        train_dataset: Training dataset
        training_args: Training arguments
        gpu_monitor: GPU monitor instance
    
    Returns:
        Dictionary with results
    """
    print(f"\n{'='*80}")
    print("Starting LoRA Fine-tuning")
    print(f"{'='*80}\n")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, not masked LM
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # Start monitoring
    gpu_monitor.start()
    start_time = time.time()
    
    # Train
    train_result = trainer.train()
    
    # Stop monitoring
    end_time = time.time()
    metrics = gpu_monitor.stop()
    
    # Calculate statistics
    runtime = end_time - start_time
    samples_per_second = len(train_dataset) * training_args.num_train_epochs / runtime
    
    print(f"\nTraining completed:")
    print(f"  Runtime: {runtime:.2f}s")
    print(f"  Samples/second: {samples_per_second:.2f}")
    print(f"  Final loss: {train_result.training_loss:.4f}")
    
    return {
        'runtime': runtime,
        'samples_per_second': samples_per_second,
        'training_loss': train_result.training_loss,
        'num_steps': train_result.global_step,
        'metrics': metrics
    }


def run_benchmark(
    model_config: Dict,
    dataset_config: Dict,
    platform: str,
    log_dir: str
):
    """
    Run complete LoRA fine-tuning benchmark
    
    Args:
        model_config: Model configuration
        dataset_config: Dataset configuration
        platform: Platform name
        log_dir: Directory for logs
    """
    model_name = model_config['name']
    batch_size = model_config['batch_size']
    gradient_accumulation_steps = model_config['gradient_accumulation_steps']
    num_epochs = model_config['num_epochs']
    max_steps = model_config['max_steps']
    lora_r = model_config['lora_r']
    lora_alpha = model_config['lora_alpha']
    lora_dropout = model_config['lora_dropout']
    use_4bit = dataset_config.get('use_4bit', True)
    max_seq_length = dataset_config['max_seq_length']
    
    # Initialize logger
    logger = BenchmarkLogger(
        log_dir=log_dir,
        platform=platform,
        benchmark_type='lora_finetune'
    )
    
    # Log configuration
    config_dict = {
        'model': model_name,
        'batch_size': batch_size,
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'num_epochs': num_epochs,
        'max_steps': max_steps,
        'lora_r': lora_r,
        'lora_alpha': lora_alpha,
        'lora_dropout': lora_dropout,
        'use_4bit': use_4bit,
        'max_seq_length': max_seq_length,
        'platform': platform
    }
    logger.log_config(config_dict)
    
    # Create model
    model, tokenizer = create_lora_model(
        model_name,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_4bit=use_4bit
    )
    
    # Prepare dataset
    train_dataset = prepare_dataset(
        dataset_config['dataset'],
        dataset_config['dataset_split'],
        tokenizer,
        max_seq_length=max_seq_length,
        max_samples=max_steps * batch_size * gradient_accumulation_steps * 2  # Approximate
    )
    
    # Training arguments
    output_dir = os.path.join(log_dir, f"lora_checkpoint_{platform}")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        logging_steps=50,
        save_strategy="steps",
        save_steps=500,
        fp16=True,
        optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
        report_to="none",  # Disable wandb/tensorboard
        remove_unused_columns=False,
    )
    
    # Initialize GPU monitor
    gpu_monitor = GPUMonitor(device_id=0, sample_interval=0.5)
    
    # Run training
    try:
        result = run_training(
            model, tokenizer, train_dataset,
            training_args, gpu_monitor
        )
        
        # Get GPU summary
        gpu_summary = gpu_monitor.get_summary()
        
        # Calculate effective batch size
        effective_batch_size = batch_size * gradient_accumulation_steps
        
        # Create result entry
        result_dict = create_result_dict(
            model_name=model_name,
            batch_size=effective_batch_size,
            num_samples=len(train_dataset),
            runtime_s=result['runtime'],
            throughput=result['samples_per_second'],
            gpu_summary=gpu_summary,
            platform=platform,
            additional_metrics={
                'training_loss': result['training_loss'],
                'num_steps': result['num_steps'],
                'lora_r': lora_r,
                'lora_alpha': lora_alpha,
                'use_4bit': use_4bit,
                'method': 'QLoRA' if use_4bit else 'LoRA'
            }
        )
        
        # Log results
        logger.log_results([result_dict])
        logger.log_gpu_metrics(result['metrics'])
        print_result_table([result_dict])
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("Benchmark completed!")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning Benchmark")
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
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    lora_config = config['finetuning']['lora']
    
    dataset_config = {
        'dataset': lora_config['dataset'],
        'dataset_split': lora_config['dataset_split'],
        'max_seq_length': lora_config['max_seq_length'],
        'use_4bit': lora_config.get('use_4bit', True)
    }
    
    # Determine which models to benchmark
    if args.model:
        model_configs = [m for m in lora_config['models'] if m['name'] == args.model]
        if not model_configs:
            print(f"Error: Model {args.model} not found in config")
            sys.exit(1)
    else:
        model_configs = lora_config['models']
    
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