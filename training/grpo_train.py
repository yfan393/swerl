"""
training/grpo_train.py
======================
GRPO (Group Relative Policy Optimization) training.
"""
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

def train(config_path: str) -> None:
    """
    Run GRPO training loop.
    
    Args:
        config_path: Path to training config YAML
    """
    from utils.io_utils import read_yaml, read_jsonl
    
    config = read_yaml(config_path)
    
    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
        )
        from trl import GRPOConfig, GRPOTrainer
        from peft import LoraConfig, get_peft_model
    except ImportError as e:
        logger.warning(f"Missing dependency: {e}, using basic training")
        return
    
    # Load config
    model_name = config.get("model", {}).get("name_or_path", "gpt2")
    train_file = config.get("paths", {}).get("train_file", "data/processed/train.jsonl")
    output_dir = config.get("training", {}).get("output_dir", "outputs/grpo/")
    
    # Load training data
    records = read_jsonl(train_file)
    if not records:
        logger.error(f"No training data found in {train_file}")
        return
    
    logger.info(f"Loaded {len(records)} training records")
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Apply LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
    )
    model = get_peft_model(model, lora_config)
    
    # GRPO config
    grpo_config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=config.get("grpo", {}).get("num_epochs", 3),
        per_device_train_batch_size=config.get("training", {}).get("per_device_train_batch_size", 4),
        learning_rate=config.get("training", {}).get("learning_rate", 5e-5),
    )
    
    # Train
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=records,
        tokenizer=tokenizer,
    )
    
    trainer.train()
    
    logger.info(f"Training complete. Model saved to {output_dir}")
