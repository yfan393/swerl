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
        from datasets import Dataset
    except ImportError as e:
        raise RuntimeError(
            f"Cannot run GRPO training: missing dependency {e}.\n"
            f"Install with: pip install transformers trl peft datasets"
        )
    
    # Load config
    model_name = config.get("model", {}).get("name_or_path", "gpt2")
    train_file = config.get("paths", {}).get("train_file", "data/processed/train.jsonl")
    output_dir = config.get("paths", {}).get("output_dir", "outputs/grpo/")
    
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
    # Note: max_steps takes precedence over num_train_epochs
    max_steps = config.get("grpo", {}).get("max_steps", 300)
    num_train_epochs = config.get("grpo", {}).get("num_train_epochs", None)

    grpo_config = GRPOConfig(
        output_dir=output_dir,
        max_steps=max_steps,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=config.get("training", {}).get("per_device_train_batch_size", 8), 
        learning_rate=config.get("training", {}).get("learning_rate", 5e-5),
    )
     
    # Train
    # Convert list of dicts to HuggingFace Dataset
    train_dataset = Dataset.from_list(records)

    # Create reward function
    from reward.reward_fn import SWERLRewardFunction
    reward_fn = SWERLRewardFunction(
        alpha=reward_cfg.get("alpha", 0.3),
        use_lint=True,
        continuous_correctness=reward_cfg.get("continuous_correctness", True),
        use_matcher_correctness=reward_cfg.get("use_matcher_correctness", True),
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        reward_funcs=reward_fn,
    )
    
    trainer.train()
    
    logger.info(f"Training complete. Model saved to {output_dir}")
