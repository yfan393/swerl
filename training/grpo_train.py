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
    grpo_cfg = config.get("grpo", {})
    log_cfg = config.get("logging", {})

    max_steps = grpo_cfg.get("max_steps", 300)
    num_train_epochs = grpo_cfg.get("num_train_epochs", 1)

    grpo_config = GRPOConfig(
        output_dir=output_dir,
        max_steps=max_steps,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=grpo_cfg.get("per_device_train_batch_size", config.get("training", {}).get("per_device_train_batch_size", 32)),
        learning_rate=grpo_cfg.get("learning_rate", config.get("training", {}).get("learning_rate", 5e-5)),
        gradient_accumulation_steps=grpo_cfg.get("gradient_accumulation_steps", 1),
        logging_steps=grpo_cfg.get("logging_steps", 10),
        save_steps=grpo_cfg.get("save_steps", 100),
        report_to=log_cfg.get("report_to", "none"),
        lr_scheduler_type=grpo_cfg.get("lr_scheduler", "linear"),
        warmup_steps=grpo_cfg.get("warmup_steps", 0),
        max_grad_norm=grpo_cfg.get("max_grad_norm", 1.0),
        weight_decay=grpo_cfg.get("weight_decay", 0.0),
        bf16=False,
        fp16=False,
    )

    # Load reward config
    reward_cfg = config.get("reward", {})

    # Preprocess records: build prompts and ensure required fields
    from agent.prompts import build_messages
    import json

    processed_records = []
    for record in records:
        problem_statement = record.get("problem_statement", "")
        code_context = record.get("code_context", "")
        oracle_new_content = record.get("oracle_new_content", {})

        # Build prompt using same method as SFT
        messages = build_messages(problem_statement, code_context)
        prompt_str = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Create processed record with required fields for GRPOTrainer and reward function
        processed_record = {
            "prompt": prompt_str,
            "code_context": json.dumps(code_context) if isinstance(code_context, dict) else code_context,
            "oracle_new_content": json.dumps(oracle_new_content) if isinstance(oracle_new_content, dict) else oracle_new_content,
        }
        # Preserve metadata
        for key in ["instance_id", "repo", "problem_statement"]:
            if key in record:
                processed_record[key] = record[key]

        processed_records.append(processed_record)

    logger.info(f"Processed {len(processed_records)} records with prompts")

    # Train
    # Convert list of dicts to HuggingFace Dataset
    train_dataset = Dataset.from_list(processed_records)

    # Create reward function
    from reward.reward_fn import SWERLRewardFunction
    reward_fn = SWERLRewardFunction(
        alpha=reward_cfg.get("alpha", 0.3),
        use_lint=True,
        continuous_correctness=reward_cfg.get("continuous_correctness", True),
        use_matcher_correctness=reward_cfg.get("use_matcher_correctness", True),
    )
    # Add __name__ attribute for GRPOTrainer logging
    reward_fn.__name__ = "SWERLReward"

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        reward_funcs=[reward_fn],
    )
    
    trainer.train()
    
    logger.info(f"Training complete. Model saved to {output_dir}")
