"""
training/grpo_train.py
======================
GRPO (Group Relative Policy Optimization) training.
"""
import inspect
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _supports_parameter(cls, parameter: str) -> bool:
    return parameter in inspect.signature(cls).parameters


def _make_grpo_config(GRPOConfig, grpo_cfg: dict, log_cfg: dict, output_dir: str, config: dict):
    kwargs = {
        "output_dir": output_dir,
        "max_steps": grpo_cfg.get("max_steps", 300),
        "num_train_epochs": grpo_cfg.get("num_train_epochs", 1),
        "per_device_train_batch_size": grpo_cfg.get(
            "per_device_train_batch_size",
            config.get("training", {}).get("per_device_train_batch_size", 1),
        ),
        "learning_rate": grpo_cfg.get(
            "learning_rate",
            config.get("training", {}).get("learning_rate", 5e-5),
        ),
        "num_generations": grpo_cfg.get("num_generations", 2),
        "gradient_accumulation_steps": grpo_cfg.get("gradient_accumulation_steps", 1),
        "logging_steps": grpo_cfg.get("logging_steps", 10),
        "save_steps": grpo_cfg.get("save_steps", 100),
        "report_to": log_cfg.get("report_to", "none"),
        "lr_scheduler_type": grpo_cfg.get("lr_scheduler", "linear"),
        "warmup_steps": grpo_cfg.get("warmup_steps", 0),
        "max_grad_norm": grpo_cfg.get("max_grad_norm", 1.0),
        "weight_decay": grpo_cfg.get("weight_decay", 0.0),
        "bf16": False,
        "fp16": False,
    }
    optional_mappings = {
        "beta": "beta",
        "epsilon": "clip_epsilon",
        "max_completion_length": "max_new_tokens",
        "temperature": "generation_temperature",
        "optim": "optimizer",
    }
    for arg_name, cfg_name in optional_mappings.items():
        if cfg_name in grpo_cfg and _supports_parameter(GRPOConfig, arg_name):
            kwargs[arg_name] = grpo_cfg[cfg_name]
    return GRPOConfig(**kwargs)

def train(config_path: str) -> None:
    """
    Run GRPO training loop.

    Args:
        config_path: Path to training config YAML
    """
    from utils.io_utils import read_yaml, read_jsonl

    config = read_yaml(config_path)

    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
        from trl import GRPOConfig, GRPOTrainer
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
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
    model_cfg = config.get("model", {})

    # Load training data
    records = read_jsonl(train_file)
    if not records:
        logger.error(f"No training data found in {train_file}")
        return

    logger.info(f"Loaded {len(records)} training records")

    # Load model and tokenizer
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(model_cfg.get("torch_dtype", "float16"), torch.float16)
    quant_config = None
    if model_cfg.get("load_in_4bit", False):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        quantization_config=quant_config,
        device_map="auto" if model_cfg.get("load_in_4bit", False) else None,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if model_cfg.get("load_in_4bit", False):
        model = prepare_model_for_kbit_training(model)

    # Apply LoRA
    sft_final_dir = Path(config.get("sft_baseline", {}).get("output_dir", "")) / "final"
    if config.get("grpo", {}).get("init_from_sft", True) and (sft_final_dir / "adapter_config.json").exists():
        logger.info("Initializing GRPO from SFT adapter: %s", sft_final_dir)
        model = PeftModel.from_pretrained(model, str(sft_final_dir), is_trainable=True)
    elif model_cfg.get("use_lora", True):
        lora_cfg = model_cfg.get("lora", {})
        lora_config = LoraConfig(
            r=lora_cfg.get("r", 8),
            lora_alpha=lora_cfg.get("lora_alpha", 16),
            target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
            lora_dropout=lora_cfg.get("lora_dropout", 0.05),
            bias=lora_cfg.get("bias", "none"),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    # GRPO config
    # Note: max_steps takes precedence over num_train_epochs
    grpo_cfg = config.get("grpo", {})
    log_cfg = config.get("logging", {})

    per_device_train_batch_size = grpo_cfg.get(
        "per_device_train_batch_size",
        config.get("training", {}).get("per_device_train_batch_size", 1),
    )
    num_generations = grpo_cfg.get("num_generations", 2)
    gradient_accumulation_steps = grpo_cfg.get("gradient_accumulation_steps", 1)

    # TRL's GRPOConfig requires:
    #   generation_batch_size = per_device_train_batch_size
    #                           * gradient_accumulation_steps
    #                           * num_processes
    # to be divisible by num_generations. We don't know num_processes here, but
    # per_device_train_batch_size * gradient_accumulation_steps must already be a
    # multiple of num_generations for any single-process or multi-process setup.
    effective = per_device_train_batch_size * gradient_accumulation_steps
    if effective % num_generations != 0:
        raise ValueError(
            f"GRPO config error: per_device_train_batch_size "
            f"({per_device_train_batch_size}) * gradient_accumulation_steps "
            f"({gradient_accumulation_steps}) = {effective}, which is not "
            f"divisible by num_generations ({num_generations}). "
            f"Increase per_device_train_batch_size or gradient_accumulation_steps "
            f"to a multiple of num_generations. Note: per_device_train_batch_size "
            f"in TRL counts TOTAL (prompt × generation) samples per device, so it "
            f"must be at least num_generations for a single unique prompt per step."
        )

    grpo_config = _make_grpo_config(GRPOConfig, grpo_cfg, log_cfg, output_dir, config)

    # Load reward config
    reward_cfg = config.get("reward", {})

    # Preprocess records: build prompts and ensure required fields
    from agent.prompts import build_messages
    import json

    processed_records = []
    for record in records:
        problem_statement = record.get("problem_statement", "")
        prompt_context = record.get("code_context", "")
        file_contents = record.get("file_contents", {})
        oracle_new_content = record.get("oracle_new_content", {})

        # Build prompt using same method as SFT
        messages = build_messages(problem_statement, prompt_context)
        prompt_str = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Create processed record with required fields for GRPOTrainer and reward function
        processed_record = {
            "prompt": prompt_str,
            "code_context": json.dumps(file_contents),
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
