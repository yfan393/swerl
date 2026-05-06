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
        # Try both TRL parameter names for max generation length. Different
        # TRL versions expose this as max_completion_length OR max_new_tokens
        # OR neither (defaulting to 256). The build-then-fallback loop below
        # will retry without unsupported keys.
        "max_completion_length": "max_new_tokens",
        "max_new_tokens": "max_new_tokens",
        "temperature": "generation_temperature",
        "optim": "optimizer",
        # Sampling-stability params: filtering the distribution before
        # multinomial sampling avoids the
        # "probability tensor contains either inf, nan or element < 0" CUDA assert
        # that fires when raw softmax of an unfiltered ~150k-token vocab
        # produces a numerical spike.
        "top_p": "top_p",
        "top_k": "top_k",
        "min_p": "min_p",
        "repetition_penalty": "repetition_penalty",
    }
    for arg_name, cfg_name in optional_mappings.items():
        if cfg_name in grpo_cfg and _supports_parameter(GRPOConfig, arg_name):
            kwargs[arg_name] = grpo_cfg[cfg_name]

    # If neither max_completion_length nor max_new_tokens was accepted by
    # the inspect.signature gate, fall back to setting the most likely TRL
    # name unconditionally. We'll catch a TypeError and strip it on retry.
    if (
        "max_new_tokens" in grpo_cfg
        and "max_completion_length" not in kwargs
        and "max_new_tokens" not in kwargs
    ):
        kwargs["max_completion_length"] = grpo_cfg["max_new_tokens"]

    logger.info(
        "Constructing GRPOConfig with: "
        + ", ".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
    )

    # Build with retry: if a kwarg isn't supported by this TRL version,
    # strip it and try again. This keeps the function tolerant of TRL
    # API drift between versions without losing the params we need.
    while True:
        try:
            return GRPOConfig(**kwargs)
        except TypeError as exc:
            msg = str(exc)
            removed = None
            for key in list(kwargs.keys()):
                if key in msg and "unexpected keyword argument" in msg:
                    kwargs.pop(key)
                    removed = key
                    break
            if removed is None:
                raise
            logger.warning(
                "GRPOConfig does not accept %r in this TRL version; dropping and retrying.",
                removed,
            )


def _select_model_dtype(torch, model_cfg: dict, grpo_cfg: dict):
    requested = grpo_cfg.get("torch_dtype", model_cfg.get("torch_dtype", "float16"))
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(requested, torch.float16)

    if grpo_cfg.get("force_model_dtype"):
        return torch_dtype
    if model_cfg.get("load_in_4bit", False):
        return torch_dtype
    if torch_dtype is torch.float16:
        logger.info(
            "Using float32 for non-quantized GRPO generation instead of requested "
            "float16. Set grpo.force_model_dtype=true to override."
        )
        return torch.float32
    return torch_dtype


def _stabilize_generation_config(model, tokenizer, grpo_cfg: dict | None = None) -> None:
    generation_config = getattr(model, "generation_config", None)
    if generation_config is None:
        return
    generation_config.pad_token_id = tokenizer.pad_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id
    # remove_invalid_values clamps NaN/Inf in logits before softmax. Without
    # this, a single rogue activation can poison the entire probability tensor
    # and trigger the "probability tensor contains either inf, nan or element < 0"
    # CUDA assert inside torch.multinomial.
    generation_config.remove_invalid_values = True
    # Mirror sampling-stability params from the user's grpo config onto the
    # model's own generation_config, so any code path that doesn't go through
    # GRPOConfig (e.g. fallback HF generate) still gets a filtered distribution.
    grpo_cfg = grpo_cfg or {}
    if "top_p" in grpo_cfg:
        generation_config.top_p = grpo_cfg["top_p"]
    if "top_k" in grpo_cfg:
        generation_config.top_k = grpo_cfg["top_k"]
    if "min_p" in grpo_cfg:
        generation_config.min_p = grpo_cfg["min_p"]
    if "repetition_penalty" in grpo_cfg:
        generation_config.repetition_penalty = grpo_cfg["repetition_penalty"]
    # Clamp temperature to a strictly-positive value; T=0 with do_sample=True
    # is another well-known cause of NaN probability tensors.
    temp = grpo_cfg.get("generation_temperature", getattr(generation_config, "temperature", 1.0))
    generation_config.temperature = max(float(temp), 1e-3)
    generation_config.do_sample = True


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
    grpo_cfg = config.get("grpo", {})
    log_cfg = config.get("logging", {})

    # Load training data
    records = read_jsonl(train_file)
    if not records:
        logger.error(f"No training data found in {train_file}")
        return

    logger.info(f"Loaded {len(records)} training records")

    # Load model and tokenizer
    torch_dtype = _select_model_dtype(torch, model_cfg, grpo_cfg)
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
    _stabilize_generation_config(model, tokenizer, grpo_cfg)

    if model_cfg.get("load_in_4bit", False):
        model = prepare_model_for_kbit_training(model)

    # Apply LoRA
    sft_final_dir = Path(config.get("sft_baseline", {}).get("output_dir", "")) / "final"
    init_from_sft = config.get("grpo", {}).get("init_from_sft", True)
    sft_adapter_present = (sft_final_dir / "adapter_config.json").exists()
    if init_from_sft and sft_adapter_present:
        logger.info("Initializing GRPO from SFT adapter: %s", sft_final_dir)
        model = PeftModel.from_pretrained(model, str(sft_final_dir), is_trainable=True)
    elif model_cfg.get("use_lora", True):
        if init_from_sft and not sft_adapter_present:
            logger.warning(
                "=" * 70 + "\n"
                "GRPO is starting from the BASE model with a fresh LoRA adapter.\n"
                "No SFT adapter was found at: %s\n"
                "\n"
                "Without an SFT-initialized policy that already knows the SWE-RL\n"
                "output schema (<think>...</think><solution>...```search/replace```\n"
                "</solution>), GRPO is very likely to produce only the format\n"
                "penalty (-1.0) for every rollout, leading to zero advantage,\n"
                "zero loss, and NaN gradients.\n"
                "\n"
                "Recommended: run the SFT baseline first:\n"
                "    python run.py sft_train --config %s\n"
                "then re-run GRPO. Or set grpo.init_from_sft=false to silence this.\n"
                + "=" * 70,
                sft_final_dir,
                config_path,
            )
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

    # Persist the final policy and tokenizer to <output_dir>/final/ so that
    # downstream commands (evaluation, the GRPO->SFT init_from_sft branch on
    # subsequent runs, etc.) can pick it up the same way they pick up the
    # SFT adapter. Without this, only the periodic checkpoint-N folders
    # exist and there is no canonical "final" artifact.
    final_dir = Path(output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving GRPO model to {final_dir}")
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info(f"Training complete. Final model saved to {final_dir}")
