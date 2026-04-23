"""Trainer assembly for HY-Embodied-0.5-X SFT.

This module exposes a single high-level entry point, :func:`run_sft`, that
wires together config loading, model + processor loading, dataset
construction, the HuggingFace :class:`~transformers.Trainer`, and our two
custom callbacks. ``cli/train.py`` is a thin shell around it.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import torch
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    set_seed,
)

from hy_embodied.training.callbacks import (
    ProgressLoggingCallback,
    SaveInferenceArtifactsCallback,
    save_inference_artifacts,
)
from hy_embodied.training.chat_template import build_sft_chat_template
from hy_embodied.training.config import SFTConfig
from hy_embodied.training.data import LazySupervisedDataset, SFTDataCollator

logger = logging.getLogger(__name__)


def detect_model_variant(model_path: str | Path) -> str:
    """Return ``'mot'`` or ``'dense'`` based on the ckpt's ``config.json``.

    With ``transformers>=4.57`` both variants are registered natively and model
    loading always goes through :class:`AutoModelForImageTextToText`. This
    helper exists only for logging and sanity checking.
    """
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_path}")
    with open(config_path) as f:
        cfg = json.load(f)
    model_type = cfg.get("model_type", "")
    if model_type == "hunyuan_vl_mot":
        return "mot"
    if model_type == "hunyuan_v1_dense":
        return "dense"
    raise ValueError(f"Unknown model_type '{model_type}' in {config_path}")


def _resolve_torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return mapping[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported torch dtype: {name}") from exc


def load_model_and_processor(config: SFTConfig):
    """Load the HY-Embodied processor + model via native transformers Auto classes.

    Requires ``transformers>=4.57`` where both ``hunyuan_vl_mot`` and
    ``hunyuan_v1_dense`` are registered natively.  We force
    ``trust_remote_code=False`` regardless of the config setting to avoid
    falling back to the ``auto_map`` entries in ``config.json`` — those point
    to python files that only exist on the HF Hub upload, not in a local ckpt
    directory.
    """
    model_path = config.model.model_name_or_path
    variant = detect_model_variant(model_path)
    logger.info("Detected model variant: %s (path: %s)", variant, model_path)

    if config.model.trust_remote_code:
        logger.warning(
            "Ignoring model.trust_remote_code=true: HY-Embodied is natively supported "
            "by the installed transformers; remote code would require "
            "configuration_hunyuan_vl_mot.py / modeling_hunyuan_vl_mot.py / "
            "processing_hunyuan_vl_mot.py alongside the ckpt, which are not shipped "
            "with local checkpoints."
        )

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=False)
    processor.chat_template = build_sft_chat_template()
    logger.info("Using SFT chat template (data-driven /think, <think>, <answer> passthrough)")

    torch_dtype = _resolve_torch_dtype(config.model.torch_dtype)

    model_kwargs: dict = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": False,
    }
    if config.model.attn_implementation:
        model_kwargs["attn_implementation"] = config.model.attn_implementation

    model = AutoModelForImageTextToText.from_pretrained(model_path, **model_kwargs)

    if config.model.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    return model, processor


def freeze_modules(model, config: SFTConfig) -> None:
    """Freeze visual tower and/or language model according to the config."""
    frozen_names: list[str] = []

    if config.model.freeze_vision_tower:
        for name, param in model.named_parameters():
            if name.startswith("model.visual"):
                param.requires_grad = False
                frozen_names.append(name)

    if config.model.freeze_language_model:
        for name, param in model.named_parameters():
            if name.startswith("model.language_model"):
                param.requires_grad = False
                frozen_names.append(name)

    logger.info("Frozen %d parameter tensors.", len(frozen_names))


def log_trainable_parameters(model) -> None:
    total = 0
    trainable = 0
    for param in model.parameters():
        numel = param.numel()
        total += numel
        if param.requires_grad:
            trainable += numel

    ratio = 100 * trainable / total if total else 0.0
    logger.info(
        "Trainable params: %s / %s (%.2f%%)",
        f"{trainable:,}",
        f"{total:,}",
        ratio,
    )


def build_datasets(processor, config: SFTConfig):
    train_dataset = LazySupervisedDataset(
        data_paths=config.data.train_data_paths,
        processor=processor,
        data_config=config.data,
        sampling_ratios=config.data.train_data_sampling_ratios,
        sampling_seed=config.train.seed,
    )

    eval_dataset = None
    if config.data.eval_data_paths:
        eval_dataset = LazySupervisedDataset(
            data_paths=config.data.eval_data_paths,
            processor=processor,
            data_config=config.data,
        )

    return train_dataset, eval_dataset


def build_training_arguments(config: SFTConfig, has_eval: bool) -> TrainingArguments:
    train_cfg = config.train

    if train_cfg.deepspeed and train_cfg.fsdp:
        raise ValueError("`train.deepspeed` and `train.fsdp` cannot be enabled at the same time.")

    os.makedirs(train_cfg.output_dir, exist_ok=True)
    with open(Path(train_cfg.output_dir) / "resolved_sft_config.json", "w") as f:
        json.dump(config.to_dict(), f, ensure_ascii=False, indent=2)

    # Newer transformers raises TypeError when fsdp is None (checks membership
    # on it), so fall back to "" when not set.
    fsdp_value = train_cfg.fsdp if train_cfg.fsdp else ""
    fsdp_config_value = train_cfg.fsdp_config if train_cfg.fsdp_config else None

    return TrainingArguments(
        output_dir=train_cfg.output_dir,
        per_device_train_batch_size=train_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=train_cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        learning_rate=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
        num_train_epochs=train_cfg.num_train_epochs,
        warmup_ratio=train_cfg.warmup_ratio,
        lr_scheduler_type=train_cfg.lr_scheduler_type,
        logging_steps=train_cfg.logging_steps,
        save_steps=train_cfg.save_steps,
        eval_steps=train_cfg.eval_steps if has_eval else None,
        save_total_limit=train_cfg.save_total_limit,
        bf16=train_cfg.bf16,
        fp16=train_cfg.fp16,
        tf32=train_cfg.tf32,
        report_to=train_cfg.report_to,
        deepspeed=train_cfg.deepspeed,
        fsdp=fsdp_value,
        fsdp_config=fsdp_config_value,
        remove_unused_columns=False,
        dataloader_num_workers=config.data.num_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=config.data.num_workers > 0,
        logging_first_step=True,
        save_safetensors=True,
        save_only_model=train_cfg.save_only_model,
        eval_strategy="steps" if has_eval else "no",
        do_eval=has_eval,
        do_train=True,
        gradient_checkpointing=config.model.gradient_checkpointing,
        ddp_find_unused_parameters=train_cfg.ddp_find_unused_parameters,
        resume_from_checkpoint=train_cfg.resume_from_checkpoint,
    )


def run_sft(config: SFTConfig) -> None:
    """Run a full SFT training job described by ``config``."""
    set_seed(config.train.seed)

    model, processor = load_model_and_processor(config)
    freeze_modules(model, config)
    log_trainable_parameters(model)
    save_inference_artifacts(processor, config.model.model_name_or_path, config.train.output_dir)

    train_dataset, eval_dataset = build_datasets(processor, config)
    data_collator = SFTDataCollator(
        pad_token_id=processor.tokenizer.pad_token_id,
        pad_to_multiple_of=config.data.pad_to_multiple_of,
    )
    training_args = build_training_arguments(config, has_eval=eval_dataset is not None)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[
            SaveInferenceArtifactsCallback(processor, config.model.model_name_or_path),
            ProgressLoggingCallback(),
        ],
    )

    trainer.train(resume_from_checkpoint=config.train.resume_from_checkpoint)
    trainer.save_model()
    save_inference_artifacts(processor, config.model.model_name_or_path, config.train.output_dir)


__all__ = [
    "build_datasets",
    "build_training_arguments",
    "detect_model_variant",
    "freeze_modules",
    "load_model_and_processor",
    "log_trainable_parameters",
    "run_sft",
]
