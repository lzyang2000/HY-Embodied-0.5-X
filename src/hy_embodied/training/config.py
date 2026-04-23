"""Structured configuration for SFT training.

The config is a nested dataclass loaded from a YAML/JSON file.  Three sections:

* ``model``  : where to load weights from, precision, freezing options.
* ``data``   : data files, sampling ratios, length / pixel budgets.
* ``train``  : optimizer, scheduler, logging, distributed strategy.

See ``configs/sft/example_small.yaml`` for the single reference SFT config
(its training / optimizer defaults match the release recipe) and
``docs/training.md`` for a field-by-field reference.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _load_structured_file(path: str) -> dict[str, Any]:
    config_path = Path(path)
    suffix = config_path.suffix.lower()

    if suffix == ".json":
        return json.loads(config_path.read_text())

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Reading YAML config requires `pyyaml` to be installed.") from exc
        return yaml.safe_load(config_path.read_text())

    raise ValueError(f"Unsupported config format: {config_path}")


def _update_dataclass(instance: Any, values: dict[str, Any] | None) -> Any:
    if not values:
        return instance

    for key, value in values.items():
        if not hasattr(instance, key):
            raise ValueError(f"Unknown config key `{key}` for {type(instance).__name__}")
        setattr(instance, key, value)
    return instance


def _ensure_list(value: str | list[str]) -> list[str]:
    if isinstance(value, list):
        return value
    return [value]


@dataclass
class ModelConfig:
    """Where the backbone weights come from and how to load them."""

    model_name_or_path: str = "ckpts/HY-Embodied-0.5-X"
    torch_dtype: str = "bfloat16"
    attn_implementation: str | None = None
    # Kept for backwards compatibility. The training code always loads through
    # natively registered classes in transformers>=4.57 and will warn if set to
    # True, because local checkpoints don't ship the auto_map python files.
    trust_remote_code: bool = False
    freeze_vision_tower: bool = False
    freeze_language_model: bool = False
    gradient_checkpointing: bool = True


@dataclass
class DataConfig:
    """Training / evaluation data sources and shape controls."""

    train_data_paths: list[str] = field(default_factory=lambda: ["data_examples/data_demo.jsonl"])
    train_data_sampling_ratios: dict[str, float] = field(default_factory=dict)
    eval_data_paths: list[str] = field(default_factory=list)
    max_length: int = 8192
    max_multimodal_tokens: int | None = None
    skip_overlong_samples: bool = True
    skip_broken_samples: bool = True
    image_min_pixels: int | None = 28 * 28
    image_max_pixels: int | None = 1024 * 1024
    video_min_pixels: int | None = 28 * 28
    video_max_pixels: int | None = 512 * 512
    video_fps: float | None = 2.0
    max_frames: int | None = 32
    num_workers: int = 4
    pad_to_multiple_of: int | None = 8


@dataclass
class TrainConfig:
    """Optimizer / scheduler / distributed / logging settings."""

    output_dir: str = "outputs/hy-embodied-sft"
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_train_epochs: float = 1.0
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 2
    seed: int = 42
    bf16: bool = True
    fp16: bool = False
    tf32: bool = True
    report_to: list[str] = field(default_factory=lambda: ["tensorboard"])
    deepspeed: str | None = None
    fsdp: str | list[str] | None = None
    fsdp_config: dict[str, Any] | str | None = None
    ddp_find_unused_parameters: bool | None = False
    save_only_model: bool = False
    resume_from_checkpoint: str | None = None


@dataclass
class SFTConfig:
    """Root config: ``model`` + ``data`` + ``train``."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_file(cls, path: str) -> "SFTConfig":
        """Load and validate an SFT config from a YAML/JSON file."""
        raw = _load_structured_file(path)
        if raw is None:
            raw = {}

        config = cls()
        _update_dataclass(config.model, raw.get("model"))
        _update_dataclass(config.data, raw.get("data"))
        _update_dataclass(config.train, raw.get("train"))

        config.data.train_data_paths = _ensure_list(config.data.train_data_paths)
        config.data.train_data_sampling_ratios = dict(config.data.train_data_sampling_ratios or {})
        config.data.eval_data_paths = (
            _ensure_list(config.data.eval_data_paths) if config.data.eval_data_paths else []
        )
        config.train.report_to = _ensure_list(config.train.report_to)

        for data_path, ratio in config.data.train_data_sampling_ratios.items():
            ratio_f = float(ratio)
            if not math.isfinite(ratio_f) or ratio_f < 0.0:
                raise ValueError(
                    f"`data.train_data_sampling_ratios[{data_path}]` must be a finite value >= 0, "
                    f"got {ratio}."
                )
        return config

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
