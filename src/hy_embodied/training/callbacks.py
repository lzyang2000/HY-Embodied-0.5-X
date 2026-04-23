"""Trainer callbacks for HY-Embodied SFT.

* :class:`SaveInferenceArtifactsCallback` — copies processor files + chat
  template + restores ``model_type`` / ``auto_map`` into every saved
  ``checkpoint-*`` directory so each checkpoint is immediately usable for
  inference.
* :class:`ProgressLoggingCallback` — concise, ETA-aware progress logs (on the
  local rank-0 process only).
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path

from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

logger = logging.getLogger(__name__)


def _restore_config_fields(source_model_path: str | Path, target_dir: Path) -> None:
    """Copy ``model_type`` and ``auto_map`` from source config into a saved config.

    ``Trainer.save_model()`` persists ``model.config`` which for this family of
    models may serialize with ``model_type="hunyuan_v1_dense"`` even for MoT
    variants.  This helper restores the original values so that downstream
    inference via ``AutoModel`` picks the right class.
    """
    src_cfg_path = Path(source_model_path) / "config.json"
    tgt_cfg_path = target_dir / "config.json"
    if not src_cfg_path.exists() or not tgt_cfg_path.exists():
        return

    with open(src_cfg_path) as f:
        src_cfg = json.load(f)
    with open(tgt_cfg_path) as f:
        tgt_cfg = json.load(f)

    changed = False
    for key in ("model_type", "auto_map"):
        if key in src_cfg and tgt_cfg.get(key) != src_cfg[key]:
            tgt_cfg[key] = src_cfg[key]
            changed = True

    if changed:
        with open(tgt_cfg_path, "w") as f:
            json.dump(tgt_cfg, f, indent=2, ensure_ascii=False)
        logger.info("Restored model_type/auto_map in %s from source model.", tgt_cfg_path)


def save_inference_artifacts(processor, source_model_path: str | Path, target_dir: str | Path) -> None:
    """Persist processor + chat templates + restored config into ``target_dir``."""
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    processor.save_pretrained(target_path)

    for filename in ("chat_template.jinja", "chat_template.json"):
        src = Path(source_model_path) / filename
        if src.exists():
            shutil.copy2(src, target_path / filename)

    _restore_config_fields(source_model_path, target_path)


class SaveInferenceArtifactsCallback(TrainerCallback):
    """Mirror inference-ready artifacts into each saved checkpoint."""

    def __init__(self, processor, source_model_path: str | Path) -> None:
        self.processor = processor
        self.source_model_path = str(source_model_path)

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if checkpoint_dir.exists():
            save_inference_artifacts(self.processor, self.source_model_path, checkpoint_dir)
        return control


class ProgressLoggingCallback(TrainerCallback):
    """Log training progress with wall-clock elapsed / ETA info."""

    def __init__(self) -> None:
        self.train_start_time: float | None = None
        self.last_logged_step = -1

    @staticmethod
    def _format_seconds(seconds: float | None) -> str:
        if seconds is None:
            return "unknown"
        total_seconds = max(0, int(seconds))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        self.train_start_time = time.time()
        self.last_logged_step = -1
        if state.is_local_process_zero:
            logger.info(
                "Training started: max_steps=%s, num_train_epochs=%s, logging_steps=%s",
                state.max_steps,
                args.num_train_epochs,
                args.logging_steps,
            )
        return control

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, float] | None = None,
        **kwargs,
    ) -> TrainerControl:
        if not state.is_local_process_zero or not logs:
            return control
        if state.global_step <= 0 or state.global_step == self.last_logged_step:
            return control

        self.last_logged_step = state.global_step
        elapsed = None if self.train_start_time is None else time.time() - self.train_start_time
        eta_seconds: float | None = None
        progress_pct = 0.0
        if elapsed is not None and state.max_steps and state.global_step > 0:
            progress_pct = min(100.0, 100.0 * state.global_step / state.max_steps)
            remaining_steps = max(0, state.max_steps - state.global_step)
            eta_seconds = elapsed / state.global_step * remaining_steps

        metric_parts: list[str] = []
        for key in ("loss", "grad_norm", "learning_rate", "epoch"):
            value = logs.get(key)
            if value is None:
                continue
            if isinstance(value, float):
                metric_parts.append(f"{key}={value:.6g}")
            else:
                metric_parts.append(f"{key}={value}")

        logger.info(
            "Train progress: step=%s/%s (%.2f%%), elapsed=%s, eta=%s%s",
            state.global_step,
            state.max_steps,
            progress_pct,
            self._format_seconds(elapsed),
            self._format_seconds(eta_seconds),
            f", {', '.join(metric_parts)}" if metric_parts else "",
        )
        return control

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        if state.is_local_process_zero:
            elapsed = None if self.train_start_time is None else time.time() - self.train_start_time
            logger.info(
                "Training ended: step=%s/%s, elapsed=%s",
                state.global_step,
                state.max_steps,
                self._format_seconds(elapsed),
            )
        return control


__all__ = [
    "ProgressLoggingCallback",
    "SaveInferenceArtifactsCallback",
    "save_inference_artifacts",
]
