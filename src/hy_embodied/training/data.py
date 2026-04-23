from __future__ import annotations

import json
import logging
import math
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from hy_embodied.training.config import DataConfig

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
USER_TOKEN_TEXT = "<｜hy_User｜>"
ASSISTANT_TOKEN_TEXT = "<｜hy_Assistant｜>"


class SampleEncodingError(RuntimeError):
    """Base class for sample encoding failures."""


class OverlongSampleError(SampleEncodingError):
    """Raised when a sample exceeds the configured token budget."""


class OversizedMultimodalSampleError(SampleEncodingError):
    """Raised when a sample exceeds the configured multimodal token budget."""


def _round_up_to_multiple(value: int, multiple: int | None) -> int:
    if multiple is None or multiple <= 1:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def _matches_subsequence(values: list[int], start: int, pattern: list[int]) -> bool:
    end = start + len(pattern)
    if end > len(values):
        return False
    return values[start:end] == pattern


def _resolve_media_path(path_or_url: str, source_path: str | None) -> str:
    if not isinstance(path_or_url, str):
        return path_or_url

    if path_or_url.startswith(("http://", "https://", "data:")):
        return path_or_url

    raw_path = Path(path_or_url)
    if raw_path.exists():
        return str(raw_path)

    candidates: list[Path] = []
    if source_path and not raw_path.is_absolute():
        candidates.append(Path(source_path).parent / raw_path)

    if path_or_url.startswith("/"):
        stripped = path_or_url.lstrip("/")
        # pass
        # 获取绝对路径

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return path_or_url


def build_assistant_labels(
    input_ids: list[int],
    assistant_token_ids: list[int],
    user_token_ids: list[int],
    eos_token_id: int,
    pad_token_id: int,
) -> list[int]:
    labels = [IGNORE_INDEX] * len(input_ids)
    predict_mode = False
    cursor = 0

    while cursor < len(input_ids):
        if _matches_subsequence(input_ids, cursor, assistant_token_ids):
            predict_mode = True
            cursor += len(assistant_token_ids)
            continue

        if _matches_subsequence(input_ids, cursor, user_token_ids):
            predict_mode = False
            cursor += len(user_token_ids)
            continue

        token_id = input_ids[cursor]
        if token_id == pad_token_id:
            predict_mode = False
        elif predict_mode:
            labels[cursor] = token_id
            if token_id == eos_token_id:
                predict_mode = False

        cursor += 1

    return labels


class LazySupervisedDataset(Dataset):
    def __init__(
        self,
        data_paths: list[str],
        processor,
        data_config: DataConfig,
        sampling_ratios: dict[str, float] | None = None,
        sampling_seed: int | None = None,
    ):
        self.processor = processor
        self.data_config = data_config
        self.samples: list[dict[str, Any]] = []
        self.failures = 0
        self.sampling_ratios = {str(Path(path)): float(ratio) for path, ratio in (sampling_ratios or {}).items()}
        self.sampling_seed = sampling_seed

        tokenizer = processor.tokenizer
        self.user_token_ids = tokenizer.encode(USER_TOKEN_TEXT, add_special_tokens=False)
        self.assistant_token_ids = tokenizer.encode(ASSISTANT_TOKEN_TEXT, add_special_tokens=False)
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.image_token_id = processor.image_token_id
        self.video_token_id = processor.video_token_id

        if not self.user_token_ids or not self.assistant_token_ids:
            raise ValueError("Failed to resolve conversation boundary tokens from tokenizer.")

        sampling_rng = random.Random(sampling_seed)
        for data_path in data_paths:
            path = Path(data_path)
            sampling_ratio = self.sampling_ratios.get(str(path), 1.0)
            valid_samples = 0
            kept_samples = 0
            with path.open("r") as f:
                for line_idx, line in enumerate(f, start=1):
                    if not line.strip():
                        continue

                    record = json.loads(line)
                    messages = record.get("messages")
                    if not isinstance(messages, list) or not messages:
                        logger.warning("Skip malformed sample at %s:%d", path, line_idx)
                        continue

                    valid_samples += 1
                    if sampling_ratio <= 0.0:
                        continue

                    whole_repeats = math.floor(sampling_ratio)
                    fractional_repeat = sampling_ratio - whole_repeats
                    keep_count = whole_repeats
                    if fractional_repeat > 0.0 and sampling_rng.random() < fractional_repeat:
                        keep_count += 1

                    if keep_count <= 0:
                        continue

                    for _ in range(keep_count):
                        sampled_record = dict(record)
                        sampled_record["_source_path"] = str(path)
                        self.samples.append(sampled_record)
                    kept_samples += keep_count

            logger.info(
                "Loaded %d sample copies from %d valid samples in %s with sampling_ratio=%.4f",
                kept_samples,
                valid_samples,
                path,
                sampling_ratio,
            )

        if not self.samples:
            raise ValueError("No valid training samples were found.")

        logger.info("Loaded %d samples from %d file(s).", len(self.samples), len(data_paths))

    def __len__(self) -> int:
        return len(self.samples)

    def _processor_kwargs(self) -> dict[str, Any]:
        images_kwargs = {}
        videos_kwargs = {}

        if self.data_config.image_min_pixels is not None:
            images_kwargs["min_pixels"] = self.data_config.image_min_pixels
        if self.data_config.image_max_pixels is not None:
            images_kwargs["max_pixels"] = self.data_config.image_max_pixels
        if self.data_config.video_min_pixels is not None:
            videos_kwargs["min_pixels"] = self.data_config.video_min_pixels
        if self.data_config.video_max_pixels is not None:
            videos_kwargs["max_pixels"] = self.data_config.video_max_pixels
        if self.data_config.video_fps is not None:
            videos_kwargs["fps"] = self.data_config.video_fps
        if self.data_config.max_frames is not None:
            videos_kwargs["num_frames"] = self.data_config.max_frames

        kwargs: dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": False,
            "return_dict": True,
            "return_tensors": "pt",
        }
        if images_kwargs:
            kwargs["images_kwargs"] = images_kwargs
        if videos_kwargs:
            kwargs["videos_kwargs"] = videos_kwargs
        return kwargs

    def _normalize_messages(self, sample: dict[str, Any]) -> list[dict[str, Any]]:
        normalized_messages: list[dict[str, Any]] = []
        source_path = sample.get("_source_path")

        for message in sample["messages"]:
            normalized_message = dict(message)
            content = message.get("content")

            if isinstance(content, list):
                normalized_content = []
                for item in content:
                    normalized_item = dict(item)
                    if "image" in normalized_item:
                        normalized_item["image"] = _resolve_media_path(normalized_item["image"], source_path)
                    if "image_url" in normalized_item:
                        normalized_item["image_url"] = _resolve_media_path(normalized_item["image_url"], source_path)
                    if "video" in normalized_item:
                        normalized_item["video"] = _resolve_media_path(normalized_item["video"], source_path)
                    normalized_content.append(normalized_item)
                normalized_message["content"] = normalized_content

            normalized_messages.append(normalized_message)

        return normalized_messages

    def _count_multimodal_tokens(self, input_ids: torch.Tensor) -> int:
        return int(((input_ids == self.image_token_id) | (input_ids == self.video_token_id)).sum().item())

    def _encode_sample(self, sample: dict[str, Any]) -> dict[str, torch.Tensor]:
        encoded = self.processor.apply_chat_template(
            self._normalize_messages(sample),
            **self._processor_kwargs(),
        )

        input_ids = encoded["input_ids"][0]
        attention_mask = encoded["attention_mask"][0]
        labels = torch.tensor(
            build_assistant_labels(
                input_ids=input_ids.tolist(),
                assistant_token_ids=self.assistant_token_ids,
                user_token_ids=self.user_token_ids,
                eos_token_id=self.eos_token_id,
                pad_token_id=self.pad_token_id,
            ),
            dtype=torch.long,
        )

        if (labels != IGNORE_INDEX).sum().item() == 0:
            raise ValueError(f"Sample `{sample.get('id', 'unknown')}` does not contain any assistant targets.")

        if input_ids.numel() > self.data_config.max_length:
            raise OverlongSampleError(
                f"Sample `{sample.get('id', 'unknown')}` exceeds max_length: "
                f"{input_ids.numel()} > {self.data_config.max_length}"
            )

        if self.data_config.max_multimodal_tokens is not None:
            multimodal_tokens = self._count_multimodal_tokens(input_ids)
            if multimodal_tokens > self.data_config.max_multimodal_tokens:
                raise OversizedMultimodalSampleError(
                    f"Sample `{sample.get('id', 'unknown')}` exceeds max_multimodal_tokens: "
                    f"{multimodal_tokens} > {self.data_config.max_multimodal_tokens}"
                )

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        optional_tensor_keys = (
            "pixel_values",
            "image_grid_thw",
            "pixel_values_videos",
            "video_grid_thw",
            "mm_token_type_ids",
        )
        for key in optional_tensor_keys:
            value = encoded.get(key)
            if value is not None:
                batch[key] = value

        return batch

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        max_attempts = min(max(64, self.data_config.num_workers * 8), len(self.samples))
        last_error: Exception | None = None

        for attempt in range(max_attempts):
            if attempt < 8:
                sample_index = (index + attempt) % len(self.samples)
            else:
                sample_index = random.randrange(len(self.samples))
            sample = self.samples[sample_index]
            try:
                return self._encode_sample(sample)
            except (ValueError, RuntimeError, OSError, KeyError, IndexError) as exc:
                last_error = exc
                self.failures += 1
                if not (self.data_config.skip_broken_samples or self.data_config.skip_overlong_samples):
                    raise

                is_overlong = isinstance(exc, (OverlongSampleError, OversizedMultimodalSampleError))
                if is_overlong and not self.data_config.skip_overlong_samples:
                    raise
                if (not is_overlong) and not self.data_config.skip_broken_samples:
                    raise

                logger.warning("Skip sample `%s`: %s", sample.get("id", "unknown"), exc)

        raise RuntimeError(f"Unable to fetch a valid sample after {max_attempts} attempts.") from last_error


class SFTDataCollator:
    def __init__(self, pad_token_id: int, pad_to_multiple_of: int | None = 8):
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def _pad_1d(self, tensor: torch.Tensor, target_len: int, pad_value: int) -> torch.Tensor:
        if tensor.size(0) == target_len:
            return tensor
        pad_len = target_len - tensor.size(0)
        padding = torch.full((pad_len,), pad_value, dtype=tensor.dtype)
        return torch.cat([tensor, padding], dim=0)

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_len = max(feature["input_ids"].size(0) for feature in features)
        max_len = _round_up_to_multiple(max_len, self.pad_to_multiple_of)

        batch = {
            "input_ids": torch.stack(
                [self._pad_1d(feature["input_ids"], max_len, self.pad_token_id) for feature in features]
            ),
            "attention_mask": torch.stack(
                [self._pad_1d(feature["attention_mask"], max_len, 0) for feature in features]
            ),
            "labels": torch.stack(
                [self._pad_1d(feature["labels"], max_len, IGNORE_INDEX) for feature in features]
            ),
        }

        if any("mm_token_type_ids" in feature for feature in features):
            batch["mm_token_type_ids"] = torch.stack(
                [
                    self._pad_1d(
                        feature["mm_token_type_ids"][0] if "mm_token_type_ids" in feature else torch.zeros_like(feature["input_ids"]),
                        max_len,
                        0,
                    )
                    for feature in features
                ]
            )

        for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
            values = [feature[key] for feature in features if key in feature]
            if values:
                batch[key] = torch.cat(values, dim=0)

        return batch
