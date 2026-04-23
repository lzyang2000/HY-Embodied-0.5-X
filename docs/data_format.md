# Data Format

HY-Embodied-0.5-X is trained on a single unified JSONL schema. Every line is
a standalone sample of the form:

```json
{
  "id": "optional-sample-id",
  "type": "optional-label-used-only-for-bookkeeping",
  "messages": [
    {
      "role": "system",                 
      "content": "optional system prompt"
    },
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "path/or/url"},
        {"type": "video", "video": "path/or/url"},
        {"type": "text",  "text": "User instruction /think"}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "<think>\n...\n</think>\n<answer>\n...\n</answer>"}
      ]
    }
  ]
}
```

## Fields

| Field | Required | Description |
|---|---|---|
| `messages` | yes | List of dialogue turns in chronological order. |
| `messages[].role` | yes | One of `system`, `user`, `assistant`, `tool`. |
| `messages[].content` | yes | Either a plain string, or a list of content items. |
| `id` | no | Opaque identifier used in logs. |
| `type` | no | Free-form tag (e.g. `direct_answer` / `cot` / `short_cot`). Ignored by the loader. |

### Content items

Inside a `user` message's `content` list, each item is one of:

| Item | Required fields | Notes |
|---|---|---|
| Image | `{"type": "image", "image": "<path>"}` | `path` may be a local path (absolute or repo-relative), an HTTP(s) URL, or a `data:image/...;base64,...` URI. |
| Video | `{"type": "video", "video": "<path>"}` | Same path semantics as image. |
| Text | `{"type": "text", "text": "..."}` | UTF-8. Preserve `/think` / `/no_think` suffixes when applicable. |

`assistant` messages currently only support a single `{"type": "text"}` item —
the chat template does not expand non-text assistant content.

## Think / no_think modes

HY-Embodied-0.5-X supports two interleaved inference modes:

* **Think mode.** The user instruction ends with ` /think`. The assistant
  output contains a non-empty reasoning block wrapped in `<think>...</think>`,
  followed by an `<answer>...</answer>` block with the final answer.
* **Direct answer.** The user instruction ends with ` /no_think`. The
  assistant output uses an empty think block `<think>\n\n</think>` followed
  by `<answer>...</answer>`.

The SFT chat template
(`hy_embodied.training.chat_template.build_sft_chat_template`) is **data
driven**: it does not auto-append the `/think` / `/no_think` suffix, and does
not wrap the assistant output with extra `<think>` / `<answer>` tags. Your
training data **must** already contain those markers exactly as above.

## Coordinate conventions

For grounding / pointing / trajectory tasks, coordinates are normalized to the
integer range `(0, 1000)` on both axes, relative to each image's own
resolution:

| Task | Typical format |
|---|---|
| Point | `(x, y)` or `[(x1, y1), ...]` |
| Bbox | `[xmin, ymin, xmax, ymax]` |
| Trajectory | `[<point>(x1, y1)</point>, <point>(x2, y2)</point>, ...]` |

## Assistant-only loss masking

During SFT, only tokens inside `<｜hy_Assistant｜> ... <eos>` spans contribute
to the loss. User / system / role-marker tokens have `label = -100`
(`IGNORE_INDEX`). The masking logic lives in
`hy_embodied.training.data.build_assistant_labels` and scans for the role
tokens at the tokenizer id level, so multi-turn conversations are correctly
supervised on every assistant turn.

## Size and budget controls

`DataConfig` exposes several knobs that act at the per-sample level, before
the Trainer ever sees the batch:

| Option | Meaning |
|---|---|
| `max_length` | Hard cap on tokens per sample; overlong samples are skipped (or raise). |
| `max_multimodal_tokens` | Optional cap on multimodal (image/video) placeholder tokens. |
| `image_min_pixels` / `image_max_pixels` | Passed through to the processor as `images_kwargs`. |
| `video_min_pixels` / `video_max_pixels` / `video_fps` / `max_frames` | Passed through to the video processor. |
| `skip_overlong_samples` / `skip_broken_samples` | Whether to silently skip and continue, or raise. |
| `pad_to_multiple_of` | Applied by the collator when padding to a batch. |

## Sampling ratios

`DataConfig.train_data_sampling_ratios` maps each JSONL path to a float:

* `ratio < 1` — downsample (keep each record with probability `ratio`).
* `ratio == 1` — keep everything.
* `ratio > 1` — upsample by repetition, supporting fractional parts
  (e.g. `1.42` ≈ keep all + 42 % extra copies).
* `ratio == 0` — skip the file entirely.

Ratios are applied during dataset construction using a deterministic
`random.Random(seed)` so training is reproducible across runs.
