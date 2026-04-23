# Training

HY-Embodied-0.5-X ships a complete, minimal SFT pipeline built on
HuggingFace `Trainer` plus DeepSpeed ZeRO-2 / FSDP. This page covers how to
run training, how configs work, and what gets saved.

## TL;DR

```bash
# 1. prepare environment (once)
bash setup_env.sh
conda activate hy_embodied_x

# 2. fetch weights (once)
huggingface-cli download tencent/HY-Embodied-0.5-X --local-dir ckpts/HY-Embodied-0.5-X

# 3. single GPU smoke test (no torchrun needed)
CUDA_VISIBLE_DEVICES=0 python -m hy_embodied.cli.train \
    --config configs/sft/example_small_single_gpu.yaml

# 4. or launch the 1-node 8-GPU training script (DeepSpeed ZeRO-2)
bash scripts/run_sft_1node_8gpu.sh

# 5. multi-node
bash scripts/run_sft_4node_8gpu.sh
```

## Entry points

Three equivalent ways to launch training:

| Style | Command |
|---|---|
| Python CLI | `python -m hy_embodied.cli.train --config <cfg.yaml>` |
| Console script (after `pip install`) | `hy-embodied-train --config <cfg.yaml>` |
| Library API | `from hy_embodied.training import SFTConfig, run_sft; run_sft(SFTConfig.from_file(path))` |

## Configs

Two reference configs are shipped with the repo:

| File | Purpose |
|---|---|
| `configs/sft/example_small_single_gpu.yaml` | **Single-GPU config** with DeepSpeed disabled. Can be launched with plain `python -m` (no `torchrun` required). Best for quick validation and debugging. |
| `configs/sft/example_small.yaml` | **Multi-GPU config** with DeepSpeed ZeRO-2 enabled. Must be launched via `torchrun` or `accelerate`. Its training / optimizer defaults match the release recipe; the only thing users are expected to edit for real training is the `data` section (`train_data_paths` + `train_data_sampling_ratios`). |

Both configs train on `data_examples/data_demo.jsonl` out of the box, so
the default commands run end-to-end with no external data.

Config files are plain YAML and map 1:1 onto three nested dataclasses:

```
SFTConfig
├── model:  ModelConfig  # where weights come from, freezing, dtype
├── data:   DataConfig   # jsonl paths, sampling, length budgets
└── train:  TrainConfig  # optimizer, scheduler, distributed strategy
```

See `src/hy_embodied/training/config.py` for the authoritative field list.

## Config reference

Every YAML key listed below maps directly to a dataclass field in
`src/hy_embodied/training/config.py`. Values shown are the defaults used in
`configs/sft/example_small.yaml` (i.e. the release recipe); unless noted,
you should only need to touch the `data` section.

### `model.*`

| Field | Default | What it does |
|---|---|---|
| `model_name_or_path` | `ckpts/HY-Embodied-0.5-X` | Local checkpoint dir or HF Hub repo id. Accepts either a local path produced by `huggingface-cli download` or a string like `tencent/HY-Embodied-0.5-X` (downloaded on demand). |
| `torch_dtype` | `bfloat16` | Compute dtype for model parameters. Use `float16` only on GPUs without bf16 support (e.g. V100). |
| `attn_implementation` | `null` | Forwarded to `from_pretrained`. `null` auto-picks FA2 when `flash_attn` is installed. Override to `sdpa` / `eager` for debugging. |
| `trust_remote_code` | `false` | Kept for compatibility. The trainer force-disables it regardless, because local checkpoints don't ship the `auto_map` python files. |
| `gradient_checkpointing` | `true` | Recompute activations on the backward pass. Trades ~20% throughput for ~3x memory; strongly recommended for this 4B model. |
| `freeze_vision_tower` | `false` | If true, freeze all parameters under `model.visual` (the SigLIP tower). |
| `freeze_language_model` | `false` | If true, freeze all parameters under `model.language_model`. Combine with `freeze_vision_tower=false` to train only the vision side. |

### `data.*`

| Field | Default | What it does |
|---|---|---|
| `train_data_paths` | `["data_examples/data_demo.jsonl"]` | **Edit this.** List of JSONL files (schema in `docs/data_format.md`). Paths are relative to the repo root unless absolute. |
| `train_data_sampling_ratios` | `{}` | **Edit this.** Per-file resampling ratio: `<1` downsamples, `=1` keeps all (default if unset), `>1` upsamples via repetition, `=0` skips the file. |
| `eval_data_paths` | `[]` | Optional JSONL files for `trainer.evaluate`. Same schema as training data. |
| `max_length` | `15000` | Max tokens per sample after tokenization. Samples exceeding this are either skipped (see below) or raise. |
| `max_multimodal_tokens` | `10000` | Upper bound on vision/video tokens per sample. `null` disables the cap. |
| `skip_overlong_samples` | `true` | Skip (with a WARNING) samples longer than `max_length` / `max_multimodal_tokens`. If false, overlong samples abort the run. |
| `skip_broken_samples` | `true` | Skip (with a WARNING) samples whose image/video fails to load (missing path, corrupted file, …). |
| `image_min_pixels` | `784` | Lower bound (in pixels) for image resize; processor upsamples below this. |
| `image_max_pixels` | `1048576` | Upper bound (in pixels) for image resize; processor downsamples above this. Larger = finer detail but more vision tokens. |
| `video_min_pixels` / `video_max_pixels` | `784` / `262144` | Same as above but applied per video frame. |
| `video_fps` | `2.0` | Decode videos at this frame rate before sampling. |
| `max_frames` | `32` | Cap on the number of frames per clip after `video_fps` sampling. |
| `num_workers` | `4` | DataLoader workers per rank. Raise for slow storage, set to 0 when debugging. |
| `pad_to_multiple_of` | `8` | Pad sequences to a multiple of this value; 8 keeps tensor cores happy under bf16. |

### `train.*`

| Field | Default | What it does |
|---|---|---|
| `output_dir` | `outputs/hy-embodied-sft` | Where checkpoints, logs, and `resolved_sft_config.json` go. |
| `per_device_train_batch_size` | `1` | Micro-batch per GPU. Keep at 1 for long multimodal sequences. |
| `per_device_eval_batch_size` | `1` | Micro-batch per GPU for `trainer.evaluate`. |
| `gradient_accumulation_steps` | `2` | Effective global batch = this × `per_device_train_batch_size` × world_size. On 1 node × 8 GPUs → 16; on 4 nodes × 8 GPUs (release run) → 64. For a quick smoke test on tiny data you can lower this to 1 so at least one optimizer step fires. |
| `learning_rate` | `2.0e-5` | AdamW peak LR. |
| `weight_decay` | `0.01` | AdamW weight decay. |
| `num_train_epochs` | `1` | Passes over the training corpus. |
| `warmup_ratio` | `0.03` | Fraction of total steps used for linear warmup. |
| `lr_scheduler_type` | `cosine` | Decay shape; `cosine` / `linear` / `constant` / `constant_with_warmup` are all supported. |
| `logging_steps` | `10` | Print / push metrics every N optimizer steps. |
| `save_steps` | `1000` | Save a checkpoint every N optimizer steps. |
| `eval_steps` | `1000` | Run evaluation every N steps (only if `eval_data_paths` is set). |
| `save_total_limit` | `3` | Keep the latest N checkpoints on disk; older ones are pruned. |
| `seed` | `42` | Global RNG seed (data shuffle, dropout, init). |
| `bf16` / `fp16` | `true` / `false` | Mixed-precision training flavor. Must agree with `model.torch_dtype`. |
| `tf32` | `true` | Allow TF32 matmul on Ampere+; ~20% speedup at no accuracy cost. |
| `report_to` | `[tensorboard]` | Metric backends. Use `[]` to disable, add `"wandb"` if configured. |
| `deepspeed` | `configs/deepspeed/zero2.json` | ZeRO-2 config path. Set to `null` for plain DDP, or switch to FSDP by setting `fsdp` + `fsdp_config`. Mutually exclusive with `fsdp`. |
| `fsdp` | `null` | e.g. `"full_shard auto_wrap"` to enable FSDP. |
| `fsdp_config` | `null` | Path to an FSDP YAML (e.g. `configs/fsdp/full_shard_auto_wrap.yaml`). |
| `ddp_find_unused_parameters` | `false` | Set to `true` only if some module is conditionally unused in forward. |
| `save_only_model` | `false` | If true, skip optimizer / scheduler / RNG state. Smaller checkpoints but can't resume mid-epoch. |
| `resume_from_checkpoint` | `null` | Path to a previous `checkpoint-*` directory to resume from. |

### Model loading

`transformers >= 4.57` ships native support for the `hunyuan_vl_mot` and
`hunyuan_v1_dense` model types. The training code:

1. Reads `config.json` in the checkpoint directory and logs the detected
   variant.
2. Always loads via `AutoModelForImageTextToText.from_pretrained(...,
   trust_remote_code=False)` — even if the config sets
   `trust_remote_code: true`, we force it off and warn, because local
   checkpoints do not ship the `auto_map` python files that the HF Hub
   upload references.

### Chat template

The training chat template differs from the inference one: training data
already contains `/think` / `/no_think` suffixes and `<think>` / `<answer>`
wrappers, so the SFT template passes them through verbatim rather than
re-injecting them. See `docs/data_format.md` and
`hy_embodied.training.chat_template`.

## Distributed strategies

* **DDP**: default, no `deepspeed` / `fsdp` set.
* **DeepSpeed ZeRO-2**: set `train.deepspeed: configs/deepspeed/zero2.json`.
  Most cost-effective for the 4B-parameter model family.
* **FSDP**: set `train.fsdp: full_shard auto_wrap` and
  `train.fsdp_config: configs/fsdp/full_shard_auto_wrap.yaml`.
* The two are mutually exclusive — the trainer raises if both are set.

### Accelerate launcher

Accelerate configs under `configs/accelerate/*.yaml` cover the common
single-node / multi-node combinations. Switch the launcher via:

```bash
LAUNCHER=accelerate \
ACCELERATE_CONFIG_FILE=configs/accelerate/sft_1node_8gpu_zero2.yaml \
bash scripts/run_sft_1node_8gpu.sh
```

## Outputs

Inside `train.output_dir` you get, after each checkpoint save:

```
<output_dir>/
├── resolved_sft_config.json           # the final parsed config (for reproducibility)
├── checkpoint-100/
│   ├── model-00001-of-000NN.safetensors
│   ├── ...
│   ├── config.json                    # model_type / auto_map restored from source
│   ├── chat_template.jinja            # inference-ready template
│   ├── preprocessor_config.json
│   └── tokenizer.json
└── checkpoint-200/
    └── ...
```

Each checkpoint directory is **immediately usable for inference** — feed it
to `HyEmbodiedPipeline.from_pretrained()` or
`python -m hy_embodied.cli.infer --model <output_dir>/checkpoint-200 ...`.

## Freezing modules

`ModelConfig.freeze_vision_tower` / `freeze_language_model` freeze parameters
by prefix:

* `model.visual` — the SigLIP-based vision tower.
* `model.language_model` — the LM backbone.

Freezing is applied after model load; trainable parameter counts are logged.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `OSError: ... does not appear to have a file named configuration_hunyuan_vl_mot.py` | `trust_remote_code=True` caused transformers to look for `auto_map` python files not shipped with the local ckpt. | Set `model.trust_remote_code: false` (default). The trainer also force-disables it. |
| `FileNotFoundError: config.json not found in <path>` | Wrong `model_name_or_path`, or the ckpt wasn't fully downloaded. | Re-run `huggingface-cli download tencent/HY-Embodied-0.5-X --local-dir ckpts/HY-Embodied-0.5-X`. |
| Out-of-memory with ZeRO-2 on short batches | `gradient_checkpointing` disabled, or too-large `image_max_pixels`. | Enable `gradient_checkpointing`, reduce `image_max_pixels`, or switch to FSDP. |
| Every sample warns "Skip sample ... exceeds max_length" | `max_length` is too small for your data. | Increase `data.max_length` or shorten prompts. |
