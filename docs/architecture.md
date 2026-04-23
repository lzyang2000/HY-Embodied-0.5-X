# Repository architecture

The repository is organized as a standard Python project with a source
package under `src/hy_embodied/` and data / configs / scripts at the top
level:

```
HY-Embodied-0.5-X/
├── README.md
├── LICENSE
├── pyproject.toml                    # packaging metadata, console scripts
├── requirements.txt                  # pinned training / inference deps
├── setup_env.sh                      # one-click conda + pip setup
├── .gitignore
│
├── src/hy_embodied/                  # Python package (importable after `pip install`)
│   ├── __init__.py
│   ├── cli/                          # command-line entry points
│   │   ├── train.py                  # `python -m hy_embodied.cli.train`
│   │   └── infer.py                  # `python -m hy_embodied.cli.infer`
│   ├── training/                     # SFT trainer + data pipeline
│   │   ├── config.py                 # dataclasses (ModelConfig/DataConfig/TrainConfig)
│   │   ├── chat_template.py          # SFT chat template (data-driven)
│   │   ├── data.py                   # LazySupervisedDataset + SFTDataCollator
│   │   ├── callbacks.py              # SaveInferenceArtifactsCallback, ProgressLoggingCallback
│   │   └── trainer.py                # model/processor loading + run_sft entry
│   └── inference/
│       └── pipeline.py               # HyEmbodiedPipeline + GenerationConfig
│
├── configs/
│   ├── sft/
│   │   └── example_small.yaml        # reference SFT config (uses data_examples/ by default)
│   ├── accelerate/                   # accelerate launcher configs
│   ├── deepspeed/                    # ZeRO configs
│   └── fsdp/                         # FSDP configs
│
├── scripts/
│   ├── run_sft_1node_8gpu.sh
│   └── run_sft_4node_8gpu.sh
│
├── data_examples/                    # small JSONL samples per capability (see README)
│   ├── README.md
│   └── *.jsonl
│
├── docs/
│   ├── architecture.md               # this file
│   ├── data_format.md
│   ├── training.md
│   └── inference.md
│
├── assets/                           # images used in README / demos
├── ckpts/                            # (gitignored) `huggingface-cli download` target
├── outputs/                          # (gitignored) training runs
│
└── inference.py                      # backward-compat shim for `python inference.py`
```

## Import rules

* User-facing code goes in `src/hy_embodied/`.
* Never import from `configs/`, `scripts/`, `data_examples/`, or `docs/`.
* `training/` must not import from `inference/` and vice versa — the two
  subsystems are independent modules that share only the chat template.
* `cli/*.py` files are **entry points only**; they parse args, set up
  logging, and delegate to library code in `training/` or `inference/`.

## Dependency direction

```
cli/train.py  ─▶  training/trainer.py
                      │
                      ├─▶ training/config.py
                      ├─▶ training/chat_template.py
                      ├─▶ training/data.py
                      └─▶ training/callbacks.py

cli/infer.py  ─▶  inference/pipeline.py
```

## Model loading philosophy

Since `transformers >= 4.57` natively registers `HunYuanVLMoT{Config,Model,Processor}`
and `HunYuanDenseV1Config`, **no local modelling code is needed**. Both the
trainer and the inference pipeline:

1. Always disable `trust_remote_code` at the `AutoProcessor` /
   `AutoModelForImageTextToText` calls.
2. Ship a training-specific chat template that is only used at training time.
3. Copy the original `chat_template.jinja` and restore `model_type` /
   `auto_map` into every checkpoint, so saved artifacts are
   interchangeable with the upstream release.

That means bumping transformers to a newer version transparently upgrades
our model implementation — we do not fork or vendor any modelling code.

## Adding a new capability

1. Author a JSONL file following `docs/data_format.md`.
2. Drop a small (<= 5 samples) representative slice into `data_examples/`
   and document it in `data_examples/README.md`.
3. Add a new section to your training config's `data.train_data_paths` and
   `data.train_data_sampling_ratios`.
4. If the capability introduces new tokens or structured outputs, document
   them in `docs/data_format.md`.
