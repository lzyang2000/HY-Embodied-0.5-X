# Inference

Two entry points are provided: a command-line tool for quick sanity checks
and a small Python library (`HyEmbodiedPipeline`) for programmatic use.

## CLI

```bash
# Auto-download from HF Hub
python -m hy_embodied.cli.infer --image assets/demo.jpg --prompt "Describe this image"

# Use a local checkpoint
python -m hy_embodied.cli.infer \
    --model ckpts/HY-Embodied-0.5-X \
    --image assets/demo.jpg \
    --prompt "Describe this image"

# Disable thinking mode
python -m hy_embodied.cli.infer \
    --model ckpts/HY-Embodied-0.5-X \
    --image assets/demo.jpg \
    --prompt "Describe this image" \
    --no-thinking

# Batch demo
python -m hy_embodied.cli.infer --model ckpts/HY-Embodied-0.5-X --batch
```

After `pip install -e .` the CLI also ships as a console script:

```bash
hy-embodied-infer --image assets/demo.jpg --prompt "Describe this image"
```

The legacy `python inference.py` command is kept as a thin shim over the
same code path.

## Python API

```python
import torch
from hy_embodied.inference import GenerationConfig, HyEmbodiedPipeline

pipe = HyEmbodiedPipeline.from_pretrained(
    "ckpts/HY-Embodied-0.5-X",
    device="cuda",
    torch_dtype=torch.bfloat16,
)

# Single-image
out = pipe.generate(
    "Describe the image in detail.",
    image="assets/demo.jpg",
    generation_config=GenerationConfig(
        max_new_tokens=32768,
        temperature=0.05,
        enable_thinking=True,
    ),
)
print(out)

# Multi-image
out = pipe.generate(
    "What changed between these frames?",
    images=["frame_0.jpg", "frame_1.jpg"],
)

# Video
out = pipe.generate("Summarize the clip.", video="clip.mp4")

# Batch
messages_batch = [
    pipe.build_messages("Describe this.", image="a.jpg"),
    pipe.build_messages("What should I do next?"),
]
outs = pipe.generate_batch(messages_batch)
```

`HyEmbodiedPipeline.from_pretrained` accepts either a local directory or a
Hugging Face Hub repo id. If the given path is a local directory and
contains a `chat_template.jinja` / `chat_template.json`, that template
overrides the one bundled with the processor — this is what makes fine-tuned
checkpoints usable at inference time without extra plumbing.

## Coordinate & response format

All grounding outputs are normalized to `(0, 1000)` on each axis. In thinking
mode the assistant wraps its response as:

```
<think>
reasoning...
</think>
<answer>
final answer (possibly JSON with bbox / point / trajectory fields)
</answer>
```

In `--no-thinking` mode the `<think>` block is empty.

See `docs/data_format.md` for the exact schema and coordinate conventions.

## OpenAI-compatible API Server

HY-Embodied-0.5-X ships with a built-in API server that exposes an
**OpenAI-compatible** `/v1/chat/completions` endpoint, so you can use the
standard OpenAI Python SDK, `curl`, or any compatible client to interact
with the model.

### Launching the Server

```bash
# Option 1: shell script (recommended)
bash scripts/run_server.sh

# Option 2: Python module
python -m hy_embodied.cli.server \
    --model ckpts/HY-Embodied-0.5-X \
    --host 0.0.0.0 \
    --port 8080

# Option 3: console script (after `pip install -e ".[serve]"`)
hy-embodied-server --model ckpts/HY-Embodied-0.5-X --port 8080
```

### Server CLI Options

| Flag              | Default                        | Description                             |
|-------------------|--------------------------------|-----------------------------------------|
| `--model`         | `tencent/HY-Embodied-0.5-X`   | Local path or HF Hub repo id            |
| `--device`        | `cuda`                         | Device to run the model on              |
| `--host`          | `0.0.0.0`                      | Bind address                            |
| `--port`          | `8080`                         | Bind port                               |
| `--dtype`         | `bfloat16`                     | Precision (`bfloat16`/`float16`/`float32`) |
| `--model-name`    | basename of `--model`          | Name shown in `/v1/models`              |
| `--workers`       | `1`                            | Uvicorn workers (1 recommended for GPU) |

### Endpoints

| Method | Path                     | Description                    |
|--------|--------------------------|--------------------------------|
| GET    | `/`                      | Server info                    |
| GET    | `/health`                | Health check                   |
| GET    | `/v1/models`             | List available models          |
| POST   | `/v1/chat/completions`   | Chat completion (main endpoint)|
| GET    | `/docs`                  | Swagger UI (auto-generated)    |

### Client Examples

#### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="any")

# Text-only
resp = client.chat.completions.create(
    model="HY-Embodied-0.5-X",
    messages=[{"role": "user", "content": "How to open a fridge?"}],
)
print(resp.choices[0].message.content)

# With image (URL)
resp = client.chat.completions.create(
    model="HY-Embodied-0.5-X",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
            {"type": "text", "text": "What should the robot do next?"},
        ],
    }],
)

# With image (base64)
import base64
with open("demo.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
resp = client.chat.completions.create(
    model="HY-Embodied-0.5-X",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": "Describe this scene."},
        ],
    }],
)

# Streaming
stream = client.chat.completions.create(
    model="HY-Embodied-0.5-X",
    messages=[{"role": "user", "content": "Plan how to clean the table."}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

#### curl

```bash
# Non-streaming
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "HY-Embodied-0.5-X",
    "messages": [{"role":"user","content":"Hello!"}]
  }'

# Streaming
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "HY-Embodied-0.5-X",
    "messages": [{"role":"user","content":"Hello!"}],
    "stream": true
  }'
```

### Custom Parameters

In addition to standard OpenAI parameters, the server supports:

| Parameter          | Type   | Default | Description                          |
|--------------------|--------|---------|--------------------------------------|
| `enable_thinking`  | bool   | `true`  | Enable/disable thinking (CoT) mode   |

Example:

```python
resp = client.chat.completions.create(
    model="HY-Embodied-0.5-X",
    messages=[{"role": "user", "content": "Describe."}],
    extra_body={"enable_thinking": False},
)
```

### Dependencies

The server requires `fastapi` and `uvicorn`. Install via:

```bash
pip install -e ".[serve]"
# or
pip install fastapi uvicorn[standard]
```
