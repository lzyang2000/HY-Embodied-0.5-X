# Local Install Notes

Validated end-to-end install on an x86_64 workstation (Ubuntu 22.04, RTX 4090 24 GB, system gcc 11.4, system CUDA 12.6 at `/usr/local/cuda`, Miniconda). Treat this as an addendum to `setup_env.sh` — the script works, but two pitfalls will bite a clean install in 2026.

## TL;DR — the command sequence that worked

```bash
# 1. Env + CUDA toolkit
conda create -n hy_embodied_x python=3.12 -y
conda activate hy_embodied_x
conda install -c 'nvidia/label/cuda-12.6.3' cuda-toolkit -y

# 2. setup_env.sh WILL fail on flash_attn 2.8.3 (see Pitfall #1).
#    Run it anyway — it installs everything up through the flash_attn step
#    and then errors out. That's fine.
bash setup_env.sh || true

# 3. Install a prebuilt flash_attn wheel instead of compiling 2.8.3 from source
pip install 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1%2Bcu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl'

# 4. Finish the steps setup_env.sh skipped after it bailed
pip install "git+https://github.com/huggingface/transformers@9293856c419762ebf98fbe2bd9440f9ce7069f1a"
pip install accelerate==1.12.0 deepspeed==0.18.4 timm==1.0.21 liger-kernel==0.7.0 \
    opencv-python==4.13.0.90 pydantic==2.12.5 tensorboard==2.20.0 \
    fastapi==0.115.12 "uvicorn[standard]==0.34.3"

# 5. Install repo + weights
pip install -e ".[serve]"
hf download tencent/HY-Embodied-0.5-X --local-dir ckpts/HY-Embodied-0.5-X

# 6. Smoke test
python -m hy_embodied.cli.infer \
    --model ckpts/HY-Embodied-0.5-X \
    --image assets/demo.jpg \
    --prompt 'Point to the fruit. Output JSON: {"steps":[{"points":[[x,y]]}]}' \
    --no-thinking

# 7. Launch the OpenAI-compatible server on :8080
bash scripts/run_server.sh
```

Expected smoke-test output (pointing prompt on `assets/demo.jpg`):
```
<answer>
{"steps":[{"points":[[217, 553]]}]}
```

## Pitfall #1 — flash_attn 2.8.3 won't compile against conda's gcc-14

`setup_env.sh` pins `flash_attn==2.8.3` and installs it with `--no-build-isolation`, which builds from source. On a fresh setup the build fails with:

```
/usr/local/cuda/include/crt/host_config.h:143:2:
error: #error -- unsupported GNU version! gcc versions later than 13 are not supported!
```

**Why:** `conda install cuda-toolkit` from the `nvidia` channel pulls in `cxx-compiler`, which pulls gcc 14.3 into the env. CUDA 12.6's `host_config.h` has a hard cap at gcc 13. Activating the env puts conda's gcc first on `PATH`, so flash_attn's build picks it up instead of the system gcc 11.4.

**Fix (what the TL;DR does):** skip source compilation and install a prebuilt wheel. Dao-AILab publishes wheels per `(flash_attn, torch, cuda, python, cxx11abi)` combo at [github.com/Dao-AILab/flash-attention/releases](https://github.com/Dao-AILab/flash-attention/releases). The important constraints are:

| PyTorch from `setup_env.sh` | 2.10.0+cu126 (cxx11abi=TRUE) |
|---|---|
| Python | 3.12 (cp312) |
| Platform | linux_x86_64 |

flash_attn 2.8.3 has no torch 2.10 wheel. 2.8.1 has one. The API surface we use (`flash_attn_func`, `flash_attn_varlen_func`) is stable across 2.8.x, so 2.8.1 is a drop-in.

**Other fixes if you need 2.8.3 exactly:** point the build at system gcc instead:
```bash
CC=/usr/bin/gcc CXX=/usr/bin/g++ CUDA_HOME=/usr/local/cuda \
    pip install flash_attn==2.8.3 --no-build-isolation
```
This compiles cleanly but takes ~15 min on an 8-core box. Keep `MAX_JOBS=4` unless you have >64 GB RAM.

## Pitfall #2 — `setup_env.sh` bails hard when flash_attn fails

The script uses `set -e`. When flash_attn fails in Step 6, Steps 7–8 (transformers from the pinned commit, and the 40-ish runtime deps) never run, leaving the env half-populated. You need to install a prebuilt flash_attn wheel first (Step 3 above), then finish the remaining pip installs manually (Step 4). Re-running `setup_env.sh` after flash_attn is installed would work in principle, but it tries to re-install `flash_attn==2.8.3` and fails again.

If you want an idempotent full-reinstall path, consider editing `setup_env.sh` locally to guard Step 6 with `pip show flash_attn || pip install ...`, or split the script into two.

## Pitfall #3 — `run_server.sh` binds to `x86_64-conda-linux-gnu`

Starting the server in an activated `hy_embodied_x` env fails with:

```
ERROR:    [Errno -5] No address associated with hostname
```

**Why:** `scripts/run_server.sh` defaults its bind address via `HOST="${HOST:-0.0.0.0}"`. But conda's `cxx-compiler` package (pulled in transitively by `cuda-toolkit`, same thing that causes Pitfall #1) exports `HOST=x86_64-conda-linux-gnu` on env activation — that's the C compiler target triple, not a hostname. The shell substitution sees `HOST` as non-empty and hands the compiler triple to uvicorn.

**Fix — pick one:**

```bash
# Override per-invocation
HOST=0.0.0.0 bash scripts/run_server.sh

# Or clear it in the shell
unset HOST && bash scripts/run_server.sh

# Or edit scripts/run_server.sh to force the default
#   HOST=0.0.0.0            (drop the :- fallback)
```

Any of the three works. The per-invocation override is the least invasive.

## Disk / memory notes

- Total footprint of the env + weights: ~25 GB (conda env ~12 GB, weights 7.1 GB, CUDA toolkit ~4 GB).
- Weight download is ~8 GB over `hf download`; on a fast link it's under a minute.
- `bash setup_env.sh` without the flash_attn workaround still burns ~15 min attempting the compile before failing; using the prebuilt wheel saves that.
- Peak RAM during the PyTorch + transformers install is well under 8 GB. Source-compiling flash_attn is the only real RAM hog (~10 GB with `MAX_JOBS=4`).

## System prerequisites worth stating explicitly

- **System gcc ≤ 13** at `/usr/bin/gcc` (11.4 works, 14+ is blocked by CUDA 12.6).
- **System CUDA 12.6** headers somewhere the build can find them (the standard `/usr/local/cuda` symlink is the safe path). flash_attn's build prefers system CUDA over the conda-env one when `CUDA_HOME` is unset.
- **NVIDIA driver ≥ 560** (driver must support the CUDA 12.6 runtime bundled in the PyTorch wheel; any recent driver is fine).
- **≥ 16 GB VRAM** for bf16 inference; the 4B/2B-active MoE fits in ~8 GB at bf16 but the loader transiently needs more.
- **Python 3.12** exactly — the pinned transformers commit and several wheels only ship cp312.

## Verifying the install

```bash
conda activate hy_embodied_x
python - <<'PY'
import torch, transformers, flash_attn, timm
print("torch       :", torch.__version__, "cuda:", torch.cuda.is_available())
print("transformers:", transformers.__version__)
print("flash_attn  :", flash_attn.__version__)
print("timm        :", timm.__version__)
PY
```

Expected:
```
torch       : 2.10.0+cu126 cuda: True
transformers: 4.57.0
flash_attn  : 2.8.1
timm        : 1.0.21
```

## Server + API check

```bash
bash scripts/run_server.sh &            # binds 0.0.0.0:8080
curl -s http://127.0.0.1:8080/v1/models # {"object":"list","data":[...]}
```

Pointing-style OpenAI request:
```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "HY-Embodied-0.5-X",
    "messages": [{"role": "user", "content": [
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,'"$(base64 -w0 assets/demo.jpg)"'"}},
      {"type": "text", "text": "Point to the fruit. Output JSON: {\"steps\":[{\"points\":[[x,y]]}]} with coordinates normalized to (0, 1000)."}
    ]}],
    "max_tokens": 256,
    "enable_thinking": false
  }' | python -c 'import json,sys;print(json.load(sys.stdin)["choices"][0]["message"]["content"])'
```
