# B200 FlashAttention Test Images

Run these commands from the repository root:

```bash
cd /NS/llm-pretraining/work/afkhan/GPT-NeoX-Dev/TorchRun_Support/gpt-neox
```

## Build Images

```bash
docker build -f containers/docker/Dockerfile.B200.FA -t gpt-neox:b200-fa .
docker build -f containers/docker/Dockerfile.B200.FA4 -t gpt-neox:b200-fa4 .
```

## Test FlashAttention 2 Image

This checks CUDA, the installed `flash-attn` package, and one forward pass through the FA2 API used by GPT-NeoX.

```bash
docker run --rm --gpus all --ipc=host \
  -v "$PWD":/home/mchorse/gpt-neox \
  -w /home/mchorse/gpt-neox \
  gpt-neox:b200-fa \
  python - <<'PY'
import torch
from importlib.metadata import version
from flash_attn.flash_attn_interface import flash_attn_func

assert torch.cuda.is_available(), "CUDA is not available"
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0))
print("flash-attn:", version("flash-attn"))

q = torch.randn(2, 128, 16, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn(2, 128, 16, 128, device="cuda", dtype=torch.bfloat16)
v = torch.randn(2, 128, 16, 128, device="cuda", dtype=torch.bfloat16)
out = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=True)
torch.cuda.synchronize()
print("FA2 output:", tuple(out.shape), out.dtype)
PY
```

## Test FlashAttention 4 Image

This checks CUDA 13, the `flash-attn-4[cu13]` install, and one forward pass through the FA4 CuTeDSL API used by GPT-NeoX. The first run may spend extra time compiling kernels.

```bash
docker run --rm --gpus all --ipc=host \
  -v "$PWD":/home/mchorse/gpt-neox \
  -w /home/mchorse/gpt-neox \
  gpt-neox:b200-fa4 \
  python - <<'PY'
import torch
from importlib.metadata import version
from flash_attn.cute import flash_attn_func

assert torch.cuda.is_available(), "CUDA is not available"
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0))
print("flash-attn-4:", version("flash-attn-4"))

q = torch.randn(2, 128, 16, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn(2, 128, 16, 128, device="cuda", dtype=torch.bfloat16)
v = torch.randn(2, 128, 16, 128, device="cuda", dtype=torch.bfloat16)
out = flash_attn_func(q, k, v, softmax_scale=None, causal=True)
if isinstance(out, tuple):
    out = out[0]
torch.cuda.synchronize()
print("FA4 output:", tuple(out.shape), out.dtype)
PY
```

## Test GPT-NeoX Backend Selection

These checks verify that the GPT-NeoX config default remains FA2 and that the FA4 package is visible in the FA4 image.

```bash
docker run --rm --gpus all --ipc=host \
  -v "$PWD":/home/mchorse/gpt-neox \
  -w /home/mchorse/gpt-neox \
  gpt-neox:b200-fa \
  python - <<'PY'
from megatron.neox_arguments.neox_args import NeoXArgsModel

assert NeoXArgsModel().flash_attention_backend == "flash_attn_2"
print("GPT-NeoX default flash backend: flash_attn_2")
PY

docker run --rm --gpus all --ipc=host \
  -v "$PWD":/home/mchorse/gpt-neox \
  -w /home/mchorse/gpt-neox \
  gpt-neox:b200-fa4 \
  python - <<'PY'
from importlib.metadata import version
from megatron.neox_arguments.neox_args import NeoXArgsModel

assert NeoXArgsModel().flash_attention_backend == "flash_attn_2"
print("GPT-NeoX default flash backend:", NeoXArgsModel().flash_attention_backend)
print("flash-attn-4:", version("flash-attn-4"))
PY
```

## Optional Interactive Shell

```bash
docker run --rm -it --gpus all --ipc=host \
  -v "$PWD":/home/mchorse/gpt-neox \
  -w /home/mchorse/gpt-neox \
  gpt-neox:b200-fa4 bash
```
