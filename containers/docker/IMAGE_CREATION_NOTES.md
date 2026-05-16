# Docker Image Creation Notes

This note reflects the current Docker files in `gpt-neox/containers/docker`.

## Current files

- `Dockerfile`: upstream repo Dockerfile
- `Dockerfile.h100-te`: H100-oriented variant
- `Dockerfile.b200-te`: B200-oriented variant
- `docker-compose.yml`: checked-in compose file

## Important repo detail

Build from the `gpt-neox` repo root, not from `containers/docker`.

The Dockerfiles copy `requirements/*` and `megatron/fused_kernels/`, so the build context must be the repo root:

```bash
cd /NS/llm-pretraining/work/afkhan/GPT-NeoX-Dev/TorchRun_Support/gpt-neox
```

## H100 image

Use `containers/docker/Dockerfile.h100-te`.

What it does:

- keeps the upstream `nvcr.io/nvidia/pytorch:24.02-py3` base
- adds the extra system packages needed for the custom build path
- installs:
  - `requirements.txt`
  - `requirements-onebitadam.txt`
  - `wandb==0.16.6`
  - `transformer-engine[pytorch]==1.12`
  - `flash-attn==2.5.6` built from source
  - `protobuf==3.20.*`
- sets `TORCH_CUDA_ARCH_LIST=9.0`
- pins W&B to `0.16.6` so it remains compatible with the image's `protobuf==3.20.*` pin
- installs FlashAttention from source instead of using the wheel path from `requirements-transformerengine.txt`, because prebuilt wheels are more likely to hit ABI mismatches on the older `24.02` NGC PyTorch base

Build:

```bash
docker build -f containers/docker/Dockerfile.h100-te -t gpt-neox:h100-te .
```

Quick verification:

```bash
docker run --rm --gpus all --ipc=host gpt-neox:h100-te nvidia-smi
docker run --rm --gpus all --ipc=host gpt-neox:h100-te python -c "import torch, wandb, flash_attn, transformer_engine; print(torch.__version__)"
```

## B200 image

Use `containers/docker/Dockerfile.b200-te`.

What it does:

- switches the base image to `nvcr.io/nvidia/pytorch:25.04-py3`
- keeps the repo requirement installs for:
  - `requirements.txt`
  - `requirements-onebitadam.txt`
  - `wandb==0.16.6`
- sets `TORCH_CUDA_ARCH_LIST=10.0`
- relies on the newer NGC base image for Transformer Engine instead of installing `requirements-transformerengine.txt`
- rewrites the repo's `jinja2==3.1.4` pin to `jinja2==3.1.6` during the image build to match the NGC base-image pip constraint
- does not force the old `protobuf==3.20.*` pin, because the `25.04` base image constrains protobuf to a newer version
- removes `chardet` after install to avoid the `requests` dependency warning emitted during W&B imports in this base image
- pins W&B to `0.16.6` so it stays compatible with the base image's protobuf constraint and avoids the newer telemetry/protobuf import breakage

Build:

```bash
docker build -f containers/docker/Dockerfile.b200-te -t gpt-neox:b200-te .
```

Quick verification:

```bash
docker run --rm --gpus all --ipc=host gpt-neox:b200-te nvidia-smi
docker run --rm --gpus all --ipc=host gpt-neox:b200-te python -c "import torch, wandb, transformer_engine; print(torch.__version__)"
```

## Running the container

Set your data and checkpoint paths first:

```bash
export NEOX_DATA_PATH=/path/to/data
export NEOX_CHECKPOINT_PATH=/path/to/checkpoints
export WANDB_API_KEY=your_wandb_key
```

Example interactive run:

```bash
docker run --rm -it \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  -v $NEOX_DATA_PATH:/home/mchorse/data \
  -v $NEOX_CHECKPOINT_PATH:/home/mchorse/chk \
  -v $(pwd):/home/mchorse/gpt-neox \
  gpt-neox:h100-te \
  bash
```

Swap `gpt-neox:h100-te` for `gpt-neox:b200-te` when using the B200 image.

## Publishing Docker images to GitHub

GitHub's container registry uses `ghcr.io`.

Authenticate first. For command-line pushes, use a GitHub personal access token that can write packages:

```bash
export GHCR_USER=your-github-username
export GHCR_PAT=your-token
echo $GHCR_PAT | docker login ghcr.io -u $GHCR_USER --password-stdin
```

Choose an image namespace. A common pattern is:

```bash
export GHCR_NAMESPACE=your-github-username-or-org
```

Tag and push the H100 image:

```bash
docker tag gpt-neox:h100-te ghcr.io/$GHCR_NAMESPACE/gpt-neox:h100-te
docker push ghcr.io/$GHCR_NAMESPACE/gpt-neox:h100-te
```

Tag and push the B200 image:

```bash
docker tag gpt-neox:b200-te ghcr.io/$GHCR_NAMESPACE/gpt-neox:b200-te
docker push ghcr.io/$GHCR_NAMESPACE/gpt-neox:b200-te
```

If you also want a floating tag such as `latest-h100` or `latest-b200`:

```bash
docker tag gpt-neox:h100-te ghcr.io/$GHCR_NAMESPACE/gpt-neox:latest-h100
docker push ghcr.io/$GHCR_NAMESPACE/gpt-neox:latest-h100

docker tag gpt-neox:b200-te ghcr.io/$GHCR_NAMESPACE/gpt-neox:latest-b200
docker push ghcr.io/$GHCR_NAMESPACE/gpt-neox:latest-b200
```

These Dockerfiles already set `org.opencontainers.image.source`, which helps when associating the published container package with a GitHub repository.

If you prefer GitHub Actions instead of local `docker push`, publish to `ghcr.io/${{ github.repository }}` with `GITHUB_TOKEN`.

## Compose note

The checked-in `docker-compose.yml` still uses `context: .`, which resolves to `containers/docker`. That does not match these Dockerfiles.

If you use Compose, update the build context to the repo root, for example:

```yaml
services:
  gpt-neox:
    image: gpt-neox:h100-te
    build:
      context: ../..
      dockerfile: containers/docker/Dockerfile.h100-te
    command: bash
    gpus: all
    ipc: host
    shm_size: 16g
    ulimits:
      memlock:
        soft: -1
        hard: -1
    volumes:
      - ${NEOX_DATA_PATH}:/home/mchorse/data
      - ${NEOX_CHECKPOINT_PATH}:/home/mchorse/chk
      - ../..:/home/mchorse/gpt-neox
```

## Torchrun note

This branch includes `deepy_torchrun.py`, so after entering the container you can launch from the repo root with:

```bash
python deepy_torchrun.py train.py /path/to/config.yml
```

## Apptainer equivalents

Matching Apptainer definition files are available under `containers/apptainer`:

- `gpt-neox-h100-te.def`
- `gpt-neox-b200-te.def`

Build them from the `gpt-neox` repo root so the `%files` section can stage `requirements/` and `megatron/fused_kernels/` into the image:

```bash
cd /NS/llm-pretraining/work/afkhan/GPT-NeoX-Dev/TorchRun_Support/gpt-neox
apptainer build containers/apptainer/gpt-neox-h100-te.sif containers/apptainer/gpt-neox-h100-te.def
apptainer build containers/apptainer/gpt-neox-b200-te.sif containers/apptainer/gpt-neox-b200-te.def
```

If you already built the Docker images and want `.sif` files from those images instead of rebuilding from the `.def` files, you can convert directly from the local Docker daemon:

```bash
cd /NS/llm-pretraining/work/afkhan/GPT-NeoX-Dev/TorchRun_Support/gpt-neox
apptainer build containers/apptainer/gpt-neox-h100-te-from-docker.sif docker-daemon://gpt-neox:h100-te
apptainer build containers/apptainer/gpt-neox-b200-te-from-docker.sif docker-daemon://gpt-neox:b200-te
```

If you pushed the Docker images to a registry, you can convert from the registry instead:

```bash
apptainer build gpt-neox-h100-te.sif docker://your-registry/gpt-neox:h100-te
apptainer build gpt-neox-b200-te.sif docker://your-registry/gpt-neox:b200-te
```

Run them with:

```bash
apptainer exec --nv containers/apptainer/gpt-neox-h100-te.sif \
  python -c "import torch, wandb, flash_attn, transformer_engine; print(torch.__version__)"
apptainer exec --nv containers/apptainer/gpt-neox-b200-te.sif \
  python -c "import torch, wandb, transformer_engine; print(torch.__version__)"
```

For `.sif` files produced from the Docker images, use the same `apptainer exec --nv ...` pattern with the generated filenames.
