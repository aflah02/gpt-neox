# ThunderKittens Attention Integration for GPT-NeoX

## Summary

Compatibility is conditional but promising for H100/H200 BF16 training.

| Area | Assessment |
|---|---|
| Hardware | Compatible with Hopper SM90; not portable to Ampere/Blackwell without separate kernels. |
| Attention | Causal MHA and GQA supported for head dimensions 64 and 128. |
| Training | Forward/backward kernels exist, but require a NeoX-owned autograd wrapper. |
| Unsupported initially | FP16/FP32, dropout, ALiBi/RPE score bias, sliding windows, arbitrary masks, KV-cache/decode, custom softmax scaling. |
| Toolchain | Requires CUDA 12.8+, PyTorch 2.8+, C++20; the inspected shell has CUDA 12.4, no PyTorch, and no visible GPU. |
| Packaging | TK 2.x is header-only with individually compiled kernels, not an installable Python package. [Upstream requirements](https://github.com/HazyResearch/ThunderKittens/blob/1c3920d993404dd49a6d4c7267ea11d583bd5c68/README.md) |

GPT-NeoX already has the correct integration seam in `megatron/model/transformer.py`: per-layer backend selection, fused Q/K/V dispatch, GQA, tensor parallelism, and an output layout matching TK after one permutation.

## Implementation Changes

- Pin ThunderKittens commit `1c3920d993404dd49a6d4c7267ea11d583bd5c68` as `third_party/ThunderKittens` submodule and preserve its MIT license.
- Add a reproducible SM90 extension build using C++20 and `compute_90a/sm_90a`, exposed through `python -m megatron.fused_kernels.thunderkittens.build`.
- Keep the submodule pristine: copy the upstream [H100 attention source](https://github.com/HazyResearch/ThunderKittens/blob/1c3920d993404dd49a6d4c7267ea11d583bd5c68/kernels/attention/mha_h100/mha_h100.cu) into the build cache, verify its checksum, and apply a tracked NeoX patch that:
  - removes device/stream synchronizations;
  - launches on PyTorch's current CUDA stream with a device guard;
  - checks launch errors asynchronously;
  - validates CUDA device, BF16 dtype, contiguity, rank, shapes, head ratio, head dimension, and sequence alignment.
- Extend NeoX argument validation with the public attention type `thunderkittens`. Fail fast—never silently fall back—unless:
  - precision is BF16;
  - head dimension is 64 or 128;
  - attention dropout is zero;
  - attention is causal and full-window;
  - positional encoding adds no attention-score bias;
  - standard `1/sqrt(head_dim)` scaling is used;
  - cache inference is disabled.
- Add a custom `torch.autograd.Function`:
  - permute NeoX `[S,B,H,D]` tensors to contiguous TK `[B,H,S,D]`;
  - retain reduced K/V heads for GQA;
  - zero-pad training sequences to the next multiple of 768;
  - call TK forward and save padded Q/K/V, output, and LSE;
  - pad the output gradient, call TK backward, slice gradients to the original length, and return them in the input dtype.
- The 768 quantum is inferred from the forward and backward launch divisors in the upstream source. It must be verified because the inference demo only pads to 192 and does not exercise autograd. [Upstream inference integration](https://github.com/HazyResearch/ThunderKittens/blob/1c3920d993404dd49a6d4c7267ea11d583bd5c68/demos/llama/src/model/transformers_modeling_llama.py)
- Upgrade the Hopper runtime to Python 3.12, PyTorch 2.8.0+cu128, CUDA 12.8+, GCC/G++ 11+, Ninja, and FlashAttention 2.8.3. First certify the existing global/FlashAttention paths and pinned DeeperSpeed dependency in that runtime.
- Keep checkpoints unchanged: the backend introduces no parameters or state-dict keys. Existing attention defaults remain unchanged.

## Test and Performance Plan

- Compare TK against FP64 SDPA and FlashAttention for:
  - head dimensions 64 and 128;
  - MHA and GQA;
  - aligned length 768 and padded production lengths 2048, 4096, and 8192;
  - forward outputs and Q/K/V gradients.
- Require TK numerical error to remain within the FlashAttention BF16 error envelope against the same FP64 reference; test finite values and padding invariance explicitly.
- Add negative tests for unsupported dtype, head size, dropout, ALiBi/RPE, sliding window, cache use, nonstandard scaling, missing submodule, wrong CUDA version, and non-SM90 devices.
- Run one- and two-GPU training smoke tests covering tensor parallel MHA/GQA, activation checkpointing, optimizer steps, checkpoint save/resume, and mixed per-layer attention configuration.
- Benchmark complete NeoX layout conversion, padding, forward, and backward—not the raw kernel alone—against FlashAttention 2.8.3. Use D64/D128, MHA/GQA, and 2K/4K/8K contexts.
- Release gate:
  - TK must beat FlashAttention in the attention microbenchmark for every release-blocking shape;
  - median end-to-end tokens/sec across three steady-state runs must be no more than 2% below FlashAttention;
  - profiling must show no device-wide synchronization or new communication stalls.

## Assumptions

- First milestone targets H100/H200 BF16 full-sequence training and evaluation.
- Both 64- and 128-dimensional MHA/GQA paths are release-blocking.
- Cached generation and all other ThunderKittens kernels remain out of scope.
- The backend remains experimental and opt-in until correctness and no-regression gates pass.
