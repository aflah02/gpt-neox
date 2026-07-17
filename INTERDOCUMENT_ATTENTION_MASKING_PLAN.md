# Interdocument Attention Masking Plan

## Summary

Interdocument attention masking fits naturally into GPT-NeoX's existing mask-building path, but optimized attention backends need explicit handling. The codebase previously exposed this feature as `reset_attention_mask`; commit `0fb2c09` removed it in 2021 because it was unused. Restoring the historical implementation unchanged would silently fail with current Flash Attention and upper-triangular fused softmax.

## Recommended semantics

Add the following configuration flag:

```yaml
reset_attention_mask: true
```

The default should remain `false`, preserving current behavior.

When enabled, the tokenizer's EOD token marks the end of a document:

```text
doc A ... EOD | doc B ...
```

Tokens after EOD cannot attend to EOD or anything preceding it. EOD itself remains part of the preceding document, matching historical Megatron behavior.

This should remain independent from:

```yaml
eod_mask_loss: true
```

`eod_mask_loss` suppresses the loss for predicting the first token of the next document from the EOD position. Users will commonly want both flags.

Position IDs should not be reset as part of this feature. That is a separate policy, particularly relevant for learned absolute embeddings.

## Required changes

### 1. Configuration

Add `reset_attention_mask: bool = False` next to `eod_mask_loss` in `megatron/neox_arguments/neox_args.py`, then regenerate `configs/neox_arguments.md`.

The historical name is preferable because it matches Megatron terminology and this repository's former API.

### 2. Mask construction

Extend `get_ltor_masks_and_position_ids()` in `megatron/utils.py` with the flag.

Conceptually:

```python
is_eod = data.eq(eod_token)
document_ids = is_eod.cumsum(dim=-1) - is_eod.long()

cross_document = document_ids[:, :, None] != document_ids[:, None, :]
attention_mask = causal_or_sliding_mask | cross_document[:, None]
```

The subtraction assigns EOD to the preceding document. The result changes from `[1, 1, S, S]` to `[B, 1, S, S]` only when enabled.

### 3. Thread the flag through batching

Pass the flag from:

- `_get_batch()` in `megatron/training.py`
- `get_batch_sequential()` in `megatron/training.py`
- Optionally, text-generation batching in `megatron/text_generation_utils.py` if generated EOD boundaries should also reset context

Curriculum slicing already slices every batch dimension correctly, although its comments assume a singleton batch dimension.

### 4. Fix fused-softmax selection

The upper-triangular fused kernel in `megatron/model/fused_softmax.py` explicitly ignores the supplied mask. Therefore, when interdocument masking is enabled:

- Select the general masked-softmax fusion, or
- Fall back to the PyTorch masked softmax.

Leaving `scaled_upper_triang_masked_softmax_fusion` active would produce incorrect training.

### 5. Handle Flash Attention separately

Flash Attention currently does not receive the attention mask in `megatron/model/transformer.py`.

The efficient solution is to represent each packed document as a separate variable-length sequence and call the already-imported `flash_attn_varlen_func`:

1. Flatten Q/K/V across the microbatch.
2. Build `cu_seqlens` from EOD boundaries and batch-row boundaries.
3. Set `max_seqlen` to the longest document segment.
4. Run causal variable-length Flash Attention.
5. Reshape the output back to `[S, B, H, D]`.

Passing a dense `[B, S, S]` mask to Flash Attention is not supported and would lose its primary memory advantage.

### 6. Backend compatibility

For a first correctness release, support standard `global` attention and reject unsupported configurations clearly.

Additional implementation work and testing are needed for:

- Transformer Engine MHA
- DeepSpeed sparse/local attention
- Flash/Triton ALiBi fallback
- Mixed attention configurations

Mamba and RWKV ignore attention masks. gMLP's spatial gating also mixes sequence positions independently of the attention mask, so these configurations should be rejected when the flag is enabled.

## Dataset requirement

Packed documents are concatenated in `megatron/data/gpt2_dataset.py`, but the dataset output does not include boundary metadata. Consequently, the implementation must infer boundaries from EOD tokens.

Preprocessing only inserts those tokens when `tools/datasets/preprocess_data.py` is run with `--append-eod`. This requirement should be documented prominently. Without EOD tokens, the flag cannot distinguish documents.

## Overhead

With the flag disabled, overhead is effectively zero beyond one branch.

### Conventional dense attention

- Mask construction: `O(B * S^2)` once per batch.
- Mask memory: `B * S^2` bytes because PyTorch boolean values occupy one byte.
- Attention FLOPs remain unchanged; cross-document scores are computed and then masked.
- Upper-triangular softmax fusion must be replaced by general-mask fusion, which may affect throughput.

| Sequence length | Microbatch | Current mask | Interdocument mask |
|---:|---:|---:|---:|
| 2,048 | 4 | 4 MiB | 16 MiB |
| 4,096 | 4 | 16 MiB | 64 MiB |
| 8,192 | 4 | 64 MiB | 256 MiB |

Pipeline parallelism can also transmit this larger mask between stages.

### Flash Attention variable-length path

- Boundary scan: `O(B * S)`.
- Metadata: approximately `O(number of documents)`.
- Possible Q/K/V layout-copy cost: `O(B * S * hidden_size)`.
- Attention compute can decrease substantially: it becomes proportional to `sum(document_length^2)` rather than `S^2`.

With several short documents, segmented variable-length Flash Attention can be faster than the current unsegmented Flash Attention.

## Implementation plan

1. Restore the configuration flag and dense mask behavior.
2. Force general-mask softmax and add configuration validation.
3. Add unit tests for multiple boundaries, different boundaries per batch row, consecutive EODs, sliding-window intersection, and disabled-flag regression.
4. Add an end-to-end invariance test: changing document A must not alter document B logits.
5. Test tensor parallelism, pipeline parallelism, curriculum truncation, and activation checkpointing.
6. Add the Flash Attention variable-length path and compare forward and backward results against dense attention.
7. Benchmark dense and Flash paths at representative sequence lengths and packing ratios.
8. Document `--append-eod` and the relationship with `eod_mask_loss`.

