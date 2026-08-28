# Inter-Document Attention Masking Plan

NeoX needs a packed-sequence execution path, not only a direct port of the ten-file diff in [Megatron PR #5298](https://github.com/NVIDIA/Megatron-LM/pull/5298). Megatron already had THD infrastructure for supervised fine-tuning; NeoX does not.

Do not port Megatron's context-parallel changes. The author later confirmed that context parallelism greater than one is incorrect for arbitrary document lengths. [Megatron PR #5531](https://github.com/NVIDIA/Megatron-LM/pull/5531) proposes a guard, and [issue #6156](https://github.com/NVIDIA/Megatron-LM/issues/6156) remains open. NeoX currently has no context-parallel support.

## Recommended implementation plan

### 1. Add the configuration contract

Add `inter_document_attention_masking: bool = False` to `megatron/neox_arguments/neox_args.py`, regenerate `configs/neox_arguments.md`, and validate supported combinations in `megatron/neox_arguments/arguments.py`.

Initial support should require:

- `dataset_impl: gpt2`
- `train_impl: normal`
- FlashAttention or Transformer Engine attention
- No soft-prompt tuning or curriculum sequence truncation
- No sparse, gMLP, RWKV, or Mamba layers until their cross-document state is handled

### 2. Produce document-boundary metadata

Refactor `GPT2Dataset.__getitem__` in `megatron/data/gpt2_dataset.py` to retain the length of every document fragment contributing to a sample.

Generate:

- `cu_seqlens`, covering the `seq_length` input tokens
- `max_seqlen`
- Document-local position IDs, or enough metadata to generate them later

Match Megatron's edge cases:

- Subtract the extra next-token label from the final fragment.
- Drop a resulting zero-length final fragment.
- Fold sample padding into the last fragment.
- Ensure `cu_seqlens[-1] == seq_length`.
- Pad `cu_seqlens` to `seq_length + 1` with repeated terminal values so default DataLoader collation works for samples containing different document counts.

### 3. Normalize microbatches and broadcast metadata

Extend the batch construction in `megatron/training.py` to:

- Broadcast `cu_seqlens` and `max_seqlen` separately as `int32`; the current `megatron/mpu/data.py::broadcast_data` assumes one dtype.
- Strip collation padding.
- Merge boundaries across microbatch samples by offsetting sample `i` by `i * seq_length`.
- Flatten `[micro_batch, seq]` tensors to `[1, micro_batch * seq]`.
- Reset position IDs at every document boundary.
- Skip allocation of the dense `[1, 1, S, S]` attention mask.

### 4. Thread packed metadata through the model

Generalize the pipeline state from `(hidden_states, attention_mask)` to a tensor-only context carrying:

- A dense-mask sentinel or optional mask
- `cu_seqlens`
- `max_seqlen`
- Position IDs if native RoPE needs them

Update:

- `megatron/model/word_embeddings.py::EmbeddingPipe`
- The GPT pipeline pre/post transforms in `megatron/model/gpt2_model.py`
- `megatron/model/transformer.py::ParallelTransformerLayerPipe`
- Activation-checkpoint tuple handling in `megatron/model/utils.py::SequentialWrapper`

Non-attention pipeline layers should transparently preserve the additional metadata.

### 5. Add native FlashAttention THD execution

Extend `ParallelSelfAttention` in `megatron/model/transformer.py`:

- Squeeze Q/K/V from `[T, 1, H, D]` to `[T, H, D]`.
- Call `flash_attn_varlen_func` during training with document `cu_seqlens`, `max_seqlen`, causal masking, dropout, sliding-window, and ALiBi arguments.
- Restore `[T, 1, H]` afterward.
- Preserve the existing fixed-shape path when the feature is disabled.

Rotary embeddings must reset per document. The pinned FlashAttention version already provides packed RoPE using `cu_seqlens` and `max_seqlen`; the current NeoX implementation uses flat sequence offsets and would be incorrect. See [FlashAttention's packed RoPE API](https://github.com/Dao-AILab/flash-attention/blob/v2.5.6/flash_attn/layers/rotary.py#L94-L124).

### 6. Add Transformer Engine THD execution

Update `TEMultiheadAttention` in `megatron/model/transformer_engine.py` to select `qkv_format="thd"` only for packed calls and pass `cu_seqlens`, `max_seqlen`, and a causal padding mask type.

It must return to `sbhd` for generation and ordinary batches; configuring THD permanently would break inference from the same checkpoint. Transformer Engine 1.12 already exposes the required parameters in its [MultiheadAttention forward API](https://github.com/NVIDIA/TransformerEngine/blob/v1.12/transformer_engine/pytorch/attention.py#L8680-L8699).

### 7. Handle positional embeddings

- Learned positions: use reset `position_ids`.
- Rotary: use packed RoPE, not the current flat offset.
- Sinusoidal: fix it to consume actual position-ID values or reject it initially; it currently ignores the values.
- ALiBi and sliding-window attention: verify behavior through the varlen kernel.
- RPE and sparse attention: reject initially.

### 8. Add tests and acceptance criteria

Add CPU dataset tests and GPU/distributed tests covering:

- Mid-document starts and ends, exact document-boundary endings, shortfall padding, and single-document samples.
- Microbatch sizes one and greater than one with differing document counts.
- Strictly increasing unpadded `cu_seqlens`, the correct terminal offset, and the correct `max_seqlen`.
- Tensor- and pipeline-parallel sizes 1, 2, and 4, plus activation checkpointing and sequence parallelism.
- Native FlashAttention and Transformer Engine outputs and gradients against a small dense block-diagonal reference.
- Cross-document isolation: changing document A cannot affect document B.
- Learned and rotary position resets.
- Flag-disabled regression behavior and normal inference after packed training.
- No quadratic dense attention-mask allocation.

## Separate parity work

Megatron's hybrid results depend on packed metadata reaching Mamba and linear-recurrent kernels. NeoX's current Mamba implementation in `megatron/model/mamba/mamba.py` carries convolution and SSM state across the flattened stream and cannot consume document IDs. Full hybrid parity therefore requires upgraded segmented kernels or per-document execution.

Until that work is complete, reject Mamba, RWKV, gMLP, and sparse configurations with clear validation errors.

## Suggested delivery sequence

1. Dataset metadata, collation, merging, and unit tests.
2. Native FlashAttention THD path and dense-reference correctness tests.
3. Tensor-parallel, pipeline-parallel, and activation-checkpoint metadata plumbing.
4. Transformer Engine THD path.
5. Distributed integration tests, documentation, and performance smoke tests.
6. Hybrid Mamba/linear-recurrent parity as a separate change.
