# Copyright (c) 2026, EleutherAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from megatron.data.packed_sequence import PackedSequenceModelInputs
from megatron.model.gpt2_model import (
    _post_transformer_block,
    _pre_transformer_block,
)
from megatron.model.transformer import (
    ParallelSelfAttention,
    ParallelTransformerLayer,
    ParallelTransformerLayerPipe,
)
from megatron.model.utils import SequentialWrapper
from megatron.model.word_embeddings import Embedding, EmbeddingPipe


@pytest.mark.cpu
@pytest.mark.parametrize("packed", [False, True], ids=["ordinary", "packed"])
def test_embedding_and_gpt_transforms_preserve_model_context(monkeypatch, packed):
    tokens = torch.tensor([[10, 11, 12, 13]])
    position_ids = torch.tensor([[0, 1, 0, 1]])
    hidden_states = torch.arange(8, dtype=torch.float32).view(1, 4, 2)
    attention_mask = torch.zeros(1, dtype=torch.bool)

    if packed:
        model_inputs = PackedSequenceModelInputs(
            tokens=tokens,
            position_ids=position_ids,
            cu_seqlens=torch.tensor([0, 2, 4, 4, 4], dtype=torch.int32),
            num_documents=torch.tensor(2, dtype=torch.int32),
            max_seqlen=torch.tensor(2, dtype=torch.int32),
            attention_mask=attention_mask,
        )
    else:
        model_inputs = (tokens, position_ids, attention_mask)

    embedding = EmbeddingPipe.__new__(EmbeddingPipe)
    torch.nn.Module.__init__(embedding)

    def embed(_self, input_ids, positions, tokentype_ids=None):
        assert input_ids is tokens
        assert positions is position_ids
        return hidden_states

    monkeypatch.setattr(Embedding, "forward", embed)

    embedded = embedding(model_inputs)
    assert embedded[0] is hidden_states
    assert all(
        actual is expected
        for actual, expected in zip(embedded[1:], model_inputs[2:])
    )

    transformer_input = _pre_transformer_block(embedded)
    assert torch.equal(transformer_input[0], hidden_states.transpose(0, 1))
    assert all(
        actual is expected
        for actual, expected in zip(transformer_input[1:], embedded[1:])
    )

    output = _post_transformer_block(transformer_input)
    assert torch.equal(output, hidden_states)


@pytest.mark.cpu
@pytest.mark.parametrize("packed", [False, True], ids=["ordinary", "packed"])
def test_transformer_layer_pipe_threads_model_context(monkeypatch, packed):
    hidden_states = torch.zeros((4, 1, 8))
    attention_mask = torch.zeros(1, dtype=torch.bool)
    if packed:
        context = (
            torch.tensor([0, 2, 4, 4, 4], dtype=torch.int32),
            torch.tensor(2, dtype=torch.int32),
            torch.tensor(2, dtype=torch.int32),
            attention_mask,
        )
    else:
        context = (attention_mask,)

    received = []

    def forward(
        _self,
        hidden,
        mask,
        layer_past=None,
        *,
        cu_seqlens=None,
        num_documents=None,
        max_seqlen=None,
    ):
        received.append(
            (
                hidden,
                mask,
                cu_seqlens,
                num_documents,
                max_seqlen,
            )
        )
        return hidden + 1

    monkeypatch.setattr(ParallelTransformerLayer, "forward", forward)
    layers = []
    for _ in range(2):
        layer = ParallelTransformerLayerPipe.__new__(ParallelTransformerLayerPipe)
        torch.nn.Module.__init__(layer)
        layers.append(layer)

    output = (hidden_states, *context)
    for layer in layers:
        output = layer(output)

    expected_metadata = context[:-1] if packed else (None, None, None)
    assert len(received) == len(layers)
    for layer_index, values in enumerate(received):
        assert torch.equal(values[0], hidden_states + layer_index)
        assert all(
            actual is expected
            for actual, expected in zip(
                values[1:],
                (attention_mask, *expected_metadata),
            )
        )
    assert torch.equal(output[0], hidden_states + len(layers))
    assert all(
        actual is expected for actual, expected in zip(output[1:], context)
    )


@pytest.mark.cpu
def test_deepspeed_two_stage_transport_preserves_packed_context(monkeypatch):
    """Exercise DeepSpeed's stage send/receive logic without a network backend."""

    from deepspeed.runtime.pipe import engine as pipeline_engine

    packed_context = (
        torch.arange(8, dtype=torch.float32).view(4, 1, 2),
        torch.tensor([0, 2, 4, 4, 4], dtype=torch.int32),
        torch.tensor(2, dtype=torch.int32),
        torch.tensor(2, dtype=torch.int32),
        torch.ones(1, dtype=torch.bool),
    )
    mailbox = []

    def send(tensor, _stage):
        mailbox.append(tensor.detach().clone())

    monkeypatch.setattr(pipeline_engine.p2p, "send", send)
    sender = SimpleNamespace(
        wall_clock_breakdown=lambda: False,
        pipe_buffers={"outputs": [packed_context]},
        has_attention_mask=True,
        has_bool_tensors=False,
        dynamic_shape=False,
        first_output_send=False,
        next_stage=1,
    )

    pipeline_engine.PipelineEngine._exec_send_activations(sender, 0)

    assert [tensor.dtype for tensor in mailbox] == [
        torch.float32,
        torch.int32,
        torch.int32,
        torch.int32,
        torch.float16,
    ]

    def recv(tensor, _stage):
        tensor.copy_(mailbox.pop(0))

    monkeypatch.setattr(pipeline_engine.p2p, "recv", recv)
    receive_buffers = tuple(
        torch.empty_like(
            tensor,
            dtype=torch.float16 if tensor.dtype == torch.bool else tensor.dtype,
        )
        for tensor in packed_context
    )
    receiver = SimpleNamespace(
        wall_clock_breakdown=lambda: False,
        dynamic_shape=False,
        pipe_recv_buf=receive_buffers,
        prev_stage=0,
        is_pipe_partitioned=False,
        meta_buffer=None,
        device=torch.device("cpu"),
        has_attention_mask=True,
        has_bool_tensors=False,
        pipe_buffers={"inputs": [None]},
    )

    pipeline_engine.PipelineEngine._exec_recv_activations(receiver, 0)

    received = receiver.pipe_buffers["inputs"][0]
    assert not mailbox
    assert len(received) == len(packed_context)
    for actual, expected in zip(received, packed_context):
        assert torch.equal(actual, expected)
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape


@pytest.mark.cpu
@pytest.mark.parametrize("gpt_j_residual", [False, True])
def test_transformer_layer_passes_metadata_to_attention(gpt_j_residual):
    class Attention(torch.nn.Module):
        def forward(
            self,
            hidden_states,
            attention_mask,
            layer_past=None,
            *,
            cu_seqlens=None,
            num_documents=None,
            max_seqlen=None,
        ):
            self.received = (
                attention_mask,
                cu_seqlens,
                num_documents,
                max_seqlen,
            )
            return torch.zeros_like(hidden_states), None

    class MLP(torch.nn.Module):
        def forward(self, hidden_states):
            return torch.zeros_like(hidden_states), None

    layer = ParallelTransformerLayer.__new__(ParallelTransformerLayer)
    torch.nn.Module.__init__(layer)
    layer.neox_args = SimpleNamespace(
        te_fp8_mha=False,
        te_layernorm_mlp=False,
    )
    layer.layer_past = None
    layer.hidden_dropout = 0.0
    layer.bias_dropout_fusion = False
    layer.gpt_j_residual = gpt_j_residual
    layer.gpt_j_tied = True
    layer.use_cache = False
    layer.num_experts = 1
    layer.input_layernorm = torch.nn.Identity()
    layer.post_attention_layernorm = torch.nn.Identity()
    layer.attention = Attention()
    layer.mlp = MLP()
    layer.reduce = lambda value: value
    layer.eval()

    hidden_states = torch.zeros((4, 1, 8))
    attention_mask = torch.zeros(1, dtype=torch.bool)
    cu_seqlens = torch.tensor([0, 2, 4, 4, 4], dtype=torch.int32)
    num_documents = torch.tensor(2, dtype=torch.int32)
    max_seqlen = torch.tensor(2, dtype=torch.int32)

    output = layer(
        hidden_states,
        attention_mask,
        cu_seqlens=cu_seqlens,
        num_documents=num_documents,
        max_seqlen=max_seqlen,
    )

    assert torch.equal(output, hidden_states)
    assert all(
        actual is expected
        for actual, expected in zip(
            layer.attention.received,
            (attention_mask, cu_seqlens, num_documents, max_seqlen),
        )
    )


@pytest.mark.cpu
def test_native_attention_rejects_packed_context_until_backend_support_exists():
    attention = ParallelSelfAttention.__new__(ParallelSelfAttention)
    torch.nn.Module.__init__(attention)

    with pytest.raises(NotImplementedError, match="Packed-sequence FlashAttention"):
        attention(
            torch.zeros((4, 1, 8)),
            torch.zeros(1, dtype=torch.bool),
            cu_seqlens=torch.tensor([0, 2, 4, 4, 4], dtype=torch.int32),
            num_documents=torch.tensor(2, dtype=torch.int32),
            max_seqlen=torch.tensor(2, dtype=torch.int32),
        )


@pytest.mark.cpu
def test_sequential_wrapper_preserves_packed_context_during_checkpointing():
    """Exercise the wrapper constructed by GPT2ModelPipe.to_sequential()."""

    class CheckpointedParallelTransformerLayerPipe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0))
            self.received_contexts = []

        def forward(self, args):
            hidden_states, *context = args
            self.received_contexts.append(
                tuple(tensor.detach().clone() for tensor in context)
            )
            return (hidden_states * self.weight, *context)

    class PostTransformerBlock(torch.nn.Module):
        def forward(self, args):
            self.received_context = tuple(
                tensor.detach().clone() for tensor in args[1:]
            )
            return _post_transformer_block(args)

    def run(checkpoint_interval):
        transformer_layer = CheckpointedParallelTransformerLayerPipe()
        post_transformer = PostTransformerBlock()
        model = SequentialWrapper(
            [transformer_layer, post_transformer],
            activation_checkpoint_interval=checkpoint_interval,
            activation_checkpoint_func=partial(checkpoint, use_reentrant=True),
            parent_class_name="GPT2ModelPipe",
        )

        hidden_states = torch.arange(8, dtype=torch.float32).view(4, 1, 2)
        hidden_states.requires_grad_()
        context = (
            torch.tensor([0, 2, 4, 4, 4], dtype=torch.int32),
            torch.tensor(2, dtype=torch.int32),
            torch.tensor(2, dtype=torch.int32),
            torch.zeros(1, dtype=torch.bool),
        )

        output = model((hidden_states, *context))
        output.sum().backward()

        assert torch.is_tensor(output)
        assert len(post_transformer.received_context) == len(context)
        received_contexts = [
            post_transformer.received_context,
            *transformer_layer.received_contexts,
        ]
        for received_context in received_contexts:
            for actual, expected in zip(received_context, context):
                assert torch.equal(actual, expected)
                assert actual.dtype == expected.dtype
                assert actual.shape == expected.shape
        for tensor in context:
            assert not tensor.requires_grad
            assert tensor.grad is None

        expected_layer_calls = 2 if checkpoint_interval else 1
        assert len(transformer_layer.received_contexts) == expected_layer_calls

        return (
            output.detach(),
            hidden_states.grad,
            transformer_layer.weight.grad,
        )

    direct = run(checkpoint_interval=0)
    checkpointed = run(checkpoint_interval=1)

    for direct_tensor, checkpointed_tensor in zip(direct, checkpointed):
        assert torch.equal(direct_tensor, checkpointed_tensor)
