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

from types import SimpleNamespace

import pytest
import torch

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

    expected_output = torch.ones_like(hidden_states)
    received = {}

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
        received["values"] = (
            hidden,
            mask,
            cu_seqlens,
            num_documents,
            max_seqlen,
        )
        return expected_output

    monkeypatch.setattr(ParallelTransformerLayer, "forward", forward)
    layer = ParallelTransformerLayerPipe.__new__(ParallelTransformerLayerPipe)
    torch.nn.Module.__init__(layer)

    output = layer((hidden_states, *context))

    expected_metadata = context[:-1] if packed else (None, None, None)
    assert all(
        actual is expected
        for actual, expected in zip(
            received["values"],
            (hidden_states, attention_mask, *expected_metadata),
        )
    )
    assert output[0] is expected_output
    assert all(
        actual is expected for actual, expected in zip(output[1:], context)
    )


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
