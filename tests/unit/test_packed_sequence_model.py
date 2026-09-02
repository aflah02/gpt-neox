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

import pytest
import torch

from megatron.data.packed_sequence import PackedSequenceModelInputs
from megatron.model.gpt2_model import (
    _post_transformer_block,
    _pre_transformer_block,
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
