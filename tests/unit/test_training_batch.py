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

from megatron import training


def _neox_args(enabled=True, seq_length=4):
    return SimpleNamespace(
        inter_document_attention_masking=enabled,
        seq_length=seq_length,
        tokenizer=SimpleNamespace(eod=-1),
        eod_mask_loss=False,
        sliding_window_width=None,
    )


def _batch():
    return {
        "text": torch.arange(10, dtype=torch.int64).view(2, 5),
        "cu_seqlens": torch.tensor(
            [[0, 2, 4, 4, 4], [0, 4, 4, 4, 4]], dtype=torch.int32
        ),
        "max_seqlen": torch.tensor([2, 4], dtype=torch.int32),
    }


@pytest.fixture
def local_broadcast(monkeypatch):
    calls = []

    def broadcast_data(keys, source, datatype):
        calls.append((keys, datatype))
        return {key: source[key] for key in keys}

    monkeypatch.setattr(training.mpu, "broadcast_data", broadcast_data)
    return calls


@pytest.mark.cpu
def test_get_batch_broadcasts_packed_metadata_as_int32(local_broadcast):
    data = _batch()

    training._get_batch(
        neox_args=_neox_args(),
        tokenizer=SimpleNamespace(eod=-1),
        keys=["text"],
        data=data,
        datatype=torch.int64,
    )

    assert local_broadcast == [
        (["text"], torch.int64),
        (["cu_seqlens", "max_seqlen"], torch.int32),
    ]


@pytest.mark.cpu
def test_get_batch_packed_path_skips_dense_mask_and_preserves_loss_masking(
    monkeypatch, local_broadcast
):
    data = _batch()
    data["label"] = data["text"].clone()
    data["label"][0, 2] = -1
    neox_args = _neox_args()
    neox_args.tokenizer.eod = 2
    neox_args.eod_mask_loss = True

    monkeypatch.setattr(
        training,
        "get_ltor_masks_and_position_ids",
        lambda *args, **kwargs: pytest.fail("dense mask helper was called"),
    )

    tokens, labels, loss_mask, attention_mask, position_ids = training._get_batch(
        neox_args=neox_args,
        tokenizer=neox_args.tokenizer,
        keys=["text", "label"],
        data=data,
        datatype=torch.int64,
    )

    assert torch.equal(tokens, torch.tensor([[0, 1, 2, 3, 5, 6, 7, 8]]))
    assert attention_mask.shape == (1,)
    assert attention_mask.dtype == torch.bool
    assert not attention_mask.item()
    assert torch.equal(
        labels, torch.tensor([[1, 0, 3, 4, 6, 7, 8, 9]])
    )
    assert torch.equal(
        loss_mask, torch.tensor([[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    )
    assert torch.equal(
        position_ids, torch.tensor([[0, 1, 0, 1, 0, 1, 2, 3]])
    )


@pytest.mark.cpu
def test_get_batch_preserves_exact_behavior_when_disabled(local_broadcast):
    data = _batch()
    neox_args = _neox_args(enabled=False)
    neox_args.tokenizer.eod = 2
    neox_args.eod_mask_loss = True

    tokens, labels, loss_mask, attention_mask, position_ids = training._get_batch(
        neox_args=neox_args,
        tokenizer=neox_args.tokenizer,
        keys=["text"],
        data=data,
        datatype=torch.int64,
    )

    assert local_broadcast == [(["text"], torch.int64)]
    assert torch.equal(tokens, torch.tensor([[0, 1, 2, 3], [5, 6, 7, 8]]))
    assert torch.equal(labels, torch.tensor([[1, 2, 3, 4], [6, 7, 8, 9]]))
    assert torch.equal(
        loss_mask,
        torch.tensor([[1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]]),
    )
    assert torch.equal(
        position_ids,
        torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]),
    )
    assert torch.equal(
        attention_mask,
        torch.tensor(
            [
                [
                    [
                        [False, True, True, True],
                        [False, False, True, True],
                        [False, False, False, True],
                        [False, False, False, False],
                    ]
                ]
            ]
        ),
    )


@pytest.mark.cpu
def test_get_batch_sequential_preserves_packed_mask_sentinel(monkeypatch):
    forward_input = (
        torch.tensor([[0, 1, 2, 3]]),
        torch.tensor([[0, 1, 0, 1]]),
        torch.zeros(1, dtype=torch.bool),
    )

    monkeypatch.setattr(
        training,
        "get_ltor_masks_and_position_ids",
        lambda *args, **kwargs: pytest.fail("dense mask helper was called"),
    )

    result = training.get_batch_sequential(forward_input, _neox_args())

    assert result is forward_input


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        (
            "cu_seqlens",
            torch.zeros((2, 4), dtype=torch.int32),
            "expects cu_seqlens with shape",
        ),
        (
            "max_seqlen",
            torch.zeros((2, 1), dtype=torch.int32),
            "expects max_seqlen with shape",
        ),
        (
            "cu_seqlens",
            torch.zeros((2, 5), dtype=torch.int64),
            "expects cu_seqlens to be int32",
        ),
        (
            "max_seqlen",
            torch.zeros(2, dtype=torch.int64),
            "expects max_seqlen to be int32",
        ),
    ],
)
def test_packed_metadata_shape_validation(monkeypatch, field, value, error):
    data = _batch()
    data[field] = value

    monkeypatch.setattr(
        training.mpu,
        "broadcast_data",
        lambda keys, source, datatype: {key: source[key] for key in keys},
    )

    with pytest.raises(AssertionError, match=error):
        training._broadcast_packed_sequence_metadata(
            _neox_args(), data, data["text"]
        )
