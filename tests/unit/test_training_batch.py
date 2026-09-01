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


@pytest.mark.cpu
def test_get_batch_broadcasts_packed_metadata_as_int32(monkeypatch):
    data = _batch()
    calls = []

    def broadcast_data(keys, source, datatype):
        calls.append((keys, datatype))
        return {key: source[key] for key in keys}

    monkeypatch.setattr(training.mpu, "broadcast_data", broadcast_data)

    training._get_batch(
        neox_args=_neox_args(),
        tokenizer=SimpleNamespace(eod=-1),
        keys=["text"],
        data=data,
        datatype=torch.int64,
    )

    assert calls == [
        (["text"], torch.int64),
        (["cu_seqlens", "max_seqlen"], torch.int32),
    ]


@pytest.mark.cpu
def test_get_batch_does_not_broadcast_metadata_when_disabled(monkeypatch):
    data = _batch()
    calls = []

    def broadcast_data(keys, source, datatype):
        calls.append((keys, datatype))
        return {key: source[key] for key in keys}

    monkeypatch.setattr(training.mpu, "broadcast_data", broadcast_data)

    training._get_batch(
        neox_args=_neox_args(enabled=False),
        tokenizer=SimpleNamespace(eod=-1),
        keys=["text"],
        data=data,
        datatype=torch.int64,
    )

    assert calls == [(["text"], torch.int64)]


@pytest.mark.cpu
def test_get_packed_sequence_document_lengths_removes_collation_padding():
    cu_seqlens = torch.tensor(
        [
            [0, 1, 3, 6, 6, 6, 6],
            [0, 4, 6, 6, 6, 6, 6],
        ],
        dtype=torch.int32,
    )

    document_lengths = training._get_packed_sequence_document_lengths(cu_seqlens)

    assert torch.equal(
        document_lengths, torch.tensor([1, 2, 3, 4, 2], dtype=torch.int32)
    )


@pytest.mark.cpu
def test_get_packed_sequence_document_lengths_preserves_single_document_samples():
    cu_seqlens = torch.tensor(
        [
            [0, 4, 4, 4, 4],
            [0, 2, 4, 4, 4],
        ],
        dtype=torch.int32,
    )

    document_lengths = training._get_packed_sequence_document_lengths(cu_seqlens)

    assert torch.equal(
        document_lengths, torch.tensor([4, 2, 2], dtype=torch.int32)
    )


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
