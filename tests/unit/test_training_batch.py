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
def test_get_batch_flattens_sequence_aligned_tensors_when_enabled(monkeypatch):
    data = _batch()

    monkeypatch.setattr(
        training.mpu,
        "broadcast_data",
        lambda keys, source, datatype: {key: source[key] for key in keys},
    )

    tokens, labels, loss_mask, _, position_ids = training._get_batch(
        neox_args=_neox_args(),
        tokenizer=SimpleNamespace(eod=-1),
        keys=["text"],
        data=data,
        datatype=torch.int64,
    )

    assert torch.equal(
        tokens, torch.tensor([[0, 1, 2, 3, 5, 6, 7, 8]])
    )
    assert torch.equal(
        labels, torch.tensor([[1, 2, 3, 4, 6, 7, 8, 9]])
    )
    assert torch.equal(loss_mask, torch.ones((1, 8)))
    assert torch.equal(
        position_ids, torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
    )


@pytest.mark.cpu
def test_get_batch_does_not_broadcast_metadata_when_disabled(monkeypatch):
    data = _batch()
    calls = []

    def broadcast_data(keys, source, datatype):
        calls.append((keys, datatype))
        return {key: source[key] for key in keys}

    monkeypatch.setattr(training.mpu, "broadcast_data", broadcast_data)

    tokens, labels, loss_mask, _, position_ids = training._get_batch(
        neox_args=_neox_args(enabled=False),
        tokenizer=SimpleNamespace(eod=-1),
        keys=["text"],
        data=data,
        datatype=torch.int64,
    )

    assert calls == [(["text"], torch.int64)]
    assert tokens.shape == labels.shape == loss_mask.shape == position_ids.shape == (
        2,
        4,
    )


@pytest.mark.cpu
def test_flatten_packed_sequence_tensors_preserves_row_major_alignment():
    tokens = torch.tensor([[10, 11, 12], [20, 21, 22]])
    labels = torch.tensor([[11, 12, 13], [21, 22, 23]])
    loss_mask = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    position_ids = torch.arange(3).unsqueeze(0).expand(2, -1)

    flat_tokens, flat_labels, flat_loss_mask, flat_position_ids = (
        training._flatten_packed_sequence_tensors(
            tokens, labels, loss_mask, position_ids
        )
    )

    assert torch.equal(flat_tokens, torch.tensor([[10, 11, 12, 20, 21, 22]]))
    assert torch.equal(flat_labels, torch.tensor([[11, 12, 13, 21, 22, 23]]))
    assert torch.equal(
        flat_loss_mask, torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0, 1.0]])
    )
    assert torch.equal(
        flat_position_ids, torch.tensor([[0, 1, 2, 0, 1, 2]])
    )
    assert all(
        tensor.shape == (1, 6) and tensor.is_contiguous()
        for tensor in (
            flat_tokens,
            flat_labels,
            flat_loss_mask,
            flat_position_ids,
        )
    )


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
def test_merge_packed_sequence_metadata_builds_microbatch_boundaries():
    document_lengths = torch.tensor([1, 2, 3, 4, 2], dtype=torch.int32)
    max_seqlen = torch.tensor([3, 4], dtype=torch.int32)

    merged_cu_seqlens, microbatch_max_seqlen = (
        training._merge_packed_sequence_metadata(document_lengths, max_seqlen)
    )

    assert torch.equal(
        merged_cu_seqlens,
        torch.tensor([0, 1, 3, 6, 10, 12], dtype=torch.int32),
    )
    assert merged_cu_seqlens.dtype == torch.int32
    assert microbatch_max_seqlen.dim() == 0
    assert microbatch_max_seqlen.dtype == torch.int32
    assert microbatch_max_seqlen == document_lengths.max()


@pytest.mark.cpu
def test_merge_packed_sequence_metadata_handles_one_sample():
    document_lengths = torch.tensor([2, 2], dtype=torch.int32)
    max_seqlen = torch.tensor([2], dtype=torch.int32)

    merged_cu_seqlens, microbatch_max_seqlen = (
        training._merge_packed_sequence_metadata(document_lengths, max_seqlen)
    )

    assert torch.equal(
        merged_cu_seqlens, torch.tensor([0, 2, 4], dtype=torch.int32)
    )
    assert microbatch_max_seqlen == 2


@pytest.mark.cpu
def test_pad_packed_sequence_metadata_uses_fixed_transport_shape():
    many_documents = torch.tensor(
        [0, 1, 3, 6, 10, 12], dtype=torch.int32
    )
    few_documents = torch.tensor([0, 6, 12], dtype=torch.int32)

    padded_many, many_count = training._pad_packed_sequence_metadata(
        many_documents, batch_size=2, sequence_length=6
    )
    padded_few, few_count = training._pad_packed_sequence_metadata(
        few_documents, batch_size=2, sequence_length=6
    )

    assert padded_many.shape == padded_few.shape == (13,)
    assert torch.equal(
        padded_many,
        torch.tensor(
            [0, 1, 3, 6, 10, 12, 12, 12, 12, 12, 12, 12, 12],
            dtype=torch.int32,
        ),
    )
    assert torch.equal(
        padded_few,
        torch.tensor(
            [0, 6, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
            dtype=torch.int32,
        ),
    )
    assert padded_many.dtype == padded_few.dtype == torch.int32
    assert many_count.shape == few_count.shape == ()
    assert many_count.dtype == few_count.dtype == torch.int32
    assert many_count == 5
    assert few_count == 2
    assert torch.equal(padded_many[: many_count + 1], many_documents)
    assert torch.equal(padded_few[: few_count + 1], few_documents)


@pytest.mark.cpu
def test_pad_packed_sequence_metadata_handles_maximum_document_count():
    merged_cu_seqlens = torch.arange(13, dtype=torch.int32)

    padded_cu_seqlens, num_documents = (
        training._pad_packed_sequence_metadata(
            merged_cu_seqlens, batch_size=2, sequence_length=6
        )
    )

    assert torch.equal(padded_cu_seqlens, merged_cu_seqlens)
    assert num_documents == 12


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
