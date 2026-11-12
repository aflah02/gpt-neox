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

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import megatron.model.transformer as transformer
from megatron.neox_arguments import NeoXArgs
from megatron.neox_arguments.arguments import _validate_norm_placement

pytestmark = pytest.mark.cpu

PLACEMENTS = (
    "pre_norm",
    "post_norm_outside_residual",
    "post_norm_inside_residual",
)


def arg_config(**overrides):
    config = dict(
        num_layers=1,
        hidden_size=4,
        num_attention_heads=2,
        max_position_embeddings=16,
        seq_length=8,
        train_micro_batch_size_per_gpu=1,
        lr=1.0e-3,
        global_num_gpus=1,
    )
    config.update(overrides)
    return config


def test_norm_placement_defaults_to_pre_norm():
    assert NeoXArgs.from_dict(arg_config()).norm_placement == "pre_norm"


@pytest.mark.parametrize("placement", PLACEMENTS)
def test_norm_placement_config(placement):
    args = NeoXArgs.from_dict(arg_config(norm_placement=placement))
    assert args.norm_placement == placement


def test_invalid_norm_placement():
    with pytest.raises(ValueError, match="Invalid norm_placement"):
        _validate_norm_placement("post_norm", False, False)


@pytest.mark.parametrize("placement", PLACEMENTS[1:])
@pytest.mark.parametrize("option", ("gpt_j_residual", "te_layernorm_mlp"))
def test_post_norm_rejects_incompatible_configuration(placement, option):
    with pytest.raises(ValueError, match=option + "=False"):
        NeoXArgs.from_dict(arg_config(norm_placement=placement, **{option: True}))

    # Direct layer construction must reject these before allocating parameters.
    args = SimpleNamespace(
        norm_placement=placement,
        gpt_j_residual=option == "gpt_j_residual",
        te_layernorm_mlp=option == "te_layernorm_mlp",
    )
    with pytest.raises(ValueError, match=option + "=False"):
        transformer.ParallelTransformerLayer(args, None, None, None, 1)


@pytest.mark.parametrize("gpt_j_residual", (False, True))
@pytest.mark.parametrize("te_layernorm_mlp", (False, True))
def test_pre_norm_accepts_existing_residual_and_mlp_options(
    gpt_j_residual, te_layernorm_mlp
):
    _validate_norm_placement("pre_norm", gpt_j_residual, te_layernorm_mlp)


class Projection(torch.nn.Module):
    """Small CPU sublayer exposing the same deferred-bias interface as NeoX."""

    def __init__(self, neox_args, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.tensor(
                [
                    [0.2, -0.3, 0.5, 0.7],
                    [-0.4, 0.6, 0.1, -0.2],
                    [0.8, 0.3, -0.7, 0.4],
                    [0.1, -0.5, 0.2, 0.6],
                ]
            )
        )
        self.bias = (
            torch.nn.Parameter(torch.tensor([0.3, -0.2, 0.5, 0.1]))
            if neox_args.use_bias_in_mlp
            else None
        )

    def forward(self, x):
        return torch.tanh(F.linear(x, self.weight)), self.bias


class Attention(Projection):
    def __init__(self, neox_args, use_cache=False, **kwargs):
        super().__init__(neox_args)
        self.use_cache = use_cache
        self.seen_past = []

    def forward(self, x, attention_mask, layer_past=None):
        self.seen_past.append(layer_past)
        output, bias = super().forward(x)
        if self.use_cache:
            return (output, ("present", len(self.seen_past))), bias
        return output, bias


def make_layer(monkeypatch, placement, norm="layernorm", pipe=False, **overrides):
    monkeypatch.setattr(transformer, "ParallelSelfAttention", Attention)
    monkeypatch.setattr(transformer, "ParallelMLP", Projection)
    monkeypatch.setattr(
        transformer.mpu.mappings, "reduce_from_model_parallel_region", lambda x: x
    )
    args = arg_config(
        norm=norm,
        norm_placement=placement,
        hidden_dropout=0.0,
        **overrides,
    )
    # Use the real model defaults without requiring a distributed process group.
    args = NeoXArgs.from_dict(args)
    layer_class = (
        transformer.ParallelTransformerLayerPipe
        if pipe
        else transformer.ParallelTransformerLayer
    )
    return layer_class(args, None, None, None, 1)


def reference_forward(layer, x):
    """Explicit sublayer equations, independent of the implementation helpers."""
    placement = layer.norm_placement
    for sublayer, norm in (
        (lambda value: layer.attention(value, None), layer.input_layernorm),
        (layer.mlp, layer.post_attention_layernorm),
    ):
        residual = x
        output, bias = sublayer(norm(x) if placement == "pre_norm" else x)
        if bias is not None:
            output = output + bias
        if placement == "post_norm_inside_residual":
            output = norm(output)
        x = residual + F.dropout(
            output, p=layer.hidden_dropout, training=layer.training
        )
        if placement == "post_norm_outside_residual":
            x = norm(x)
    return x


def assert_close(actual, expected):
    # Keep CPU CI compatibility with torch 1.8 (before torch.testing.assert_close).
    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, rtol=1.0e-5, atol=1.0e-6)


@pytest.mark.parametrize("placement", PLACEMENTS)
@pytest.mark.parametrize("norm", ("layernorm", "rmsnorm"))
@pytest.mark.parametrize("with_bias", (False, True))
@pytest.mark.parametrize("fused", (False, True))
def test_norm_placement_forward_and_backward(
    monkeypatch, placement, norm, with_bias, fused
):
    layer = make_layer(
        monkeypatch,
        placement,
        norm=norm,
        use_bias_in_mlp=with_bias,
        bias_dropout_fusion=fused,
    )
    reference = copy.deepcopy(layer)
    x = torch.tensor([[[0.2, -0.5, 1.3, 0.7]]], requires_grad=True)
    reference_x = x.detach().clone().requires_grad_(True)
    output = layer(x, None)
    expected = reference_forward(reference, reference_x)
    assert_close(output, expected)

    coefficients = torch.tensor([[[0.7, -0.4, 0.3, 1.2]]])
    (output * coefficients).sum().backward()
    (expected * coefficients).sum().backward()
    assert_close(x.grad, reference_x.grad)
    for (name, parameter), (reference_name, reference_parameter) in zip(
        layer.named_parameters(), reference.named_parameters()
    ):
        assert name == reference_name
        assert_close(parameter.grad, reference_parameter.grad)


@pytest.mark.parametrize("placement", PLACEMENTS[1:])
@pytest.mark.parametrize("with_bias", (False, True))
@pytest.mark.parametrize("fused", (False, True))
@pytest.mark.parametrize("training", (False, True))
def test_post_norm_dropout(monkeypatch, placement, with_bias, fused, training):
    layer = make_layer(
        monkeypatch,
        placement,
        use_bias_in_mlp=with_bias,
        bias_dropout_fusion=fused,
    )
    layer.hidden_dropout = 0.4
    layer.train(training)
    x = torch.arange(24, dtype=torch.float32).view(3, 2, 4) / 10.0
    torch.manual_seed(17)
    output = layer(x, None)
    torch.manual_seed(17)
    expected = reference_forward(layer, x)
    assert_close(output, expected)


@pytest.mark.parametrize("placement", PLACEMENTS)
@pytest.mark.parametrize("pipe", (False, True))
def test_norm_placement_cache_and_pipeline(monkeypatch, placement, pipe):
    layer = make_layer(monkeypatch, placement, pipe=pipe)
    layer.use_cache = layer.attention.use_cache = True
    x = torch.tensor([[[0.2, -0.5, 1.3, 0.7]]])
    mask = torch.zeros(1, 1, 1, 1, dtype=torch.bool)

    if pipe:
        output, output_mask = layer((x, mask))
        assert output_mask is mask
        layer((x, mask))
        assert layer.attention.seen_past == [None, ("present", 1)]
    else:
        output = layer(x, mask, layer_past=("explicit",))
        layer(x, mask)
        layer(x, mask, layer_past=("override",))
        assert layer.attention.seen_past == [
            ("explicit",),
            ("present", 1),
            ("override",),
        ]
    assert output.shape == x.shape
    assert layer.layer_past == ("present", len(layer.attention.seen_past))


@pytest.mark.parametrize("tied", (False, True))
def test_pre_norm_parallel_residual_is_preserved(monkeypatch, tied):
    layer = make_layer(monkeypatch, "pre_norm", gpt_j_residual=True, gpt_j_tied=tied)
    x = torch.tensor([[[0.2, -0.5, 1.3, 0.7]]])
    attention, attention_bias = layer.attention(layer.input_layernorm(x), None)
    mlp_input = layer.input_layernorm(x) if tied else layer.post_attention_layernorm(x)
    mlp, mlp_bias = layer.mlp(mlp_input)
    assert_close(layer(x, None), x + attention + attention_bias + mlp + mlp_bias)


def test_norm_placement_preserves_checkpoint_keys(monkeypatch):
    state = make_layer(monkeypatch, "pre_norm").state_dict()
    for placement in PLACEMENTS[1:]:
        layer = make_layer(monkeypatch, placement)
        assert set(layer.state_dict()) == set(state)
        layer.load_state_dict(state, strict=True)
