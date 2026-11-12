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

import pytest
import torch
import torch.nn.functional as F

import megatron.model.transformer as transformer
import megatron.mpu.initialize as mpu_initialize
import megatron.mpu.layers as parallel_layers
from megatron.neox_arguments import NeoXArgs
from megatron.model.utils import mark_norms_for_sequence_parallel_grad_sync
from tests.unit.test_norm_placement import arg_config, assert_close, make_layer

pytestmark = pytest.mark.cpu
RESIDUAL_PATHS = ((False, False), (True, False), (True, True))


def sandwich_config(**overrides):
    return arg_config(norm_placement="sandwich_norm", **overrides)


@pytest.mark.parametrize("norm", ("layernorm", "rmsnorm"))
def test_plain_sandwich_norm_config(norm):
    args = NeoXArgs.from_dict(sandwich_config(norm=norm))
    assert args.norm_placement == "sandwich_norm"
    assert args.sandwich_norm_depth_init is None


@pytest.mark.parametrize("norm_type", ("layernorm", "rmsnorm"))
def test_plain_sandwich_norm_gains_are_unscaled(monkeypatch, norm_type):
    layer = make_layer(monkeypatch, "sandwich_norm", norm=norm_type, num_layers=9)
    for norm in (layer.attention_output_layernorm, layer.mlp_output_layernorm):
        gain = norm.scale if norm_type == "rmsnorm" else norm.weight
        assert_close(gain, torch.ones(4))


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"norm": "scalenorm"}, "native"),
        ({"norm": "non_parametric_layernorm"}, "native"),
        ({"norm": "te_rmsnorm"}, "native"),
        ({"moe_num_experts": 2}, "dense MLPs"),
        *[
            ({name: True}, name)
            for name in (
                "te_mha",
                "te_fp8_mha",
                "te_layernorm_mlp",
                "te_rowparallel",
                "te_columnparallel",
            )
        ],
        *[
            ({"attention_config": [[[kind], 1]]}, "global/flash")
            for kind in ("mamba", "rwkv", "gmlp", "amlp", "local")
        ],
    ],
)
def test_sandwich_norm_rejects_unsupported_blocks(overrides, message):
    with pytest.raises(ValueError, match=message):
        NeoXArgs.from_dict(sandwich_config(**overrides))


@pytest.mark.parametrize(
    "depth_init",
    [
        {},
        {"attention": 1.0},
        {"attention": 1.0, "mlp": 1.0, "extra": 1.0},
        *[
            {"attention": value, "mlp": 1.0}
            for value in (-0.1, float("nan"), float("inf"), True, "1.0", None)
        ],
        {"attention": 1.0, "mlp": -0.1},
        "attention=1,mlp=1",
        [1.0, 1.0],
    ],
)
def test_sandwich_norm_rejects_invalid_depth_initialization(depth_init):
    with pytest.raises(ValueError):
        NeoXArgs.from_dict(
            sandwich_config(norm="rmsnorm", sandwich_norm_depth_init=depth_init)
        )


@pytest.mark.parametrize(
    "placement", ("pre_norm", "post_norm_inside_residual", "post_norm_outside_residual")
)
def test_depth_initialization_requires_sandwich_norm(placement):
    with pytest.raises(ValueError, match="requires `norm_placement='sandwich_norm'`"):
        NeoXArgs.from_dict(
            arg_config(
                norm="rmsnorm",
                norm_placement=placement,
                sandwich_norm_depth_init={"attention": 1, "mlp": 1},
            )
        )


def test_depth_initialization_requires_rmsnorm():
    with pytest.raises(ValueError, match="requires `norm='rmsnorm'`"):
        NeoXArgs.from_dict(
            sandwich_config(
                norm="layernorm", sandwich_norm_depth_init={"attention": 1, "mlp": 1}
            )
        )


@pytest.mark.parametrize(
    "constants",
    (
        {"attention": 1, "mlp": 1},
        {"attention": 0.283, "mlp": 0.432},
        {"attention": 0, "mlp": 0},
    ),
)
def test_depth_initialization_uses_total_model_depth(monkeypatch, constants):
    layer = make_layer(
        monkeypatch,
        "sandwich_norm",
        norm="rmsnorm",
        num_layers=9,
        sandwich_norm_depth_init=constants,
    )
    assert_close(
        layer.attention_output_layernorm.scale,
        torch.full((4,), constants["attention"] / 3.0),
    )
    assert_close(
        layer.mlp_output_layernorm.scale, torch.full((4,), constants["mlp"] / 3.0)
    )
    assert_close(layer.input_layernorm.scale, torch.ones(4))
    assert_close(layer.post_attention_layernorm.scale, torch.ones(4))
    assert (
        len(
            {
                norm.scale.data_ptr()
                for norm in (
                    layer.input_layernorm,
                    layer.post_attention_layernorm,
                    layer.attention_output_layernorm,
                    layer.mlp_output_layernorm,
                )
            }
        )
        == 4
    )


def sandwich_reference(layer, x):
    attention_input = layer.input_layernorm(x)
    attention, attention_bias = layer.attention(attention_input, None)
    if attention_bias is not None:
        attention = attention + attention_bias
    attention = layer.attention_output_layernorm(attention)
    residual = x + F.dropout(attention, layer.hidden_dropout, layer.training)
    if layer.gpt_j_residual:
        mlp_input = (
            attention_input if layer.gpt_j_tied else layer.post_attention_layernorm(x)
        )
    else:
        mlp_input = layer.post_attention_layernorm(residual)
    mlp, mlp_bias = layer.mlp(mlp_input)
    if mlp_bias is not None:
        mlp = mlp + mlp_bias
    mlp = layer.mlp_output_layernorm(mlp)
    return residual + F.dropout(mlp, layer.hidden_dropout, layer.training)


@pytest.mark.parametrize("parallel,tied", RESIDUAL_PATHS)
@pytest.mark.parametrize("norm", ("layernorm", "rmsnorm"))
@pytest.mark.parametrize(
    "attention_bias,mlp_bias",
    ((False, False), (False, True), (True, False), (True, True)),
)
@pytest.mark.parametrize("training", (False, True))
def test_sandwich_norm_forward_backward_and_dropout(
    monkeypatch, parallel, tied, norm, attention_bias, mlp_bias, training
):
    layer = make_layer(
        monkeypatch,
        "sandwich_norm",
        norm=norm,
        gpt_j_residual=parallel,
        gpt_j_tied=tied,
        use_bias_in_attn_linear=attention_bias,
        use_bias_in_mlp=mlp_bias,
        bias_dropout_fusion=True,
    )
    layer.hidden_dropout = 0.35
    layer.train(training)
    reference = copy.deepcopy(layer)
    x = torch.linspace(-1.0, 2.0, 24).view(3, 2, 4).requires_grad_(True)
    reference_x = x.detach().clone().requires_grad_(True)
    torch.manual_seed(31)
    output = layer(x, None)
    torch.manual_seed(31)
    expected = sandwich_reference(reference, reference_x)
    assert_close(output, expected)
    coefficients = torch.linspace(0.2, 1.1, 24).view_as(output)
    (output * coefficients).sum().backward()
    (expected * coefficients).sum().backward()
    assert_close(x.grad, reference_x.grad)
    for (name, parameter), (reference_name, reference_parameter) in zip(
        layer.named_parameters(), reference.named_parameters()
    ):
        assert name == reference_name
        if reference_parameter.grad is None:
            assert parameter.grad is None
        else:
            assert_close(parameter.grad, reference_parameter.grad)
    assert layer.attention.parallel_output is False
    assert layer.mlp.parallel_output is False


def test_depth_initialized_gains_are_restored_from_checkpoint(monkeypatch):
    layer = make_layer(
        monkeypatch,
        "sandwich_norm",
        norm="rmsnorm",
        num_layers=9,
        sandwich_norm_depth_init={"attention": 0.283, "mlp": 0.432},
    )
    with torch.no_grad():
        layer.attention_output_layernorm.scale.fill_(0.37)
        layer.mlp_output_layernorm.scale.fill_(-0.12)
    restored = make_layer(
        monkeypatch,
        "sandwich_norm",
        norm="rmsnorm",
        num_layers=16,
        sandwich_norm_depth_init={"attention": 1, "mlp": 1},
    )
    restored.load_state_dict(layer.state_dict(), strict=True)
    x = torch.tensor([[[0.2, -0.5, 1.3, 0.7]]])
    assert_close(restored(x, None), layer(x, None))
    assert_close(restored(x, None), sandwich_reference(restored, x))
    assert_close(restored.attention_output_layernorm.scale, torch.full((4,), 0.37))
    assert_close(restored.mlp_output_layernorm.scale, torch.full((4,), -0.12))


@pytest.mark.parametrize("parallel,tied", RESIDUAL_PATHS)
@pytest.mark.parametrize("pipe", (False, True))
def test_sandwich_norm_cache_and_pipeline(monkeypatch, parallel, tied, pipe):
    layer = make_layer(
        monkeypatch,
        "sandwich_norm",
        pipe=pipe,
        gpt_j_residual=parallel,
        gpt_j_tied=tied,
    )
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
        assert layer.attention.seen_past == [("explicit",), ("present", 1)]
    assert output.shape == x.shape
    assert layer.layer_past == ("present", 2)


@pytest.mark.parametrize("sequence_parallel", (False, True))
def test_native_gpt_j_projections_reduce_each_branch(monkeypatch, sequence_parallel):
    # Instantiate actual native projections with CPU weights and TP=2. Mock only
    # the collectives, so this verifies reduction policy without a process group.
    monkeypatch.setattr(mpu_initialize, "_MPU_WORLD_SIZE", 2)
    monkeypatch.setattr(mpu_initialize, "_MPU_RANK", 0)
    args = NeoXArgs.from_dict(
        sandwich_config(
            norm="rmsnorm",
            global_num_gpus=2,
            model_parallel_size=2,
            sequence_parallel=sequence_parallel,
            gpt_j_residual=True,
            use_cpu_initialization=True,
            hidden_dropout=0.0,
        )
    )
    layer = transformer.ParallelTransformerLayer(
        args, None, torch.nn.init.xavier_normal_, torch.nn.init.xavier_normal_, 0
    )
    assert not hasattr(layer, "reduce")
    calls = []
    other_rank_output = torch.tensor([0.4, -0.1, 0.6, -0.3])

    def reduce_branch(partial):
        calls.append(partial.clone())
        result = partial + other_rank_output
        return result.chunk(2, dim=0)[0].contiguous() if sequence_parallel else result

    monkeypatch.setattr(
        parallel_layers, "reduce_from_model_parallel_region", reduce_branch
    )
    monkeypatch.setattr(
        parallel_layers, "reduce_scatter_to_sequence_parallel_region", reduce_branch
    )
    for projection, norm in (
        (layer.attention.dense, layer.attention_output_layernorm),
        (layer.mlp.linear2, layer.mlp_output_layernorm),
    ):
        assert projection.parallel_output is False
        inputs = torch.randn(4, 1, projection.input_size_per_partition)
        partial = F.linear(inputs, projection.weight)
        output, bias = projection(inputs)
        complete = partial + other_rank_output
        if sequence_parallel:
            complete = complete.chunk(2, dim=0)[0].contiguous()
        residual = torch.randn_like(complete)
        expected = residual + norm(complete + bias)
        assert_close(
            layer._output_norm_residual(output, bias, residual, norm), expected
        )
    assert len(calls) == 2


def test_sandwich_output_norms_get_sequence_parallel_grad_hooks(monkeypatch):
    layer = make_layer(monkeypatch, "sandwich_norm", norm="rmsnorm")
    layer.neox_args.sequence_parallel = True
    mark_norms_for_sequence_parallel_grad_sync(layer, layer.neox_args)
    for norm in (layer.attention_output_layernorm, layer.mlp_output_layernorm):
        assert norm.scale._backward_hooks
