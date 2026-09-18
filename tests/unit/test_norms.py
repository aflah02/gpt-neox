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

from megatron.model.norms import get_norm


def norm_args(norm, use_bias_in_norms):
    return SimpleNamespace(
        norm=norm,
        use_bias_in_norms=use_bias_in_norms,
        layernorm_epsilon=1.0e-5,
        rms_norm_epsilon=1.0e-6,
        layernorm_fusion=False,
        rmsnorm_fusion=False,
    )


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("norm_type", "bias_parameter"),
    [("layernorm", "bias"), ("rmsnorm", "offset")],
)
@pytest.mark.parametrize("use_bias", [True, False])
def test_use_bias_in_norms_controls_norm_parameters(
    norm_type, bias_parameter, use_bias
):
    norm_class, eps = get_norm(norm_args(norm_type, use_bias))
    norm = norm_class(8, eps=eps)

    parameters = dict(norm.named_parameters())
    assert (bias_parameter in parameters) is use_bias
    assert (bias_parameter in norm.state_dict()) is use_bias


@pytest.mark.cpu
def test_biasless_layernorm_matches_zero_bias_layernorm():
    biased_class, eps = get_norm(norm_args("layernorm", True))
    biasless_class, _ = get_norm(norm_args("layernorm", False))
    biased = biased_class(4, eps=eps)
    biasless = biasless_class(4, eps=eps)
    inputs = torch.randn(2, 3, 4)

    with torch.no_grad():
        biased.weight.copy_(biasless.weight)
        biased.bias.zero_()

    torch.testing.assert_close(biasless(inputs), biased(inputs))
