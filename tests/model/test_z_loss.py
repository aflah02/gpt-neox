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

import gc
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron import mpu
from megatron.model import gpt2_model
from megatron.training import forward_step, train_step_pipe

pytestmark = pytest.mark.cpu


def _sharded_logits(device, dtype):
    # Targets and token maxima lie on different shards. Large offsets exercise
    # logsumexp stability, and the largest offset belongs to a masked token.
    logits = torch.tensor(
        [[1, 3, 2, -1, 0, 1], [-2, 0, 1, 2, 3, 0], [1, 0, 2, 3, 0, 1]],
        device=device,
        dtype=dtype,
    )
    logits = (logits + logits.new_tensor([[100], [300], [-100]])).unsqueeze(0)
    labels = torch.tensor([[4, 1, 0]], device=device)
    mask = torch.tensor([[1.0, 0.0, 1.0]], device=device)
    return logits, labels, mask


@pytest.mark.parametrize("fused", [False, True], ids=["unfused", "fused"])
def test_cross_entropy_skips_log_z_when_disabled(monkeypatch, fused):
    monkeypatch.setattr(mpu.initialize, "_MODEL_PARALLEL_GROUP", object())
    monkeypatch.setattr(dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda group: 1)
    monkeypatch.setattr(dist, "all_reduce", lambda *args, **kwargs: None)
    logits, labels, mask = _sharded_logits("cpu", torch.float32)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as profile:
        gpt2_model.cross_entropy(logits, (labels, mask), _fused=fused)

    # CE needs one log. Restoring the subtracted maximum would add it to every token.
    ops = {event.key: event.count for event in profile.key_averages()}
    assert ops["aten::log"] == 1
    assert ops.get("aten::add", 0) == 0


def _lm_forward_step(shard, labels, mask, fused, fp16_lm_cross_entropy, coef):
    args = SimpleNamespace(
        train_impl="normal",
        memory_profiling=False,
        curriculum_learning=False,
        is_pipe_parallel=False,
        fp16_lm_cross_entropy=fp16_lm_cross_entropy,
        cross_entropy_fusion=fused,
        z_loss=coef,
    )
    if fp16_lm_cross_entropy:
        mask = mask.half()
    with patch(
        "megatron.training.get_batch", return_value=(None, labels, mask, None, None)
    ):
        return forward_step(
            None, Mock(return_value=shard), args, timers=None, is_train=True
        )


def _check_sharded_lm_loss(
    rank, world_size, device, dtype, fused, fp16_lm_cross_entropy, coefficient
):
    logits, labels, mask = _sharded_logits(device, dtype)
    reference = logits.detach().float().clone().requires_grad_()
    token_losses = F.cross_entropy(reference.transpose(1, 2), labels, reduction="none")
    penalty = reference.logsumexp(dim=-1).square()
    expected_ce = token_losses[mask.bool()].mean()
    expected_z_loss = coefficient * penalty[mask.bool()].mean()
    expected = expected_ce + expected_z_loss
    expected.backward()

    # The fp16 cross entropy computes in fp16, so it needs looser tolerances.
    tols = dict(rtol=2e-3, atol=2e-3) if fp16_lm_cross_entropy else {}

    shard = logits.chunk(world_size, dim=-1)[rank].contiguous().requires_grad_()
    loss, metrics = _lm_forward_step(
        shard, labels, mask, fused, fp16_lm_cross_entropy, coefficient
    )
    loss.backward()

    torch.testing.assert_close(loss.float(), expected.detach(), **tols)
    expected_grad = reference.grad.chunk(world_size, dim=-1)[rank].to(dtype)
    torch.testing.assert_close(shard.grad, expected_grad, **tols)
    if coefficient:
        torch.testing.assert_close(
            metrics["lm_loss_without_z_loss"].float(), expected_ce.detach(), **tols
        )
        torch.testing.assert_close(metrics["z_loss"], expected_z_loss.detach(), **tols)
    else:
        assert "lm_loss_without_z_loss" not in metrics
        assert "z_loss" not in metrics

    # Evaluation runs under no_grad and reports plain cross entropy.
    shard = logits.chunk(world_size, dim=-1)[rank].contiguous()
    with torch.no_grad():
        eval_loss, eval_metrics = _lm_forward_step(
            shard, labels, mask, fused, fp16_lm_cross_entropy, coefficient
        )
    torch.testing.assert_close(eval_loss.float(), expected_ce.detach(), **tols)
    assert "lm_loss_without_z_loss" not in eval_metrics
    assert "z_loss" not in eval_metrics


def _check_log_z_kernels(rank, world_size, device):
    from megatron.fused_kernels.fused_cross_entropy import (
        fused_vocab_parallel_cross_entropy,
    )

    logits, labels, _ = _sharded_logits(device, torch.float32)
    expected_log_z = logits.logsumexp(dim=-1)
    softmax_shard = logits.softmax(dim=-1).chunk(world_size, dim=-1)[rank]
    kernels = {
        "mpu": lambda x, **kwargs: mpu.vocab_parallel_cross_entropy(
            x, labels, **kwargs
        ),
        "fused": lambda x, **kwargs: fused_vocab_parallel_cross_entropy(
            x, labels, mpu.get_model_parallel_group(), **kwargs
        ),
    }
    for name, kernel in kernels.items():
        shard = logits.chunk(world_size, dim=-1)[rank].contiguous()

        # Both kernels overwrite their input, so they get a copy.
        plain_input = shard.clone().requires_grad_()
        plain_loss = kernel(plain_input.clone())
        plain_loss.sum().backward()

        # Only the loss used: the gradient matches plain cross entropy.
        loss_input = shard.clone().requires_grad_()
        loss, log_z = kernel(loss_input.clone(), return_log_z=True)
        torch.testing.assert_close(loss, plain_loss, msg=name)
        torch.testing.assert_close(log_z, expected_log_z, msg=name)
        loss.sum().backward()
        torch.testing.assert_close(loss_input.grad, plain_input.grad, msg=name)

        # Only log(Z) used: its gradient is the softmax.
        log_z_input = shard.clone().requires_grad_()
        _, log_z = kernel(log_z_input.clone(), return_log_z=True)
        log_z.sum().backward()
        torch.testing.assert_close(log_z_input.grad, softmax_shard, msg=name)


def _run_tensor_parallel_lm_loss(rank, store_path, world_size, use_cuda):
    device = torch.device("cuda", rank) if use_cuda else torch.device("cpu")
    if use_cuda:
        torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl" if use_cuda else "gloo",
        init_method=store_path.as_uri(),
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        mpu.initialize_model_parallel(world_size)
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for fused in (False, True):
                for coefficient in (0.0, 0.025):
                    _check_sharded_lm_loss(
                        rank, world_size, device, dtype, fused, False, coefficient
                    )
        for coefficient in (0.0, 0.025):
            _check_sharded_lm_loss(
                rank, world_size, device, torch.float16, False, True, coefficient
            )
        _check_log_z_kernels(rank, world_size, device)
    finally:
        # Free the test's reference cycles (the Mock models hold the logits)
        # first. Without this, destroy_process_group intermittently hangs on gloo.
        gc.collect()
        mpu.destroy_model_parallel()
        dist.destroy_process_group()


def test_tensor_parallel_lm_loss_matches_dense_loss_and_gradients(tmp_path):
    world_size = 2
    torch.multiprocessing.spawn(
        _run_tensor_parallel_lm_loss,
        args=(
            tmp_path / "tp-store",
            world_size,
            torch.cuda.device_count() >= world_size,
        ),
        nprocs=world_size,
    )


def _pipeline_model(monkeypatch, z_loss, cross_entropy):
    from megatron.neox_arguments import NeoXArgs
    from tests.common import BASE_CONFIG

    args = NeoXArgs.from_dict(
        dict(BASE_CONFIG, global_num_gpus=1, z_loss=z_loss, cross_entropy_fusion=True)
    )
    pipeline_init = Mock(return_value=None)
    monkeypatch.setattr(gpt2_model.PipelineModule, "__init__", pipeline_init)
    monkeypatch.setattr(gpt2_model, "cross_entropy", cross_entropy)
    model = gpt2_model.GPT2ModelPipe(args)
    return model, pipeline_init.call_args.kwargs["loss_fn"]


def test_pipeline_loss_reports_z_loss_parts(monkeypatch):
    loss = torch.tensor(3.0, requires_grad=True)
    loss_without_z_loss, z_loss = torch.tensor(2.5), torch.tensor(0.5)
    cross_entropy = Mock(return_value=(loss, loss_without_z_loss, z_loss))
    model, loss_fn = _pipeline_model(monkeypatch, 0.125, cross_entropy)

    # Stages without the loss report placeholders under the same names.
    assert list(model.get_additional_losses()) == ["lm_loss_without_z_loss", "z_loss"]

    logits, labels = object(), object()
    assert loss_fn(logits, labels) is loss
    cross_entropy.assert_called_once_with(
        logits, labels, _fp16=False, _fused=True, z_loss=0.125, return_loss_parts=True
    )
    parts = model.get_additional_losses()
    assert list(parts) == ["lm_loss_without_z_loss", "z_loss"]
    torch.testing.assert_close(parts["lm_loss_without_z_loss"], loss_without_z_loss)
    torch.testing.assert_close(parts["z_loss"], z_loss)


def test_pipeline_loss_reports_nothing_without_z_loss(monkeypatch):
    loss = torch.tensor(3.0)
    cross_entropy = Mock(return_value=(loss, None, None))
    model, loss_fn = _pipeline_model(monkeypatch, 0.0, cross_entropy)

    assert loss_fn(object(), object()) is loss
    assert model.get_additional_losses() is None


@pytest.mark.parametrize("with_z_loss", [False, True])
def test_train_step_pipe_logs_additional_losses(with_z_loss):
    loss = torch.tensor(3.0)
    parts = {"lm_loss_without_z_loss": torch.tensor(2.5), "z_loss": torch.tensor(0.5)}
    engine = SimpleNamespace(
        train_batch=Mock(return_value=loss),
        agg_additional_losses=parts if with_z_loss else None,
    )
    loss_dict = train_step_pipe(SimpleNamespace(deepspeed=True), Mock(), engine, None)
    assert loss_dict == (
        {"lm_loss": loss, **parts} if with_z_loss else {"lm_loss": loss}
    )


@pytest.mark.parametrize(
    "overrides",
    [dict(z_loss=-0.1), dict(z_loss=0.1, train_impl="dpo")],
    ids=["negative", "unsupported-train-impl"],
)
def test_invalid_z_loss_is_rejected(overrides):
    from megatron.neox_arguments import NeoXArgs
    from tests.common import BASE_CONFIG

    with pytest.raises(ValueError, match="z_loss"):
        NeoXArgs.from_dict(dict(BASE_CONFIG, global_num_gpus=1, **overrides))


def _reward_model_forward_step(scores, coefficient):
    args = SimpleNamespace(
        train_impl="rm",
        is_pipe_parallel=False,
        memory_profiling=False,
        z_loss=coefficient,
    )
    mask = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    batch = (None, None, mask, None, None, None)
    with patch("megatron.training.get_batch", return_value=batch):
        return forward_step(None, Mock(return_value=scores), args, timers=None)


@pytest.mark.parametrize("coefficient", [0.0, 0.125])
def test_reward_model_z_loss_uses_selected_scores(coefficient):
    scores = torch.tensor([[[8.0], [2.0]], [[-1.0], [6.0]]], requires_grad=True)
    loss, metrics = _reward_model_forward_step(scores, coefficient)

    reference = scores.detach().clone().requires_grad_()
    chosen, rejected = reference[0, 1, 0], reference[1, 0, 0]
    expected_rm_loss = F.softplus(rejected - chosen)
    expected_z_loss = coefficient * (chosen.square() + rejected.square())
    expected = expected_rm_loss + expected_z_loss
    expected.backward()
    loss.backward()
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(scores.grad, reference.grad)
    if coefficient:
        torch.testing.assert_close(
            metrics["lm_loss_without_z_loss"], expected_rm_loss.detach()
        )
        torch.testing.assert_close(metrics["z_loss"], expected_z_loss.detach())
    else:
        assert "lm_loss_without_z_loss" not in metrics
        assert "z_loss" not in metrics

    # Evaluation runs under no_grad and leaves the z-loss out.
    with torch.no_grad():
        eval_loss, eval_metrics = _reward_model_forward_step(
            scores.detach(), coefficient
        )
    torch.testing.assert_close(eval_loss, expected_rm_loss.detach())
    assert "lm_loss_without_z_loss" not in eval_metrics
    assert "z_loss" not in eval_metrics
