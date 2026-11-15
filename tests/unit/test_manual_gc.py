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
import weakref
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from megatron import training
from megatron.neox_arguments import NeoXArgs
from tests.common import BASE_CONFIG


@pytest.fixture(autouse=True)
def automatic_gc_enabled():
    """Start every test with automatic collection on, and never leak a changed state."""
    was_enabled = gc.isenabled()
    gc.enable()
    yield
    if was_enabled:
        gc.enable()
    else:
        gc.disable()


def make_args(start_iteration=0, **overrides):
    config = deepcopy(BASE_CONFIG)
    # Avoid hardware discovery in NeoXArgs.calculate_derived during CPU unit tests.
    config["global_num_gpus"] = 1
    config.update(overrides)
    neox_args = NeoXArgs.from_dict(config)
    neox_args.iteration = start_iteration
    neox_args.do_valid = True
    neox_args.save = None
    return neox_args


def run_train(neox_args, on_train_step=None, on_validation=None, **train_kwargs):
    """Run `train()` with the training step and validation replaced by recorders.

    Returns the ordered list of events seen: ("collect", iteration) for every
    `gc.collect()` call, and ("train_step" | "validation", gc.isenabled()) for every
    training step and validation pass.
    """
    events = []
    real_collect = gc.collect

    def fake_train_step(**_kwargs):
        events.append(("train_step", gc.isenabled()))
        if on_train_step is not None:
            on_train_step()
        return {}, 0

    def fake_validation(**_kwargs):
        events.append(("validation", gc.isenabled()))
        if on_validation is not None:
            on_validation()

    def recording_collect(*args, **kwargs):
        events.append(("collect", neox_args.iteration))
        return real_collect(*args, **kwargs)

    kwargs = dict(
        neox_args=neox_args,
        timers=MagicMock(),
        model=MagicMock(),
        reference_model=None,
        optimizer=SimpleNamespace(param_groups=[], overflow=False, cur_scale=1.0),
        lr_scheduler=MagicMock(),
        train_data_iterator=MagicMock(),
        valid_data_iterator=MagicMock(),
    )
    kwargs.update(train_kwargs)

    with patch.object(
        training, "train_step", side_effect=fake_train_step
    ), patch.object(
        training, "evaluate_and_print_results", side_effect=fake_validation
    ), patch.object(
        training, "evaluate_named_data_iterators", side_effect=fake_validation
    ), patch.object(
        training, "training_log", return_value=False
    ), patch.object(
        training.gc, "collect", side_effect=recording_collect
    ):
        training.train(**kwargs)

    return events


def collections(events):
    return [iteration for kind, iteration in events if kind == "collect"]


def collector_states(events, kind):
    return [enabled for event_kind, enabled in events if event_kind == kind]


@pytest.mark.cpu
def test_manual_gc_interval_defaults_to_automatic_gc():
    neox_args = make_args(train_iters=6)
    assert neox_args.manual_gc_interval == 0

    events = run_train(neox_args)

    assert collections(events) == []
    assert collector_states(events, "train_step") == [True] * 6
    assert gc.isenabled()


@pytest.mark.cpu
def test_manual_gc_interval_rejects_negative_values():
    with pytest.raises(ValueError, match="manual_gc_interval"):
        make_args(manual_gc_interval=-1)


@pytest.mark.cpu
def test_manual_gc_collects_on_schedule_with_automatic_gc_disabled():
    events = run_train(make_args(train_iters=10, manual_gc_interval=4))

    # One full collection before the first step, then one every 4 iterations.
    assert events[0] == ("collect", 0)
    assert collections(events) == [0, 4, 8]
    assert collector_states(events, "train_step") == [False] * 10
    assert gc.isenabled()


@pytest.mark.cpu
def test_manual_gc_frees_reference_cycles_only_at_scheduled_iterations():
    class Node:
        pass

    cycles = []
    alive_at_step_start = []

    def leak_a_cycle():
        alive_at_step_start.append(sum(ref() is not None for ref in cycles))
        node = Node()
        node.itself = node
        cycles.append(weakref.ref(node))

    run_train(
        make_args(train_iters=7, manual_gc_interval=3), on_train_step=leak_a_cycle
    )

    assert alive_at_step_start == [0, 1, 2, 0, 1, 2, 0]


@pytest.mark.cpu
def test_manual_gc_schedule_is_preserved_on_resume():
    # Resuming at iteration 6 must keep collecting on multiples of 4, not on 10 and 14.
    events = run_train(
        make_args(start_iteration=6, train_iters=13, manual_gc_interval=4)
    )

    assert events[0] == ("collect", 6)
    assert collections(events)[1:] == [8, 12]


@pytest.mark.cpu
@pytest.mark.parametrize("is_pipe_parallel", [False, True])
def test_manual_gc_counts_iterations_not_gradient_accumulation_steps(is_pipe_parallel):
    neox_args = make_args(
        train_iters=6, manual_gc_interval=2, gradient_accumulation_steps=3
    )
    neox_args.is_pipe_parallel = is_pipe_parallel
    model = MagicMock()
    model.train_batch.return_value = torch.tensor(1.0)
    model.optimizer.overflow = False
    collected_at = []

    def fake_collect(*_args, **_kwargs):
        collected_at.append(neox_args.iteration)
        return 0

    # Unlike `run_train`, this exercises the real `train_step`.
    with patch.object(
        training, "forward_step", return_value=(torch.tensor(1.0), {})
    ) as forward_step, patch.object(training, "backward_step"), patch.object(
        training, "reduce_losses", side_effect=torch.stack
    ), patch.object(
        training, "training_log", return_value=False
    ), patch.object(
        training, "evaluate_and_print_results"
    ), patch.object(
        training.gc, "collect", side_effect=fake_collect
    ):
        training.train(
            neox_args=neox_args,
            timers=MagicMock(),
            model=model,
            reference_model=None,
            optimizer=SimpleNamespace(param_groups=[], overflow=False, cur_scale=1.0),
            lr_scheduler=MagicMock(),
            train_data_iterator=MagicMock(),
            valid_data_iterator=MagicMock(),
        )

    if is_pipe_parallel:
        assert model.train_batch.call_count == 6
    else:
        assert forward_step.call_count == 6 * 3
    assert collected_at == [0, 2, 4, 6]


@pytest.mark.cpu
@pytest.mark.parametrize("named_validation_sets", [False, True])
def test_manual_gc_restores_automatic_gc_for_in_loop_validation(named_validation_sets):
    valid_data_iterator = (
        {"code": MagicMock()} if named_validation_sets else MagicMock()
    )

    events = run_train(
        make_args(train_iters=4, manual_gc_interval=3, eval_interval=2),
        valid_data_iterator=valid_data_iterator,
    )

    assert [kind for kind, _ in events] == [
        "collect",
        "train_step",
        "train_step",
        "validation",
        "train_step",
        "collect",
        "train_step",
        "validation",
    ]
    assert collector_states(events, "validation") == [True, True]
    # Automatic collection is switched back off for the steps that follow validation.
    assert collector_states(events, "train_step") == [False] * 4
    assert gc.isenabled()


@pytest.mark.cpu
def test_manual_gc_keeps_collector_disabled_if_it_was_disabled_before_training():
    gc.disable()

    events = run_train(make_args(train_iters=4, manual_gc_interval=2, eval_interval=2))

    assert collections(events) == [0, 2, 4]
    assert collector_states(events, "validation") == [False, False]
    assert not gc.isenabled()


@pytest.mark.cpu
@pytest.mark.parametrize("failing_stage", ["on_train_step", "on_validation"])
def test_manual_gc_restores_automatic_gc_when_training_raises(failing_stage):
    def fail():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        run_train(
            make_args(train_iters=4, manual_gc_interval=2, eval_interval=1),
            **{failing_stage: fail},
        )

    assert gc.isenabled()
