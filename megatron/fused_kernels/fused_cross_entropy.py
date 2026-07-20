# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# The fused cross-entropy implementation below is copied verbatim from
# Megatron-Core 0.16.0, with only its imports adapted to GPT-NeoX:
# https://github.com/NVIDIA/Megatron-LM/blob/core_v0.16.0/megatron/core/fusions/fused_cross_entropy.py
#
# VocabParallelCrossEntropy is copied verbatim from:
# https://github.com/NVIDIA/Megatron-LM/blob/core_v0.16.0/megatron/core/tensor_parallel/cross_entropy.py
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

from typing import Tuple

import torch

from megatron.fused_kernels.jit import jit_fuser
from megatron.mpu.utils import VocabUtility


class VocabParallelCrossEntropy:
    """
    Computes the Cross Entropy Loss splitting the Vocab size across tensor parallel
    ranks. This implementation is used in both fused and unfused cross entropy implementations
    """

    @staticmethod
    def calculate_logits_max(
        vocab_parallel_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates logits_max."""

        vocab_parallel_logits = vocab_parallel_logits.float()
        # Maximum value along vocab dimension across all GPUs.
        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]

        return vocab_parallel_logits, logits_max

    @staticmethod
    def calculate_predicted_logits(
        vocab_parallel_logits: torch.Tensor,
        target: torch.Tensor,
        logits_max: torch.Tensor,
        vocab_start_index: int,
        vocab_end_index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculates predicted logits."""

        # In-place subtraction reduces memory pressure.
        vocab_parallel_logits -= logits_max.unsqueeze(dim=-1)

        # Create a mask of valid vocab ids (1 means it needs to be masked).
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target = target.clone() - vocab_start_index
        masked_target[target_mask] = 0

        # Get predicted-logits = logits[target].
        # For Simplicity, we convert logits to a 2-D tensor with size
        # [*, partition-vocab-size] and target to a 1-D tensor of size [*].
        partition_vocab_size = vocab_parallel_logits.size()[-1]
        logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)
        masked_target_1d = masked_target.view(-1)
        arange_1d = torch.arange(start=0, end=logits_2d.size()[0], device=logits_2d.device)
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        predicted_logits_1d = predicted_logits_1d.clone().contiguous()
        predicted_logits = predicted_logits_1d.view_as(target)
        predicted_logits[target_mask] = 0.0

        exp_logits = vocab_parallel_logits
        torch.exp(vocab_parallel_logits, out=exp_logits)
        sum_exp_logits = exp_logits.sum(dim=-1)

        return target_mask, masked_target_1d, predicted_logits, sum_exp_logits, exp_logits

    @staticmethod
    def calculate_cross_entropy_loss(
        exp_logits: torch.Tensor, predicted_logits: torch.Tensor, sum_exp_logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates cross entropy loss."""

        # Loss = log(sum(exp(logits))) - predicted-logit.
        loss = torch.log(sum_exp_logits) - predicted_logits

        # Normalize and optionally smooth logits
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))

        return exp_logits, loss

    @staticmethod
    def prepare_gradient_calculation_operands(
        softmax: torch.Tensor, target_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare gradient calculation operands."""

        # All the inputs have softmax as thier gradient.
        grad_input = softmax
        # For simplicity, work with the 2D gradient.
        partition_vocab_size = softmax.size()[-1]
        grad_2d = grad_input.view(-1, partition_vocab_size)

        # Add the gradient from matching classes.
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0], device=grad_2d.device)

        softmax_update = 1.0 - target_mask.view(-1).float()

        return grad_2d, arange_1d, softmax_update, grad_input

    @staticmethod
    def calculate_gradients(
        grad_2d: torch.Tensor,
        arange_1d: torch.Tensor,
        masked_target_1d: torch.Tensor,
        softmax_update: torch.Tensor,
        grad_input: torch.Tensor,
        grad_output: torch.Tensor,
    ) -> torch.Tensor:
        """Calculates gradients."""

        grad_2d[arange_1d, masked_target_1d] -= softmax_update

        # Finally elementwise multiplication with the output gradients.
        grad_input.mul_(grad_output.unsqueeze(dim=-1))

        return grad_input


@jit_fuser
def calculate_logits_max(vocab_parallel_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the maximum logits of the predicted tokens.
    """

    vocab_parallel_logits, logits_max = VocabParallelCrossEntropy.calculate_logits_max(
        vocab_parallel_logits
    )

    return vocab_parallel_logits, logits_max


@jit_fuser
def calculate_predicted_logits(
    vocab_parallel_logits: torch.Tensor,
    target: torch.Tensor,
    logits_max: torch.Tensor,
    vocab_start_index: int,
    vocab_end_index: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates the predicted logits for the tokens.
    """
    (target_mask, masked_target_1d, predicted_logits, sum_exp_logits, exp_logits) = (
        VocabParallelCrossEntropy.calculate_predicted_logits(
            vocab_parallel_logits, target, logits_max, vocab_start_index, vocab_end_index
        )
    )

    predicted_logits_sum_exp_logits = torch.cat((predicted_logits, sum_exp_logits))

    return target_mask, masked_target_1d, predicted_logits_sum_exp_logits, exp_logits


@jit_fuser
def calculate_cross_entropy_loss(
    exp_logits: torch.Tensor, predicted_logits_sum_exp_logits: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the final cross entropy loss for the tokens.
    """
    split_val = predicted_logits_sum_exp_logits.size()[0] // 2
    predicted_logits, sum_exp_logits = torch.split(predicted_logits_sum_exp_logits, split_val)

    exp_logits, loss = VocabParallelCrossEntropy.calculate_cross_entropy_loss(
        exp_logits, predicted_logits, sum_exp_logits
    )

    return exp_logits, loss


@jit_fuser
def calculate_gradients(
    softmax: torch.Tensor,
    grad_output: torch.Tensor,
    target_mask: torch.Tensor,
    masked_target_1d: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate the logits gradients scaled based on the CE loss
    """
    (grad_2d, arange_1d, softmax_update, grad_input) = (
        VocabParallelCrossEntropy.prepare_gradient_calculation_operands(softmax, target_mask)
    )

    grad_input = VocabParallelCrossEntropy.calculate_gradients(
        grad_2d, arange_1d, masked_target_1d, softmax_update, grad_input, grad_output
    )

    grad_input = grad_input.to(torch.bfloat16)

    return grad_input


class _VocabParallelCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vocab_parallel_logits, target, tp_group):
        """
        Forward implementation for the cross entropy loss.
        """
        vocab_parallel_logits, logits_max = calculate_logits_max(vocab_parallel_logits)
        torch.distributed.all_reduce(logits_max, op=torch.distributed.ReduceOp.MAX, group=tp_group)

        # Get the partition's vocab indices
        get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size
        partition_vocab_size = vocab_parallel_logits.size()[-1]
        vocab_start_index, vocab_end_index = get_vocab_range(
            partition_vocab_size, tp_group.rank(), tp_group.size()
        )

        (target_mask, masked_target_1d, predicted_logits_sum_exp_logits, exp_logits) = (
            calculate_predicted_logits(
                vocab_parallel_logits, target, logits_max, vocab_start_index, vocab_end_index
            )
        )

        # All reduce is needed to get the chunks from other GPUs.
        # In the fused case, tensors are batches to invoke a single
        # AllReduce call
        torch.distributed.all_reduce(
            predicted_logits_sum_exp_logits, op=torch.distributed.ReduceOp.SUM, group=tp_group
        )

        exp_logits, loss = calculate_cross_entropy_loss(exp_logits, predicted_logits_sum_exp_logits)

        # Store softmax, target-mask and masked-target for backward pass.
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward implementation for the cross entropy loss.
        """
        # Retreive tensors from the forward path.
        softmax, target_mask, masked_target_1d = ctx.saved_tensors

        grad_input = calculate_gradients(softmax, grad_output, target_mask, masked_target_1d)

        return grad_input, None, None


def fused_vocab_parallel_cross_entropy(vocab_parallel_logits, target, tp_group):
    """
    Performs cross entropy loss when logits are split across tensor parallel ranks

    Args:
        vocab_parallel_logits: logits split across tensor parallel ranks
                               dimension is [sequence_length, batch_size, hidden_size]

        target: correct vocab ids of dimseion [sequence_length, micro_batch_size]
        tp_group: the tensor parallel group over which to all reduce

    """
    return _VocabParallelCrossEntropy.apply(vocab_parallel_logits, target, tp_group)
