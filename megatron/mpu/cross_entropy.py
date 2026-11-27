# Copyright (c) 2025, EleutherAI
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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


import torch

from .initialize import get_model_parallel_group
from .initialize import get_model_parallel_rank
from .initialize import get_model_parallel_world_size
from .utils import VocabUtility


class _VocabParallelCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vocab_parallel_logits, target, return_log_z):

        # Maximum value along vocab dimension across all GPUs.
        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
        torch.distributed.all_reduce(
            logits_max,
            op=torch.distributed.ReduceOp.MAX,
            group=get_model_parallel_group(),
        )
        # Subtract the maximum value.
        vocab_parallel_logits.sub_(logits_max.unsqueeze(dim=-1))

        # Get the partition's vocab indices
        get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size
        partition_vocab_size = vocab_parallel_logits.size()[-1]
        rank = get_model_parallel_rank()
        world_size = get_model_parallel_world_size()
        vocab_start_index, vocab_end_index = get_vocab_range(
            partition_vocab_size, rank, world_size
        )

        # Create a mask of valid vocab ids (1 means it needs to be masked).
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target = target.clone() - vocab_start_index
        masked_target[target_mask] = 0

        # Get predicted-logits = logits[target].
        # For Simplicity, we convert logits to a 2-D tensor with size
        # [*, partition-vocab-size] and target to a 1-D tensor of size [*].
        logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)
        masked_target_1d = masked_target.view(-1)
        arange_1d = torch.arange(
            start=0, end=logits_2d.size()[0], device=logits_2d.device
        )
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        predicted_logits_1d = predicted_logits_1d.clone().contiguous()
        predicted_logits = predicted_logits_1d.view_as(target)
        predicted_logits[target_mask] = 0.0
        # All reduce is needed to get the chunks from other GPUs.
        torch.distributed.all_reduce(
            predicted_logits,
            op=torch.distributed.ReduceOp.SUM,
            group=get_model_parallel_group(),
        )

        # Sum of exponential of logits along vocab dimension across all GPUs.
        exp_logits = vocab_parallel_logits
        torch.exp(vocab_parallel_logits, out=exp_logits)
        sum_exp_logits = exp_logits.sum(dim=-1)
        torch.distributed.all_reduce(
            sum_exp_logits,
            op=torch.distributed.ReduceOp.SUM,
            group=get_model_parallel_group(),
        )

        # Loss = log(sum(exp(logits))) - predicted-logit.
        log_sum_exp_logits = torch.log(sum_exp_logits)
        loss = log_sum_exp_logits - predicted_logits

        # Store softmax, target-mask and masked-target for backward pass.
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)
        if return_log_z:
            ctx.set_materialize_grads(False)
            # Keep log(Z) in fp32 even when cross entropy uses a different dtype.
            if sum_exp_logits.dtype != torch.float32:
                log_sum_exp_logits = torch.log(sum_exp_logits.float())
            return loss, log_sum_exp_logits + logits_max.float()
        return loss

    @staticmethod
    def backward(ctx, grad_output, grad_log_z=None):

        # Retrieve tensors from the forward path.
        softmax, target_mask, masked_target_1d = ctx.saved_tensors

        # All the inputs have softmax as their gradient.
        grad_input = softmax
        # For simplicity, work with the 2D gradient.
        partition_vocab_size = softmax.size()[-1]
        grad_2d = grad_input.view(-1, partition_vocab_size)
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0], device=grad_2d.device)
        target_update = 1.0 - target_mask.view(-1).float()

        if grad_log_z is None:
            # Add the gradient from matching classes.
            grad_2d[arange_1d, masked_target_1d] -= target_update

            # Finally elementwise multiplication with the output gradients.
            grad_input.mul_(grad_output.unsqueeze(dim=-1))
        else:
            # d(loss)/d(logits) = softmax - onehot(target), d(log Z)/d(logits) = softmax.
            if grad_output is None:
                grad_output = torch.zeros_like(grad_log_z)
            grad_input.mul_((grad_output + grad_log_z).unsqueeze(dim=-1))
            grad_2d[arange_1d, masked_target_1d] -= target_update * grad_output.reshape(
                -1
            )

        return grad_input, None, None


def vocab_parallel_cross_entropy(vocab_parallel_logits, target, return_log_z=False):
    """Per-token cross entropy for vocab-parallel logits.

    Overwrites vocab_parallel_logits. With return_log_z, also returns each
    token's fp32 log(Z) = logsumexp(logits), whose gradient reuses the saved
    softmax, so the z-loss costs no extra memory or communication.
    """
    return _VocabParallelCrossEntropy.apply(vocab_parallel_logits, target, return_log_z)
