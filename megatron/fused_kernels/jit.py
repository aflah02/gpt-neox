# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Adapted from Megatron-Core 0.16.0:
# https://github.com/NVIDIA/Megatron-LM/blob/core_v0.16.0/megatron/core/jit.py
# The version check is kept local because GPT-NeoX does not provide
# megatron.core.utils.
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
from packaging.version import Version


def _is_torch_min_version(version):
    return Version(torch.__version__) >= Version(version)


jit_fuser = torch.jit.script
# nvFuser is deprecated in PyTorch JIT starting from 2.2


def noop_decorator(func):
    """No-op decorator"""
    return func


def enable_jit_fuser():
    """Enable the JIT fuser"""
    global jit_fuser
    try:
        if _is_torch_min_version("2.2.0a0"):
            jit_fuser = torch.compile
    except ImportError:
        jit_fuser = noop_decorator


def disable_jit_fuser():
    """Disable the JIT fuser"""
    global jit_fuser
    jit_fuser = noop_decorator


enable_jit_fuser()
