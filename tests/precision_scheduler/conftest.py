# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Shared pytest configuration for the precision-scheduler test tree."""

import os
from pathlib import Path

import pytest

VERL_ROOT = Path(__file__).resolve().parents[2]
ENV_DIR = VERL_ROOT / "scripts" / "precision_scheduler" / "env"


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "gpu_smoke: needs one free GPU and runs under run_gpu.sh; skipped unless -m gpu_smoke"
    )


def pytest_collection_modifyitems(config, items):
    """GPU smoke tests never run implicitly: deselect them unless `-m gpu_smoke` (or PS_RUN_GPU_SMOKE=1)."""
    markexpr = config.getoption("-m", default="") or ""
    if "gpu_smoke" in markexpr or os.environ.get("PS_RUN_GPU_SMOKE") == "1":
        return
    skip = pytest.mark.skip(reason="gpu_smoke tier: run with -m gpu_smoke under run_gpu.sh")
    for item in items:
        if "gpu_smoke" in item.keywords:
            item.add_marker(skip)
