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
"""Compatibility shim: the rollout-only validator moved to ``tools/validate_rollout_run.py`` (C10).

``validate_rollout_only_run(run_dir, expected_requests, steps)`` keeps its name and summary keys;
the tool adds ``--mode full_step``, the INT4 binding proof and the COMPLETE marker check.
"""

from __future__ import annotations

import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[3] / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from validate_rollout_run import main, validate_rollout_only_run, validate_rollout_run  # noqa: E402,F401

if __name__ == "__main__":
    sys.exit(main())
