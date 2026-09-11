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
"""Sender and receiver of the colocated weight-transfer socket must agree on the namespace."""

from __future__ import annotations

import pytest

from verl.workers.rollout.vllm_rollout.utils import parse_bool_env, weight_sync_namespace, zmq_handle_for


@pytest.mark.parametrize(
    ("namespace", "expected"),
    [
        ("01000000", "01000000"),
        ("dynro_bf16_g3_12345", "dynro_bf16_g3_12345"),
        ("a/b:c d", "a_b_c_d"),
    ],
)
def test_sender_and_receiver_sanitize_identically(namespace, expected):
    sender = weight_sync_namespace({"VERL_ZMQ_NAMESPACE": namespace}, default="01000000")
    receiver = weight_sync_namespace({"VERL_ZMQ_NAMESPACE": namespace}, default="0")
    assert sender == receiver == expected
    # Frozen handle format from the archived worktree (vllm_rollout.py / utils.py).
    assert zmq_handle_for(sender, 0, 1) == f"ipc:///tmp/rl-colocate-zmq-{expected}-replica-0-rank-1.sock"


def test_namespace_falls_back_to_job_id_when_env_unset():
    assert weight_sync_namespace({}, default="01000000") == "01000000"
    assert weight_sync_namespace({"VERL_ZMQ_NAMESPACE": ""}, default="job/1") == "job_1"
    # Receiver-side fallback chain: VERL_ZMQ_NAMESPACE -> VERL_RAY_JOB_ID -> "0".
    environ = {"VERL_RAY_JOB_ID": "02000000"}
    assert weight_sync_namespace(environ, default=environ.get("VERL_RAY_JOB_ID", "0")) == "02000000"


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "TRUE", " On "])
def test_parse_bool_env_truthy(value):
    assert parse_bool_env(value) is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "", "maybe"])
def test_parse_bool_env_falsy(value):
    assert parse_bool_env(value) is False


def test_parse_bool_env_default():
    assert parse_bool_env(None) is False
    assert parse_bool_env(None, default=True) is True


def test_force_shm_overrides_ipc_support():
    """ServerAdapter: use_shm = force_shm or not is_support_ipc()."""
    is_support_ipc = True
    force_shm = parse_bool_env("1")
    assert (force_shm or not is_support_ipc) is True
    force_shm = parse_bool_env(None)
    assert (force_shm or not is_support_ipc) is False
