# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Replay agent loop: recorded responses instead of generation (training-side benchmarks).

A rollout-free RL step for timing the trainer alone: the loop tokenizes the prompt exactly like
``SingleTurnAgentLoop`` and then returns the response token ids recorded for the same sample by an
earlier run (``examples/precision_scheduler/analysis/build_replay.py`` extracts them from a request
lifetime trace), so old-log-prob / ref / update see the same sequences as that run while the vLLM
engine only takes part in the weight sync. Registered through ``rollout.agent.agent_loop_config_path``::

    - name: replay_agent
      _target_: verl.experimental.agent_loop.replay_agent_loop.ReplayAgentLoop
      replay_file: /path/to/replay.json      # {"<uid>_<session_id>": [token ids...], ...}

with ``rollout.agent.default_agent_loop=replay_agent`` and ``trainer.stable_sample_uid=true`` (keys
are ``idx-<dataset index>_<n>``, the trace's ``trace_request_id``). A sample without a recorded entry
falls back to a deterministic recorded response (hash of the key) so the batch shape is preserved.
"""

import hashlib
import json
import logging
import os
from typing import Any

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_REPLAY_CACHE: dict[str, dict[str, list[int]]] = {}


def _load_replay(path: str) -> dict[str, list[int]]:
    if path not in _REPLAY_CACHE:
        with open(path) as f:
            table = json.load(f)
        if not isinstance(table, dict) or not table:
            raise ValueError(f"replay file {path} must be a non-empty {{key: token ids}} object")
        _REPLAY_CACHE[path] = {str(k): [int(t) for t in v] for k, v in table.items()}
    return _REPLAY_CACHE[path]


class ReplayAgentLoop(AgentLoopBase):
    """Single-turn loop that returns recorded response ids instead of calling the LLM server."""

    def __init__(self, *args, replay_file: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        self.replay_file = replay_file
        self.replay = _load_replay(replay_file)
        self._keys = sorted(self.replay)

    def lookup(self, key: str) -> tuple[list[int], bool]:
        """Recorded response for ``key``; (fallback response, False) when the key is unknown."""
        ids = self.replay.get(key)
        if ids is not None:
            return ids, True
        h = int(hashlib.sha1(key.encode()).hexdigest(), 16) % len(self._keys)
        return self.replay[self._keys[h]], False

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        multi_modal_data = await self.process_multi_modal_info(messages)
        if multi_modal_data:
            raise NotImplementedError("ReplayAgentLoop replays text-only prompts")
        prompt_ids = await self.apply_chat_template(messages)

        key = f"{kwargs.get('uid')}_{kwargs.get('session_id')}"
        response_ids, hit = self.lookup(key)
        if not hit:
            logger.warning("replay: no recorded response for %s, using a hashed fallback", key)
        response_ids = list(response_ids[: self.response_length])
        # The LLM server client stamps the weight version the trajectory was generated with; the trainer's
        # staleness metrics read it back as an int, so the replayed batch is tagged with the current step.
        global_steps = kwargs.get("global_steps")
        extra_fields = {"turn_scores": [], "tool_rewards": []}
        if global_steps is not None:
            extra_fields.update({"min_global_steps": int(global_steps), "max_global_steps": int(global_steps)})

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=[1] * len(response_ids),
            response_logprobs=None,
            multi_modal_data=multi_modal_data,
            num_turns=2,
            metrics={"generate_sequences": 0.0, "num_preempted": 0},
            extra_fields=extra_fields,
        )
        return output
