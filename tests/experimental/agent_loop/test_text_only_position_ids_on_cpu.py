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
"""mm_token_type_ids labelling for M-RoPE processors (Qwen3.5) in AgentLoopWorker."""

import torch

from verl.experimental.agent_loop.agent_loop import AgentLoopWorker


class _FakeQwen35Processor:
    image_token_id = 42
    video_token_id = 43

    def __init__(self):
        self.last_mm_token_type_ids = None

    def get_rope_index(
        self,
        input_ids,
        mm_token_type_ids,
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask=None,
        **kwargs,
    ):
        del image_grid_thw, video_grid_thw, attention_mask, kwargs
        self.last_mm_token_type_ids = mm_token_type_ids.clone()
        positions = torch.arange(input_ids.shape[1], dtype=input_ids.dtype)
        return positions.view(1, 1, -1).expand(3, input_ids.shape[0], -1), torch.zeros(input_ids.shape[0], 1)


def _compute(processor, input_ids, multimodal_inputs):
    worker = type("_Worker", (), {"processor": processor})()
    attention_mask = torch.ones_like(input_ids)
    return AgentLoopWorker._compute_position_ids(worker, input_ids, attention_mask, multimodal_inputs)


def test_text_only_generated_vision_token_stays_text_for_rope():
    processor = _FakeQwen35Processor()
    # Token 42 looks like <|image_pad|>, but this text-only request has no
    # image_grid_thw. It must remain token type 0.
    input_ids = torch.tensor([[11, 42, 12, 43, 13]])
    multimodal_inputs = {"mm_token_type_ids": torch.zeros_like(input_ids)}

    position_ids = _compute(processor, input_ids, multimodal_inputs)

    assert processor.last_mm_token_type_ids is not None
    assert torch.count_nonzero(processor.last_mm_token_type_ids) == 0
    assert position_ids.shape == (1, 4, input_ids.shape[1])


def test_vision_tokens_with_real_grids_are_labelled_multimodal():
    processor = _FakeQwen35Processor()
    input_ids = torch.tensor([[11, 42, 42, 12, 43, 13]])
    multimodal_inputs = {
        "mm_token_type_ids": torch.zeros_like(input_ids),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
        "video_grid_thw": torch.tensor([[1, 2, 2]]),
    }

    position_ids = _compute(processor, input_ids, multimodal_inputs)

    assert processor.last_mm_token_type_ids.tolist() == [[0, 1, 1, 0, 2, 0]]
    assert position_ids.shape == (1, 4, input_ids.shape[1])


def test_image_grid_only_labels_image_tokens_not_video_tokens():
    processor = _FakeQwen35Processor()
    input_ids = torch.tensor([[42, 43]])
    multimodal_inputs = {
        "mm_token_type_ids": torch.zeros_like(input_ids),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
    }

    _compute(processor, input_ids, multimodal_inputs)

    assert processor.last_mm_token_type_ids.tolist() == [[1, 0]]
