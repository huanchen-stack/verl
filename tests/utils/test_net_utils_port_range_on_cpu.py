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

import pytest

from verl.utils.net_utils import parse_port_range


def test_parse_port_range_ok():
    assert parse_port_range("35000:35199") == [35000, 35199]
    assert parse_port_range("1:65536") == [1, 65536]


@pytest.mark.parametrize("spec", ["0:10", "10:5", "1:70000", "abc", "5", "5:", ":5", "5:5", "a:b"])
def test_parse_port_range_rejects(spec):
    with pytest.raises(ValueError):
        parse_port_range(spec)


def test_parse_port_range_rejects_non_string():
    with pytest.raises(ValueError):
        parse_port_range(None)
