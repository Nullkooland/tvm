# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""TIM-VX reshape ops tests."""

from test_tim_vx.infrastructure import (
    build_and_run,
    verify,
    ValueRange,
)
import tvm
import tvm.testing
from tvm import relay, rpc
import pytest
import numpy as np
from typing import Tuple, Dict


@tvm.testing.requires_tim_vx
@pytest.mark.usefixtures("remote")
@pytest.mark.parametrize(
    ("dtype", "val_range", "shape", "new_shape"),
    [
        ("uint8", (0, 256), (1, 3, 16, 10), (1, 3, 160)),
        ("uint8", (0, 256), (1, 3, 16, 10),  (1, 3, 10, 16)),
        ("uint8", (0, 256), (1, 3, 16, 10),  (-1, 10)),
        ("uint8", (0, 256), (1920, ), (4, 3, 16, 10)),
        ("float32", (-1.0, 1.0),  (1, 2, 4, 8), (-1, )),
        ("int16", (-32768, 32768),  (4, 3, 6, 32), (4, -1, 32)),
    ]
)
def test_reshape(
    dtype: str,
    val_range: ValueRange,
    shape: Tuple[int, ...],
    new_shape: Tuple[int, ...],
    remote: rpc.RPCSession
):
    inputs: Dict[str, tvm.nd.NDArray] = {
        "input": tvm.nd.array(
            np.random.uniform(*val_range, size=shape).astype(dtype)
        ),
    }
    data = relay.var("input", shape=shape, dtype=dtype)
    call = relay.reshape(data, new_shape)

    tim_vx_outputs = build_and_run(
        call,
        inputs,
        params={},
        build_for_tim_vx=True,
        expected_num_cpu_ops=0,
        expected_num_tim_vx_subgraphs=1,
        remote=remote
    )

    ref_outputs = build_and_run(call, inputs, build_for_tim_vx=False)
    verify(tim_vx_outputs, ref_outputs, 0, 0)


@tvm.testing.requires_tim_vx
@pytest.mark.usefixtures("remote")
@pytest.mark.parametrize(
    ("dtype", "val_range", "shape",  "axis", "num_new_dims"),
    [
        ("float32", (0, 256), (4, 8, 8), 1, 1),
        ("uint8", (0, 256), (8, 16), 2, 2),
    ]
)
def test_expand_dims(
    dtype: str,
    val_range: ValueRange,
    shape: Tuple[int, ...],
    axis: int,
    num_new_dims: int,
    remote: rpc.RPCSession
):
    inputs: Dict[str, tvm.nd.NDArray] = {
        "input": tvm.nd.array(
            np.random.uniform(*val_range, size=shape).astype(dtype)
        ),
    }
    data = relay.var("input", shape=shape, dtype=dtype)
    call = relay.expand_dims(data, axis, num_new_dims)

    tim_vx_outputs = build_and_run(
        call,
        inputs,
        params={},
        build_for_tim_vx=True,
        expected_num_cpu_ops=0,
        expected_num_tim_vx_subgraphs=1,
        remote=remote
    )

    ref_outputs = build_and_run(call, inputs, build_for_tim_vx=False)
    verify(tim_vx_outputs, ref_outputs, 0, 0)


if __name__ == "__main__":
    pytest.main([__file__])
