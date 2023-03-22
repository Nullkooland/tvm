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
"""TIM-VX broadcast op test."""


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
    ("dtype", "val_range", "shape", "broadcast_shape"),
    [
        ("float32", (0.0, 1.0), (1, 32), (16, 32)),
        ("uint8", (0, 256), (1, 16, 1, 1), (1, 16, 40, 40)),
        ("uint8", (0, 256), (18,), (1, 72, 18)),
        ("int32", (0, 10000), (1, 16, 1), (1, 16, 8)),
    ]
)
def test_broadcast(
    dtype: str,
    val_range: ValueRange,
    shape: Tuple[int, ...],
    broadcast_shape: Tuple[int, ...],
    remote: rpc.RPCSession
):
    inputs: Dict[str, tvm.nd.NDArray] = {
        "input": tvm.nd.array(
            np.random.uniform(*val_range, size=shape).astype(dtype)
        ),
    }
    data = relay.var("input", shape=shape, dtype=dtype)
    call = relay.broadcast_to(data, broadcast_shape)

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
    ("dtype", "val_range", "shape", "repeats"),
    [
        ("float32", (0.0, 1.0), (1, 32), (16, 1)),
        ("uint8", (0, 256), (1, 16, 1, 1), (1, 1, 40, 40)),
        ("uint8", (0, 256), (18,), (200,)),
        ("uint8", (0, 256), (1, 16, 16), (1, 30, 40)),
        ("int32", (0, 10000), (1, 4, 8), (1, 16, 8)),
    ]
)
def test_tile(
    dtype: str,
    val_range: ValueRange,
    shape: Tuple[int, ...],
    repeats: Tuple[int, ...],
    remote: rpc.RPCSession
):
    inputs: Dict[str, tvm.nd.NDArray] = {
        "input": tvm.nd.array(
            np.random.uniform(*val_range, size=shape).astype(dtype)
        ),
    }
    data = relay.var("input", shape=shape, dtype=dtype)
    call = relay.tile(data, repeats)

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
