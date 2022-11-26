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
"""TIM-VX slice ops tests."""

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
from typing import Optional, Tuple, Dict, List
Axis = Optional[Tuple[int, ...] | int]


@tvm.testing.requires_tim_vx
@pytest.mark.usefixtures("remote")
@pytest.mark.parametrize(
    ("dtype", "val_range", "shape", "begin", "end", "strides", "axes",),
    [
        ("float16", (-1.0, 1.0), (512, ), (0, ), (64, ), None, (0, )),
        ("uint8", (0, 256), (1024, ), (320, ), (640, ), None, None),
        ("int8", (-128, 128), (64, 128), (4, ), (64, ), None, (0, )),
        ("uint8", (0, 256), (4, 16, 128), (8, ), (120, ), (4, ), (2, )),
        ("uint8", (0, 256), (8, 3, 64, 64), (0, 0), (64, 64), (2, 2), (2, 3)),
        ("uint8", (0, 256), (1, 3, 64, 64), (2, 0, 0),
         (3, 64, 64), (1, 2, 2), (1, 2, 3)),
    ]
)
def test_strided_slice(
    dtype: str,
    val_range: ValueRange,
    shape: Tuple[int, ...],
    begin: Tuple[int, ...],
    end: Tuple[int, ...],
    strides: Optional[Tuple[int, ...]],
    axes: Optional[Tuple[int, ...]],
    remote: rpc.RPCSession
):
    inputs: Dict[str, tvm.nd.NDArray] = {
        "input": tvm.nd.array(
            np.random.uniform(
                *val_range, size=shape
            ).astype(dtype)
        )
    }
    data = relay.var("input", shape=shape, dtype=dtype)

    call = relay.strided_slice(data, begin, end, strides, axes)

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
    ("dtype", "val_range", "shape"),
    [
        ("uint8", (0, 256), (8, 3, 32, 32)),
        ("int8", (-128, 128), (1, 3, 64, 64)),
    ]
)
def test_composite_space_to_depth_2x2(
    dtype: str,
    val_range: ValueRange,
    shape: Tuple[int, ...],
    remote: rpc.RPCSession
):
    inputs: Dict[str, tvm.nd.NDArray] = {
        "input": tvm.nd.array(
            np.random.uniform(
                *val_range, size=shape
            ).astype(dtype)
        )
    }
    data = relay.var("input", shape=shape, dtype=dtype)

    slices: List[relay.Call] = []
    for begin in ((0, 0), (0, 1), (1, 0), (1, 1)):
        slice = relay.strided_slice(
            data,
            begin=begin, end=shape[2:],
            strides=(2, 2), axes=(2, 3)
        )
        slices.append(slice)

    call = relay.concatenate(slices, axis=1)

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
