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
"""TIM-VX pool2d ops tests."""

from test_tim_vx.infrastructure import (
    build_and_run,
    verify,
    ValueRange
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
    ("dtype", "val_range", "shape", "qnn_params"),
    [
        ("float32", (-1.0, 1.0), (1, 8, 32, 32), {}),
        ("uint8", (0, 256), (1, 16, 64, 64), {
            "input_scale": relay.const(1 / 255),
            "input_zero_point": relay.const(128),
            "output_scale": relay.const(1 / 255),
            "output_zero_point": relay.const(128),
        }),
        ("int8", (-128, 128), (1, 16, 64, 64), {
            "input_scale": relay.const(1 / 127),
            "input_zero_point": relay.const(0),
            "output_scale": relay.const(1 / 127),
            "output_zero_point": relay.const(0),
        }),
    ]
)
@pytest.mark.parametrize(
    ("pool_size", "strides", "padding", "count_include_pad"),
    [
        ((2, 2), (2, 2), (0, 0, 0, 0),  True),
        ((3, 3), (2, 2), (1, 1, 1, 1), True),
        ((3, 3), (2, 2), (1, 1, 1, 1), False),
    ]
)
def test_avg_pool2d(
    dtype: str,
    val_range: ValueRange,
    shape: Tuple[int, ...],
    qnn_params: Dict[str, relay.Constant],
    pool_size: Tuple[int, int],
    strides: Tuple[int, int],
    padding: Tuple[int, int, int, int],
    count_include_pad: bool,
    remote: rpc.RPCSession,
):
    inputs: Dict[str, tvm.nd.NDArray] = {
        "input": tvm.nd.array(
            np.random.uniform(*val_range, size=shape).astype(dtype)
        ),
    }
    data = relay.var("input", shape=shape, dtype=dtype)

    if qnn_params:
        call = relay.cast(data, "int32")
        call = relay.nn.avg_pool2d(
            data=call,
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            count_include_pad=count_include_pad,
        )
        call = relay.cast(call, dtype)
    else:
        call = relay.nn.avg_pool2d(
            data=data,
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            count_include_pad=count_include_pad,
        )

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

    atol, rtol = (1, 1.0 / np.iinfo(dtype).max) if qnn_params else (1e-6, 1e-6)
    verify(tim_vx_outputs, ref_outputs, atol, rtol)


@tvm.testing.requires_tim_vx
@pytest.mark.usefixtures("remote")
@pytest.mark.parametrize(
    ("dtype", "val_range", "shape"),
    [
        ("float32", (-1.0, 1.0), (1, 8, 32, 32)),
        ("uint8", (0, 256), (1, 16, 64, 64)),
        ("int8", (-128, 128), (1, 16, 64, 64)),
    ]
)
@pytest.mark.parametrize(
    ("pool_size", "strides", "padding"),
    [
        ((2, 2), (2, 2), (0, 0, 0, 0)),
        ((3, 3), (2, 2), (1, 1, 1, 1)),
    ]
)
def test_max_pool2d(
    dtype: str,
    val_range: ValueRange,
    shape: Tuple[int, ...],
    pool_size: Tuple[int, int],
    strides: Tuple[int, int],
    padding: Tuple[int, int, int, int],
    remote: rpc.RPCSession,
):
    inputs: Dict[str, tvm.nd.NDArray] = {
        "input": tvm.nd.array(
            np.random.uniform(*val_range, size=shape).astype(dtype)
        ),
    }
    data = relay.var("input", shape=shape, dtype=dtype)

    call = relay.nn.max_pool2d(
        data=data,
        pool_size=pool_size,
        strides=strides,
        padding=padding,
    )

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
