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
"""TIM-VX elementwise binary ops tests."""

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
from typing import Tuple, Dict, Callable


@tvm.testing.requires_tim_vx
@pytest.mark.usefixtures("remote")
@pytest.mark.parametrize(
    (
        "lhs_dtype", "lhs_range", "lhs_shape",
        "rhs_dtype", "rhs_range", "rhs_shape",
        "qnn_params",
        "op"
    ),
    [
        (
            "float32", (-1e3, 1e3), (128,),
            "float32", (-1e3, 1e3), (128,),
            {},
            relay.add,
        ),
        (
            "uint8", (0, 256), (512,),
            "uint8", (0, 256), (512,),
            {
                "lhs_scale": relay.const(1 / 255),
                "lhs_zero_point": relay.const(128),
                "rhs_scale": relay.const(1 / 255),
                "rhs_zero_point": relay.const(128),
                "output_scale": relay.const(2 / 255),
                "output_zero_point": relay.const(128),
            },
            relay.qnn.op.add,
        ),
        (
            "uint8", (0, 256), (512,),
            "uint8", (0, 256), (512,),
            {
                "lhs_scale": relay.const(1 / 255),
                "lhs_zero_point": relay.const(0),
                "rhs_scale": relay.const(2 / 255),
                "rhs_zero_point": relay.const(128),
                "output_scale": relay.const(4 / 255),
                "output_zero_point": relay.const(128),
            },
            relay.qnn.op.add,
        ),
        (
            "float32", (-1e3, 1e3), (128,),
            "float32", (-1e3, 1e3), (128,),
            {},
            relay.subtract,
        ),
        (
            "uint8", (0, 256), (512,),
            "uint8", (0, 256), (512,),
            {
                "lhs_scale": relay.const(1 / 255),
                "lhs_zero_point": relay.const(0),
                "rhs_scale": relay.const(1 / 255),
                "rhs_zero_point": relay.const(0),
                "output_scale": relay.const(2 / 255),
                "output_zero_point": relay.const(128),
            },
            relay.qnn.op.subtract,
        ),
        (
            "float32", (-1e3, 1e3), (128,),
            "float32", (-1e3, 1e3), (128,),
            {},
            relay.multiply,
        ),
        (
            "uint8", (0, 256), (512,),
            "uint8", (0, 256), (512,),
            {
                "lhs_scale": relay.const(2 / 255),
                "lhs_zero_point": relay.const(128),
                "rhs_scale": relay.const(2 / 255),
                "rhs_zero_point": relay.const(0),
                "output_scale": relay.const(4 / 255),
                "output_zero_point": relay.const(128),
            },
            relay.qnn.op.mul,
        ),
        (
            "float32", (-1e3, 1e3), (128,),
            "float32", (-1e3, 1e3), (128,),
            {},
            relay.divide,
        ),
        (
            "uint8", (0, 256), (1024,),
            "uint8", (1, 10), (1024,),
            {},
            relay.floor_divide,
        ),
        (
            "int16", (-32768, 32768), (256,),
            "int16", (1, 1000), (256,),
            {},
            relay.floor_divide,
        ),
        (
            "float32", (0, 1e2), (256,),
            "float32", (-2.0, 2.0), (256,),
            {},
            relay.power,
        ),
    ],
)
def test_arithmetic(
    lhs_dtype: str,
    lhs_range: ValueRange,
    lhs_shape: Tuple[int, ...],
    rhs_dtype: str,
    rhs_range: ValueRange,
    rhs_shape: Tuple[int, ...],
    qnn_params: Dict[str, relay.Constant],
    op: Callable[[relay.Expr, relay.Expr], relay.Expr],
    remote: rpc.RPCSession,
):
    inputs: Dict[str, tvm.nd.NDArray] = {
        "lhs": tvm.nd.array(
            np.random.uniform(*lhs_range, size=lhs_shape).astype(lhs_dtype)
        ),
        "rhs": tvm.nd.array(
            np.random.uniform(*rhs_range, size=rhs_shape).astype(rhs_dtype)
        ),
    }
    lhs = relay.var("lhs", shape=lhs_shape, dtype=lhs_dtype)
    rhs = relay.var("rhs", shape=rhs_shape, dtype=rhs_dtype)

    if qnn_params:
        call = op(lhs, rhs, **qnn_params)
    else:
        call = op(lhs, rhs)

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

    atol, rtol = (1, 1.0 / np.iinfo(lhs_dtype).max) if np.issubdtype(
        np.dtype(lhs_dtype), np.integer
    ) else (1e-6, 1e-6)
    verify(tim_vx_outputs, ref_outputs, atol, rtol)


@tvm.testing.requires_tim_vx
@pytest.mark.usefixtures("remote")
@pytest.mark.parametrize(
    ("dtype", "val_range", "lhs_shape", "rhs_shape"),
    [
        ("float32", (-1e3, 1e3), (256,), (256,)),
        ("uint8", (0, 256), (1024,), (1024,)),
        ("int8", (-128, 128), (1024,), (1024,)),
        ("float32", (-1e3, 1e3), (1, 3, 8, 8), (8, 8)),
        ("uint8", (0, 256), (1, 3, 32, 32), (32, )),
        ("int8", (-128, 128), (1, 3, 32, 32), (3, 1, 1)),
    ],
)
@pytest.mark.parametrize("op", [
    relay.equal, relay.not_equal,
    relay.less, relay.less_equal,
    relay.greater, relay.greater_equal,
    relay.maximum, relay.minimum,
])
def test_compare(
    dtype: str,
    val_range: ValueRange,
    lhs_shape: Tuple[int, ...],
    rhs_shape: Tuple[int, ...],
    op: Callable[[relay.Expr, relay.Expr], relay.Expr],
    remote: rpc.RPCSession,
):
    inputs: Dict[str, tvm.nd.NDArray] = {
        "lhs": tvm.nd.array(
            np.random.uniform(*val_range, size=lhs_shape).astype(dtype)
        ),
        "rhs": tvm.nd.array(
            np.random.uniform(*val_range, size=rhs_shape).astype(dtype)
        ),
    }
    lhs = relay.var("lhs", shape=lhs_shape, dtype=dtype)
    rhs = relay.var("rhs", shape=rhs_shape, dtype=dtype)
    call = op(lhs, rhs)

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
    ("lhs_shape", "rhs_shape"),
    [
        ((1024,), (1024,)),
        ((1, 8, 32, 32,), (32, 32)),
    ],
)
@pytest.mark.parametrize("op", [
    relay.logical_and, relay.logical_or
])
def test_logical(
    lhs_shape: Tuple[int, ...],
    rhs_shape: Tuple[int, ...],
    op: Callable[[relay.Expr, relay.Expr], relay.Expr],
    remote: rpc.RPCSession,
):
    inputs: Dict[str, tvm.nd.NDArray] = {
        "lhs": tvm.nd.array(
            np.random.randint(0, 2, size=lhs_shape).astype(np.bool8)
        ),
        "rhs": tvm.nd.array(
            np.random.uniform(0, 2, size=rhs_shape).astype(np.bool8)
        ),
    }
    lhs = relay.var("lhs", shape=lhs_shape, dtype="bool")
    rhs = relay.var("rhs", shape=rhs_shape, dtype="bool")
    call = op(lhs, rhs)

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
