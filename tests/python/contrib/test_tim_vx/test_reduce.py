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
"""TIM-VX reduce ops tests."""

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
from typing import Optional, Tuple, Dict, Callable
Axis = Optional[Tuple[int, ...] | int]


@tvm.testing.requires_tim_vx
@pytest.mark.usefixtures("remote")
@pytest.mark.parametrize(
    (
        "dtype", "val_range", "shape",
        "qnn_params",
        "axis", "keep_dims", "exclude",
    ),
    [
        (
            "float32", (-1.0, 1.0),  (1, 3, 32, 32), {},
            None, False, False,  # axis, keepdims, exclude
        ),
        (
            "float32", (-1.0, 1.0),  (1, 3, 32, 32), {},
            1, False, False,  # axis, keepdims, exclude
        ),
        (
            "float32", (-1.0, 1.0),  (1, 3, 32, 32), {},
            1, False, True,  # axis, keepdims, exclude
        ),
        (
            "uint8", (0, 2),  (1, 8, 16, 16), {},
            (2, 3), False, False,  # axis, keepdims, exclude
        ),
        (
            "int16", (-100, 100),  (1, 12, 24, 24), {},
            1, False, False,  # axis, keepdims, exclude
        ),
        (
            "uint8", (0, 256),  (1, 16, 8, 8), {
                "input_scale": relay.const(1 / 255),
                "input_zero_point": relay.const(128),
                "output_scale": relay.const(8 / 255),
                "output_zero_point": relay.const(128),
            },
            (2, 3), True, False,  # axes, keepdims, exclude
        ),
        (
            "int8", (-128, 128),  (1, 16, 8, 8), {
                "input_scale": relay.const(1 / 127),
                "input_zero_point": relay.const(0),
                "output_scale": relay.const(1 / 127),
                "output_zero_point": relay.const(0),
            },
            (2, 3), True, False,  # axes, keepdims, exclude
        ),
    ]
)
def test_sum(
    dtype: str,
    val_range: ValueRange,
    shape: Tuple[int, ...],
    qnn_params: Dict[str, relay.Constant],
    axis: Axis,
    keep_dims: bool,
    exclude: bool,
    remote: rpc.RPCSession,
):
    inputs: Dict[str, tvm.nd.NDArray] = {
        "input": tvm.nd.array(
            np.random.uniform(*val_range, size=shape).astype(dtype)
        ),
    }
    data = relay.var("input", shape=shape, dtype=dtype)

    if qnn_params:
        call = relay.qnn.op.dequantize(
            data,
            input_scale=qnn_params["input_scale"],
            input_zero_point=qnn_params["input_zero_point"]
        )
        call = relay.sum(call, axis, keep_dims, exclude)
        call = relay.qnn.op.quantize(
            call,
            output_scale=qnn_params["output_scale"],
            output_zero_point=qnn_params["output_zero_point"],
            out_dtype=dtype
        )
    else:
        call = relay.sum(data, axis, keep_dims, exclude)

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
    (
        "dtype", "val_range", "shape", "qnn_params",
        "axis", "keep_dims", "exclude",
    ),
    [
        (
            "float32", (-1.0, 1.0),  (1, 3, 32, 32), {},
            None, False, False,  # axis, keepdims, exclude
        ),
        (
            "float32", (-1.0, 1.0),  (1, 3, 32, 32), {},
            1, False, False,  # axis, keepdims, exclude
        ),
        (
            "float32", (-1.0, 1.0),  (1, 3, 32, 32), {},
            1, False, True,  # axis, keepdims, exclude
        ),
        (
            "int16", (0, 100),  (1, 8, 16, 16), {},
            (2, 3), False, False,  # axis, keepdims, exclude
        ),
        (
            "uint8", (0, 256),  (1, 16, 32, 32), {
                "input_scale": relay.const(2 / 255),
                "input_zero_point": relay.const(128),
                "output_scale": relay.const(2 / 255),
                "output_zero_point": relay.const(128),
            },
            (2, 3), True, False,  # axes, keepdims, exclude
        ),
        (
            "int8", (-128, 128),  (1, 16, 32, 32), {
                "input_scale": relay.const(1 / 127),
                "input_zero_point": relay.const(0),
                "output_scale": relay.const(1 / 127),
                "output_zero_point": relay.const(0),
            },
            (2, 3), True, False,  # axes, keepdims, exclude
        ),
    ]
)
def test_mean(
    dtype: str,
    val_range: ValueRange,
    shape: Tuple[int, ...],
    qnn_params: Dict[str, relay.Constant],
    axis: Axis,
    keep_dims: bool,
    exclude: bool,
    remote: rpc.RPCSession,
):
    inputs: Dict[str, tvm.nd.NDArray] = {
        "input": tvm.nd.array(
            np.random.uniform(*val_range, size=shape).astype(dtype)
        ),
    }
    data = relay.var("input", shape=shape, dtype=dtype)

    if qnn_params:
        call = relay.qnn.op.requantize(
            data, **qnn_params, out_dtype="int32"
        )
        call = relay.mean(call, axis, keep_dims, exclude)
        call = relay.qnn.op.requantize(
            call, **qnn_params, out_dtype=dtype
        )
    else:
        call = relay.mean(data, axis, keep_dims, exclude)

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

    atol, rtol = (1, 1.0 / np.iinfo(dtype).max) if np.issubdtype(
        np.dtype(dtype), np.integer
    ) else (1e-6, 1e-6)
    verify(tim_vx_outputs, ref_outputs, atol, rtol)


@tvm.testing.requires_tim_vx
@pytest.mark.usefixtures("remote")
@pytest.mark.parametrize(
    ("dtype", "val_range", "shape", "axis", "keep_dims", "exclude"),
    [
        ("float32", (-1.0, 1.0),  (1, 3, 32, 32), None, False, False),
        ("float32", (-1.0, 1.0),  (1, 3, 32, 32), 1, False, True),
        ("float32", (-1.0, 1.0),  (1, 3, 32, 32), 1, True, False),
        ("uint8", (0, 256),  (1, 64, 16, 16), (2, 3), False, False),
        ("int16", (-3600, 1200),  (1, 12, 24, 24), 1, False, False),
    ]
)
@pytest.mark.parametrize("op", [relay.min, relay.max])
def test_minmax(
    dtype: str,
    val_range: ValueRange,
    shape: Tuple[int, ...],
    axis: Axis,
    keep_dims: bool,
    exclude: bool,
    op: Callable[[relay.Expr, Axis, bool, bool], relay.Expr],
    remote: rpc.RPCSession,
):
    inputs: Dict[str, tvm.nd.NDArray] = {
        "input": tvm.nd.array(
            np.random.uniform(*val_range, size=shape).astype(dtype)
        ),
    }
    data = relay.var("input", shape=shape, dtype=dtype)

    call = op(data, axis, keep_dims, exclude)

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

    atol, rtol = (0, 0) if np.issubdtype(
        np.dtype(dtype), np.integer
    ) else (1e-6, 1e-6)
    verify(tim_vx_outputs, ref_outputs, atol, rtol)


@tvm.testing.requires_tim_vx
@pytest.mark.usefixtures("remote")
@pytest.mark.parametrize(
    ("shape", "axis", "keep_dims", "exclude"),
    [
        ((1024,), None, False, False),
        ((1, 8, 64, 64), 1, True, False),
        ((1, 8, 64, 64), 1, False, True),
        ((1, 8, 64, 64), (2, 3), False, False),
    ]
)
@pytest.mark.parametrize("op", [relay.all, relay.any])
def test_logical(
    shape: Tuple[int, ...],
    axis: Axis,
    keep_dims: bool,
    exclude: bool,
    op: Callable[[relay.Expr, Axis, bool, bool], relay.Expr],
    remote: rpc.RPCSession,
):
    inputs: Dict[str, tvm.nd.NDArray] = {
        "input": tvm.nd.array(
            np.random.randint(0, 2, size=shape).astype(np.bool_)
        ),
    }
    data = relay.var("input", shape=shape, dtype="bool")

    call = op(data, axis, keep_dims, exclude)

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
    ("dtype", "val_range", "shape", "axis", "keep_dims", "exclude"),
    [
        ("float32", (-1.0, 1.0), (256,), None, False, False),
        ("float32", (-1.0, 1.0), (1, 8, 32, 32), 1, False, False),
        ("uint8", (0, 256), (4, 8, 64, 64), 1, False, False),
        ("int8", (-128, 128), (32, 128), 0, False, False),
    ]
)
@pytest.mark.parametrize("op", [relay.argmin, relay.argmax])
def test_argreduce(
    dtype: str,
    val_range: ValueRange,
    shape: Tuple[int, ...],
    axis: Axis,
    keep_dims: bool,
    exclude: bool,
    op: Callable[[relay.Expr, Axis, bool, bool], relay.Expr],
    remote: rpc.RPCSession,
):
    inputs: Dict[str, tvm.nd.NDArray] = {
        "input": tvm.nd.array(
            np.random.uniform(*val_range, size=shape).astype(dtype)
        ),
    }
    data = relay.var("input", shape=shape, dtype=dtype)

    call = op(data, axis, keep_dims, exclude)

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
    ("dtype", "val_range", "shape", "k",),
    [
        ("float32", (-1.0, 1.0), (256,), 5),
        ("float16", (-1.0, 1.0), (512,), 128),
        ("uint8", (0, 256), (1024,), 16),
        ("uint8", (0, 256), (8, 1024), 5),
        ("int8", (-128, 128), (1024,), 16),
    ]
)
def test_topk(
    dtype: str,
    val_range: ValueRange,
    shape: Tuple[int, ...],
    k: int,
    remote: rpc.RPCSession,
):
    inputs: Dict[str, tvm.nd.NDArray] = {
        "input": tvm.nd.array(
            np.random.uniform(*val_range, size=shape).astype(dtype)
        ),
    }
    data = relay.var("input", shape=shape, dtype=dtype)

    call = relay.topk(data, k).tuple_value

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
        ("float32", (-1.0, 1.0), (1, 8, 32, 32)),
        ("int16", (0, 1000), (1, 4, 32, 32)),
        ("uint8", (0, 256), (1, 16, 64, 64)),
        ("int8", (-128, 128), (1, 16, 64, 64)),
    ]
)
def test_global_avg_pool2d(
    dtype: str,
    val_range: ValueRange,
    shape: Tuple[int, ...],
    remote: rpc.RPCSession,
):
    inputs: Dict[str, tvm.nd.NDArray] = {
        "input": tvm.nd.array(
            np.random.uniform(*val_range, size=shape).astype(dtype)
        ),
    }
    data = relay.var("input", shape=shape, dtype=dtype)

    if np.issubdtype(np.dtype(dtype), np.integer):
        call = relay.cast(data, "int32")
        call = relay.nn.global_avg_pool2d(call)
        call = relay.cast(call, dtype)
    else:
        call = relay.nn.global_avg_pool2d(data)

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

    atol, rtol = (1, 1.0 / np.iinfo(dtype).max) if np.issubdtype(
        np.dtype(dtype), np.integer
    ) else (1e-6, 1e-6)
    verify(tim_vx_outputs, ref_outputs, atol, rtol)


if __name__ == "__main__":
    pytest.main([__file__])
