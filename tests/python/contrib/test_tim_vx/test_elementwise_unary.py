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
"""TIM-VX elementwise unary ops tests."""

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
    ("dtype", "val_range", "shape", "op"),
    [
        ("float32", (-1e3, 1e3), (256,), relay.negative),
        ("int8", (-128, 128), (1024,), relay.negative),
        ("int16", (-32768, 32768), (512,), relay.negative),
        ("float32", (-1e3, 1e3), (256,),  relay.abs),
        ("int8", (-128, 128), (1024,), relay.abs),
        ("int16", (-32768, 32768), (512,), relay.abs),
        ("float32", (-1.0, 1.0), (256,), relay.exp),
        ("float32", (1e-6, 1e6), (256,), relay.log),
        ("float32", (0.0, 1e3), (256,), relay.sqrt),
        ("float32", (0.0, 1e3), (256,), relay.rsqrt),
        ("float32", (-1e3, 1e3), (256,), relay.round),
        ("float32", (-1e3, 1e3), (256,), relay.floor),
        ("float32", (-1e3, 1e3), (256,), relay.ceil),
        ("bool", (0, 2), (1024,), relay.logical_not),
    ],
)
def test_arithmetic(
    dtype: str,
    val_range: ValueRange,
    shape: Tuple[int, ...],
    op: Callable[[relay.Expr], relay.Expr],
    remote: rpc.RPCSession,
):
    inputs: Dict[str, tvm.nd.NDArray] = {
        "input": tvm.nd.array(
            np.random.uniform(*val_range, size=shape).astype(dtype)
        ),
    }
    data = relay.var(
        "input",
        shape=shape,
        dtype=dtype
    )

    call = op(data)

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
    ("input_dtype", "output_dtype", "val_range", "shape"),
    [
        ("float32", "int32", (-1e6, 1e6), (256,)),
        ("int32", "float32", (-1e6, 1e6), (256,)),
        ("float32", "uint8", (0.0, 2e2), (256,)),
        ("float32", "int8", (-1e2, 1e2), (256,)),
        ("int32", "uint8", (0, 256), (256,)),
        ("uint8", "int32", (0, 1000), (256,)),
        ("int8", "uint8", (0, 128), (1024,)),
        ("uint8", "int8", (0, 128), (1024,)),
    ],
)
def test_cast(
    input_dtype: str,
    output_dtype: str,
    val_range: ValueRange,
    shape: Tuple[int, ...],
    remote: rpc.RPCSession,
):
    inputs: Dict[str, tvm.nd.NDArray] = {
        "input": tvm.nd.array(
            np.random.uniform(*val_range, size=shape).astype(input_dtype)
        ),
    }
    data = relay.var(
        "input",
        shape=shape,
        dtype=input_dtype
    )

    call = relay.cast(data, output_dtype)

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
        ("float32", (-1e3, 1e3), (256,)),
        ("uint8", (0, 16), (1024,)),
        ("int8", (-10, 10), (1024,)),
    ],
)
def test_square(
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
    call = data * data

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
        ("float32", (-1e3, 1e3), (256,)),
        ("float32", (1e-3, 1.0), (256,)),
    ],
)
def test_reciprocal(
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
    one = relay.const(1, dtype=dtype)
    data = relay.var("input", shape=shape, dtype=dtype)
    call = one / data

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
    verify(tim_vx_outputs, ref_outputs, 1e-6, 1e-6)


@tvm.testing.requires_tim_vx
@pytest.mark.usefixtures("remote")
@pytest.mark.parametrize(
    ("dtype", "val_range", "shape", "clip_range"),
    [
        ("float32", (-1.0, 1.0), (256,), (-0.25, 0.5)),
        ("int16", (-10000, 10000), (512,), (-5120, 3840)),
        ("uint8", (0, 256), (1024,), (40, 160)),
        ("int8", (-128, 128), (1024,), (-72, 72)),
    ],
)
def test_clip(
    dtype: str,
    val_range: ValueRange,
    shape: Tuple[int, ...],
    clip_range: Tuple[float, float],
    remote: rpc.RPCSession,
):
    inputs: Dict[str, tvm.nd.NDArray] = {
        "input": tvm.nd.array(
            np.random.uniform(*val_range, size=shape).astype(dtype)
        ),
    }
    data = relay.var(
        "input",
        shape=shape,
        dtype=dtype
    )

    call = relay.clip(data, *clip_range)

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
    ("dtype", "val_range", "shape", "qnn_params", "alpha"),
    [
        ("float32", (-1e3, 1e3), (256,), {}, 0.2),
        ("uint8", (0, 256), (1024,), {
            "input_scale": relay.const(4 / 255),
            "input_zero_point": relay.const(128),
            "output_scale": relay.const(2 / 255),
            "output_zero_point": relay.const(32),
        }, 0.2),
        ("int8", (-128, 128), (1024,), {
            "input_scale": relay.const(1 / 127),
            "input_zero_point": relay.const(0),
            "output_scale": relay.const(1 / 127),
            "output_zero_point": relay.const(0),
        }, 0.2),
    ],
)
def test_leaky_relu(
    dtype: str,
    val_range: ValueRange,
    shape: Tuple[int, ...],
    qnn_params: Dict[str, relay.Constant],
    alpha: float,
    remote: rpc.RPCSession,
):
    inputs: Dict[str, tvm.nd.NDArray] = {
        "input": tvm.nd.array(
            np.random.uniform(*val_range, size=shape).astype(dtype)
        ),
    }
    data = relay.var(
        "input",
        shape=shape,
        dtype=dtype
    )

    if qnn_params:
        call = relay.qnn.op.leaky_relu(data, alpha, **qnn_params)
    else:
        call = relay.nn.leaky_relu(data, alpha)

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
    ("dtype", "val_range", "shape", "qnn_params"),
    [
        ("float32", (-1e3, 1e3), (256,), {}),
        ("uint8", (0, 256), (1024,), {
            "scale": relay.const(20 / 255),
            "zero_point": relay.const(128),
            "output_scale": relay.const(1 / 255),
            "output_zero_point": relay.const(0),
        }),
        ("int8", (-128, 128), (1024,), {
            "scale": relay.const(10 / 127),
            "zero_point": relay.const(0),
            "output_scale": relay.const(1 / 127),
            "output_zero_point": relay.const(0),
        }),
    ],
)
def test_sigmoid(
    dtype: str,
    val_range: ValueRange,
    shape: Tuple[int, ...],
    qnn_params: Dict[str, relay.Constant],
    remote: rpc.RPCSession,
):
    inputs: Dict[str, tvm.nd.NDArray] = {
        "input": tvm.nd.array(
            np.random.uniform(*val_range, size=shape).astype(dtype)
        ),
    }
    data = relay.var(
        "input",
        shape=shape,
        dtype=dtype
    )

    if qnn_params:
        call = relay.qnn.op.sigmoid(data, **qnn_params)
    else:
        call = relay.sigmoid(data)

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
    ("dtype", "val_range", "shape", "qnn_params"),
    [
        ("float32", (-1e3, 1e3), (256,), {}),
        ("uint8", (0, 256), (1024,), {
            "scale": relay.const(64 / 255),
            "zero_point": relay.const(128),
            "output_scale": relay.const(16 / 255),
            "output_zero_point": relay.const(4),
        })
    ],
)
def test_swish(
    dtype: str,
    val_range: ValueRange,
    shape: Tuple[int, ...],
    qnn_params: Dict[str, relay.Constant],
    remote: rpc.RPCSession,
):
    inputs: Dict[str, tvm.nd.NDArray] = {
        "input": tvm.nd.array(
            np.random.uniform(*val_range, size=shape).astype(dtype)
        ),
    }
    data = relay.var(
        "input",
        shape=shape,
        dtype=dtype
    )

    if qnn_params:
        sigmoid_scale = relay.const(1 / 255)
        sigmoid_zero_point = relay.const(0)
        sigmoid = relay.qnn.op.sigmoid(
            data,
            scale=qnn_params["scale"],
            zero_point=qnn_params["zero_point"],
            output_scale=sigmoid_scale,
            output_zero_point=sigmoid_zero_point,
        )
        call = relay.qnn.op.mul(
            lhs=data,
            rhs=sigmoid,
            lhs_scale=qnn_params["scale"],
            lhs_zero_point=qnn_params["zero_point"],
            rhs_scale=sigmoid_scale,
            rhs_zero_point=sigmoid_zero_point,
            output_scale=qnn_params["output_scale"],
            output_zero_point=qnn_params["output_zero_point"],
        )
    else:
        call = data * relay.sigmoid(data)

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
    ("dtype", "val_range", "shape", "qnn_params", "axis"),
    [
        ("float32", (-1e3, 1e3), (256,), {}, 0),
        ("float32", (-1e3, 1e3), (1, 32, 64), {}, 1),
        ("uint8", (0, 256), (1024,), {
            "input_scale": relay.const(64 / 255),
            "input_zero_point": relay.const(128),
            "output_scale": relay.const(1 / 255),
            "output_zero_point": relay.const(0),
        }, 0),
        ("uint8", (0, 256), (1, 16, 4, 6300), {
            "input_scale": relay.const(51 / 255),
            "input_zero_point": relay.const(180),
            "output_scale": relay.const(1 / 255),
            "output_zero_point": relay.const(0),
        }, 1),
    ],
)
def test_softmax(
    dtype: str,
    val_range: ValueRange,
    shape: Tuple[int, ...],
    qnn_params: Dict[str, relay.Constant],
    axis: int,
    remote: rpc.RPCSession,
):
    inputs: Dict[str, tvm.nd.NDArray] = {
        "input": tvm.nd.array(
            np.random.uniform(*val_range, size=shape).astype(dtype)
        ),
    }
    data = relay.var(
        "input",
        shape=shape,
        dtype=dtype
    )

    if qnn_params:
        call = relay.qnn.op.dequantize(
            data=data,
            input_scale=qnn_params["input_scale"],
            input_zero_point=qnn_params["input_zero_point"]
        )
        call = relay.nn.softmax(call, axis=axis)
        call = relay.qnn.op.quantize(
            data=call,
            output_scale=qnn_params["output_scale"],
            output_zero_point=qnn_params["output_zero_point"]
        )
    else:
        call = relay.nn.softmax(data, axis=axis)

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


if __name__ == "__main__":
    pytest.main([__file__])
