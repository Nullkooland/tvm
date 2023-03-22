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
"""TIM-VX tensor concatenate/split ops tests."""

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
from typing import Tuple, List, Dict


@tvm.testing.requires_tim_vx
@pytest.mark.usefixtures("remote")
@pytest.mark.parametrize(
    ("dtype", "axis", "input_entries", "output_qnn_params"),
    [
        (
            "float32", 1,  # dtype, axis
            [
                ((1, 30), (-1.0, 1.0), {}),
                ((1, 20), (-1.0, 1.0), {}),
                ((1, 10), (-1.0, 1.0), {}),
            ],  # inputs
            {},  # output qnn params
        ),
        (
            "uint8", 1,  # dtype, axis
            [
                ((1, 12, 24, 16), (0, 256), {}),
                ((1, 18, 24, 16), (0, 256), {}),
            ],  # inputs
            {},  # output qnn params
        ),
        (
            "int16", 3,  # dtype, axis
            [
                ((1, 4, 10, 18), (0, 256), {}),
                ((1, 4, 10, 12), (0, 256), {}),
            ],  # inputs
            {},  # output qnn params
        ),
        (
            "uint8", 1,  # dtype, axis
            [
                ((1, 38, 16, 16), (0, 256), {
                 "input_scale": relay.const(1 / 255),
                 "input_zero_point": relay.const(0),
                 }),
                ((1, 42, 16, 16), (0, 256), {
                 "input_scale": relay.const(2 / 255),
                 "input_zero_point": relay.const(128),
                 }),
            ],  # inputs
            {
                "output_scale": relay.const(2 / 255),
                "output_zero_point": relay.const(128)
            },  # output qnn params
        ),
        (
            "int8", 1,  # dtype, axis
            [
                ((1, 38, 16, 16), (-128, 128), {
                 "input_scale": relay.const(1 / 127),
                 "input_zero_point": relay.const(0),
                 }),
                ((1, 42, 16, 16), (-128, 128), {
                 "input_scale": relay.const(2 / 127),
                 "input_zero_point": relay.const(0),
                 }),
            ],  # inputs
            {
                "output_scale": relay.const(2 / 127),
                "output_zero_point": relay.const(0)
            },  # output qnn params
        ),
    ]
)
def test_concat(
    dtype: str,
    input_entries: List[Tuple[
        Tuple[int, ...],
        ValueRange,
        Dict[str, relay.Constant]
    ]],
    output_qnn_params: Dict[str, relay.Constant],
    axis: int,
    remote: rpc.RPCSession,
):
    inputs: Dict[str, tvm.nd.NDArray] = {}
    input_vars: List[relay.Var] = []
    input_scales: List[relay.Constant] = []
    input_zero_points: List[relay.Constant] = []

    for i, (shape, val_range, qnn_params) in enumerate(input_entries):
        input_name = f"input_{i}"
        inputs[input_name] = tvm.nd.array(
            np.random.uniform(*val_range, size=shape).astype(dtype)
        )

        input_var = relay.var(input_name, shape=shape, dtype=dtype)

        if output_qnn_params:
            input_scales.append(qnn_params["input_scale"])
            input_zero_points.append(qnn_params["input_zero_point"])

        input_vars.append(input_var)

    if output_qnn_params:
        call = relay.qnn.op.concatenate(
            input_vars,
            input_scales=input_scales,
            input_zero_points=input_zero_points,
            output_scale=output_qnn_params["output_scale"],
            output_zero_point=output_qnn_params["output_zero_point"],
            axis=axis,
        )
    else:
        call = relay.concatenate(input_vars, axis)

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

    atol, rtol = (
        1, 1.0 / np.iinfo(dtype).max) if output_qnn_params else (1e-6, 1e-6)
    verify(tim_vx_outputs, ref_outputs, atol, rtol)


@tvm.testing.requires_tim_vx
@pytest.mark.usefixtures("remote")
@pytest.mark.parametrize(
    ("dtype", "val_range", "shape", "axis", "num_stacks"),
    [
        ("float32", (-1.0, 1.0), (16, 16), 0, 3),
        ("float32", (-1.0, 1.0), (16, 16), 1, 3),
        ("float32", (-1.0, 1.0), (16, 16), 2, 3),
        ("uint8", (0, 256), (3, 24, 24), 0, 16),
    ]
)
def test_stack(
    dtype: str,
    val_range: ValueRange,
    shape: Tuple[int, ...],
    axis: int,
    num_stacks: int,
    remote: rpc.RPCSession,
):
    inputs: Dict[str, tvm.nd.NDArray] = {}
    input_vars: List[relay.Var] = []

    for i in range(num_stacks):
        input_name = f"input_{i}"
        inputs[input_name] = tvm.nd.array(
            np.random.uniform(*val_range, size=shape).astype(dtype)
        )

        input_var = relay.var(input_name, shape=shape, dtype=dtype)
        input_vars.append(input_var)

    call = relay.stack(input_vars, axis)

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
    ("dtype", "val_range", "shape", "axis", "indices_or_num_sections"),
    [
        ("float32", (-1.0, 1.0), (16, 64), 0, (4, 8, 12)),
        ("float32", (-1.0, 1.0), (4, 85), 1, (2, 4)),
        ("int16", (-10000, 10000), (8, 64), 1, 4),
        ("uint8", (0, 256), (1, 3, 32, 32), 1, 3),
    ]
)
def test_split(
    dtype: str,
    val_range: ValueRange,
    shape: Tuple[int, ...],
    axis: int,
    indices_or_num_sections: Tuple[int, ...] | int,
    remote: rpc.RPCSession,
):

    inputs: Dict[str, tvm.nd.NDArray] = {
        "input": tvm.nd.array(
            np.random.uniform(*val_range, size=shape).astype(dtype)
        )
    }
    input_var = relay.var("input", shape=shape, dtype=dtype)

    call = relay.split(
        input_var,
        axis=axis,
        indices_or_sections=indices_or_num_sections,
    ).tuple_value

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
