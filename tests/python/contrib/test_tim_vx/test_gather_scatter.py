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
"""TIM-VX gather-scatter ops tests."""


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
    ("dtype", "val_range", "shape",),
    [
        ("float32", (-1.0, 1.0), (256,),),
        ("uint8", (0, 256), (1024,),),
        ("int8", (-128, 128), (1024,),),
        ("uint8", (0, 256), (32, 128),),
        ("uint8", (0, 256), (1, 8, 32, 32),),
    ]
)
def test_where(
    dtype: str,
    val_range: ValueRange,
    shape: Tuple[int, ...],
    remote: rpc.RPCSession
):
    inputs: Dict[str, tvm.nd.NDArray] = {
        "cond": tvm.nd.array(
            np.random.randint(
                0, 2, size=shape
            ).astype(np.bool8)
        ),
        "x": tvm.nd.array(
            np.random.uniform(
                *val_range, size=shape
            ).astype(dtype)
        ),
        "y": tvm.nd.array(
            np.random.uniform(
                *val_range, size=shape
            ).astype(dtype)
        ),
    }
    cond = relay.var("cond", shape=shape, dtype="bool")
    x = relay.var("x", shape=shape, dtype=dtype)
    y = relay.var("y", shape=shape, dtype=dtype)

    call = relay.where(cond, x, y)

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
    ("dtype", "val_range", "shape", "indices_shape", "axis", "num_batch_dims",),
    [
        ("uint8", (0, 256), (8, 128), (4, ), 0, 0,),
        ("uint8", (0, 256), (128, 8), (4, ), 1, 0,),
        ("uint8", (0, 256), (4, 16, 8, 8), (4, ), 1, 0,),
        ("int8", (-128, 128), (8, 128), (3, 4), 0, 0,),
        ("uint8", (0, 256), (5, 8, 128), (5, 4), 1, 1,),
    ]
)
def test_take(
    dtype: str,
    val_range: ValueRange,
    shape: Tuple[int, ...],
    indices_shape: Tuple[int, ...],
    axis: int,
    num_batch_dims: int,
    remote: rpc.RPCSession
):
    inputs: Dict[str, tvm.nd.NDArray] = {
        "input": tvm.nd.array(
            np.random.uniform(
                *val_range, size=shape
            ).astype(dtype)
        ),
        "indices": tvm.nd.array(
            np.random.randint(
                low=0,
                high=shape[axis],
                size=indices_shape,
                dtype=np.int32
            )
        )
    }
    data = relay.var("input", shape=shape, dtype=dtype)
    indices = relay.var("indices", shape=indices_shape, dtype="int32")

    call = relay.take(data, indices, axis, num_batch_dims)

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
    ("dtype", "val_range", "shape", "indices_shape", "axis",),
    [
        ("uint8", (0, 256), (64, 64), (8, 64), 0,),
        ("int8", (-128, 128), (64, 64), (64, 8), -1,),
        ("uint8", (0, 256), (8, 3, 16, 16), (8, 1, 16, 16), 1,),
    ]
)
def test_gather(
    dtype: str,
    val_range: ValueRange,
    shape: Tuple[int, ...],
    indices_shape: Tuple[int, ...],
    axis: int,
    remote: rpc.RPCSession
):
    inputs: Dict[str, tvm.nd.NDArray] = {
        "input": tvm.nd.array(
            np.random.uniform(
                *val_range, size=shape
            ).astype(dtype)
        ),
        "indices": tvm.nd.array(
            np.random.randint(
                low=0,
                high=shape[axis],
                size=indices_shape,
                dtype=np.int32
            )
        )
    }
    data = relay.var("input", shape=shape, dtype=dtype)
    indices = relay.var("indices", shape=indices_shape, dtype="int32")

    call = relay.gather(data, axis, indices)

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
