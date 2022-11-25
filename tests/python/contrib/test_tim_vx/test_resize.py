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
"""TIM-VX resize ops tests."""

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
    ("dtype", "val_range", "shape",
        "size", "method", "coordinate_transformation_mode"
     ),
    [
        (
            "float32", (-1.0, 1.0), (1, 8, 32, 32),
            (64, 64), "nearest_neighbor", "half_pixel"
        ),
        (
            "float32", (-1.0, 1.0), (1, 3, 32, 32),
            (16, 16), "linear", "half_pixel"
        ),
        (
            "uint8", (0, 256), (1, 8, 64, 64),
            (128, 128), "nearest_neighbor", "half_pixel"
        ),
        (
            "uint8", (0, 256), (1, 8, 64, 64),
            (128, 128), "nearest_neighbor", "asymmetric"
        ),
        (
            "uint8", (0, 256), (1, 8, 64, 64),
            (128, 128), "linear", "half_pixel"
        ),
        (
            "uint8", (0, 256), (1, 16, 48, 48),
            (16, 16), "linear", "half_pixel"
        ),
        (
            "uint8", (0, 256), (1, 32, 9, 9),
            (17, 17), "linear", "align_corners"
        ),
        (
            "uint8", (0, 256), (1, 32, 27, 27),
            (9, 9), "nearest_neighbor", "align_corners"
        ),
        (
            "int8", (-128, 128), (1, 8, 64, 64),
            (128, 128), "nearest_neighbor", "half_pixel"
        ),
        (
            "int16", (-32768, 32768), (1, 8, 32, 32),
            (64, 64), "nearest_neighbor", "half_pixel"
        ),
    ]
)
def test_resize2d(
    dtype: str,
    val_range: ValueRange,
    shape: Tuple[int, ...],
    size: Tuple[int, int],
    method: str,
    coordinate_transformation_mode: str,
    remote: rpc.RPCSession,
):
    inputs: Dict[str, tvm.nd.NDArray] = {
        "input": tvm.nd.array(
            np.random.uniform(*val_range, size=shape).astype(dtype)
        ),
    }
    data = relay.var("input", shape=shape, dtype=dtype)

    call = relay.image.resize2d(
        data,
        size=size,
        method=method,
        coordinate_transformation_mode=coordinate_transformation_mode,
        rounding_method="round"
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

    atol, rtol = (1, 1.0 / np.iinfo(dtype).max) if np.issubdtype(
        np.dtype(dtype), np.integer
    ) else (1e-6, 1e-6)
    verify(tim_vx_outputs, ref_outputs, atol, rtol)


if __name__ == "__main__":
    pytest.main([__file__])
