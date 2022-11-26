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
"""TIM-VX conv2d op tests."""

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
from typing import Sequence, Tuple, Dict


def _get_requantize_scale(input_scale: relay.Constant, kernel_scale: relay.Constant) -> relay.Constant:
    requantize_scale = input_scale.data.numpy() * kernel_scale.data.numpy()
    return relay.const(requantize_scale)


def _get_kernel_info(kernel_shape: Sequence[int], kernel_layout: str) -> Tuple[Sequence[int], int]:
    kernel_size = ()
    channels = -1
    if kernel_layout == "OIHW":
        kernel_size = kernel_shape[2:]
        channels = kernel_shape[0]
    elif kernel_layout == "HWIO":
        kernel_size = kernel_shape[:2]
        channels = kernel_shape[3]
    elif kernel_layout == "OHWI":
        kernel_size = kernel_shape[1:3]
        channels = kernel_shape[0]
    return kernel_size, channels


@tvm.testing.requires_tim_vx
@pytest.mark.usefixtures("remote")
@pytest.mark.parametrize(
    (
        "input_dtype",  "input_range", "input_shape", "input_layout",
        "kernel_dtype",  "kernel_range", "kernel_shape", "kernel_layout",
        "bias_dtype", "bias_range",
        "output_dtype",
        "qnn_params",
        "strides", "padding", "dilation", "groups",
    ),
    [
        (
            "float32",  (-1.0, 1.0),  (1, 3, 64, 64), "NCHW",  # input
            "float32",  (-2.0, 2.0), (16, 3, 3, 3), "OIHW",  # kernel
            "float32", (0.0, 1.0),  # bias
            "float32",  # output
            {},  # qnn params,
            (1, 1), (1, 1, 1, 1), (1, 1), 1,  # normal
        ),
        (
            "float32", (-1.0, 1.0), (1, 16, 32, 32), "NCHW",  # input
            "float32", (-2.0, 2.0), (16, 1, 3, 3), "OIHW",  # kernel
            "float32", (0.0, 1.0),  # bias
            "float32",  # output -> (1, 16, 32, 32)
            {},  # qnn params,
            (1, 1), (1, 1, 1, 1), (1, 1), 16,  # depthwise
        ),
        (
            "float32",  (-1.0, 1.0), (1, 16, 32, 32), "NCHW",  # input
            "float32", (-2.0, 2.0), (8, 4, 3, 3), "OIHW",  # kernel
            "float32", (0.0, 1.0),  # bias
            "float32",  # output -> (1, 8, 32, 32)
            {},  # qnn params,
            (1, 1), (1, 1, 1, 1), (1, 1), 4,  # 4-grouped
        ),
        (
            "float32", (-1.0, 1.0), (1, 3, 64, 64), "NCHW",  # input
            "float32", (-2.0, 2.0), (16, 3, 2, 2), "OIHW",  # kernel
            "float32", (0.0, 1.0),  # bias
            "float32",  # output
            {},  # qnn params,
            (2, 2), (0, 0, 0, 0), (1, 1), 1,  # strided: 2 + no padding
        ),
        (
            "float32", (-1.0, 1.0), (1, 3, 64, 64), "NCHW",  # input
            "float32", (-2.0, 2.0), (16, 3, 3, 3), "OIHW",  # kernel
            "float32", (0.0, 1.0),  # bias
            "float32",  # output
            {},  # qnn params,
            (1, 1), (2, 2, 2, 2), (2, 2), 1,  # dilated: 2 + edge padding of 2
        ),
        (
            "float32", (-1.0, 1.0), (1, 3, 64, 64), "NCHW",  # input
            "float32", (-2.0, 2.0), (16, 3, 3, 3), "OIHW",  # kernel
            "float32", (0.0, 1.0),  # bias
            "float32",  # output
            {},  # qnn params,
            (1, 1), (0, 0, 3, 3), (1, 1), 1,  # asymmetric padding
        ),
        (
            "uint8", (0, 256), (1, 3, 128, 128), "NCHW",  # input
            "uint8", (0, 256), (8, 3, 3, 3), "OIHW",  # kernel
            "int32", (-1e4, 1e4),  # bias
            "uint8",   # output
            {
                "input_scale": relay.const(2 / 255),
                "input_zero_point": relay.const(128),
                "kernel_scale": relay.const(1 / 255),
                "kernel_zero_point": relay.const(128),
                "output_scale": relay.const(4 / 255),
                "output_zero_point": relay.const(128),
            },  # qnn params,
            (1, 1), (1, 1, 1, 1), (1, 1), 1,  # normal
        ),
        (
            "uint8", (0, 256), (1, 3, 128, 128), "NCHW",  # input
            "int8", (-128, 128), (8, 3, 3, 3), "OIHW",  # kernel
            "int32", (-1e4, 1e4),  # bias
            "uint8",   # output
            {
                "input_scale": relay.const(1 / 255),
                "input_zero_point": relay.const(128),
                "kernel_scale": relay.const(1 / 127),
                "kernel_zero_point": relay.const(0),
                "output_scale": relay.const(4 / 255),
                "output_zero_point": relay.const(128),
            },  # qnn params,
            (1, 1), (1, 1, 1, 1), (1, 1), 1,  # normal
        ),
        # (
        #     "uint8", (0, 256), (1, 3, 128, 128), "NCHW",  # input
        #     "int8", (-128, 128), (4, 3, 3, 3), "OIHW",  # kernel
        #     "int32", (-1e4, 1e4),  # bias
        #     "uint8",   # output
        #     {
        #         "input_scale": relay.const(1 / 255),
        #         "input_zero_point": relay.const(128),
        #         "kernel_scale": relay.const([
        #             1 / 127,
        #             2 / 127,
        #             3 / 127,
        #             4 / 127,
        #         ]),
        #         "kernel_zero_point": relay.const(0),
        #         "output_scale": relay.const(12 / 255),
        #         "output_zero_point": relay.const(128),
        #     },  # qnn params (per-channel),
        #     (1, 1), (1, 1, 1, 1), (1, 1), 1,  # normal
        # )
    ]
)
@pytest.mark.parametrize(
    "has_bias",
    [False, True]
)
def test_conv2d(
    input_dtype: str,
    input_range: ValueRange,
    input_shape: Tuple[int, int, int, int],
    input_layout: str,
    kernel_dtype: str,
    kernel_range: ValueRange,
    kernel_shape: Tuple[int, int, int, int],
    kernel_layout: str,
    bias_dtype: str,
    bias_range: ValueRange,
    output_dtype: str,
    qnn_params: Dict[str, relay.Constant],
    strides: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int, int, int],
    groups: int,
    has_bias: bool,
    remote: rpc.RPCSession,
):
    inputs: Dict[str, tvm.nd.NDArray] = {
        "input": tvm.nd.array(
            np.random.uniform(
                *input_range,
                size=input_shape
            ).astype(input_dtype)
        ),
    }

    data = relay.var(
        "input",
        shape=input_shape,
        dtype=input_dtype
    )

    weight = relay.const(
        np.random.uniform(
            *kernel_range,
            size=kernel_shape
        ).astype(kernel_dtype)
    )

    kernel_size, channels = _get_kernel_info(kernel_shape, kernel_layout)
    if qnn_params:
        call = relay.qnn.op.conv2d(
            data=data,
            kernel=weight,
            input_scale=qnn_params["input_scale"],
            input_zero_point=qnn_params["input_zero_point"],
            kernel_scale=qnn_params["kernel_scale"],
            kernel_zero_point=qnn_params["kernel_zero_point"],
            kernel_size=kernel_size,
            channels=channels,
            strides=strides,
            padding=padding,
            dilation=dilation,
            groups=groups,
            data_layout=input_layout,
            kernel_layout=kernel_layout,
            out_dtype=bias_dtype
        )
    else:
        call = relay.nn.conv2d(
            data=data,
            weight=weight,
            kernel_size=kernel_size,
            channels=channels,
            strides=strides,
            padding=padding,
            dilation=dilation,
            groups=groups,
            data_layout=input_layout,
            kernel_layout=kernel_layout,
            out_dtype=bias_dtype
        )

    if has_bias:
        bias = relay.const(
            np.random.uniform(
                *bias_range,
                size=(channels, )
            ).astype(bias_dtype)
        )
        call = relay.nn.bias_add(call, bias)

    if qnn_params:
        requantize_scale = _get_requantize_scale(
            qnn_params["input_scale"], qnn_params["kernel_scale"]
        )
        call = relay.qnn.op.requantize(
            call,
            input_scale=requantize_scale,
            input_zero_point=relay.const(0, dtype="int32"),
            output_scale=qnn_params["output_scale"],
            output_zero_point=qnn_params["output_zero_point"],
            out_dtype=output_dtype,
            axis=input_layout.find("C"),
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

    atol, rtol = (0, 0) if qnn_params else (1e-5, 1e-5)
    verify(tim_vx_outputs, ref_outputs, atol, rtol)


@tvm.testing.requires_tim_vx
@pytest.mark.usefixtures("remote")
@pytest.mark.parametrize(
    (
        "features_dtype", "features_range", "features_shape", "features_layout",
        "template_dtype", "template_range", "template_shape", "template_layout",
        "bias_dtype", "bias_range",
        "output_dtype",
        "qnn_params",
        "strides", "padding",
    ),
    [
        (
            "float32", (-1.0, 1.0), (1, 4, 32, 32), "NCHW",  # input
            "float32", (-1.0, 1.0), (4, 1, 24, 24), "OIHW",  # kernel
            "float32", (0.0, 1.0),  # bias
            "float32",  # output
            {},  # qnn params,
            (1, 1), (0, 0, 0, 0),  # normal
        ),
        (
            "uint8", (0, 256), (1, 16, 32, 32), "NCHW",  # input
            "uint8", (0, 256), (16, 1, 24, 24), "OIHW",  # kernel
            "int32", (0, 10000),  # bias
            "uint8",  # output
            {
                "input_scale": relay.const(2 / 255),
                "input_zero_point": relay.const(128),
                "kernel_scale": relay.const(2 / 255),
                "kernel_zero_point": relay.const(128),
                "output_scale": relay.const(2 / 255),
                "output_zero_point": relay.const(128),
            },  # qnn params,
            (1, 1), (0, 0, 0, 0),  # normal
        ),
    ]
)
@pytest.mark.parametrize(
    "has_bias",
    [False]
)
def test_xcorr2d(
    features_dtype: str,
    features_range: ValueRange,
    features_shape: Tuple[int, int, int, int],
    features_layout: str,
    template_dtype: str,
    template_range: ValueRange,
    template_shape: Tuple[int, int, int, int],
    template_layout: str,
    bias_dtype: str,
    bias_range: ValueRange,
    output_dtype: str,
    qnn_params: Dict[str, relay.Constant],
    strides: Tuple[int, int],
    padding: Tuple[int, int, int, int],
    has_bias: bool,
    remote: rpc.RPCSession,
):
    inputs: Dict[str, tvm.nd.NDArray] = {
        "features": tvm.nd.array(
            np.random.uniform(
                *features_range,
                size=features_shape
            ).astype(features_dtype)
        ),
        "template": tvm.nd.array(
            np.random.uniform(
                *template_range,
                size=template_shape
            ).astype(template_dtype)
        ),
    }

    features = relay.var(
        "features",
        shape=features_shape,
        dtype=features_dtype
    )

    template = relay.var(
        "template",
        shape=template_shape,
        dtype=template_dtype
    )

    kernel_size, channels = _get_kernel_info(template_shape, template_layout)
    if qnn_params:
        requantize_scale = _get_requantize_scale(
            qnn_params["input_scale"], qnn_params["kernel_scale"]
        )
        qnn_params["requantize_scale"] = requantize_scale
        qnn_params["requantize_zero_point"] = relay.const(0, dtype="int32")

        call = relay.qnn.op.conv2d(
            data=features,
            kernel=template,
            input_zero_point=qnn_params["input_zero_point"],
            kernel_zero_point=qnn_params["kernel_zero_point"],
            input_scale=qnn_params["input_scale"],
            kernel_scale=qnn_params["kernel_scale"],
            kernel_size=kernel_size,
            channels=channels,
            strides=strides,
            padding=padding,
            groups=channels,
            data_layout=features_layout,
            kernel_layout=template_layout,
            out_dtype=bias_dtype
        )
    else:
        call = relay.nn.conv2d(
            data=features,
            weight=template,
            kernel_size=kernel_size,
            channels=channels,
            strides=strides,
            padding=padding,
            groups=channels,
            data_layout=features_layout,
            kernel_layout=template_layout,
            out_dtype=bias_dtype
        )

    if has_bias:
        bias = relay.const(
            np.random.uniform(
                *bias_range,
                size=(channels, )
            ).astype(bias_dtype)
        )
        call = relay.nn.bias_add(call, bias)

    if qnn_params:
        call = relay.qnn.op.requantize(
            call,
            input_scale=qnn_params["requantize_scale"],
            input_zero_point=qnn_params["requantize_zero_point"],
            output_scale=qnn_params["output_scale"],
            output_zero_point=qnn_params["output_zero_point"],
            out_dtype=output_dtype,
            axis=features_layout.find("C"),
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

    atol, rtol = (0, 0) if qnn_params else (1e-5, 1e-5)
    verify(tim_vx_outputs, ref_outputs, atol, rtol)


if __name__ == "__main__":
    pytest.main([__file__])
