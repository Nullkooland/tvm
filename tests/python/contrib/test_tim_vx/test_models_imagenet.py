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
"""TIM-VX ImageNet classification models tests."""

from typing import Dict
import pytest

import numpy as np

import torch
import torch.utils
from torchvision.io.image import ImageReadMode, read_image
from torchvision.transforms import functional_tensor as F

import tvm
import tvm.testing
from tvm import relay, rpc
from tvm.contrib.download import download_testdata

from test_tim_vx import qnn_utils
from test_tim_vx.infrastructure import build_and_run


@pytest.fixture(scope="module", name="input_tensor")
def get_input_tensor() -> torch.Tensor:
    TEST_IMG_URL = "https://s3.amazonaws.com/model-server/inputs/kitten.jpg"
    img_path = download_testdata(TEST_IMG_URL, "kitten.jpg", module="data")
    img = read_image(img_path, ImageReadMode.RGB)
    img = F.resize(img, [256, 256], interpolation="bicubic", antialias=True)
    img = F.crop(img, 32, 32, 224, 224)
    return torch.unsqueeze(img, dim=0)  # -> [1, 3, 224, 224]


@tvm.testing.requires_tim_vx
@pytest.mark.usefixtures("input_tensor")
@pytest.mark.usefixtures("remote")
@pytest.mark.parametrize(
    "onnx_name",
    [
        "resnet_18_quint8_ppq.onnx",
        "mobilenet_v2_quint8_ort.onnx",
        "shufflenet_v2_qint8.onnx",
        "regnet_x_400mf_quint8.onnx",
    ]
)
def test_onnx(
    onnx_name: str,
    input_tensor: torch.Tensor,
    remote: rpc.RPCSession
):
    import onnx
    # Put testing ONNX models to "$HOME/.tvm_test_data/onnx/".
    onnx_path = download_testdata("", onnx_name, module="onnx")
    onnx_mod = onnx.load(onnx_path)
    mod, params = relay.frontend.from_onnx(
        model=onnx_mod,
        shape={"input": input_tensor.shape}
    )

    inputs: Dict[str, tvm.nd.NDArray] = {
        "input": tvm.nd.array(input_tensor),
    }

    with tvm.transform.PassContext(opt_level=3):
        optimization_pass = tvm.transform.Sequential([
            relay.transform.InferType(),
            relay.transform.FakeQuantizationToInteger(),
            relay.transform.FoldConstant(fold_qnn=True),
            qnn_utils.InsertImagePreProcess(nhwc_to_nchw=False),
            qnn_utils.FoldQnnBinaryOpExplicitRequantize(),
            qnn_utils.InsertClassificationPostProcess(k=5),
        ])
        mod = optimization_pass(mod)

    tim_vx_outputs = build_and_run(
        mod,
        inputs,
        params=params,
        build_for_tim_vx=True,
        expected_num_cpu_ops=0,
        expected_num_tim_vx_subgraphs=1,
        remote=remote
    )

    ref_outputs = build_and_run(
        mod,
        inputs,
        params=params,
        build_for_tim_vx=False
    )

    tim_vx_cls = tim_vx_outputs[1].numpy()
    ref_cls = ref_outputs[1].numpy()
    assert np.all(tim_vx_cls == ref_cls)


if __name__ == "__main__":
    pytest.main([__file__])
