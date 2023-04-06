/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/runtime/contrib/tim_vx/ovx_runtime.cc
 * \brief OpenVX NBG runtime.
 */

#include "ovx_runtime.h"

#include <VX/vx.h>
#include <VX/vx_api.h>
#include <VX/vx_khr_import_kernel.h>
#include <VX/vx_khr_nn.h>

#include <fstream>

#include "nbg_parser.h"

namespace tvm {
namespace runtime {
namespace contrib {
namespace tim_vx {

OpenVXRuntime::OpenVXRuntime(const std::string& symbol_name, std::vector<char> nbg_buffer)
    : symbol_name_(symbol_name), nbg_buffer_(std::move(nbg_buffer)) {
  // Create OpenVX context.
  context_ = vxCreateContext();
  ICHECK(context_) << "Failed to create OpenVX context";
}

OpenVXRuntime::~OpenVXRuntime() {
  for (auto& tensor : input_tensors_) {
    vxReleaseTensor(&tensor);
  }
  for (auto& tensor : output_tensors_) {
    vxReleaseTensor(&tensor);
  }

  vxReleaseNode(&nbg_node_);
  vxReleaseKernel(&nbg_kernel_);
  vxReleaseGraph(&graph_);
  vxReleaseContext(&context_);
}

PackedFunc OpenVXRuntime::GetFunction(const std::string& name,
                                      const ObjectPtr<Object>& sptr_to_self) {
  if (name == "get_symbol") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = symbol_name_; });
  }
  if (name == symbol_name_) {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      // Initialize the subgraph.
      std::call_once(
          this->initialized_, [this](const TVMArgs& args) { this->Init(args); }, args);
      // Execute the subgraph.
      this->Run();
    });
  }
  // LOG(WARNING) << "Function: " << name << " not implemented";
  return PackedFunc(nullptr);
}

void OpenVXRuntime::Init(const TVMArgs& args) {
  NBGParser parser(nbg_buffer_.data(), nbg_buffer_.size());

  size_t num_inputs = parser.QueryNetworkNumInputs();
  size_t num_outputs = parser.QueryNetworkNumOutputs();

  ICHECK_EQ(args.size(), num_inputs + num_outputs)
      << "Number of passed args is mismatched with the NBG input/output tensors";

  VXStatus status;
  graph_ = vxCreateGraph(context_);
  status = vxGetStatus(reinterpret_cast<VXRef>(graph_));
  ICHECK_EQ(status, VX_SUCCESS) << "Failed to create OpenVX graph";

  nbg_kernel_ =
      vxImportKernelFromURL(context_, VX_VIVANTE_IMPORT_KERNEL_FROM_POINTER, nbg_buffer_.data());
  status = vxGetStatus(reinterpret_cast<VXRef>(nbg_kernel_));
  ICHECK_EQ(status, VX_SUCCESS) << "Failed to import NBG kernel";

  nbg_node_ = vxCreateGenericNode(graph_, nbg_kernel_);
  status = vxGetStatus(reinterpret_cast<VXRef>(nbg_node_));
  ICHECK_EQ(status, VX_SUCCESS) << "Failed to create NBG node";

  // Parse input tensor specs.
  input_tensors_.reserve(num_inputs);
  for (size_t i = 0; i < num_inputs; i++) {
    auto arg = args[i];
    NDArray nd_array;
    if (arg.type_code() == TVMArgTypeCode::kTVMDLTensorHandle) {
      nd_array = NDArray::FromExternalDLTensor(*arg);
    } else if (arg.type_code() == TVMArgTypeCode::kTVMNDArrayHandle) {
      nd_array = arg.AsObjectRef<NDArray>();
    } else {
      LOG(FATAL) << "Expect NDArray or DLTensor as arguments";
    }

    VXTensorCreateParams params = {};

    // Query buffer type.
    auto buffer_type = parser.QueryInputBufferType(i);
    ICHECK_EQ(buffer_type, NBGBufferType::TENSOR) << "Only supports tensor as input";

    // Query data type.
    auto nbg_data_type = parser.QueryInputDataType(i);
    auto vx_data_type = ConvertDataType(nbg_data_type);
    ICHECK_NE(vx_data_type, VXDataType::VX_TYPE_INVALID)
        << "Unsupported NBG data type: " << static_cast<int>(nbg_data_type);
    params.data_format = vx_data_type;

    // Query shape.
    size_t num_dims = parser.QueryInputNumDims(i);
    auto dims = parser.QueryInputDims(i);
    params.num_of_dims = num_dims;
    params.sizes = dims.data();

    // Query quantization info.
    auto nbg_quant_type = parser.QueryInputQuantType(i);
    auto vx_quant_type = ConvertQuantType(nbg_quant_type);
    params.quant_format = vx_quant_type;

    if (vx_quant_type == VXQuantType::VX_QUANT_AFFINE_SCALE) {
      float scale = parser.QueryInputQuantScale(i);
      int zero_point = parser.QueryInputQuantZeroPoint(i);
      params.quant_data.affine.scale = scale;
      params.quant_data.affine.zeroPoint = zero_point;
    } else if (vx_quant_type == VXQuantType::VX_QUANT_DYNAMIC_FIXED_POINT) {
      int fl = parser.QueryInputQuantDfpPos(i);
      params.quant_data.dfp.fixed_point_pos = static_cast<int8_t>(fl);
    }

    // Compute OpenVX tensor data strides.
    std::array<uint32_t, dims.size()> strides;
    strides[0] = GetDataTypeBytes(vx_data_type);
    for (size_t j = 1; j < num_dims; j++) {
      strides[j] = strides[j - 1] * dims[j - 1];
    }
    VXTensorAddressing addressing = vxCreateTensorAddressing(context_, dims.data(), strides.data(),
                                                             static_cast<uint8_t>(num_dims));

    // Create OpenVX tensor.
    VXTensor tensor = nullptr;
    if (nd_array->device.device_type == DLDeviceType::kDLCPU) {
      tensor = vxCreateTensorFromHandle2(context_, &params, sizeof(VXTensorCreateParams),
                                         addressing, nd_array->data, VX_MEMORY_TYPE_HOST);
    } else if (nd_array->device.device_type == DLDeviceType::kDLExtDev) {
      tensor = vxCreateTensorFromHandle2(context_, &params, sizeof(VXTensorCreateParams),
                                         addressing, nd_array->data, VX_MEMORY_TYPE_DMABUF);
    }

    status = vxGetStatus(reinterpret_cast<VXRef>(tensor));
    ICHECK_EQ(status, VX_SUCCESS) << "Failed to create OpenVX tensor";

    status = vxSetParameterByIndex(nbg_node_, i, reinterpret_cast<VXRef>(tensor));
    ICHECK_EQ(status, VX_SUCCESS) << "Failed to bind input tensor:" << i;

    input_tensors_.push_back(tensor);
    vxReleaseTensorAddressing(&addressing);
  }

  // Parse output tensor specs.
  output_tensors_.reserve(num_outputs);
  for (size_t i = 0; i < num_outputs; i++) {
    auto arg = args[num_inputs + i];
    NDArray nd_array;
    if (arg.type_code() == TVMArgTypeCode::kTVMDLTensorHandle) {
      nd_array = NDArray::FromExternalDLTensor(*arg);
    } else if (arg.type_code() == TVMArgTypeCode::kTVMNDArrayHandle) {
      nd_array = arg.AsObjectRef<NDArray>();
    } else {
      LOG(FATAL) << "Expect NDArray or DLTensor as arguments";
    }

    VXTensorCreateParams params = {};

    // Query buffer type.
    auto buffer_type = parser.QueryOutputBufferType(i);
    ICHECK_EQ(buffer_type, NBGBufferType::TENSOR) << "Only supports tensor as output";

    // Query data type.
    auto nbg_data_type = parser.QueryOutputDataType(i);
    auto vx_data_type = ConvertDataType(nbg_data_type);
    ICHECK_NE(vx_data_type, VXDataType::VX_TYPE_INVALID)
        << "Unsupported NBG data type: " << static_cast<int>(nbg_data_type);
    params.data_format = vx_data_type;

    // Query shape.
    size_t num_dims = parser.QueryOutputNumDims(i);
    auto dims = parser.QueryOutputDims(i);
    params.num_of_dims = num_dims;
    params.sizes = dims.data();

    // Query quantization info.
    auto nbg_quant_type = parser.QueryOutputQuantType(i);
    auto vx_quant_type = ConvertQuantType(nbg_quant_type);
    params.quant_format = vx_quant_type;

    if (vx_quant_type == VXQuantType::VX_QUANT_AFFINE_SCALE) {
      float scale = parser.QueryOutputQuantScale(i);
      int zero_point = parser.QueryOutputQuantZeroPoint(i);
      params.quant_data.affine.scale = scale;
      params.quant_data.affine.zeroPoint = zero_point;
    } else if (vx_quant_type == VXQuantType::VX_QUANT_DYNAMIC_FIXED_POINT) {
      int fl = parser.QueryOutputQuantDfpPos(i);
      params.quant_data.dfp.fixed_point_pos = static_cast<int8_t>(fl);
    }

    // Compute OpenVX tensor data strides.
    std::array<uint32_t, dims.size()> strides;
    strides[0] = GetDataTypeBytes(vx_data_type);
    for (size_t j = 1; j < num_dims; j++) {
      strides[j] = strides[j - 1] * dims[j - 1];
    }
    VXTensorAddressing addressing = vxCreateTensorAddressing(context_, dims.data(), strides.data(),
                                                             static_cast<uint8_t>(num_dims));

    // Create OpenVX tensor.
    VXTensor tensor = nullptr;
    if (nd_array->device.device_type == DLDeviceType::kDLCPU) {
      tensor = vxCreateTensorFromHandle2(context_, &params, sizeof(VXTensorCreateParams),
                                         addressing, nd_array->data, VX_MEMORY_TYPE_HOST);
    } else if (nd_array->device.device_type == DLDeviceType::kDLExtDev) {
      tensor = vxCreateTensorFromHandle2(context_, &params, sizeof(VXTensorCreateParams),
                                         addressing, nd_array->data, VX_MEMORY_TYPE_DMABUF);
    }

    status = vxGetStatus(reinterpret_cast<VXRef>(tensor));
    ICHECK_EQ(status, VX_SUCCESS) << "Failed to create OpenVX tensor";

    status = vxSetParameterByIndex(nbg_node_, num_inputs + i, reinterpret_cast<VXRef>(tensor));
    ICHECK_EQ(status, VX_SUCCESS) << "Failed to bind output tensor:" << i;

    output_tensors_.push_back(tensor);
    vxReleaseTensorAddressing(&addressing);
  }

  status = vxVerifyGraph(graph_);
  ICHECK_EQ(status, VX_SUCCESS) << "Failed to verify OpenVX graph";
}

void OpenVXRuntime::Run() {
  VXStatus status;
  for (auto& tensor : input_tensors_) {
    status = vxFlushHandle(reinterpret_cast<VXRef>(tensor));
    ICHECK_EQ(status, VX_SUCCESS) << "Failed to flush input buffer";
  }

  status = vxProcessGraph(graph_);
  ICHECK_EQ(status, VX_SUCCESS) << "Failed to run OpenVX graph";

  for (auto& tensor : output_tensors_) {
    void* ptr;
    status = vxSwapTensorHandle(tensor, nullptr, &ptr);
    ICHECK_EQ(status, VX_SUCCESS) << "Failed to invalidate output buffer";
  }
}

void OpenVXRuntime::SaveToBinary(dmlc::Stream* stream) {
  // Save symbol name.
  stream->Write(symbol_name_);
  // Save NBG.
  stream->Write(nbg_buffer_);
}

void OpenVXRuntime::SaveToFile(const std::string& file_name, const std::string& format) {
  std::ofstream fs(file_name, std::ios::out | std::ios::binary);
  ICHECK(fs.is_open()) << "Cannot open NBG file: " << file_name;
  fs.write(nbg_buffer_.data(), nbg_buffer_.size());
}

Module OpenVXRuntime::LoadFromBinary(void* raw_stream) {
  auto* stream = reinterpret_cast<dmlc::Stream*>(raw_stream);
  // Read symbol name.
  std::string symbol_name;
  ICHECK(stream->Read(&symbol_name)) << "Failed to load symbol name";
  // Read NBG.
  std::vector<char> nbg_buffer;
  ICHECK(stream->Read(&nbg_buffer)) << "Failed to load NBG";

  DLOG(INFO) << "NBG size: " << nbg_buffer.size() << " bytes";
  DLOG(INFO) << "Symbol name: " << symbol_name;

  auto n = make_object<OpenVXRuntime>(symbol_name, std::move(nbg_buffer));
  return Module(n);
}

TVM_REGISTER_GLOBAL("runtime.tim_vx_runtime_create")
    .set_body_typed([](String symbol_name, const void* p_nbg_buffer, size_t nbg_size) {
      std::vector<char> nbg_buffer(nbg_size);
      std::copy_n(reinterpret_cast<const char*>(p_nbg_buffer), nbg_size, nbg_buffer.begin());
      auto n = make_object<OpenVXRuntime>(symbol_name, nbg_buffer);
      return runtime::Module(n);
    });

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_tim_vx").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = OpenVXRuntime::LoadFromBinary(args[0]);
});

}  // namespace tim_vx
}  // namespace contrib
}  // namespace runtime
}  // namespace tvm