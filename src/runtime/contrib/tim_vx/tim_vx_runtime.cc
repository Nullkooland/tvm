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
 * \file src/runtime/contrib/tim_vx/tim_vx_runtime.cc
 * \brief TIM-VX NBG runtime.
 */

#include "tim_vx_runtime.h"

#include <tim/vx/ops/nbg.h>

#include <fstream>

#include "nbg_parser.h"

namespace tvm {
namespace runtime {
namespace contrib {
namespace tim_vx {

TimVxRuntime::TimVxRuntime(const std::string& symbol_name, std::vector<char> nbg_buffer)
    : symbol_name_(symbol_name), nbg_buffer_(std::move(nbg_buffer)) {
  // Create TIM-VX context.
  context_ = tim::vx::Context::Create();
  ICHECK(context_) << "Failed to create TIM-VX context.";
}

PackedFunc TimVxRuntime::GetFunction(const std::string& name,
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

void TimVxRuntime::Init(const TVMArgs& args) {
  NBGParser parser(nbg_buffer_.data(), nbg_buffer_.size());

  size_t num_inputs = parser.QueryNetworkNumInputs();
  size_t num_outputs = parser.QueryNetworkNumOutputs();

  ICHECK_EQ(args.size(), num_inputs + num_outputs)
      << "Number of passed args is mismatched with the NBG input/output tensors.";

  graph_ = context_->CreateGraph();
  ICHECK(graph_) << "Failed to create TIM-VX graph for inference.";

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

    // Query buffer type.
    auto buffer_type = parser.QueryInputBufferType(i);
    ICHECK_EQ(buffer_type, NBGBufferType::TENSOR) << "Only supports  tensor as input";

    // Query data type.
    auto nbg_data_type = parser.QueryInputDataType(i);

    auto vx_data_type = ConvertDataType(nbg_data_type);
    ICHECK(vx_data_type != tim::vx::DataType::UNKNOWN)
        << "Unsupported NBG data type: " << static_cast<int>(nbg_data_type);

    // Query shape.
    size_t num_dims = parser.QueryInputNumDims(i);
    auto dims = parser.QueryInputDims(i);
    auto shape = tim::vx::ShapeType(num_dims);
    std::copy_n(dims.cbegin(), num_dims, shape.begin());

    auto spec = tim::vx::TensorSpec(vx_data_type, shape, tim::vx::TensorAttribute::INPUT);

    // Query quantization info.
    auto nbg_quant_type = parser.QueryInputQuantType(i);
    auto vx_quant_type = ConvertQuantType(nbg_quant_type);
    if (vx_quant_type == tim::vx::QuantType::ASYMMETRIC) {
      float scale = parser.QueryInputQuantScale(i);
      int zero_point = parser.QueryInputQuantZeroPoint(i);
      auto quant_info = tim::vx::Quantization(vx_quant_type, scale, zero_point);
      spec.SetQuantization(quant_info);
    } else if (vx_quant_type == tim::vx::QuantType::DYNAMIC_FIXED_POINT) {
      int fl = parser.QueryInputQuantDfpPos(i);
      auto quant_info = tim::vx::Quantization(vx_quant_type, static_cast<int8_t>(fl));
      spec.SetQuantization(quant_info);
    }

    if (nd_array->device.device_type == DLDeviceType::kDLCPU) {
      input_tensors_.push_back(graph_->CreateIOTensor(spec, nd_array->data));
    } else if (nd_array->device.device_type == DLDeviceType::kDLExtDev) {
      auto dma_buf = tim::vx::DmaBufferDesc{.fd = reinterpret_cast<int64_t>(nd_array->data)};
      input_tensors_.push_back(graph_->CreateTensor(spec, dma_buf));
    }
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

    // Query buffer type.
    auto buffer_type = parser.QueryOutputBufferType(i);
    ICHECK_EQ(buffer_type, NBGBufferType::TENSOR) << "Only supports  tensor as output";

    // Query data type.
    auto nbg_data_type = parser.QueryOutputDataType(i);

    auto vx_data_type = ConvertDataType(nbg_data_type);
    ICHECK(vx_data_type != tim::vx::DataType::UNKNOWN)
        << "Unsupported NBG data type: " << static_cast<int>(nbg_data_type);

    // Query shape.
    size_t num_dims = parser.QueryOutputNumDims(i);
    auto dims = parser.QueryOutputDims(i);
    auto shape = tim::vx::ShapeType(num_dims);
    std::copy_n(dims.cbegin(), num_dims, shape.begin());

    auto spec = tim::vx::TensorSpec(vx_data_type, shape, tim::vx::TensorAttribute::OUTPUT);

    // Query quantization info.
    auto nbg_quant_type = parser.QueryOutputQuantType(i);
    auto vx_quant_type = ConvertQuantType(nbg_quant_type);
    if (vx_quant_type == tim::vx::QuantType::ASYMMETRIC) {
      float scale = parser.QueryOutputQuantScale(i);
      int zero_point = parser.QueryOutputQuantZeroPoint(i);
      auto quant_info = tim::vx::Quantization(vx_quant_type, scale, zero_point);
      spec.SetQuantization(quant_info);
    } else if (vx_quant_type == tim::vx::QuantType::DYNAMIC_FIXED_POINT) {
      int fl = parser.QueryOutputQuantDfpPos(i);
      auto quant_info = tim::vx::Quantization(vx_quant_type, static_cast<int8_t>(fl));
      spec.SetQuantization(quant_info);
    }

    if (nd_array->device.device_type == DLDeviceType::kDLCPU) {
      output_tensors_.push_back(graph_->CreateIOTensor(spec, nd_array->data));
    } else if (nd_array->device.device_type == DLDeviceType::kDLExtDev) {
      auto dma_buf = tim::vx::DmaBufferDesc{.fd = reinterpret_cast<int64_t>(nd_array->data)};
      output_tensors_.push_back(graph_->CreateTensor(spec, dma_buf));
    }
  }

  // Create NBG node.
  auto nbg_node =
      graph_->CreateOperation<tim::vx::ops::NBG>(nbg_buffer_.data(), num_inputs, num_outputs);
  // Bind inputs/outputs.
  nbg_node->BindInputs(input_tensors_);
  nbg_node->BindOutputs(output_tensors_);

  ICHECK(graph_->Compile()) << "Failed to compile TIM-VX graph";
  ICHECK(graph_->Run()) << "Failed to pre-run TIM-VX graph";
}

void TimVxRuntime::Run() {
  for (auto& tensor : input_tensors_) {
    tensor->FlushCacheForHandle();
  }

  ICHECK(graph_->Run()) << "Failed to run TIM-VX graph";

  for (auto& tensor : output_tensors_) {
    tensor->InvalidateCacheForHandle();
  }
}

void TimVxRuntime::SaveToBinary(dmlc::Stream* stream) {
  // Save symbol name.
  stream->Write(symbol_name_);
  // Save NBG.
  stream->Write(nbg_buffer_);
}

void TimVxRuntime::SaveToFile(const std::string& file_name, const std::string& format) {
  std::ofstream fs(file_name, std::ios::out | std::ios::binary);
  ICHECK(fs.is_open()) << "Cannot open NBG file: " << file_name;
  fs.write(nbg_buffer_.data(), nbg_buffer_.size());
}

Module TimVxRuntime::LoadFromBinary(void* raw_stream) {
  auto* stream = reinterpret_cast<dmlc::Stream*>(raw_stream);
  // Read symbol name.
  std::string symbol_name;
  ICHECK(stream->Read(&symbol_name)) << "Failed to load symbol name";
  // Read NBG.
  std::vector<char> nbg_buffer;
  ICHECK(stream->Read(&nbg_buffer)) << "Failed to load NBG";

  DLOG(INFO) << "NBG size: " << nbg_buffer.size() << " bytes";
  DLOG(INFO) << "Symbol name: " << symbol_name;

  auto n = make_object<TimVxRuntime>(symbol_name, std::move(nbg_buffer));
  return Module(n);
}

TVM_REGISTER_GLOBAL("runtime.tim_vx_runtime_create")
    .set_body_typed([](String symbol_name, const void* p_nbg_buffer, size_t nbg_size) {
      std::vector<char> nbg_buffer(nbg_size);
      std::copy_n(reinterpret_cast<const char*>(p_nbg_buffer), nbg_size, nbg_buffer.begin());
      auto n = make_object<TimVxRuntime>(symbol_name, nbg_buffer);
      return runtime::Module(n);
    });

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_tim_vx").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = TimVxRuntime::LoadFromBinary(args[0]);
});

}  // namespace tim_vx
}  // namespace contrib
}  // namespace runtime
}  // namespace tvm