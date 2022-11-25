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
 * \file src/runtime/contrib/tim_vx/viplite_runtime.cc
 * \brief VIPLite NBG runtime.
 */

#include "viplite_runtime.h"

#include <fstream>

namespace tvm {
namespace runtime {
namespace contrib {
namespace tim_vx {

VIPLiteNetwork::VIPLiteNetwork(const void* nbg_buffer, size_t nbg_size) {
  vip_status_e status = vip_create_network(
      nbg_buffer, nbg_size, vip_create_network_type_e::VIP_CREATE_NETWORK_FROM_MEMORY, &network_);
  ICHECK_EQ(status, vip_status_e::VIP_SUCCESS) << "Failed to create VIPLite network from NBG";

  vip_query_network(network_, vip_network_property_e::VIP_NETWORK_PROP_INPUT_COUNT, &num_inputs_);
  vip_query_network(network_, vip_network_property_e::VIP_NETWORK_PROP_OUTPUT_COUNT, &num_outputs_);

  name_ = std::string(64, '\0');
  vip_query_network(network_, vip_network_property_e::VIP_NETWORK_PROP_NETWORK_NAME, name_.data());
}

VIPLiteNetwork::~VIPLiteNetwork() {
  if (network_ == nullptr) {
    return;
  }
  vip_finish_network(network_);
  vip_destroy_network(network_);
  network_ = nullptr;
}

VIPLiteBufferParams VIPLiteNetwork::QueryInput(size_t index) const {
  VIPLiteBufferParams params = {};
  vip_query_input(network_, index, vip_buffer_property_e::VIP_BUFFER_PROP_DATA_FORMAT,
                  &params.data_format);
  vip_query_input(network_, index, vip_buffer_property_e::VIP_BUFFER_PROP_NUM_OF_DIMENSION,
                  &params.num_of_dims);
  vip_query_input(network_, index, vip_buffer_property_e::VIP_BUFFER_PROP_SIZES_OF_DIMENSION,
                  &params.sizes);
  vip_query_input(network_, index, vip_buffer_property_e::VIP_BUFFER_PROP_QUANT_FORMAT,
                  &params.quant_format);
  switch (params.quant_format) {
    case VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT:
      vip_query_input(network_, index, VIP_BUFFER_PROP_FIXED_POINT_POS,
                      &params.quant_data.dfp.fixed_point_pos);
      break;
    case VIP_BUFFER_QUANTIZE_TF_ASYMM:
      vip_query_input(network_, index, VIP_BUFFER_PROP_TF_SCALE, &params.quant_data.affine.scale);
      vip_query_input(network_, index, VIP_BUFFER_PROP_TF_ZERO_POINT,
                      &params.quant_data.affine.zeroPoint);
      break;
    default:
      break;
  }

  return params;
}

VIPLiteBufferParams VIPLiteNetwork::QueryOutput(size_t index) const {
  VIPLiteBufferParams params = {};
  vip_query_output(network_, index, vip_buffer_property_e::VIP_BUFFER_PROP_DATA_FORMAT,
                   &params.data_format);
  vip_query_output(network_, index, vip_buffer_property_e::VIP_BUFFER_PROP_NUM_OF_DIMENSION,
                   &params.num_of_dims);
  vip_query_output(network_, index, vip_buffer_property_e::VIP_BUFFER_PROP_SIZES_OF_DIMENSION,
                   &params.sizes);
  vip_query_output(network_, index, vip_buffer_property_e::VIP_BUFFER_PROP_QUANT_FORMAT,
                   &params.quant_format);
  switch (params.quant_format) {
    case VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT:
      vip_query_output(network_, index, VIP_BUFFER_PROP_FIXED_POINT_POS,
                       &params.quant_data.dfp.fixed_point_pos);
      break;
    case VIP_BUFFER_QUANTIZE_TF_ASYMM:
      vip_query_output(network_, index, VIP_BUFFER_PROP_TF_SCALE, &params.quant_data.affine.scale);
      vip_query_output(network_, index, VIP_BUFFER_PROP_TF_ZERO_POINT,
                       &params.quant_data.affine.zeroPoint);
      break;
    default:
      break;
  }

  return params;
}

void VIPLiteNetwork::BindInputs(const std::vector<VIPLiteBuffer>& input_buffers) {
  for (size_t i = 0; i < GetNumInputs(); i++) {
    vip_status_e status = vip_set_input(network_, i, input_buffers[i].buffer_);
    ICHECK_EQ(status, vip_status_e::VIP_SUCCESS) << "Failed to set network input:" << i;
  }
}

void VIPLiteNetwork::BindOutputs(const std::vector<VIPLiteBuffer>& output_buffers) {
  for (size_t i = 0; i < GetNumOutputs(); i++) {
    vip_status_e status = vip_set_output(network_, i, output_buffers[i].buffer_);
    ICHECK_EQ(status, vip_status_e::VIP_SUCCESS) << "Failed to set network output:" << i;
  }
}

void VIPLiteNetwork::Prepare() {
  vip_status_e status = vip_prepare_network(network_);
  ICHECK_EQ(status, vip_status_e::VIP_SUCCESS) << "Failed to prepare VIPLite network";
}

void VIPLiteNetwork::Run() {
  vip_status_e status = vip_run_network(network_);
  ICHECK_EQ(status, vip_status_e::VIP_SUCCESS) << "Failed to run VIPLite network";
}

VIPLiteBuffer::VIPLiteBuffer(VIPLiteBufferParams params, void* buffer, size_t size) {
  params.memory_type = vip_buffer_memory_type_e::VIP_BUFFER_MEMORY_TYPE_DEFAULT;
  vip_status_e status = vip_create_buffer_from_handle(&params, buffer, size, &buffer_);
  ICHECK_EQ(status, vip_status_e::VIP_SUCCESS) << "Failed to create VIPLite buffer from handle";
}

VIPLiteBuffer::VIPLiteBuffer(VIPLiteBufferParams params, int fd) {
  params.memory_type = vip_buffer_memory_type_e::VIP_BUFFER_MEMORY_TYPE_DMA_BUF;
  vip_status_e status = vip_create_buffer_from_fd(&params, fd, 0, &buffer_);
  ICHECK_EQ(status, vip_status_e::VIP_SUCCESS) << "Failed to create VIPLite buffer from dma-buf fd";
}

VIPLiteBuffer::~VIPLiteBuffer() {
  vip_destroy_buffer(buffer_);
  buffer_ = nullptr;
}

void VIPLiteBuffer::Flush() {
  vip_status_e status =
      vip_flush_buffer(buffer_, vip_buffer_operation_type_e::VIP_BUFFER_OPER_TYPE_FLUSH);
  ICHECK_EQ(status, vip_status_e::VIP_SUCCESS) << "Failed to flush VIPLite input buffer";
}

void VIPLiteBuffer::Invalidate() {
  vip_status_e status =
      vip_flush_buffer(buffer_, vip_buffer_operation_type_e::VIP_BUFFER_OPER_TYPE_INVALIDATE);
  ICHECK_EQ(status, vip_status_e::VIP_SUCCESS) << "Failed to invalidate VIPLite output buffer";
}

VIPLiteRuntime::VIPLiteRuntime(const std::string& symbol_name, std::vector<char> nbg_buffer)
    : symbol_name_(symbol_name), nbg_buffer_(std::move(nbg_buffer)) {
  ICHECK_EQ(vip_init(), vip_status_e::VIP_SUCCESS) << "Failed to initialize VIPLite";
}

VIPLiteRuntime::~VIPLiteRuntime() {
  network_.reset();
  input_buffers_.clear();
  output_buffers_.clear();
  vip_destroy();
}

PackedFunc VIPLiteRuntime::GetFunction(const std::string& name,
                                       const ObjectPtr<Object>& sptr_to_self) {
  if (name == "get_symbol") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = symbol_name_; });
  } else if (name == symbol_name_) {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      // Initialize the subgraph.
      std::call_once(
          this->initialized_, [this](const TVMArgs& args) { this->Init(args); }, args);
      // Execute the subgraph.
      this->Run();
    });
  } else {
    return PackedFunc(nullptr);
  }
}

void VIPLiteRuntime::Init(const TVMArgs& args) {
  network_ = std::make_unique<VIPLiteNetwork>(nbg_buffer_.data(), nbg_buffer_.size());
  size_t num_inputs = network_->GetNumInputs();
  size_t num_outputs = network_->GetNumOutputs();

  ICHECK_EQ(args.size(), num_inputs + num_outputs)
      << "Number of passed args is mismatched with the NBG input/output tensors.";

  input_buffers_.reserve(num_inputs);
  for (size_t i = 0; i < num_inputs; i++) {
    auto arg = args[i];
    DLTensor dl_tensor;
    if (arg.type_code() == TVMArgTypeCode::kTVMDLTensorHandle) {
      dl_tensor = *arg;
    } else if (arg.type_code() == TVMArgTypeCode::kTVMNDArrayHandle) {
      auto nd_array = arg.AsObjectRef<NDArray>();
      dl_tensor = *nd_array.operator->();
    } else {
      LOG(FATAL) << "Expect NDArray or DLTensor as arguments";
    }

    auto params = network_->QueryInput(i);
    input_buffers_.emplace_back(params, dl_tensor.data, GetDataSize(dl_tensor));
  }

  output_buffers_.reserve(num_outputs);
  for (size_t i = 0; i < num_outputs; i++) {
    auto arg = args[num_inputs + i];
    DLTensor dl_tensor;
    if (arg.type_code() == TVMArgTypeCode::kTVMDLTensorHandle) {
      dl_tensor = *arg;
    } else if (arg.type_code() == TVMArgTypeCode::kTVMNDArrayHandle) {
      auto nd_array = arg.AsObjectRef<NDArray>();
      dl_tensor = *nd_array.operator->();
    } else {
      LOG(FATAL) << "Expect NDArray or DLTensor as arguments";
    }

    auto params = network_->QueryOutput(i);
    output_buffers_.emplace_back(params, dl_tensor.data, GetDataSize(dl_tensor));
  }

  network_->Prepare();
  network_->BindInputs(input_buffers_);
  network_->BindOutputs(output_buffers_);
}

void VIPLiteRuntime::Run() {
  for (auto& buffer : input_buffers_) {
    buffer.Flush();
  }

  network_->Run();

  for (auto& buffer : output_buffers_) {
    buffer.Invalidate();
  }
}

void VIPLiteRuntime::SaveToBinary(dmlc::Stream* stream) {
  // Save symbol name.
  stream->Write(symbol_name_);
  // Save NBG.
  stream->Write(nbg_buffer_);
}

void VIPLiteRuntime::SaveToFile(const std::string& file_name, const std::string& format) {
  std::ofstream fs(file_name, std::ios::out | std::ios::binary);
  ICHECK(fs.is_open()) << "Cannot open NBG file: " << file_name;
  fs.write(nbg_buffer_.data(), nbg_buffer_.size());
}

Module VIPLiteRuntime::LoadFromBinary(void* raw_stream) {
  auto* stream = reinterpret_cast<dmlc::Stream*>(raw_stream);
  // Read symbol name.
  std::string symbol_name;
  ICHECK(stream->Read(&symbol_name)) << "Failed to load symbol name";
  // Read NBG.
  std::vector<char> nbg_buffer;
  ICHECK(stream->Read(&nbg_buffer)) << "Failed to load NBG";

  auto n = make_object<VIPLiteRuntime>(symbol_name, std::move(nbg_buffer));
  return Module(n);
}

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_tim_vx").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = VIPLiteRuntime::LoadFromBinary(args[0]);
});

}  // namespace tim_vx
}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
