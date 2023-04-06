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
 * \file src/runtime/contrib/tim_vx/nbg_parser.cc
 * \brief NBG format parser.
 */

#include "nbg_parser.h"

#include <tvm/runtime/logging.h>

#include <algorithm>
#include <array>

namespace tvm {
namespace runtime {
namespace contrib {
namespace tim_vx {

NBGParser::NBGParser(const char* nbg_data, size_t nbg_size) {
  ReadBinFixed(nbg_data);
  ReadBinDynamic(nbg_data);
}

void NBGParser::ReadBinFixed(const char* data) {
  fixed_.header = *reinterpret_cast<const GcVipBinHeader*>(data);
  data += sizeof(GcVipBinHeader);

  // Check NBG magic.
  ICHECK(fixed_.header.magic == NBG_MAGIC) << "Invalid NBG file";

  // Check NBG version.
  ICHECK_LE(fixed_.header.version, NBG_FORMAT_VERSION) << "NBG format version not supported";
  ICHECK_GE(fixed_.header.version, NBG_FORMAT_MIN_VERSION) << "NBG format version not supported";

  fixed_.memory_pool = *reinterpret_cast<const GcVipBinMemoryPool*>(data);
  data += sizeof(GcVipBinMemoryPool);

  fixed_.sram = *reinterpret_cast<const GcVipBinSRAM*>(data);
  data += sizeof(GcVipBinSRAM);

  fixed_.input_table = *reinterpret_cast<const GcVipBinEntry*>(data);
  data += sizeof(GcVipBinEntry);

  fixed_.output_table = *reinterpret_cast<const GcVipBinEntry*>(data);
  data += sizeof(GcVipBinEntry);

  fixed_.layer_table = *reinterpret_cast<const GcVipBinEntry*>(data);
  data += sizeof(GcVipBinEntry);
}

void NBGParser::ReadBinDynamic(const char* data) {
  // Read input table.
  if (fixed_.input_table.size > 0) {
    size_t num_inputs = fixed_.input_table.size / sizeof(GcVipBinInoutEntry);
    inputs_.resize(num_inputs);
    const auto* inputs_data =
        reinterpret_cast<const GcVipBinInoutEntry*>(data + fixed_.input_table.offset);
    std::copy_n(inputs_data, num_inputs, inputs_.data());
  }
  // Read output table.
  if (fixed_.output_table.size > 0) {
    size_t num_outputs = fixed_.output_table.size / sizeof(GcVipBinInoutEntry);
    outputs_.resize(num_outputs);
    const auto* outputs_data =
        reinterpret_cast<const GcVipBinInoutEntry*>(data + fixed_.output_table.offset);
    std::copy_n(outputs_data, num_outputs, outputs_.data());
  }
  // Read layer table.
  if (fixed_.layer_table.size > 0) {
    size_t num_layers = fixed_.layer_table.size / sizeof(GcVipBinLayer);
    layers_.resize(num_layers);
    const auto* layers_data =
        reinterpret_cast<const GcVipBinLayer*>(data + fixed_.layer_table.offset);
    std::copy_n(layers_data, num_layers, layers_.data());
  }
}

size_t NBGParser::QueryNetworkNumInputs() const { return inputs_.size(); };

size_t NBGParser::QueryNetworkNumOutputs() const { return outputs_.size(); };

std::string NBGParser::QueryNetworkName() const {
  return std::string(fixed_.header.network_name.data());
};

uint32_t NBGParser::QueryNetworkHardwareTarget() const { return fixed_.header.hw_target; };

size_t NBGParser::QueryInputNumDims(size_t index) const { return inputs_[index].dim_count; };

std::array<uint32_t, NBGParser::NBG_MAX_NUM_DIMS> NBGParser::QueryInputDims(size_t index) const {
  return inputs_[index].dims;
};

NBGDataType NBGParser::QueryInputDataType(size_t index) const { return inputs_[index].data_type; };

NBGQuantType NBGParser::QueryInputQuantType(size_t index) const {
  return inputs_[index].quant_type;
};

NBGBufferType NBGParser::QueryInputBufferType(size_t index) const {
  return inputs_[index].buffer_type;
};

float NBGParser::QueryInputQuantScale(size_t index) const { return inputs_[index].scale; };

int NBGParser::QueryInputQuantZeroPoint(size_t index) const { return inputs_[index].zero_point; };

int NBGParser::QueryInputQuantDfpPos(size_t index) const { return inputs_[index].fixed_pos; };

std::string NBGParser::QueryInputName(size_t index) const {
  return std::string(inputs_[index].name.data());
};

size_t NBGParser::QueryOutputNumDims(size_t index) const { return outputs_[index].dim_count; };

std::array<uint32_t, NBGParser::NBG_MAX_NUM_DIMS> NBGParser::QueryOutputDims(size_t index) const {
  return outputs_[index].dims;
};

NBGDataType NBGParser::QueryOutputDataType(size_t index) const {
  return outputs_[index].data_type;
};

NBGQuantType NBGParser::QueryOutputQuantType(size_t index) const {
  return outputs_[index].quant_type;
};

NBGBufferType NBGParser::QueryOutputBufferType(size_t index) const {
  return outputs_[index].buffer_type;
};

float NBGParser::QueryOutputQuantScale(size_t index) const { return outputs_[index].scale; };

int NBGParser::QueryOutputQuantZeroPoint(size_t index) const { return outputs_[index].zero_point; };

int NBGParser::QueryOutputQuantDfpPos(size_t index) const { return outputs_[index].fixed_pos; };

std::string NBGParser::QueryOutputName(size_t index) const {
  return std::string(outputs_[index].name.data());
};

}  // namespace tim_vx
}  // namespace contrib
}  // namespace runtime
}  // namespace tvm