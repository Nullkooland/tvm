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
 * \file src/runtime/contrib/tim_vx/nbg_parser.h
 * \brief NBG format parser.
 */
#ifndef TVM_RUNTIME_CONTRIB_TIM_VX_NBG_PARSER_H_
#define TVM_RUNTIME_CONTRIB_TIM_VX_NBG_PARSER_H_

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace tvm {
namespace runtime {
namespace contrib {
namespace tim_vx {

enum NBGQuantType {
  /*! \brief Not quantized . */
  NONE = 0,
  /*! \brief The data is quantized with dynamic fixed point format. */
  DYNAMIC_FIXED_POINT = 1,
  /*! \brief The data is quantized with asymmetric affine format. */
  AFFINE_ASYMMETRIC = 2
};

enum NBGDataType {
  /*! \brief A float type of data. */
  FP32 = 0,
  /*! \brief A half float type of data. */
  FP16 = 1,
  /*! \brief A 8 bit unsigned integer type of data. */
  UINT8 = 2,
  /*! \brief A 8 bit signed integer type of data. */
  INT8 = 3,
  /*! \brief A 16 bit unsigned integer type of data. */
  UINT16 = 4,
  /*! \brief A 16 signed integer type of data. */
  INT16 = 5,
  /*! \brief A char type of data. */
  CHAR = 6,
  /*! \brief A bfloat 16 type of data. */
  BFP16 = 7,
  /*! \brief A 32 bit integer type of data. */
  INT32 = 8,
  /*! \brief A 32 bit unsigned signed integer type of data. */
  UINT32 = 9,
  /*! \brief A 64 bit signed integer type of data. */
  INT64 = 10,
  /*! \brief A 64 bit unsigned integer type of data. */
  UINT64 = 11,
  /*! \brief A 64 bit float type of buffer data. */
  FP64 = 12,
};

enum NBGBufferType {
  /*! \brief A tensor type of buffer data. */
  TENSOR = 0,
  /*! \brief A image type of buffer data. */
  IMAGE = 1,
  /*! \brief A array type of buffer data. */
  ARRAY = 2,
  /*! \brief A scalar type of buffer data. */
  SCALAR = 3,
};

class NBGParser {
 public:
  static constexpr size_t NBG_MAX_NUM_DIMS = 6;

  NBGParser(const char* nbg_data, size_t nbg_size);

  /*! \brief Network queries. */
  size_t QueryNetworkNumInputs() const;
  size_t QueryNetworkNumOutputs() const;
  std::string QueryNetworkName() const;
  uint32_t QueryNetworkHardwareTarget() const;

  /*! \brief Input queries. */
  size_t QueryInputNumDims(size_t index) const;
  std::array<uint32_t, NBG_MAX_NUM_DIMS> QueryInputDims(size_t index) const;
  NBGDataType QueryInputDataType(size_t index) const;
  NBGQuantType QueryInputQuantType(size_t index) const;
  NBGBufferType QueryInputBufferType(size_t index) const;
  float QueryInputQuantScale(size_t index) const;
  int QueryInputQuantZeroPoint(size_t index) const;
  int QueryInputQuantDfpPos(size_t index) const;
  std::string QueryInputName(size_t index) const;

  /*! \brief Output queries. */
  size_t QueryOutputNumDims(size_t index) const;
  std::array<uint32_t, NBG_MAX_NUM_DIMS> QueryOutputDims(size_t index) const;
  NBGDataType QueryOutputDataType(size_t index) const;
  NBGQuantType QueryOutputQuantType(size_t index) const;
  NBGBufferType QueryOutputBufferType(size_t index) const;
  float QueryOutputQuantScale(size_t index) const;
  int QueryOutputQuantZeroPoint(size_t index) const;
  int QueryOutputQuantDfpPos(size_t index) const;
  std::string QueryOutputName(size_t index) const;

 private:
  static constexpr size_t NBG_FORMAT_VERSION = 0x00010016;
  static constexpr size_t NBG_FORMAT_MIN_VERSION = 0x0001000B;
  static constexpr size_t NBG_VERSION_MAJOR = 1;
  static constexpr size_t NBG_VERSION_MINOR = 1;
  static constexpr size_t NBG_VERSION_SUB_MINOR = 2;
  static constexpr size_t NBG_NETWORK_NAME_SIZE = 64;
  static constexpr size_t NBG_LAYER_NAME_SIZE = 64;
  static constexpr size_t NBG_MAX_IO_NAME_LENGTH = 64;
  static constexpr std::array<char, 4> NBG_MAGIC = {'V', 'P', 'M', 'N'};

  struct GcVipBinFeatureDatabase {
    uint32_t hi_reorder_fix : 1;
    uint32_t ocb_counter : 1;
    uint32_t nn_command_size : 2;
    uint32_t change_ppu_param : 1;
    uint32_t reserved : 27;
    uint32_t num_pixel_pipes;
    uint8_t core_count;
    uint8_t device_id;
    uint8_t axi_bus_width_index;
    uint8_t reserved3;
    std::array<uint32_t, 13> vsi_reserved;
    std::array<uint32_t, 48> customer_reserved;
  };

  struct GcVipBinHeader {
    std::array<char, 4> magic;
    uint32_t version;
    uint32_t hw_target;
    std::array<char, NBG_NETWORK_NAME_SIZE> network_name;
    uint32_t layer_count;
    uint32_t operation_count;
    uint32_t input_count;
    uint32_t output_count;
    GcVipBinFeatureDatabase feature_db;
  };

  struct GcVipBinMemoryPool {
    uint32_t size;
    uint32_t alignment;
    uint32_t base;
  };

  struct GcVipBinSRAM {
    uint32_t axi_sram_base;
    uint32_t axi_sram_size;
    uint32_t vip_sram_base;
    uint32_t vip_sram_size;
  };

  struct GcVipBinEntry {
    uint32_t offset;
    uint32_t size;
  };

  struct GcVipBinFixed {
    GcVipBinHeader header;
    GcVipBinMemoryPool memory_pool;
    GcVipBinSRAM sram;
    GcVipBinEntry input_table;
    GcVipBinEntry output_table;
    GcVipBinEntry layer_table;
  };

  struct GcVipBinInoutEntry {
    uint32_t dim_count;
    std::array<uint32_t, NBG_MAX_NUM_DIMS> dims;
    NBGDataType data_type;
    NBGBufferType buffer_type;
    NBGQuantType quant_type;
    int32_t fixed_pos;
    float scale;
    int32_t zero_point;
    std::array<char, NBG_MAX_IO_NAME_LENGTH> name;
  };

  struct GcVipBinLayer {
    std::array<char, NBG_LAYER_NAME_SIZE> name;
    uint32_t id;
    uint32_t operation_count;
    uint32_t uid;
  };

  void ReadBinFixed(const char* data);
  void ReadBinDynamic(const char* data);

  GcVipBinFixed fixed_;
  std::vector<GcVipBinInoutEntry> inputs_;
  std::vector<GcVipBinInoutEntry> outputs_;
  std::vector<GcVipBinLayer> layers_;
};

}  // namespace tim_vx
}  // namespace contrib
}  // namespace runtime
}  // namespace tvm

#endif