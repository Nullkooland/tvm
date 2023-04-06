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
 * \file src/runtime/contrib/tim_vx/ovx_runtime.h
 * \brief OpenVX NBG runtime.
 */
#ifndef TVM_RUNTIME_CONTRIB_TIM_VX_OVX_RUNTIME_H_
#define TVM_RUNTIME_CONTRIB_TIM_VX_OVX_RUNTIME_H_

#include <VX/vx.h>
#include <VX/vx_khr_nn.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "nbg_parser.h"

namespace tvm {
namespace runtime {
namespace contrib {
namespace tim_vx {

class OpenVXRuntime final : public ModuleNode {
 public:
  /*! \brief Get the module type key. */
  virtual const char* type_key() const override { return "tim_vx"; };

  explicit OpenVXRuntime(const std::string& symbol_name, std::vector<char> nbg_buffer);
  ~OpenVXRuntime();

  /*!
   * \brief Get a packed function.
   * \param name The name/symbol of the function.
   * \param sptr_to_self The pointer to the module node.
   * \return The packed function.
   */
  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) override;

  /*!
   * \brief Save the OpenVX runtime module to binary stream.
   * \param stream The DMLC stream.
   */
  void SaveToBinary(dmlc::Stream* stream) override;

  /*!
   * \brief Save the OpenVX NBG to file.
   * \param path Path to the NBG file.
   */
  void SaveToFile(const std::string& file_name, const std::string& format) override;

  /*!
   * \brief Load a OpenVX runtime module from binary stream.
   * \param raw_stream The DMLC stream.
   */
  static Module LoadFromBinary(void* raw_stream);

 private:
  using VXContext = vx_context;
  using VXGraph = vx_graph;
  using VXTensor = vx_tensor;
  using VXTensorCreateParams = vx_tensor_create_params_t;
  using VXTensorAddressing = vx_tensor_addressing;
  using VXKernel = vx_kernel;
  using VXNode = vx_node;
  using VXRef = vx_reference;
  using VXStatus = vx_status;
  using VXDataType = vx_type_e;
  using VXQuantType = vx_quantized_format_e;

  /*! \brief Bind input/output TVM tensors to OpenVX tensors. */
  void Init(const TVMArgs& args);

  /*! \brief Run inference. */
  void Run();

  /*! \brief Convert NBG data type to OpenVX data type. */
  static inline VXDataType ConvertDataType(NBGDataType data_type) {
    switch (data_type) {
      case NBGDataType::FP32:
        return VXDataType::VX_TYPE_FLOAT32;
      case NBGDataType::FP16:
        return VXDataType::VX_TYPE_FLOAT16;
      case NBGDataType::BFP16:
        return VXDataType::VX_TYPE_BFLOAT16;
      case NBGDataType::UINT8:
        return VXDataType::VX_TYPE_UINT8;
      case NBGDataType::UINT16:
        return VXDataType::VX_TYPE_UINT16;
      case NBGDataType::UINT32:
        return VXDataType::VX_TYPE_UINT32;
      case NBGDataType::UINT64:
        return VXDataType::VX_TYPE_UINT64;
      case NBGDataType::INT8:
        return VXDataType::VX_TYPE_INT8;
      case NBGDataType::INT16:
        return VXDataType::VX_TYPE_INT16;
      case NBGDataType::INT32:
        return VXDataType::VX_TYPE_INT32;
      case NBGDataType::INT64:
        return VXDataType::VX_TYPE_INT64;
      case NBGDataType::CHAR:
        return VXDataType::VX_TYPE_CHAR;
      default:
        return VXDataType::VX_TYPE_INVALID;
    }
  }

  /*! \brief Convert NBG quantization type to OpenVX quantization type. */
  static inline VXQuantType ConvertQuantType(NBGQuantType quant_type) {
    switch (quant_type) {
      case NBGQuantType::NONE:
        return VXQuantType::VX_QUANT_NONE;
      case NBGQuantType::AFFINE_ASYMMETRIC:
        return VXQuantType::VX_QUANT_AFFINE_SCALE;
      case NBGQuantType::DYNAMIC_FIXED_POINT:
        return VXQuantType::VX_QUANT_DYNAMIC_FIXED_POINT;
      default:
        return VXQuantType::VX_QUANT_NONE;
    }
  }

  /*! \brief Get number of bytes of given OpenVX data type. */
  static size_t GetDataTypeBytes(VXDataType dtype) {
    switch (dtype) {
      case VXDataType::VX_TYPE_INT8:
      case VXDataType::VX_TYPE_UINT8:
      case VXDataType::VX_TYPE_BOOL8:
      case VXDataType::VX_TYPE_CHAR:
        return 1;
      case VXDataType::VX_TYPE_INT16:
      case VXDataType::VX_TYPE_UINT16:
      case VXDataType::VX_TYPE_FLOAT16:
      case VXDataType::VX_TYPE_BFLOAT16:
        return 2;
      case VXDataType::VX_TYPE_INT32:
      case VXDataType::VX_TYPE_UINT32:
      case VXDataType::VX_TYPE_FLOAT32:
        return 4;
      case VXDataType::VX_TYPE_INT64:
      case VXDataType::VX_TYPE_UINT64:
      case VXDataType::VX_TYPE_FLOAT64:
        return 8;
      default:
        return 0;
    }
  }

  /*! \brief The flag indicating whether the module is initialized. */
  std::once_flag initialized_;

  /*! \brief The OpenVX context for management of all OpenVX objects. */
  VXContext context_;
  /*! \brief The OpenVX graph for execution. */
  VXGraph graph_;
  /*! \brief The OpenVX NBG node. */
  VXNode nbg_node_;
  /*! \brief The OpenVX NBG kernel. */
  VXKernel nbg_kernel_;
  /*! \brief The OpenVX input tensors. */
  std::vector<VXTensor> input_tensors_;
  /*! \brief The OpenVX output tensors. */
  std::vector<VXTensor> output_tensors_;

  /*! \brief The only subgraph func name of the module. */
  std::string symbol_name_;
  /*! \brief The NBG buffer. */
  std::vector<char> nbg_buffer_;
};

}  // namespace tim_vx
}  // namespace contrib
}  // namespace runtime
}  // namespace tvm

#endif