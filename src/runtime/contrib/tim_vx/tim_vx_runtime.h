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
 * \file src/runtime/contrib/tim_vx/tim_vx_runtime.h
 * \brief TIM-VX NBG runtime.
 */
#ifndef TVM_RUNTIME_CONTRIB_TIM_VX_TIM_VX_RUNTIME_H_
#define TVM_RUNTIME_CONTRIB_TIM_VX_TIM_VX_RUNTIME_H_

#include <tim/utils/nbg_parser.h>
#include <tim/vx/context.h>
#include <tim/vx/graph.h>
#include <tim/vx/tensor.h>
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

namespace tvm {
namespace runtime {
namespace contrib {
namespace tim_vx {

class TimVxRuntime final : public ModuleNode {
 public:
  /*! \brief Get the module type key. */
  virtual const char* type_key() const override { return "tim_vx"; };

  explicit TimVxRuntime(const std::string& symbol_name, std::vector<char> nbg_buffer);

  /*!
   * \brief Get a packed function.
   * \param name The name/symbol of the function.
   * \param sptr_to_self The pointer to the module node.
   * \return The packed function.
   */
  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) override;

  /*!
   * \brief Save the TIM-VX runtime module to binary stream.
   * \param stream The DMLC stream.
   */
  void SaveToBinary(dmlc::Stream* stream) override;

  /*!
   * \brief Save the TIM-VX NBG to file.
   * \param path Path to the NBG file.
   */
  void SaveToFile(const std::string& file_name, const std::string& format) override;

  /*!
   * \brief Load a TIM-VX runtime module from binary stream.
   * \param raw_stream The DMLC stream.
   */
  static Module LoadFromBinary(void* raw_stream);

 private:
  /*! \brief Bind input/output TVM tensors to TIM-VX tensors. */
  void Init(const TVMArgs& args);

  /*! \brief Run inference. */
  void Run();

  /*! \brief Convert NBG datatype to TIM-VX datatype. */
  static inline tim::vx::DataType ConvertDataType(nbg_buffer_format_e data_type) {
    switch (data_type) {
      case nbg_buffer_format_e::NBG_BUFFER_FORMAT_FP32:
        return tim::vx::DataType::FLOAT32;
      case nbg_buffer_format_e::NBG_BUFFER_FORMAT_FP16:
        return tim::vx::DataType::FLOAT16;
      case nbg_buffer_format_e::NBG_BUFFER_FORMAT_UINT8:
        return tim::vx::DataType::UINT8;
      case nbg_buffer_format_e::NBG_BUFFER_FORMAT_UINT16:
        return tim::vx::DataType::UINT16;
      case nbg_buffer_format_e::NBG_BUFFER_FORMAT_UINT32:
        return tim::vx::DataType::UINT32;
      case nbg_buffer_format_e::NBG_BUFFER_FORMAT_INT8:
        return tim::vx::DataType::INT8;
      case nbg_buffer_format_e::NBG_BUFFER_FORMAT_INT16:
        return tim::vx::DataType::INT16;
      case nbg_buffer_format_e::NBG_BUFFER_FORMAT_INT32:
        return tim::vx::DataType::INT32;
      default:
        return tim::vx::DataType::UNKNOWN;
    }
  }

  /*! \brief Convert NBG quantization type to TIM-VX quantization type. */
  static inline tim::vx::QuantType ConvertQuantType(nbg_buffer_quantize_format_e quant_type) {
    switch (quant_type) {
      case nbg_buffer_quantize_format_e::NBG_BUFFER_QUANTIZE_NONE:
        return tim::vx::QuantType::NONE;
      case nbg_buffer_quantize_format_e::NBG_BUFFER_QUANTIZE_AFFINE_ASYMMETRIC:
        return tim::vx::QuantType::ASYMMETRIC;
      case nbg_buffer_quantize_format_e::NBG_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT:
        return tim::vx::QuantType::DYNAMIC_FIXED_POINT;
      default:
        return tim::vx::QuantType::NONE;
    }
  }

  /*! \brief The flag indicating whether the module is initialized. */
  std::once_flag initialized_;

  /*! \brief The TIM-VX context for management of all TIM-VX objects. */
  std::shared_ptr<tim::vx::Context> context_;
  /*! \brief The TIM-VX graph for execution. */
  std::shared_ptr<tim::vx::Graph> graph_;
  /*! \brief The TIM-VX input tensors. */
  std::vector<std::shared_ptr<tim::vx::Tensor>> input_tensors_;
  /*! \brief The TIM-VX output tensors. */
  std::vector<std::shared_ptr<tim::vx::Tensor>> output_tensors_;

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