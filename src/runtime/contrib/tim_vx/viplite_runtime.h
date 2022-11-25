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
 * \file src/runtime/contrib/tim_vx/viplite_runtime.h
 * \brief VIPLite NBG runtime.
 */
#ifndef TVM_RUNTIME_CONTRIB_TIM_VX_VIPLITE_RUNTIME_H_
#define TVM_RUNTIME_CONTRIB_TIM_VX_VIPLITE_RUNTIME_H_

#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <vip_lite.h>

#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace tvm {
namespace runtime {
namespace contrib {
namespace tim_vx {

using VIPLiteBufferParams = vip_buffer_create_params_t;

class VIPLiteBuffer {
 public:
  explicit VIPLiteBuffer(VIPLiteBufferParams params, void* buffer, size_t size);
  explicit VIPLiteBuffer(VIPLiteBufferParams params, int fd);
  ~VIPLiteBuffer();

  void Flush();
  void Invalidate();

 private:
  vip_buffer buffer_;

  friend class VIPLiteNetwork;
};

class VIPLiteNetwork {
 public:
  explicit VIPLiteNetwork(const void* nbg_buffer, size_t nbg_size);
  ~VIPLiteNetwork();

  const std::string_view GetName() const { return name_; }
  size_t GetNumInputs() const { return num_inputs_; }
  size_t GetNumOutputs() const { return num_outputs_; }
  VIPLiteBufferParams QueryInput(size_t index) const;
  VIPLiteBufferParams QueryOutput(size_t index) const;

  void BindInputs(const std::vector<VIPLiteBuffer>& input_buffers);
  void BindOutputs(const std::vector<VIPLiteBuffer>& output_buffers);
  void Prepare();
  void Run();

 private:
  vip_network network_;
  std::string name_;
  size_t num_inputs_;
  size_t num_outputs_;
};

class VIPLiteRuntime final : public ModuleNode {
 public:
  /*! \brief Get the module type key. */
  virtual const char* type_key() const override { return "tim_vx"; };

  explicit VIPLiteRuntime(const std::string& symbol_name, std::vector<char> nbg_buffer);
  ~VIPLiteRuntime();

  /*!
   * \brief Get a packed function.
   * \param name The name/symbol of the function.
   * \param sptr_to_self The pointer to the module node.
   * \return The packed function.
   */
  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) override;

  /*!
   * \brief Save the VIPLite runtime module to binary stream.
   * \param stream The DMLC stream.
   */
  void SaveToBinary(dmlc::Stream* stream) override;

  /*!
   * \brief Save the NBG to file.
   * \param path Path to the NBG file.
   */
  void SaveToFile(const std::string& file_name, const std::string& format) override;

  /*!
   * \brief Load a VIPLite runtime module from binary stream.
   * \param raw_stream The DMLC stream.
   */
  static Module LoadFromBinary(void* raw_stream);

 private:
  /*! \brief Bind input/output TVM tensors to VIPLite buffers. */
  void Init(const TVMArgs& args);

  /*! \brief Run inference. */
  void Run();

  /*! \brief The flag indicating whether the module has been initialized. */
  std::once_flag initialized_;

  /*! \brief The VIPLite network. */
  std::unique_ptr<VIPLiteNetwork> network_;
  /*! \brief The VIPLite input buffers. */
  std::vector<VIPLiteBuffer> input_buffers_;
  /*! \brief The VIPLite output buffers. */
  std::vector<VIPLiteBuffer> output_buffers_;

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