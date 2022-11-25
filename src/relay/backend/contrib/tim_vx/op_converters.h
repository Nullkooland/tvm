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
 * \file src/relay/backend/contrib/tim_vx/op_converters.h
 * \brief TVM Relay op -> TIM-VX op converters.
 */

#ifndef TVM_RELAY_BACKEND_CONTRIB_TIM_VX_OP_CONVERTERS_H_
#define TVM_RELAY_BACKEND_CONTRIB_TIM_VX_OP_CONVERTERS_H_

#include <tim/vx/graph.h>
#include <tim/vx/tensor.h>
#include <tvm/relay/expr.h>

#include <memory>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "utils.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace tim_vx {

/*! \brief Base abstract class of TVM Relay op -> TIM-VX op converter. */
class TimVxOpConverter {
 public:
  explicit TimVxOpConverter() = default;
  virtual ~TimVxOpConverter() = default;

  virtual TimVxOp Convert(TimVxGraph graph, const CallNode* call,
                          TimVxTensorSpecList& in_tensor_specs,
                          TimVxTensorSpecList& out_tensor_specs) = 0;

  /*! \brief The memo of Relay-builtin and composite op name to the corresponding converter. */
  using Memo = std::unordered_map<std::string_view, std::unique_ptr<TimVxOpConverter>>;
  /*! \brief Get the memo of all op converters. */
  static const Memo GetMemo();
};

}  // namespace tim_vx
}  // namespace contrib
}  // namespace relay
}  // namespace tvm

#endif