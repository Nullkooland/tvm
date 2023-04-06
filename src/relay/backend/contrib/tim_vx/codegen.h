
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
 * \file src/relay/backend/contrib/tim_vx/codegen.h
 * \brief The TVM Relay -> TIM-VX NBG compilation pass.
 */

#ifndef TVM_RELAY_BACKEND_CONTRIB_TIM_VX_CODEGEN_H_
#define TVM_RELAY_BACKEND_CONTRIB_TIM_VX_CODEGEN_H_

#include <tim/vx/context.h>
#include <tim/vx/graph.h>
#include <tim/vx/tensor.h>
#include <tvm/relay/attrs/image.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/attrs/reduce.h>
#include <tvm/relay/type.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../../utils.h"
#include "op_converters.h"
#include "utils.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace tim_vx {

/*! \brief Type aliases. */
using TensorSpecMemo = std::unordered_map<Expr, TimVxTensorSpec, ObjectPtrHash, ObjectPtrEqual>;
using TupleTensorSpecsMemo =
    std::unordered_map<Expr, TimVxTensorSpecList, ObjectPtrHash, ObjectPtrEqual>;
using TensorMemo = std::unordered_map<Expr, TimVxTensor, ObjectPtrHash, ObjectPtrEqual>;
using TupleTensorsMemo = std::unordered_map<Expr, TimVxTensorList, ObjectPtrHash, ObjectPtrEqual>;
using OpMemo = std::unordered_map<Expr, TimVxOp, ObjectPtrHash, ObjectPtrEqual>;
using OpSet = std::unordered_set<Expr, ObjectPtrHash, ObjectPtrEqual>;

class TimVxGraphBuilder final : private MixedModeVisitor {
 public:
  explicit TimVxGraphBuilder(TimVxGraph graph) : graph_(graph) {}

  void Build(const Function& func);

 private:
  void VisitExpr_(const VarNode* var_node) override;
  void VisitExpr_(const ConstantNode* var_node) override;
  void VisitExpr_(const TupleNode* tuple_node) override;
  void VisitExpr_(const TupleGetItemNode* tuple_item_node) override;
  void VisitExpr_(const CallNode* call_node) override;
  void VisitLeaf(const Expr& expr) override;

  TimVxTensorSpecList GetOpInputTensorSpecs(const CallNode* call);
  TimVxTensorSpecList CreateOpOutputTensorSpecs(const CallNode* call);

  /*! \brief The TIM-VX graph for compilation. */
  TimVxGraph graph_;
  /*! \brief The output Relay Expr set. */
  OpSet output_op_;
  /*! \brief The memo of Relay Expr to TIM-VX op. */
  OpMemo op_memo_;
  /*! \brief The memo of Relay Expr to TIM-VX tensor spec. */
  TensorSpecMemo tensor_spec_memo_;
  /*! \brief The memo of Relay tuple-like Expr to list of TIM-VX tensor specs. */
  TupleTensorSpecsMemo tuple_tensor_specs_memo_;

  /*! \brief The TVM Relay op -> TIM-VX op converters memo. */
  static const TimVxOpConverter::Memo OP_CONVERTERS_MEMO;

  /*! \brief Allow TimVxTensorBinder to access private memos. */
  friend class TimVxTensorBinder;
};

class TimVxTensorBinder final : private MixedModeVisitor {
 public:
  explicit TimVxTensorBinder(TimVxGraph graph, TimVxGraphBuilder&& builder)
      : graph_(graph),
        op_memo_(std::move(builder.op_memo_)),
        tensor_spec_memo_(std::move(builder.tensor_spec_memo_)),
        tuple_tensor_specs_memo_(std::move(builder.tuple_tensor_specs_memo_)) {}

  void Bind(const Function& func);

 private:
  void VisitExpr_(const TupleNode* tuple_node) override;
  void VisitExpr_(const TupleGetItemNode* tuple_item_node) override;
  void VisitExpr_(const CallNode* call_node) override;
  void VisitLeaf(const Expr& expr) override;

  TimVxTensorList FindOrCreateOpInputTensors(const CallNode* call);
  TimVxTensorList CreateOpOutputTensors(const CallNode* call);

  /*! \brief The TIM-VX graph for compilation. */
  TimVxGraph graph_;
  /*! \brief The memo of Relay Expr to TIM-VX op. */
  OpMemo op_memo_;
  /*! \brief The memo of Relay Expr to TIM-VX tensor spec. */
  TensorSpecMemo tensor_spec_memo_;
  /*! \brief The memo of Relay tuple-like Expr to list of TIM-VX tensor specs. */
  TupleTensorSpecsMemo tuple_tensor_specs_memo_;
  /*! \brief The memo of Relay Expr to TIM-VX tensor. */
  TensorMemo tensor_memo_;
  /*! \brief The memo of Relay tuple-like Expr to list of TIM-VX tensors. */
  TupleTensorsMemo tuple_tensors_memo_;
};

/*!
 * \brief Compile a runtime module for TIM-VX runtime.
 *
 * \param ref The ext_func Relay expression/module to be executed using extern ops.
 * \return The compiled runtime module.
 */
runtime::Module TimVxCompile(const ObjectRef& ref);

TVM_REGISTER_GLOBAL("relay.ext.tim_vx").set_body_typed(TimVxCompile);

/*! \brief All constants are handled by the NBG, so the runtime module won't store them. */
TVM_REGISTER_GLOBAL("relay.ext.tim_vx.constant_updater")
    .set_body_typed([](Expr expr, std::string symbol) { return Map<String, runtime::NDArray>(); });

/*!
 * \brief Check whether TIM-VX runtime is enabled.
 *
 * \return True if TIM-VX runtime is enabled, False if not.
 */
inline constexpr bool IsTimVxRuntimeEnabled() {
#if TVM_GRAPH_EXECUTOR_TIM_VX | TVM_GRAPH_EXECUTOR_OPENVX
  return true;
#else
  return false;
#endif
}

TVM_REGISTER_GLOBAL("relay.op.is_tim_vx_runtime_enabled").set_body_typed(IsTimVxRuntimeEnabled);

/*! \brief Attributes to store the compiler options for TIM-VX */
struct TimVxCompilerConfigNode : public tvm::AttrsNode<TimVxCompilerConfigNode> {
  String npu_target;
  bool relax_mode;
  int debug_level;
  String debug_dir;

  TVM_DECLARE_ATTRS(TimVxCompilerConfigNode, "ext.attrs.TimVxCompilerConfigNode") {
    TVM_ATTR_FIELD(npu_target)
        .describe("Target VeriSilicon Vivante NPU hardware target")
        .set_default("");
    TVM_ATTR_FIELD(relax_mode)
        .describe("Whether to execute fp32 operations in bf16")
        .set_default(false);
    TVM_ATTR_FIELD(debug_level).describe("The deriver debug log level.").set_default(0);
    TVM_ATTR_FIELD(debug_dir)
        .describe("The output directory for debug info from NPU driver.")
        .set_default(".");
  }
};

class TimVxCompilerConfig : public Attrs {
 public:
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(TimVxCompilerConfig, Attrs, TimVxCompilerConfigNode);
};

TVM_REGISTER_NODE_TYPE(TimVxCompilerConfigNode);
TVM_REGISTER_PASS_CONFIG_OPTION("relay.ext.tim_vx.options", TimVxCompilerConfig);

}  // namespace tim_vx
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
#endif