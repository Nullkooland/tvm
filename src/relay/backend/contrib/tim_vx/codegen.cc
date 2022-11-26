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
 * \file src/relay/backend/contrib/tim_vx/codegen.cc
 * \brief The TVM Relay -> TIM-VX NBG compilation pass.
 */

#include "codegen.h"

#include <tim/vx/compile_option.h>
#include <tim/vx/context.h>
#include <tim/vx/graph.h>
#include <tim/vx/operation.h>
#include <tim/vx/tensor.h>
#include <tvm/relay/attrs/image.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/attrs/reduce.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/container/string.h>

#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "op_converters.h"
#include "utils.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace tim_vx {

// Initialize op converters memo for TimVxGraphBuilder.
const TimVxOpConverter::Memo TimVxGraphBuilder::OP_CONVERTERS_MEMO = TimVxOpConverter::GetMemo();

void TimVxGraphBuilder::Build(const Function& func) {
  // Mark output ops.
  if (const auto* tuple = func->body.as<TupleNode>()) {
    for (const auto& expr : tuple->fields) {
      output_op_.emplace(expr);
    }
  } else if (const auto* tuple_item = func->body.as<TupleGetItemNode>()) {
    output_op_.emplace(tuple_item->tuple);
  } else if (const auto* call = func->body.as<CallNode>()) {
    output_op_.emplace(GetRef<Expr>(call));
  }

  MixedModeVisitor::VisitExpr(func->body);
}

void TimVxGraphBuilder::VisitExpr_(const VarNode* var) {
  DLOG(INFO) << "[var] " << var->name_hint() << " -> " << var->checked_type();

  const auto& var_type = var->checked_type();
  if (const auto* tensor_type = var_type.as<TensorTypeNode>()) {
    auto spec = std::make_shared<tim::vx::TensorSpec>(ConvertDataType(tensor_type->dtype),
                                                      ConvertShape(tensor_type->shape),
                                                      tim::vx::TensorAttribute::INPUT);
    tensor_spec_memo_.emplace(GetRef<Expr>(var), std::move(spec));
  } else if (const auto* tuple_type = var_type.as<TupleTypeNode>()) {
    TimVxTensorSpecList tuple_tensor_specs;
    tuple_tensor_specs.reserve(tuple_type->fields.size());

    for (const auto& field_type : tuple_type->fields) {
      const auto* tensor_type = field_type.as<TensorTypeNode>();
      auto item_spec = std::make_shared<tim::vx::TensorSpec>(ConvertDataType(tensor_type->dtype),
                                                             ConvertShape(tensor_type->shape),
                                                             tim::vx::TensorAttribute::INPUT);
      tuple_tensor_specs.push_back(std::move(item_spec));
    }
    tuple_tensor_specs_memo_.emplace(GetRef<Expr>(var), std::move(tuple_tensor_specs));
  } else {
    LOG(FATAL) << "Only supports TensorType or TupleType, but got " << var_type->GetTypeKey();
  }
}

void TimVxGraphBuilder::VisitExpr_(const ConstantNode* constant) {
  DLOG(INFO) << "[const] "
             << " -> " << constant->checked_type();

  const auto* tensor_type = constant->checked_type().as<TensorTypeNode>();
  auto spec = std::make_shared<tim::vx::TensorSpec>(ConvertDataType(tensor_type->dtype),
                                                    ConvertShape(tensor_type->shape),
                                                    tim::vx::TensorAttribute::CONSTANT);
  tensor_spec_memo_.emplace(GetRef<Expr>(constant), std::move(spec));
}

void TimVxGraphBuilder::VisitExpr_(const TupleNode* tuple) {
  DLOG(INFO) << tuple->fields;

  TimVxTensorSpecList tuple_tensor_specs;
  tuple_tensor_specs.reserve(tuple->fields.size());
  for (const auto& expr : tuple->fields) {
    tuple_tensor_specs.push_back(tensor_spec_memo_[expr]);
  }
  tuple_tensor_specs_memo_.emplace(GetRef<Expr>(tuple), std::move(tuple_tensor_specs));
}

void TimVxGraphBuilder::VisitExpr_(const TupleGetItemNode* tuple_item) {
  DLOG(INFO) << tuple_item->tuple;

  auto& tuple_tensor_specs = tuple_tensor_specs_memo_[tuple_item->tuple];
  tensor_spec_memo_.emplace(GetRef<Expr>(tuple_item), tuple_tensor_specs[tuple_item->index]);
}

void TimVxGraphBuilder::VisitLeaf(const Expr& expr) {
  if (expr->IsInstance<FunctionNode>()) {
    // Do not traverse into composite op's function body.
    return;
  }
  MixedModeVisitor::VisitLeaf(expr);
}

void TimVxGraphBuilder::VisitExpr_(const CallNode* call) {
  std::string_view op_name;
  if (const auto* op = call->op.as<OpNode>()) {
    op_name = std::string_view(op->name.data(), op->name.length());
    DLOG(INFO) << "[op] " << op_name << " -> " << call->checked_type();
  } else if (const auto* func = call->op.as<FunctionNode>()) {
    String composite_op_name = func->GetAttr<String>(attr::kComposite).value();
    DLOG(INFO) << "[op_comp] " << composite_op_name << " -> " << call->checked_type();
    const auto prefix = std::string_view(composite_op_name.data(), 7);
    ICHECK_EQ(prefix, "tim_vx.") << "Invalid composite op for TIM-VX: " << composite_op_name;
    op_name = std::string_view(composite_op_name.data() + 7, composite_op_name.length() - 7);
  } else {
    LOG(FATAL) << "Only supports op or composite func, but got " << call->op->GetTypeKey();
  }

  // Find op converter.
  const auto it = TimVxGraphBuilder::OP_CONVERTERS_MEMO.find(op_name);
  ICHECK(it != TimVxGraphBuilder::OP_CONVERTERS_MEMO.cend())
      << "No converter implemented for op: " << op_name;
  const auto& [_, op_converter] = *it;

  // Collect input tensors for the current op.
  TimVxTensorSpecList in_tensor_specs = GetOpInputTensorSpecs(call);
  // Create ouput tensors for the current op.
  TimVxTensorSpecList out_tensor_specs = CreateOpOutputTensorSpecs(call);

  // Do Relay -> TIM-VX op conversion using current call and context.
  TimVxOp vx_op = op_converter->Convert(graph_, call, in_tensor_specs, out_tensor_specs);

  // Put op output tensor spec(s) into the memo.
  if (out_tensor_specs.size() > 1) {
    tuple_tensor_specs_memo_[GetRef<Expr>(call)] = std::move(out_tensor_specs);
  } else {
    tensor_spec_memo_[GetRef<Expr>(call)] = std::move(out_tensor_specs[0]);
  }

  op_memo_.emplace(GetRef<Expr>(call), std::move(vx_op));
}

TimVxTensorSpecList TimVxGraphBuilder::GetOpInputTensorSpecs(const CallNode* call) {
  TimVxTensorSpecList in_tensor_specs;
  for (const auto& arg : call->args) {
    if (const auto* tuple_type = arg->checked_type().as<TupleTypeNode>()) {
      for (const auto& item_spec : tuple_tensor_specs_memo_[arg]) {
        in_tensor_specs.push_back(item_spec);
      }
    } else {
      in_tensor_specs.push_back(tensor_spec_memo_[arg]);
    }
  }
  return in_tensor_specs;
}

TimVxTensorSpecList TimVxGraphBuilder::CreateOpOutputTensorSpecs(const CallNode* call) {
  bool is_graph_output = output_op_.find(GetRef<Expr>(call)) != output_op_.end();
  TimVxTensorSpecList out_tensor_specs;

  // Create op output tensors based on op return types.
  const auto op_out_type = call->checked_type();
  if (const auto* tensor_type = op_out_type.as<TensorTypeNode>()) {
    auto spec = std::make_shared<tim::vx::TensorSpec>(
        ConvertDataType(tensor_type->dtype), ConvertShape(tensor_type->shape),
        is_graph_output ? tim::vx::TensorAttribute::OUTPUT : tim::vx::TensorAttribute::TRANSIENT);
    out_tensor_specs.push_back(std::move(spec));
  } else if (const auto* tuple_type = op_out_type.as<TupleTypeNode>()) {
    out_tensor_specs.reserve(tuple_type->fields.size());
    for (const auto& field_type : tuple_type->fields) {
      const auto* tensor_type = field_type.as<TensorTypeNode>();
      auto item_spec = std::make_shared<tim::vx::TensorSpec>(
          ConvertDataType(tensor_type->dtype), ConvertShape(tensor_type->shape),
          is_graph_output ? tim::vx::TensorAttribute::OUTPUT : tim::vx::TensorAttribute::TRANSIENT);
      out_tensor_specs.push_back(std::move(item_spec));
    }
  } else {
    LOG(FATAL) << "Only supports TensorType or TupleType, but got " << op_out_type->GetTypeKey();
  }
  return out_tensor_specs;
}

void TimVxTensorBinder::Bind(const Function& func) { MixedModeVisitor::VisitExpr(func->body); }

void TimVxTensorBinder::VisitLeaf(const Expr& expr) {
  if (expr->IsInstance<FunctionNode>()) {
    // Do not traverse into composite op's function body.
    return;
  }
  MixedModeVisitor::VisitLeaf(expr);
}

void TimVxTensorBinder::VisitExpr_(const TupleNode* tuple) {
  TimVxTensorList tuple_tensors;
  tuple_tensors.reserve(tuple->fields.size());
  for (const auto& expr : tuple->fields) {
    tuple_tensors.push_back(tensor_memo_[expr]);
  }
  tuple_tensors_memo_.emplace(GetRef<Expr>(tuple), std::move(tuple_tensors));
}

void TimVxTensorBinder::VisitExpr_(const TupleGetItemNode* tuple_item) {
  auto& tuple_tensors = tuple_tensors_memo_[tuple_item->tuple];
  tensor_memo_.emplace(GetRef<Expr>(tuple_item), tuple_tensors[tuple_item->index]);
}

void TimVxTensorBinder::VisitExpr_(const CallNode* call) {
  // Get the corresponding TIM-VX op.
  TimVxOp vx_op = op_memo_[GetRef<Expr>(call)];
  // Find or create input tensors for the current op.
  TimVxTensorList in_tensors = FindOrCreateOpInputTensors(call);
  // Create ouput tensors for the current op.
  TimVxTensorList out_tensors = CreateOpOutputTensors(call);

  // Bind input/output tensors to the op.
  vx_op->BindInputs(in_tensors);
  vx_op->BindOutputs(out_tensors);
}

TimVxTensorList TimVxTensorBinder::FindOrCreateOpInputTensors(const CallNode* call) {
  TimVxTensorList in_tensors;

  const auto& arg0 = call->args[0];
  if (const auto* tuple_type = arg0->checked_type().as<TupleTypeNode>()) {
    if (const auto it = tuple_tensors_memo_.find(arg0); it != tuple_tensors_memo_.cend()) {
      const auto& [_, in_tensors] = *it;
      return in_tensors;
    }

    const auto& in_tuple_specs = tuple_tensor_specs_memo_.at(arg0);
    in_tensors.reserve(tuple_type->fields.size());
    for (const auto& field_spec : in_tuple_specs) {
      in_tensors.push_back(graph_->CreateTensor(*field_spec));
      ;
    }

    tuple_tensors_memo_.emplace(arg0, in_tensors);
  } else {
    in_tensors.reserve(call->args.size());
    for (const auto& arg : call->args) {
      if (const auto it = tensor_memo_.find(arg); it != tensor_memo_.cend()) {
        const auto& [_, tensor] = *it;
        in_tensors.push_back(tensor);
        continue;
      }

      const auto& arg_spec = tensor_spec_memo_.at(arg);
      if (arg_spec->shape_.empty()) {
        continue;
      }

      std::shared_ptr<tim::vx::Tensor> in_tensor;
      if (const auto* constant = arg.as<ConstantNode>()) {
        ICHECK(arg_spec->attr_ == tim::vx::TensorAttribute::CONSTANT)
            << "Should be a constant TIM-VX tensor";
        in_tensor = graph_->CreateTensor(*arg_spec, constant->data->data);
      } else {
        in_tensor = graph_->CreateTensor(*arg_spec);
      }

      tensor_memo_.emplace(arg, in_tensor);
      in_tensors.push_back(std::move(in_tensor));
    }
  }

  return in_tensors;
}

TimVxTensorList TimVxTensorBinder::CreateOpOutputTensors(const CallNode* call) {
  TimVxTensorList out_tensors;

  if (call->checked_type()->IsInstance<TupleTypeNode>()) {
    const auto& out_tuple_specs = tuple_tensor_specs_memo_.at(GetRef<Expr>(call));
    out_tensors.reserve(out_tuple_specs.size());
    for (const auto& item_spec : out_tuple_specs) {
      out_tensors.push_back(graph_->CreateTensor(*item_spec));
      ;
    }

    tuple_tensors_memo_.emplace(GetRef<Expr>(call), out_tensors);
  } else {
    const auto& out_spec = tensor_spec_memo_.at(GetRef<Expr>(call));
    auto out_tensor = graph_->CreateTensor(*out_spec);

    tensor_memo_.emplace(GetRef<Expr>(call), out_tensor);
    out_tensors.push_back(std::move(out_tensor));
  }

  return out_tensors;
}

runtime::Module TimVxCompile(const ObjectRef& ref) {
  ICHECK(ref->IsInstance<FunctionNode>()) << "The input ref is expected to be a Relay function.";
  auto func = Downcast<Function>(ref);
  auto symbol_name = func->GetAttr<String>(tvm::attr::kGlobalSymbol).value();

  auto context = tim::vx::Context::Create();
  ICHECK(context) << "Failed to create TIM-VX context for compilation";

  auto pass_context = transform::PassContext::Current();
  auto config = pass_context
                    ->GetConfig<TimVxCompilerConfig>("relay.ext.tim_vx.options",
                                                     AttrsWithDefaultValues<TimVxCompilerConfig>())
                    .value();

  // Set environment variables to configure the NPU driver.
  setenv("VSIMULATOR_CONFIG", config->npu_target.c_str(), 1);
  setenv("VIV_VX_DEBUG_LEVEL", std::to_string(config->debug_level).c_str(), 1);
  setenv("PATH_ASSETS", config->debug_dir.c_str(), 1);

  auto compile_options = tim::vx::CompileOption::DefaultOptions;
  compile_options.setRelaxMode(config->relax_mode);

  auto graph = context->CreateGraph(compile_options);

  // Build TIM-VX graph from partitioned subgraph function.
  auto graph_builder = TimVxGraphBuilder(graph);
  graph_builder.Build(func);

  auto tensor_binder = TimVxTensorBinder(graph, std::move(graph_builder));
  tensor_binder.Bind(func);

  // Compile NBG, get binary size.
  size_t nbg_size;
  ICHECK(graph->CompileToBinary(nullptr, &nbg_size)) << "Failed to compile to NBG";
  // Fill in NBG buffer.
  std::vector<char> nbg_buffer(nbg_size);
  ICHECK(graph->CompileToBinary(nbg_buffer.data(), &nbg_size)) << "Failed to fill in NBG buffer";
  nbg_buffer.resize(nbg_size);

  const auto* pf = runtime::Registry::Get("runtime.tim_vx_runtime_create");
  ICHECK(pf != nullptr) << "Cannot find TIM-VX runtime module create function";

  void* p_nbg_buffer = nbg_buffer.data();
  runtime::Module lib = (*pf)(symbol_name, p_nbg_buffer, nbg_size);
  return lib;
}

}  // namespace tim_vx
}  // namespace contrib
}  // namespace relay
}  // namespace tvm