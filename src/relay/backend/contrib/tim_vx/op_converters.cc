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
 * \file src/relay/backend/contrib/tim_vx/memo.cc
 * \brief TVM Relay op -> TIM-VX op converters.
 */

#include "op_converters.h"

#include <tim/vx/ops.h>
#include <tvm/relay/attrs/image.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/attrs/reduce.h>
#include <tvm/relay/attrs/vision.h>
#include <tvm/relay/function.h>
#include <tvm/relay/qnn/attrs.h>

#include <numeric>

#include "../../utils.h"
#include "utils.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace tim_vx {

namespace ops = tim::vx::ops;

/*!
 * \brief Converter class template for NN ops.
 * \note Op format: nn.op(input, weight, optional[bias]) -> output.
 * \tparam TOp A TIM-VX operator type inherited from tim::vx::Operation.
 */
template <typename TOp>
class NNOpConverter final : public TimVxOpConverter {
  static_assert(std::is_base_of<tim::vx::Operation, TOp>(),
                "TOp must inherit from tim::vx::Operation");

 public:
  TimVxOp Convert(TimVxGraph graph, const CallNode* call, TimVxTensorSpecList& in_tensor_specs,
                  TimVxTensorSpecList& out_tensor_specs) override;
};

/*!
 * \brief Converter class (fully specialized template) for nn.conv2d op.
 * \note Op format: nn.conv2d(input, weight, optional[bias]) -> output.
 */
template <>
class NNOpConverter<ops::Conv2d> final : public TimVxOpConverter {
 public:
  TimVxOp Convert(TimVxGraph graph, const CallNode* call, TimVxTensorSpecList& in_tensor_specs,
                  TimVxTensorSpecList& out_tensor_specs) override {
    const auto* func = call->op.as<FunctionNode>();
    const auto* inner_call = func->body.as<CallNode>();

    // Skip nn.requantize for qnn conv2d.
    if (backend::IsOp(inner_call, "qnn.requantize")) {
      inner_call = inner_call->args[0].as<CallNode>();
    }

    // Skip bias_add if exists.
    if (in_tensor_specs.size() == 3) {
      inner_call = inner_call->args[0].as<CallNode>();
    }

    const auto* attrs = inner_call->attrs.as<Conv2DAttrs>();
    int channels = attrs->channels.as<IntImmNode>()->value;
    int groups = attrs->groups;
    int multiplier = (groups > 1) ? channels / groups : 0;
    bool is_grouped = groups > 1 && groups != channels;

    auto kernel_size = ConvertIndexArray<uint32_t, 2>(attrs->kernel_size);
    auto strides = ConvertIndexArray<uint32_t, 2>(attrs->strides);
    auto dilation = ConvertIndexArray<uint32_t, 2>(attrs->dilation);
    auto tvm_padding = ConvertIndexArray<uint32_t, 4>(attrs->padding);
    auto vx_padding = ConvertPadding(tvm_padding);

    if (is_grouped) {
      return graph->CreateOperation<ops::GroupedConv2d>(vx_padding, strides, dilation, groups);
    } else {
      return graph->CreateOperation<ops::Conv2d>(channels, tim::vx::PadType::AUTO, kernel_size,
                                                 strides, dilation, vx_padding, multiplier);
    }
  }
};

/*!
 * \brief Converter class (fully specialized template) for nn.conv2d_transpose op.
 * \note Op format: nn.conv2d_transpose(input, weight, optional[bias]) -> output.
 */
template <>
class NNOpConverter<ops::DeConv2d> final : public TimVxOpConverter {
 public:
  TimVxOp Convert(TimVxGraph graph, const CallNode* call, TimVxTensorSpecList& in_tensor_specs,
                  TimVxTensorSpecList& out_tensor_specs) override {
    const auto* func = call->op.as<FunctionNode>();
    const auto* op = func->body.as<CallNode>();

    // Skip nn.requantize for qnn conv2d_transpose.
    if (backend::IsOp(op, "qnn.requantize")) {
      op = op->args[0].as<CallNode>();
    }

    // Skip bias_add if exists.
    if (in_tensor_specs.size() == 3) {
      op = op->args[0].as<CallNode>();
    }

    const auto* attrs = op->attrs.as<Conv2DTransposeAttrs>();
    int channels = attrs->channels.as<IntImmNode>()->value;
    int groups = attrs->groups;
    int multiplier = (groups > 1) ? channels / groups : 0;

    auto kernel_size = ConvertIndexArray<uint32_t, 2>(attrs->kernel_size);
    auto strides = ConvertIndexArray<uint32_t, 2>(attrs->strides);
    auto dilation = ConvertIndexArray<uint32_t, 2>(attrs->dilation);
    auto tvm_padding = ConvertIndexArray<uint32_t, 4>(attrs->padding);
    auto vx_padding = ConvertPadding(tvm_padding);

    return graph->CreateOperation<ops::DeConv2d>(channels, tim::vx::PadType::AUTO, kernel_size,
                                                 strides, dilation, vx_padding, multiplier);
  }
};

/*!
 * \brief Converter class (fully specialized template) for nn.dense op.
 * \note Op format: nn.dense(input, weight, optional[bias]) -> output.
 */
template <>
class NNOpConverter<ops::FullyConnected> final : public TimVxOpConverter {
 public:
  TimVxOp Convert(TimVxGraph graph, const CallNode* call, TimVxTensorSpecList& in_tensor_specs,
                  TimVxTensorSpecList& out_tensor_specs) override {
    const auto* func = call->op.as<FunctionNode>();
    const auto* op = func->body.as<CallNode>();

    // Skip nn.requantize for qnn fc.
    if (backend::IsOp(op, "qnn.requantize")) {
      op = op->args[0].as<CallNode>();
    }

    // Skip bias_add if exists.
    if (in_tensor_specs.size() == 3) {
      op = op->args[0].as<CallNode>();
    }

    const auto* attrs = op->attrs.as<DenseAttrs>();
    uint32_t channels = static_cast<uint32_t>(attrs->units.as<IntImmNode>()->value);

    return graph->CreateOperation<ops::FullyConnected>(0, channels);
  }
};

/*!
 * \brief Converter class template for unary ops.
 * \note Op format: op(input) -> output.
 * \tparam TOp A TIM-VX operator type inherited from tim::vx::Operation.
 */
template <typename TOp>
class UnaryOpConverter final : public TimVxOpConverter {
  static_assert(std::is_base_of<tim::vx::Operation, TOp>(),
                "TOp must inherit from tim::vx::Operation");

 public:
  TimVxOp Convert(TimVxGraph graph, const CallNode* call, TimVxTensorSpecList& in_tensor_specs,
                  TimVxTensorSpecList& out_tensor_specs) override {
    return graph->CreateOperation<TOp>();
  }
};

/*!
 * \brief Converter class (fully specialized template) for nn.leaky_relu op.
 * \note Op format: nn.leaky_relu(input) -> output.
 */
template <>
class UnaryOpConverter<ops::LeakyRelu> final : public TimVxOpConverter {
 public:
  TimVxOp Convert(TimVxGraph graph, const CallNode* call, TimVxTensorSpecList& in_tensor_specs,
                  TimVxTensorSpecList& out_tensor_specs) override {
    const auto* attrs = call->attrs.as<LeakyReluAttrs>();

    return graph->CreateOperation<ops::LeakyRelu>(static_cast<float>(attrs->alpha));
  }
};

/*!
 * \brief Converter class (fully specialized template) for clip op.
 * \note Op format: clip(input) -> output.
 */
template <>
class UnaryOpConverter<ops::Clip> final : public TimVxOpConverter {
 public:
  TimVxOp Convert(TimVxGraph graph, const CallNode* call, TimVxTensorSpecList& in_tensor_specs,
                  TimVxTensorSpecList& out_tensor_specs) override {
    const auto* attrs = call->attrs.as<ClipAttrs>();

    return graph->CreateOperation<ops::Clip>(static_cast<float>(attrs->a_min),
                                             static_cast<float>(attrs->a_max));
  }
};

/*!
 * \brief Converter class (fully specialized template) for reshape op.
 * \note Op format: reshape(input) -> output.
 */
template <>
class UnaryOpConverter<ops::Reshape> final : public TimVxOpConverter {
 public:
  TimVxOp Convert(TimVxGraph graph, const CallNode* call, TimVxTensorSpecList& in_tensor_specs,
                  TimVxTensorSpecList& out_tensor_specs) override {
    const auto* attrs = call->attrs.as<ReshapeAttrs>();
    auto vx_new_shape = ConvertShape(attrs->newshape);

    return graph->CreateOperation<ops::Reshape>(vx_new_shape);
  }
};

/*!
 * \brief Converter class (fully specialized template) for transpose op.
 * \note Op format: transpose(input) -> output.
 */
template <>
class UnaryOpConverter<ops::Transpose> final : public TimVxOpConverter {
 public:
  TimVxOp Convert(TimVxGraph graph, const CallNode* call, TimVxTensorSpecList& in_tensor_specs,
                  TimVxTensorSpecList& out_tensor_specs) override {
    const auto* attrs = call->attrs.as<TransposeAttrs>();
    const auto& tvm_perm = Downcast<Array<Integer>>(attrs->axes);

    uint32_t rank = in_tensor_specs[0]->shape_.size();
    auto vx_perm = ConvertAxes<uint32_t>(tvm_perm, rank);

    return graph->CreateOperation<ops::Transpose>(vx_perm);
  }
};

/*!
 * \brief Converter class (fully specialized template) for nn.space_to_depth op.
 * \note Op format: nn.space_to_depth(input) -> output.
 */
template <>
class UnaryOpConverter<ops::SpaceToDepth> final : public TimVxOpConverter {
 public:
  TimVxOp Convert(TimVxGraph graph, const CallNode* call, TimVxTensorSpecList& in_tensor_specs,
                  TimVxTensorSpecList& out_tensor_specs) override {
    const auto* attrs = call->attrs.as<SubPixelAttrs>();
    std::vector<int> block_size(2, attrs->block_size);

    return graph->CreateOperation<ops::SpaceToDepth>(block_size);
  }
};

/*!
 * \brief Converter class (fully specialized template) for split op.
 * \note Op format: split(input) -> [outputs, ...].
 */
template <>
class UnaryOpConverter<ops::Split> final : public TimVxOpConverter {
 public:
  TimVxOp Convert(TimVxGraph graph, const CallNode* call, TimVxTensorSpecList& in_tensor_specs,
                  TimVxTensorSpecList& out_tensor_specs) override {
    const auto& input_shape = in_tensor_specs[0]->shape_;
    const auto* attrs = call->attrs.as<SplitAttrs>();
    uint32_t vx_axis = ConvertAxis<uint32_t>(attrs->axis, input_shape.size());

    const auto& indices = Downcast<Array<Integer>>(attrs->indices_or_sections);

    // Map from split indices to slices sizes.
    std::vector<uint32_t> slices(indices.size() + 1);
    slices.front() = static_cast<uint32_t>(indices.front()->value);
    for (size_t i = 1; i < indices.size(); i++) {
      slices[i] = static_cast<uint32_t>(indices[i]->value - indices[i - 1]->value);
    }
    slices.back() = input_shape[vx_axis] - static_cast<uint32_t>(indices.back()->value);

    return graph->CreateOperation<ops::Split>(vx_axis, slices);
  }
};

/*!
 * \brief Converter class (fully specialized template) for sum op.
 * \note Op format: sum(input) -> output.
 */
template <>
class UnaryOpConverter<ops::ReduceSum> final : public TimVxOpConverter {
 public:
  TimVxOp Convert(TimVxGraph graph, const CallNode* call, TimVxTensorSpecList& in_tensor_specs,
                  TimVxTensorSpecList& out_tensor_specs) override {
    const auto* attrs = call->attrs.as<ReduceAttrs>();
    uint32_t rank = in_tensor_specs[0]->shape_.size();
    auto vx_axes = ConvertAxes<int>(attrs->axis, rank);

    return graph->CreateOperation<ops::ReduceSum>(vx_axes, attrs->keepdims);
  }
};

/*!
 * \brief Converter class (fully specialized template) for prod op.
 * \note Op format: prod(input) -> output.
 */
template <>
class UnaryOpConverter<ops::ReduceProd> final : public TimVxOpConverter {
 public:
  TimVxOp Convert(TimVxGraph graph, const CallNode* call, TimVxTensorSpecList& in_tensor_specs,
                  TimVxTensorSpecList& out_tensor_specs) override {
    const auto* attrs = call->attrs.as<ReduceAttrs>();
    uint32_t rank = in_tensor_specs[0]->shape_.size();
    auto vx_axes = ConvertAxes<int>(attrs->axis, rank);

    return graph->CreateOperation<ops::ReduceProd>(vx_axes, attrs->keepdims);
  }
};

/*!
 * \brief Converter class (fully specialized template) for mean op.
 * \note Op format: mean(input) -> output.
 */
template <>
class UnaryOpConverter<ops::ReduceMean> final : public TimVxOpConverter {
 public:
  TimVxOp Convert(TimVxGraph graph, const CallNode* call, TimVxTensorSpecList& in_tensor_specs,
                  TimVxTensorSpecList& out_tensor_specs) override {
    const auto* attrs = call->attrs.as<ReduceAttrs>();
    uint32_t rank = in_tensor_specs[0]->shape_.size();
    auto vx_axes = ConvertAxes<int>(attrs->axis, rank);

    return graph->CreateOperation<ops::ReduceMean>(vx_axes, attrs->keepdims);
  }
};

/*!
 * \brief Converter class (fully specialized template) for min op.
 * \note Op format: min(input) -> output.
 */
template <>
class UnaryOpConverter<ops::ReduceMin> final : public TimVxOpConverter {
 public:
  TimVxOp Convert(TimVxGraph graph, const CallNode* call, TimVxTensorSpecList& in_tensor_specs,
                  TimVxTensorSpecList& out_tensor_specs) override {
    const auto* attrs = call->attrs.as<ReduceAttrs>();
    uint32_t rank = in_tensor_specs[0]->shape_.size();
    auto vx_axes = ConvertAxes<int>(attrs->axis, rank);

    return graph->CreateOperation<ops::ReduceMin>(vx_axes, attrs->keepdims);
  }
};

/*!
 * \brief Converter class (fully specialized template) for max op.
 * \note Op format: max(input) -> output.
 */
template <>
class UnaryOpConverter<ops::ReduceMax> final : public TimVxOpConverter {
 public:
  TimVxOp Convert(TimVxGraph graph, const CallNode* call, TimVxTensorSpecList& in_tensor_specs,
                  TimVxTensorSpecList& out_tensor_specs) override {
    const auto* attrs = call->attrs.as<ReduceAttrs>();
    uint32_t rank = in_tensor_specs[0]->shape_.size();
    auto vx_axes = ConvertAxes<int>(attrs->axis, rank);

    return graph->CreateOperation<ops::ReduceMax>(vx_axes, attrs->keepdims);
  }
};

/*!
 * \brief Converter class (fully specialized template) for logical all op.
 * \note Op format: all(input) -> output.
 */
template <>
class UnaryOpConverter<ops::ReduceAll> final : public TimVxOpConverter {
 public:
  TimVxOp Convert(TimVxGraph graph, const CallNode* call, TimVxTensorSpecList& in_tensor_specs,
                  TimVxTensorSpecList& out_tensor_specs) override {
    const auto* attrs = call->attrs.as<ReduceAttrs>();
    uint32_t rank = in_tensor_specs[0]->shape_.size();
    auto vx_axes = ConvertAxes<int>(attrs->axis, rank);

    return graph->CreateOperation<ops::ReduceAll>(vx_axes, attrs->keepdims);
  }
};

/*!
 * \brief Converter class (fully specialized template) for logical any op.
 * \note Op format: any(input) -> output.
 */
template <>
class UnaryOpConverter<ops::ReduceAny> final : public TimVxOpConverter {
 public:
  TimVxOp Convert(TimVxGraph graph, const CallNode* call, TimVxTensorSpecList& in_tensor_specs,
                  TimVxTensorSpecList& out_tensor_specs) override {
    const auto* attrs = call->attrs.as<ReduceAttrs>();
    uint32_t rank = in_tensor_specs[0]->shape_.size();
    auto vx_axes = ConvertAxes<int>(attrs->axis, rank);

    return graph->CreateOperation<ops::ReduceAny>(vx_axes, attrs->keepdims);
  }
};

/*!
 * \brief Converter class (fully specialized template) for argmin op.
 * \note Op format: argmin(input) -> output.
 */
template <>
class UnaryOpConverter<ops::ArgMin> final : public TimVxOpConverter {
 public:
  TimVxOp Convert(TimVxGraph graph, const CallNode* call, TimVxTensorSpecList& in_tensor_specs,
                  TimVxTensorSpecList& out_tensor_specs) override {
    const auto* attrs = call->attrs.as<ArgReduceAttrs>();
    uint32_t rank = in_tensor_specs[0]->shape_.size();
    int vx_axis = ConvertAxis<int>(attrs->axis[0]->value, rank);

    return graph->CreateOperation<ops::ArgMin>(vx_axis);
  }
};

/*!
 * \brief Converter class (fully specialized template) for argmax op.
 * \note Op format: argmax(input) -> output.
 */
template <>
class UnaryOpConverter<ops::ArgMax> final : public TimVxOpConverter {
 public:
  TimVxOp Convert(TimVxGraph graph, const CallNode* call, TimVxTensorSpecList& in_tensor_specs,
                  TimVxTensorSpecList& out_tensor_specs) override {
    const auto* attrs = call->attrs.as<ArgReduceAttrs>();
    uint32_t rank = in_tensor_specs[0]->shape_.size();
    int vx_axis = ConvertAxis<int>(attrs->axis[0]->value, rank);

    return graph->CreateOperation<ops::ArgMax>(vx_axis);
  }
};

/*!
 * \brief Converter class (fully specialized template) for nn.[max/avg]_pool2d op.
 * \note Op format: nn.[max/avg]_pool2d(input) -> output.
 */
template <>
class UnaryOpConverter<ops::Pool2d> final : public TimVxOpConverter {
 public:
  TimVxOp Convert(TimVxGraph graph, const CallNode* call, TimVxTensorSpecList& in_tensor_specs,
                  TimVxTensorSpecList& out_tensor_specs) override {
    std::array<uint32_t, 4> vx_padding;
    std::array<uint32_t, 2> strides;
    std::array<uint32_t, 2> kernel_size;
    tim::vx::PoolType pool_type = tim::vx::PoolType::MAX;
    tim::vx::RoundType round_type = tim::vx::RoundType::FLOOR;

    if (backend::IsOp(call, "nn.avg_pool2d")) {
      const auto* attrs = call->attrs.as<AvgPool2DAttrs>();
      strides = ConvertIndexArray<uint32_t, 2>(attrs->strides);
      kernel_size = ConvertIndexArray<uint32_t, 2>(attrs->pool_size);
      auto tvm_padding = ConvertIndexArray<uint32_t, 4>(attrs->padding);
      vx_padding = ConvertPadding(tvm_padding);

      pool_type =
          attrs->count_include_pad ? tim::vx::PoolType::AVG : tim::vx::PoolType::AVG_ANDROID;
      round_type = attrs->ceil_mode ? tim::vx::RoundType::CEILING : tim::vx::RoundType::FLOOR;
    } else if (backend::IsOp(call, "nn.max_pool2d")) {
      const auto* attrs = call->attrs.as<MaxPool2DAttrs>();
      strides = ConvertIndexArray<uint32_t, 2>(attrs->strides);
      kernel_size = ConvertIndexArray<uint32_t, 2>(attrs->pool_size);
      auto tvm_padding = ConvertIndexArray<uint32_t, 4>(attrs->padding);
      vx_padding = ConvertPadding(tvm_padding);

      pool_type = tim::vx::PoolType::MAX;
      round_type = attrs->ceil_mode ? tim::vx::RoundType::CEILING : tim::vx::RoundType::FLOOR;
    } else {
      LOG(FATAL) << "Unsupported pooling op: " << call->op;
    }

    return graph->CreateOperation<ops::Pool2d>(pool_type, vx_padding, kernel_size, strides,
                                               round_type);
  }
};

/*!
 * \brief Converter class (fully specialized template) for image.resize2d op.
 * \note Op format: image.resize2d(input) -> output.
 */
template <>
class UnaryOpConverter<ops::Resize> final : public TimVxOpConverter {
 public:
  TimVxOp Convert(TimVxGraph graph, const CallNode* call, TimVxTensorSpecList& in_tensor_specs,
                  TimVxTensorSpecList& out_tensor_specs) override {
    const auto* attrs = call->attrs.as<Resize2DAttrs>();
    auto method = ConvertResizeType(attrs->method);
    auto size = ConvertIndexArray<int, 2>(attrs->size);
    bool align_corners = attrs->coordinate_transformation_mode == "align_corners";
    bool half_pixel_centers = attrs->coordinate_transformation_mode == "half_pixel";

    return graph->CreateOperation<ops::Resize>(method, 0.0F, align_corners, half_pixel_centers,
                                               size[0], size[1]);
  }
};

/*!
 * \brief Converter class (fully specialized template) for take op.
 * \note Op format: take(input, indices) -> output.
 */
template <>
class UnaryOpConverter<ops::Gather> final : public TimVxOpConverter {
 public:
  TimVxOp Convert(TimVxGraph graph, const CallNode* call, TimVxTensorSpecList& in_tensor_specs,
                  TimVxTensorSpecList& out_tensor_specs) override {
    const auto* attrs = call->attrs.as<TakeAttrs>();
    uint32_t rank = in_tensor_specs[0]->shape_.size();
    int vx_axis = ConvertAxis<int>(attrs->axis->value, rank);
    int num_batch_dims = attrs->batch_dims->value;

    return graph->CreateOperation<ops::Gather>(vx_axis, num_batch_dims);
  }
};

/*!
 * \brief Converter class (fully specialized template) for gather op.
 * \note Op format: gather(input, indices) -> output.
 */
template <>
class UnaryOpConverter<ops::GatherElements> final : public TimVxOpConverter {
 public:
  TimVxOp Convert(TimVxGraph graph, const CallNode* call, TimVxTensorSpecList& in_tensor_specs,
                  TimVxTensorSpecList& out_tensor_specs) override {
    const auto* attrs = call->attrs.as<GatherAttrs>();
    uint32_t rank = in_tensor_specs[0]->shape_.size();
    int vx_axis = ConvertAxis<int>(attrs->axis->value, rank);

    return graph->CreateOperation<ops::GatherElements>(vx_axis);
  }
};

/*!
 * \brief Converter class template for binary ops.
 * \note Op format: op(lhs, rhs) -> output.
 * \tparam TOp A TIM-VX operator type inherited from tim::vx::Operation.
 */
template <typename TOp>
class BinaryOpConverter final : public TimVxOpConverter {
  static_assert(std::is_base_of<tim::vx::Operation, TOp>(),
                "TOp must inherit from tim::vx::Operation");

 public:
  TimVxOp Convert(TimVxGraph graph, const CallNode* call, TimVxTensorSpecList& in_tensor_specs,
                  TimVxTensorSpecList& out_tensor_specs) override {
    return graph->CreateOperation<TOp>();
  }
};

/*!
 * \brief Converter class template for tuple-consuming ops.
 * \note Op format: op([inputs, ...]) -> output.
 * \tparam TOp A TIM-VX operator type inherited from tim::vx::Operation.
 */
template <typename TOp>
class TupleOpConverter final : public TimVxOpConverter {
  static_assert(std::is_base_of<tim::vx::Operation, TOp>(),
                "TOp must inherit from tim::vx::Operation");

 public:
  TimVxOp Convert(TimVxGraph graph, const CallNode* call, TimVxTensorSpecList& in_tensor_specs,
                  TimVxTensorSpecList& out_tensor_specs) override {
    return graph->CreateOperation<TOp>();
  }
};

/*!
 * \brief Converter class (fully specialized template) for concatenate op.
 * \note Op format: concatenate([inputs, ...]) -> output.
 */
template <>
class TupleOpConverter<ops::Concat> final : public TimVxOpConverter {
 public:
  TimVxOp Convert(TimVxGraph graph, const CallNode* call, TimVxTensorSpecList& in_tensor_specs,
                  TimVxTensorSpecList& out_tensor_specs) override {
    const auto* attrs = call->attrs.as<ConcatenateAttrs>();
    uint32_t rank = in_tensor_specs[0]->shape_.size();
    uint32_t vx_axis = ConvertAxis<uint32_t>(attrs->axis, rank);

    return graph->CreateOperation<ops::Concat>(vx_axis, in_tensor_specs.size());
  }
};

/*! \brief QNN op quantization type.  */
enum class OpQuantFormat {
  NONE,
  TRANSPARENT,
  EXPLICIT,
  QDQ,
  I32_CAST,
};

/*!
 * \brief Converter class template for QNN ops.
 * \tparam TBaseOpConverter An op converter type that implements TimVxOpConverter.
 * \tparam Q Op quantization format.
 */
template <typename TBaseOpConverter, OpQuantFormat Q>
class QnnWrapper final : public TimVxOpConverter {
  static_assert(std::is_base_of<TimVxOpConverter, TBaseOpConverter>(),
                "TBaseOpConverter must implement TVXOpConverter.");
};

/*!
 * \brief Converter class template for quantization-transparent unary ops.
 * \note Op format: op(input) -> output, where input and output have the same quantization info.
 * \tparam TOp A TIM-VX operator type inherited from tim::vx::Operation.
 */
template <typename TOp>
class QnnWrapper<UnaryOpConverter<TOp>, OpQuantFormat::TRANSPARENT> final
    : public TimVxOpConverter {
 public:
  TimVxOp Convert(TimVxGraph graph, const CallNode* call, TimVxTensorSpecList& in_tensor_specs,
                  TimVxTensorSpecList& out_tensor_specs) override {
    // Pass on quantization info from input to output.
    const auto& quant_info = in_tensor_specs[0]->quantization_;
    out_tensor_specs[0]->SetQuantization(const_cast<tim::vx::Quantization&>(quant_info));

    return base_converter_.Convert(graph, call, in_tensor_specs, out_tensor_specs);
  }

 private:
  UnaryOpConverter<TOp> base_converter_;
};

/*!
 * \brief Converter class for quantization-transparent split op.
 * \note Op format: split(input) -> [outputs, ...],
 * where input and outputs have the same quantization info.
 */
template <>
class QnnWrapper<UnaryOpConverter<ops::Split>, OpQuantFormat::TRANSPARENT> final
    : public TimVxOpConverter {
 public:
  TimVxOp Convert(TimVxGraph graph, const CallNode* call, TimVxTensorSpecList& in_tensor_specs,
                  TimVxTensorSpecList& out_tensor_specs) override {
    // Pass on quantization info from input to outputs.
    const auto& quant_info = in_tensor_specs[0]->quantization_;
    for (auto& out_spec : out_tensor_specs) {
      out_spec->SetQuantization(const_cast<tim::vx::Quantization&>(quant_info));
    }

    return base_converter_.Convert(graph, call, in_tensor_specs, out_tensor_specs);
  }

 private:
  UnaryOpConverter<ops::Split> base_converter_;
};

/*!
 * \brief Converter class template for Relay-builtin quantized unary ops.
 * \note Op format: qnn.op(input, in_scale, in_zp, out_scale, out_zp) -> output.
 * \tparam TOp A TIM-VX operator type inherited from tim::vx::Operation.
 */
template <typename TOp>
class QnnWrapper<UnaryOpConverter<TOp>, OpQuantFormat::EXPLICIT> final : public TimVxOpConverter {
 public:
  TimVxOp Convert(TimVxGraph graph, const CallNode* call, TimVxTensorSpecList& in_tensor_specs,
                  TimVxTensorSpecList& out_tensor_specs) override {
    const auto* in_scale = call->args[1].as<ConstantNode>();
    const auto* in_zero_point = call->args[2].as<ConstantNode>();
    const auto* out_scale = call->args[3].as<ConstantNode>();
    const auto* out_zero_point = call->args[4].as<ConstantNode>();

    auto in_quant_info = ConvertQuantization(in_scale, in_zero_point, -1);
    auto out_quant_info = ConvertQuantization(out_scale, out_zero_point, -1);

    in_tensor_specs[0]->SetQuantization(in_quant_info);
    out_tensor_specs[0]->SetQuantization(out_quant_info);

    for (size_t i = 1; i < 5; i++) {
      in_tensor_specs[i]->shape_.clear();
    }

    return base_converter_.Convert(graph, call, in_tensor_specs, out_tensor_specs);
  }

 private:
  UnaryOpConverter<TOp> base_converter_;
};

/*!
 * \brief Converter class for quantized unary.swish op.
 * \note Op format: qnn.swish(input) = qnn.sigmoid(input) *  qnn.mul(input).
 */
template <>
class QnnWrapper<UnaryOpConverter<ops::Swish>, OpQuantFormat::EXPLICIT> final
    : public TimVxOpConverter {
 public:
  TimVxOp Convert(TimVxGraph graph, const CallNode* call, TimVxTensorSpecList& in_tensor_specs,
                  TimVxTensorSpecList& out_tensor_specs) override {
    const auto* func = call->op.as<FunctionNode>();
    const auto* qnn_mul = func->body.as<CallNode>();

    const auto* in_scale = qnn_mul->args[2].as<ConstantNode>();
    const auto* in_zero_point = qnn_mul->args[3].as<ConstantNode>();
    const auto* out_scale = qnn_mul->args[6].as<ConstantNode>();
    const auto* out_zero_point = qnn_mul->args[7].as<ConstantNode>();

    auto in_quant_info = ConvertQuantization(in_scale, in_zero_point, -1);
    auto out_quant_info = ConvertQuantization(out_scale, out_zero_point, -1);

    in_tensor_specs[0]->SetQuantization(in_quant_info);
    out_tensor_specs[0]->SetQuantization(out_quant_info);

    return base_converter_.Convert(graph, call, in_tensor_specs, out_tensor_specs);
  }

 private:
  UnaryOpConverter<ops::Swish> base_converter_;
};

/*!
 * \brief Converter class template for composite quantized unary ops with QDQ structure.
 * \note Op format: dq(input, in_scale, in_zp) -> fp_op(*) -> q(*, out_scale, out_zp) -> output.
 * \tparam TOp A TIM-VX operator type inherited from tim::vx::Operation.
 */
template <typename TOp>
class QnnWrapper<UnaryOpConverter<TOp>, OpQuantFormat::QDQ> final : public TimVxOpConverter {
 public:
  TimVxOp Convert(TimVxGraph graph, const CallNode* call, TimVxTensorSpecList& in_tensor_specs,
                  TimVxTensorSpecList& out_tensor_specs) override {
    const auto* func = call->op.as<FunctionNode>();
    const auto* inner_call = func->body.as<CallNode>();

    const auto* quantize = inner_call;
    const auto* quantize_attrs = quantize->attrs.as<qnn::QuantizeAttrs>();
    inner_call = inner_call->args[0].as<CallNode>();

    const auto* fp_op = inner_call;
    inner_call = inner_call->args[0].as<CallNode>();

    const auto* dequantize = inner_call;
    const auto* dequantize_attrs = dequantize->attrs.as<qnn::DequantizeAttrs>();

    const auto* in_scale = dequantize->args[1].as<ConstantNode>();
    const auto* in_zero_point = dequantize->args[2].as<ConstantNode>();
    uint32_t in_rank = in_tensor_specs[0]->shape_.size();
    int in_channel_dim = ConvertAxis<int>(dequantize_attrs->axis, in_rank);

    const auto* out_scale = quantize->args[1].as<ConstantNode>();
    const auto* out_zero_point = quantize->args[2].as<ConstantNode>();
    uint32_t out_rank = out_tensor_specs[0]->shape_.size();
    int out_channel_dim = ConvertAxis<int>(quantize_attrs->axis, out_rank);

    auto in_quant_info = ConvertQuantization(in_scale, in_zero_point, in_channel_dim);
    auto out_quant_info = ConvertQuantization(out_scale, out_zero_point, out_channel_dim);

    in_tensor_specs[0]->SetQuantization(in_quant_info);
    out_tensor_specs[0]->SetQuantization(out_quant_info);

    return base_converter_.Convert(graph, fp_op, in_tensor_specs, out_tensor_specs);
  }

 private:
  UnaryOpConverter<TOp> base_converter_;
};

/*!
 * \brief Converter class template for composite quantized unary ops with pre/post i32 casts.
 * \note Op format: cast(input, i32) -> i32_op(*) -> cast(*, in_dtype) -> output.
 * \tparam TOp A TIM-VX operator type inherited from tim::vx::Operation.
 */
template <typename TOp>
class QnnWrapper<UnaryOpConverter<TOp>, OpQuantFormat::I32_CAST> final : public TimVxOpConverter {
 public:
  TimVxOp Convert(TimVxGraph graph, const CallNode* call, TimVxTensorSpecList& in_tensor_specs,
                  TimVxTensorSpecList& out_tensor_specs) override {
    const auto* func = call->op.as<FunctionNode>();
    const auto* inner_call = func->body.as<CallNode>();
    const CallNode* i32_op = nullptr;

    if (backend::IsOp(inner_call, "qnn.requantize")) {
      // Skip output requantize op.
      inner_call = inner_call->args[0].as<CallNode>();
      i32_op = inner_call;
      // Skip i32 op.
      inner_call = inner_call->args[0].as<CallNode>();
      const auto* requantize = inner_call;

      const auto* requantize_attrs = requantize->attrs.as<qnn::RequantizeAttrs>();
      uint32_t rank = in_tensor_specs[0]->shape_.size();
      int channel_dim = ConvertAxis<int>(requantize_attrs->axis, rank);

      const auto* in_scale = requantize->args[1].as<ConstantNode>();
      const auto* in_zero_point = requantize->args[2].as<ConstantNode>();
      const auto* out_scale = requantize->args[3].as<ConstantNode>();
      const auto* out_zero_point = requantize->args[4].as<ConstantNode>();

      auto in_quant_info = ConvertQuantization(in_scale, in_zero_point, channel_dim);
      auto out_quant_info = ConvertQuantization(out_scale, out_zero_point, channel_dim);

      in_tensor_specs[0]->SetQuantization(in_quant_info);
      out_tensor_specs[0]->SetQuantization(out_quant_info);
    } else {
      // Pass on quantization info from input to output.
      const auto& quant_info = in_tensor_specs[0]->quantization_;
      out_tensor_specs[0]->SetQuantization(const_cast<tim::vx::Quantization&>(quant_info));

      // Skip output cast op.
      inner_call = inner_call->args[0].as<CallNode>();
      i32_op = inner_call;
    }

    return base_converter_.Convert(graph, i32_op, in_tensor_specs, out_tensor_specs);
  }

 private:
  UnaryOpConverter<TOp> base_converter_;
};

/*!
 * \brief Converter class template for Relay-builtin quantized binary ops.
 * \note Op format: qnn.op(lhs, rhs,
 * lhs_scale, lhs_zp, rhs_scale, rhs_zp, out_scale, out_zp) -> output.
 * \tparam TOp A TIM-VX operator type inherited from tim::vx::Operation.
 */
template <typename TOp>
class QnnWrapper<BinaryOpConverter<TOp>, OpQuantFormat::EXPLICIT> final : public TimVxOpConverter {
 public:
  TimVxOp Convert(TimVxGraph graph, const CallNode* call, TimVxTensorSpecList& in_tensor_specs,
                  TimVxTensorSpecList& out_tensor_specs) override {
    const auto* lhs_scale = call->args[2].as<ConstantNode>();
    const auto* lhs_zero_point = call->args[3].as<ConstantNode>();
    const auto* rhs_scale = call->args[4].as<ConstantNode>();
    const auto* rhs_zero_point = call->args[5].as<ConstantNode>();
    const auto* out_scale = call->args[6].as<ConstantNode>();
    const auto* out_zero_point = call->args[7].as<ConstantNode>();

    auto lhs_quant_info = ConvertQuantization(lhs_scale, lhs_zero_point, -1);
    auto rhs_quant_info = ConvertQuantization(rhs_scale, rhs_zero_point, -1);
    auto out_quant_info = ConvertQuantization(out_scale, out_zero_point, -1);

    in_tensor_specs[0]->SetQuantization(lhs_quant_info);
    in_tensor_specs[1]->SetQuantization(rhs_quant_info);
    out_tensor_specs[0]->SetQuantization(out_quant_info);

    for (size_t i = 2; i < 8; i++) {
      in_tensor_specs[i]->shape_.clear();
    }

    return base_converter_.Convert(graph, call, in_tensor_specs, out_tensor_specs);
  }

 private:
  BinaryOpConverter<TOp> base_converter_;
};

/*!
 * \brief Converter class template for Relay-builtin quantized tuple-consuming ops.
 * \note Op format: qnn.op([inputs, ...], [input_scales, ...], [input_zps, ...],
 * output_scale, output_zp) -> output.
 * \tparam TOp A TIM-VX operator type inherited from tim::vx::Operation.
 */
template <typename TOp>
class QnnWrapper<TupleOpConverter<TOp>, OpQuantFormat::EXPLICIT> final : public TimVxOpConverter {
 public:
  TimVxOp Convert(TimVxGraph graph, const CallNode* call, TimVxTensorSpecList& in_tensor_specs,
                  TimVxTensorSpecList& out_tensor_specs) override {
    size_t num_inputs = call->args[0]->checked_type().as<TupleTypeNode>()->fields.size();

    const auto& in_scales = call->args[1].as<TupleNode>();
    const auto& in_zero_points = call->args[2].as<TupleNode>();
    const auto* out_scale = call->args[3].as<ConstantNode>();
    const auto* out_zero_point = call->args[4].as<ConstantNode>();

    for (size_t i = 0; i < num_inputs; i++) {
      const auto* in_scale = in_scales->fields[i].as<ConstantNode>();
      const auto* in_zero_point = in_zero_points->fields[i].as<ConstantNode>();

      auto in_quant_info = ConvertQuantization(in_scale, in_zero_point, -1);
      in_tensor_specs[i]->SetQuantization(in_quant_info);
    }

    auto out_quant_info = ConvertQuantization(out_scale, out_zero_point, -1);
    out_tensor_specs[0]->SetQuantization(out_quant_info);

    for (size_t i = num_inputs; i < in_tensor_specs.size(); i++) {
      in_tensor_specs[i]->shape_.clear();
    }

    return base_converter_.Convert(graph, call, in_tensor_specs, out_tensor_specs);
  }

 private:
  TupleOpConverter<TOp> base_converter_;
};

/*!
 * \brief Converter class template for Relay-builtin QNN ops.
 * \note Op format: qnn.op(input, input_zp, input_scale, output_scale, output_zp) -> output.
 * \tparam TOp A TIM-VX operator type inherited from tim::vx::Operation.
 */
template <typename TOp>
class QnnWrapper<NNOpConverter<TOp>, OpQuantFormat::EXPLICIT> final : public TimVxOpConverter {
 public:
  TimVxOp Convert(TimVxGraph graph, const CallNode* call, TimVxTensorSpecList& in_tensor_specs,
                  TimVxTensorSpecList& out_tensor_specs) override {
    const auto* func = call->op.as<FunctionNode>();
    const auto* inner_call = func->body.as<CallNode>();

    const auto* requantize = inner_call;
    inner_call = inner_call->args[0].as<CallNode>();

    // Skip bias_add if exists.
    if (in_tensor_specs.size() == 3) {
      inner_call = inner_call->args[0].as<CallNode>();
    }

    const auto* qnn_op = inner_call;
    int kernel_channel_dim = GetChannelDim(qnn_op);

    const auto* in_scale = qnn_op->args[4].as<ConstantNode>();
    const auto* in_zero_point = qnn_op->args[2].as<ConstantNode>();
    const auto* kernel_scale = qnn_op->args[5].as<ConstantNode>();
    const auto* kernel_zero_point = qnn_op->args[3].as<ConstantNode>();

    const auto* bias_scale = requantize->args[1].as<ConstantNode>();
    const auto* bias_zero_point = requantize->args[2].as<ConstantNode>();
    const auto* out_scale = requantize->args[3].as<ConstantNode>();
    const auto* out_zero_point = requantize->args[4].as<ConstantNode>();

    auto in_quant_info = ConvertQuantization(in_scale, in_zero_point, -1);
    auto kernel_quant_info =
        ConvertQuantization(kernel_scale, kernel_zero_point, kernel_channel_dim);
    auto out_quant_info = ConvertQuantization(out_scale, out_zero_point, -1);

    in_tensor_specs[0]->SetQuantization(in_quant_info);
    in_tensor_specs[1]->SetQuantization(kernel_quant_info);
    if (in_tensor_specs.size() == 3) {
      auto bias_quant_info = ConvertQuantization(bias_scale, bias_zero_point, 0);
      in_tensor_specs[2]->SetQuantization(bias_quant_info);
    }
    out_tensor_specs[0]->SetQuantization(out_quant_info);

    return base_converter_.Convert(graph, call, in_tensor_specs, out_tensor_specs);
  }

 private:
  static int GetChannelDim(const CallNode* qnn_op);
  NNOpConverter<TOp> base_converter_;
};

template <>
int QnnWrapper<NNOpConverter<ops::Conv2d>, OpQuantFormat::EXPLICIT>::GetChannelDim(
    const CallNode* qnn_op) {
  const auto* attrs = qnn_op->attrs.as<Conv2DAttrs>();
  auto kernel_layout = std::string_view(attrs->kernel_layout->data, attrs->kernel_layout->size);
  return 3 - kernel_layout.find('O');
}

template <>
int QnnWrapper<NNOpConverter<ops::DeConv2d>, OpQuantFormat::EXPLICIT>::GetChannelDim(
    const CallNode* qnn_op) {
  const auto* attrs = qnn_op->attrs.as<Conv2DTransposeAttrs>();
  return 3 - attrs->kernel_layout.find('O');
}

template <>
int QnnWrapper<NNOpConverter<ops::FullyConnected>, OpQuantFormat::EXPLICIT>::GetChannelDim(
    const CallNode* qnn_op) {
  return 1;
}

/*!
 * \brief Converter class for qnn.quantize op.
 * \note Op format: qnn.quantize(input, output_scale, output_zp) -> output.
 */
class QuantizeConverter final : public TimVxOpConverter {
 public:
  TimVxOp Convert(TimVxGraph graph, const CallNode* call, TimVxTensorSpecList& in_tensor_specs,
                  TimVxTensorSpecList& out_tensor_specs) override {
    const auto* attrs = call->attrs.as<qnn::QuantizeAttrs>();
    uint32_t rank = in_tensor_specs[0]->shape_.size();
    int channel_dim = ConvertAxis<int>(attrs->axis, rank);

    const auto* out_scale = call->args[1].as<ConstantNode>();
    const auto* out_zero_point = call->args[2].as<ConstantNode>();

    // Set output quantization info.
    auto quant_info = ConvertQuantization(out_scale, out_zero_point, channel_dim);
    out_tensor_specs[0]->SetQuantization(quant_info);

    for (size_t i = 1; i < 3; i++) {
      in_tensor_specs[i]->shape_.clear();
    }

    return graph->CreateOperation<ops::DataConvert>();
  }
};

/*!
 * \brief Converter class for qnn.dequantize op.
 * \note Op format: qnn.dequantize(input, input_scale, input_zp) -> output.
 */
class DequantizeConverter final : public TimVxOpConverter {
 public:
  TimVxOp Convert(TimVxGraph graph, const CallNode* call, TimVxTensorSpecList& in_tensor_specs,
                  TimVxTensorSpecList& out_tensor_specs) override {
    const auto* attrs = call->attrs.as<qnn::DequantizeAttrs>();
    uint32_t rank = in_tensor_specs[0]->shape_.size();
    int channel_dim = ConvertAxis<int>(attrs->axis, rank);

    const auto* in_scale = call->args[1].as<ConstantNode>();
    const auto* in_zero_point = call->args[2].as<ConstantNode>();

    // Set input quantization info.
    auto quant_info = ConvertQuantization(in_scale, in_zero_point, channel_dim);
    in_tensor_specs[0]->SetQuantization(quant_info);

    for (size_t i = 1; i < 3; i++) {
      in_tensor_specs[i]->shape_.clear();
    }

    return graph->CreateOperation<ops::DataConvert>();
  }
};

const TimVxOpConverter::Memo TimVxOpConverter::GetMemo() {
  Memo memo;

  /* NN ops. */
  memo.emplace("nn.conv2d", std::make_unique<NNOpConverter<ops::Conv2d>>());
  memo.emplace("qnn.conv2d",
               std::make_unique<QnnWrapper<NNOpConverter<ops::Conv2d>, OpQuantFormat::EXPLICIT>>());

  memo.emplace("nn.conv2d_transpose", std::make_unique<NNOpConverter<ops::DeConv2d>>());
  memo.emplace(
      "qnn.conv2d_transpose",
      std::make_unique<QnnWrapper<NNOpConverter<ops::DeConv2d>, OpQuantFormat::EXPLICIT>>());

  memo.emplace("nn.dense", std::make_unique<NNOpConverter<ops::FullyConnected>>());
  memo.emplace(
      "qnn.dense",
      std::make_unique<QnnWrapper<NNOpConverter<ops::FullyConnected>, OpQuantFormat::EXPLICIT>>());

  /* Activation functions. */
  memo.emplace(
      "nn.relu",
      std::make_unique<QnnWrapper<UnaryOpConverter<ops::Relu>, OpQuantFormat::TRANSPARENT>>());

  memo.emplace("nn.leaky_relu", std::make_unique<UnaryOpConverter<ops::LeakyRelu>>());
  memo.emplace(
      "qnn.leaky_relu",
      std::make_unique<QnnWrapper<UnaryOpConverter<ops::LeakyRelu>, OpQuantFormat::EXPLICIT>>());

  memo.emplace("sigmoid", std::make_unique<UnaryOpConverter<ops::Sigmoid>>());
  memo.emplace(
      "qnn.sigmoid",
      std::make_unique<QnnWrapper<UnaryOpConverter<ops::Sigmoid>, OpQuantFormat::EXPLICIT>>());

  memo.emplace("tanh", std::make_unique<UnaryOpConverter<ops::Tanh>>());
  memo.emplace(
      "qnn.tanh",
      std::make_unique<QnnWrapper<UnaryOpConverter<ops::Tanh>, OpQuantFormat::EXPLICIT>>());

  memo.emplace("nn.swish", std::make_unique<UnaryOpConverter<ops::Swish>>());
  memo.emplace(
      "qnn.swish",
      std::make_unique<QnnWrapper<UnaryOpConverter<ops::Swish>, OpQuantFormat::EXPLICIT>>());
  memo.emplace(
      "qnn.hardswish",
      std::make_unique<QnnWrapper<UnaryOpConverter<ops::HardSwish>, OpQuantFormat::EXPLICIT>>());

  memo.emplace("erf", std::make_unique<UnaryOpConverter<ops::Erf>>());
  memo.emplace("qnn.erf",
               std::make_unique<QnnWrapper<UnaryOpConverter<ops::Erf>, OpQuantFormat::EXPLICIT>>());

  /* Elementwise unary ops. */
  memo.emplace("negative", std::make_unique<UnaryOpConverter<ops::Neg>>());

  memo.emplace("abs", std::make_unique<UnaryOpConverter<ops::Abs>>());
  memo.emplace("qnn.abs",
               std::make_unique<QnnWrapper<UnaryOpConverter<ops::Abs>, OpQuantFormat::EXPLICIT>>());

  memo.emplace("exp", std::make_unique<UnaryOpConverter<ops::Exp>>());
  memo.emplace("qnn.exp",
               std::make_unique<QnnWrapper<UnaryOpConverter<ops::Exp>, OpQuantFormat::EXPLICIT>>());

  memo.emplace("log", std::make_unique<UnaryOpConverter<ops::Log>>());
  memo.emplace("qnn.log",
               std::make_unique<QnnWrapper<UnaryOpConverter<ops::Log>, OpQuantFormat::EXPLICIT>>());

  memo.emplace("sqrt", std::make_unique<UnaryOpConverter<ops::Sqrt>>());
  memo.emplace(
      "qnn.sqrt",
      std::make_unique<QnnWrapper<UnaryOpConverter<ops::Sqrt>, OpQuantFormat::EXPLICIT>>());

  memo.emplace("rsqrt", std::make_unique<UnaryOpConverter<ops::Rsqrt>>());
  memo.emplace(
      "qnn.rsqrt",
      std::make_unique<QnnWrapper<UnaryOpConverter<ops::Rsqrt>, OpQuantFormat::EXPLICIT>>());

  memo.emplace("square", std::make_unique<UnaryOpConverter<ops::Square>>());
  memo.emplace("reciprocal", std::make_unique<UnaryOpConverter<ops::Rcp>>());

  memo.emplace("sin", std::make_unique<UnaryOpConverter<ops::Sin>>());

  memo.emplace("round", std::make_unique<UnaryOpConverter<ops::Round>>());
  memo.emplace("floor", std::make_unique<UnaryOpConverter<ops::Floor>>());
  memo.emplace("ceil", std::make_unique<UnaryOpConverter<ops::Ceil>>());

  memo.emplace("logical_not", std::make_unique<UnaryOpConverter<ops::LogicalNot>>());

  memo.emplace("clip", std::make_unique<UnaryOpConverter<ops::Clip>>());

  /* Elementwise binary ops. */
  memo.emplace("add", std::make_unique<BinaryOpConverter<ops::Add>>());
  memo.emplace(
      "qnn.add",
      std::make_unique<QnnWrapper<BinaryOpConverter<ops::Add>, OpQuantFormat::EXPLICIT>>());

  memo.emplace("subtract", std::make_unique<BinaryOpConverter<ops::Sub>>());
  memo.emplace(
      "qnn.subtract",
      std::make_unique<QnnWrapper<BinaryOpConverter<ops::Sub>, OpQuantFormat::EXPLICIT>>());

  memo.emplace("multiply", std::make_unique<BinaryOpConverter<ops::Multiply>>());
  memo.emplace(
      "qnn.mul",
      std::make_unique<QnnWrapper<BinaryOpConverter<ops::Multiply>, OpQuantFormat::EXPLICIT>>());

  memo.emplace("divide", std::make_unique<BinaryOpConverter<ops::Div>>());
  memo.emplace("floor_divide", std::make_unique<BinaryOpConverter<ops::FloorDiv>>());

  memo.emplace("power", std::make_unique<BinaryOpConverter<ops::Pow>>());

  memo.emplace("equal", std::make_unique<BinaryOpConverter<ops::Equal>>());
  memo.emplace("not_equal", std::make_unique<BinaryOpConverter<ops::NotEqual>>());
  memo.emplace("less", std::make_unique<BinaryOpConverter<ops::Less>>());
  memo.emplace("less_equal", std::make_unique<BinaryOpConverter<ops::LessOrEqual>>());
  memo.emplace("greater", std::make_unique<BinaryOpConverter<ops::Greater>>());
  memo.emplace("greater_equal", std::make_unique<BinaryOpConverter<ops::GreaterOrEqual>>());

  memo.emplace("minimum", std::make_unique<BinaryOpConverter<ops::Minimum>>());
  memo.emplace("maximum", std::make_unique<BinaryOpConverter<ops::Maximum>>());

  memo.emplace("logical_and", std::make_unique<BinaryOpConverter<ops::LogicalAnd>>());
  memo.emplace("logical_or", std::make_unique<BinaryOpConverter<ops::LogicalOr>>());

  /* Reduce ops. */
  memo.emplace("sum", std::make_unique<UnaryOpConverter<ops::ReduceSum>>());
  memo.emplace(
      "qnn.sum",
      std::make_unique<QnnWrapper<UnaryOpConverter<ops::ReduceSum>, OpQuantFormat::QDQ>>());

  memo.emplace("prod", std::make_unique<UnaryOpConverter<ops::ReduceProd>>());
  memo.emplace(
      "qnn.prod",
      std::make_unique<QnnWrapper<UnaryOpConverter<ops::ReduceProd>, OpQuantFormat::QDQ>>());

  memo.emplace("mean", std::make_unique<UnaryOpConverter<ops::ReduceMean>>());
  memo.emplace(
      "qnn.mean",
      std::make_unique<QnnWrapper<UnaryOpConverter<ops::ReduceMean>, OpQuantFormat::I32_CAST>>());

  memo.emplace(
      "min",
      std::make_unique<QnnWrapper<UnaryOpConverter<ops::ReduceMin>, OpQuantFormat::TRANSPARENT>>());
  memo.emplace(
      "max",
      std::make_unique<QnnWrapper<UnaryOpConverter<ops::ReduceMax>, OpQuantFormat::TRANSPARENT>>());

  memo.emplace("all", std::make_unique<UnaryOpConverter<ops::ReduceAll>>());
  memo.emplace("any", std::make_unique<UnaryOpConverter<ops::ReduceAny>>());

  memo.emplace("argmin", std::make_unique<UnaryOpConverter<ops::ArgMin>>());
  memo.emplace("argmax", std::make_unique<UnaryOpConverter<ops::ArgMax>>());

  /* Neighborhood ops. */
  memo.emplace(
      "nn.max_pool2d",
      std::make_unique<QnnWrapper<UnaryOpConverter<ops::Pool2d>, OpQuantFormat::TRANSPARENT>>());

  memo.emplace("nn.avg_pool2d", std::make_unique<UnaryOpConverter<ops::Pool2d>>());
  memo.emplace(
      "qnn.avg_pool2d",
      std::make_unique<QnnWrapper<UnaryOpConverter<ops::Pool2d>, OpQuantFormat::I32_CAST>>());

  memo.emplace(
      "image.resize2d",
      std::make_unique<QnnWrapper<UnaryOpConverter<ops::Resize>, OpQuantFormat::TRANSPARENT>>());

  /* Tensor data rearrangements. */
  memo.emplace(
      "reshape",
      std::make_unique<QnnWrapper<UnaryOpConverter<ops::Reshape>, OpQuantFormat::TRANSPARENT>>());

  memo.emplace(
      "transpose",
      std::make_unique<QnnWrapper<UnaryOpConverter<ops::Transpose>, OpQuantFormat::TRANSPARENT>>());

  memo.emplace("nn.space_to_depth",
               std::make_unique<
                   QnnWrapper<UnaryOpConverter<ops::SpaceToDepth>, OpQuantFormat::TRANSPARENT>>());

  memo.emplace("concatenate", std::make_unique<TupleOpConverter<ops::Concat>>());
  memo.emplace(
      "qnn.concatenate",
      std::make_unique<QnnWrapper<TupleOpConverter<ops::Concat>, OpQuantFormat::EXPLICIT>>());

  memo.emplace(
      "split",
      std::make_unique<QnnWrapper<UnaryOpConverter<ops::Split>, OpQuantFormat::TRANSPARENT>>());

  /* Gather-Scatter ops. */
  memo.emplace(
      "take",
      std::make_unique<QnnWrapper<UnaryOpConverter<ops::Gather>, OpQuantFormat::TRANSPARENT>>());
  memo.emplace(
      "gather",
      std::make_unique<
          QnnWrapper<UnaryOpConverter<ops::GatherElements>, OpQuantFormat::TRANSPARENT>>());

  /* Datatype conversions. */
  memo.emplace("cast", std::make_unique<UnaryOpConverter<ops::Cast>>());
  memo.emplace("qnn.quantize", std::make_unique<QuantizeConverter>());
  memo.emplace("qnn.dequantize", std::make_unique<DequantizeConverter>());
  memo.emplace(
      "qnn.requantize",
      std::make_unique<QnnWrapper<UnaryOpConverter<ops::DataConvert>, OpQuantFormat::EXPLICIT>>());

  return memo;
}

}  // namespace tim_vx
}  // namespace contrib
}  // namespace relay
}  // namespace tvm