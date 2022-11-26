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
 * \file src/relay/backend/contrib/tim_vx/utils.h
 * \brief Utility helper methods used by TIM-VX codegen.
 */

#ifndef TVM_RUNTIME_CONTRIB_TIM_VX_UTILS_H_
#define TVM_RUNTIME_CONTRIB_TIM_VX_UTILS_H_

#include <tim/vx/context.h>
#include <tim/vx/graph.h>
#include <tim/vx/tensor.h>
#include <tvm/relay/expr.h>

#include <algorithm>
#include <array>
#include <string_view>
#include <vector>

namespace tvm {
namespace relay {
namespace contrib {
namespace tim_vx {

/*! \brief Type aliases. */
using TimVxContext = std::shared_ptr<tim::vx::Context>;
using TimVxGraph = std::shared_ptr<tim::vx::Graph>;
using TimVxOp = std::shared_ptr<tim::vx::Operation>;
using TimVxTensor = std::shared_ptr<tim::vx::Tensor>;
using TimVxTensorSpec = std::shared_ptr<tim::vx::TensorSpec>;
using TimVxTensorList = std::vector<TimVxTensor>;
using TimVxTensorSpecList = std::vector<TimVxTensorSpec>;

/*!
 * \brief Convert string vector to index array.
 * \tparam T The index datatype.
 * \tparam Dim The index dimensions.
 */
template <typename T, size_t Dim>
inline std::array<T, Dim> ConvertIndexArray(const Array<IndexExpr>& index_array) {
  static_assert(std::is_integral<T>(), "Index value must be integer");
  std::array<T, Dim> index;
  std::transform(index_array.begin(), index_array.end(), index.begin(),
                 [](const IndexExpr& val) { return static_cast<T>(val.as<IntImmNode>()->value); });
  return index;
}

/*! \brief Read the channel axis for a given TIM-VX data layout. */
inline std::array<uint32_t, 4> ConvertPadding(std::array<uint32_t, 4> tvm_padding) {
  return {
      tvm_padding[1],  // left.
      tvm_padding[3],  // right.
      tvm_padding[0],  // top.
      tvm_padding[2],  // bottom.
  };
}

/*! \brief Convert string to TIM-VX resize type enum. */
inline tim::vx::ResizeType ConvertResizeType(const std::string_view method) {
  if (method == "area") {
    return tim::vx::ResizeType::AREA;
  } else if (method == "linear") {
    return tim::vx::ResizeType::BILINEAR;
  } else if (method == "nearest_neighbor") {
    return tim::vx::ResizeType::NEAREST_NEIGHBOR;
  }
  // Fall back to NEAREST_NEIGHBOR method.
  return tim::vx::ResizeType::NEAREST_NEIGHBOR;
}

/*! \brief Convert TVM shape array to TIM-VX shape (uint32 vector). */
inline tim::vx::ShapeType ConvertShape(const Array<PrimExpr>& tvm_shape) {
  size_t rank = tvm_shape.size();
  if (rank == 0) {
    return {1};
  }

  tim::vx::ShapeType vx_shape(rank);
  // Reverse channel order because TIM-VX uses column-major data layout.
  // e.g.: NCHW -> WHCN, NHWC -> CWHN.
  for (size_t i = 0; i < rank; i++) {
    vx_shape[rank - i - 1] = static_cast<uint32_t>(tvm_shape[i].as<IntImmNode>()->value);
  }
  return vx_shape;
}

/*! \brief Convert shape array to TIM-VX shape (uint32 vector). */
inline tim::vx::ShapeType ConvertShape(const Array<Integer>& tvm_shape) {
  size_t rank = tvm_shape.size();
  tim::vx::ShapeType vx_shape(rank);

  // Reverse channel order because TIM-VX uses column-major layout.
  // e.g.: NCHW -> WHCN, NHWC -> CWHN.
  for (size_t i = 0; i < rank; i++) {
    vx_shape[rank - i - 1] = static_cast<uint32_t>(tvm_shape[i]->value);
  }
  return vx_shape;
}

/*! \brief Convert TVM axis to TIM-VX axis. */
template <typename T>
inline T ConvertAxis(int tvm_axis, uint32_t rank) {
  static_assert(std::is_integral<T>(), "Requires integer type.");
  // Map negative axis into [0, rank) range.
  tvm_axis += (tvm_axis < 0) ? rank : 0;
  // Map TVM axis to TIM-VX axis.
  return static_cast<T>((rank - 1) - tvm_axis);
}

/*! \brief Convert TVM axes to TIM-VX axes. */
template <typename T>
inline std::vector<T> ConvertAxes(const Array<Integer>& tvm_axes, uint32_t rank) {
  static_assert(std::is_integral<T>(), "Requires integer type.");
  std::vector<T> vx_axes(tvm_axes.size());
  for (size_t i = 0; i < tvm_axes.size(); i++) {
    int64_t tvm_axis = tvm_axes[i]->value;
    // Map negative axis into [0, rank) range.
    tvm_axis += (tvm_axis < 0) ? rank : 0;
    // Map TVM axis to TIM-VX axis.
    vx_axes[(tvm_axes.size() - 1) - i] = static_cast<T>((rank - 1) - tvm_axis);
  }

  return vx_axes;
}

/*! \brief Convert TVM runtime data type to TIM-VX data type. */
inline tim::vx::DataType ConvertDataType(tvm::DataType tvm_dtype) {
  switch (tvm_dtype.code()) {
    case tvm::DataType::kUInt:
      switch (tvm_dtype.bits()) {
        case 1:
          return tim::vx::DataType::BOOL8;
        case 8:
          return tim::vx::DataType::UINT8;
        case 16:
          return tim::vx::DataType::UINT16;
        case 32:
          return tim::vx::DataType::UINT32;
      }
    case tvm::DataType::kInt:
      switch (tvm_dtype.bits()) {
        case 8:
          return tim::vx::DataType::INT8;
        case 16:
          return tim::vx::DataType::INT16;
        case 32:
          return tim::vx::DataType::INT32;
      }
    case tvm::DataType::kFloat:
      switch (tvm_dtype.bits()) {
        case 16:
          return tim::vx::DataType::FLOAT16;
        case 32:
          return tim::vx::DataType::FLOAT32;
      }
  }
  return tim::vx::DataType::UNKNOWN;
}

/*! \brief Convert related info to TIM-VX tensor quantization instance. */
inline tim::vx::Quantization ConvertQuantization(const ConstantNode* scale,
                                                 const ConstantNode* zero_point, int channel_dim) {
  std::vector<float> scale_val;
  std::vector<int> zero_point_val;
  auto quant_type = tim::vx::QuantType::NONE;

  if (scale->is_scalar()) {
    ICHECK(zero_point->is_scalar())
        << "The zero point must be a scalar for per-tensor quantization";

    quant_type = tim::vx::QuantType::ASYMMETRIC;
    channel_dim = -1;

    scale_val.push_back(*reinterpret_cast<const float*>(scale->data->data));
    zero_point_val.push_back(*reinterpret_cast<const int*>(zero_point->data->data));
  } else {
    ICHECK_EQ(scale->data->ndim, 1)
        << "The scale for per-channel quantization must have dimension of 1";

    quant_type = tim::vx::QuantType::SYMMETRIC_PER_CHANNEL;
    size_t num_channels = scale->data->shape[0];

    scale_val.resize(num_channels);
    std::copy_n(reinterpret_cast<const float*>(scale->data->data), num_channels, scale_val.begin());
    // In symmetric quantization, zero points must be all 0.
    zero_point_val.resize(num_channels, 0);
  }

  return tim::vx::Quantization(quant_type, channel_dim, scale_val, zero_point_val);
}

}  // namespace tim_vx
}  // namespace contrib
}  // namespace relay
}  // namespace tvm

#endif