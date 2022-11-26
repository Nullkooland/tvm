# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""TIM-VX supported operators."""
from typing import Optional, Callable, Sequence, Tuple, List, Dict
from itertools import repeat
from functools import reduce
import operator
import numpy as np

import tvm
from tvm import relay
from tvm.relay import transform
from tvm.relay.expr import Expr, Call
from tvm.relay.function import Function
from tvm.relay.expr_functor import ExprMutator
from tvm.relay.build_module import bind_params_by_name
from tvm.ir.tensor_type import TensorType
from tvm.relay.op import op_attrs

from ...dataflow_pattern import (
    wildcard,
    is_constant,
    is_op,
    is_tuple,
)
from .register import register_pattern_table, get_pattern_table


@transform.function_pass(opt_level=0)
class RemoveInputQuantizeOps(ExprMutator):
    """A pass to remove qnn.quantize ops on inputs."""

    def transform_function(
        self,
        func: Function,
        mod: tvm.runtime.Module,
        ctx: tvm.transform.PassContext
    ) -> Function:
        return self.visit_function(func)

    def visit_function(self, func: Function) -> Function:
        self.params = func.params
        self.param_to_dtype_map = {param: "" for param in func.params}

        new_body = self.visit(func.body)
        binds = {
            param: relay.var(
                name_hint=param.name_hint,
                shape=param.type_annotation.shape,
                dtype=dtype
            ) for param, dtype in self.param_to_dtype_map.items()
        }
        new_body = relay.bind(new_body, binds)

        return Function(
            params=list(binds.values()),
            body=new_body,
            ret_type=func.ret_type,
            type_params=func.type_params,
            attrs=func.attrs
        )

    def visit_call(self, call: Call) -> Expr:
        if call.args[0] in self.param_to_dtype_map and call.op.name == "qnn.quantize":
            dtype = call.attrs.out_dtype
            self.param_to_dtype_map[call.args[0]] = dtype
            return call.args[0]

        return super().visit_call(call)


@tvm.ir.register_op_attr(op_name="transpose", attr_key="tim_vx.legalize")
def legal_transpose(
    attrs: op_attrs.TransposeAttrs,
    inputs: Sequence[Expr],
    types: Sequence[TensorType]
) -> Expr:
    data = inputs[0]
    rank = len(types[0].concrete_shape)
    perm: Tuple[int, ...] = tuple(attrs.axes) if attrs.axes \
        else tuple(reversed(range(rank)))

    return relay.transpose(data, perm)


@tvm.ir.register_op_attr(op_name="reshape", attr_key="tim_vx.legalize")
def legal_reshape(
    attrs: op_attrs.ReshapeAttrs,
    inputs: Sequence[Expr],
    types: Sequence[TensorType]
) -> Expr:
    data = inputs[0]
    old_shape = types[0].concrete_shape
    num_elements = reduce(operator.mul, old_shape)
    new_shape = list(attrs.newshape)
    if 0 in new_shape:
        assert len(new_shape) == len(old_shape)
        for i in range(len(new_shape)):
            if new_shape[i] == 0:
                new_shape[i] = old_shape[i]

    if -1 in new_shape:
        i = new_shape.index(-1)
        dims_left = reduce(operator.mul, new_shape[:i], 1)
        dims_right = reduce(operator.mul, new_shape[i+1:], 1)
        new_shape[i] = num_elements // (dims_left * dims_right)

    return relay.reshape(data, newshape=new_shape)


@tvm.ir.register_op_attr(op_name="nn.batch_flatten", attr_key="tim_vx.legalize")
def legal_batch_flatten(
    attrs: None,
    inputs: Sequence[Expr],
    types: Sequence[TensorType]
) -> Expr:
    data = inputs[0]
    old_shape = types[0].concrete_shape
    num_batches = old_shape[0]
    num_elements_per_batch = reduce(operator.mul, old_shape[1:])
    return relay.reshape(data, newshape=(num_batches, num_elements_per_batch))


@tvm.ir.register_op_attr(op_name="expand_dims", attr_key="tim_vx.legalize")
def legal_expand_dims(
    attrs: op_attrs.ExpandDimsAttrs,
    inputs: Sequence[Expr],
    types: Sequence[TensorType]
) -> Expr:
    data = inputs[0]
    expanded_shape = list(types[0].concrete_shape)
    axis = int(attrs.axis)
    num_newaxis = int(attrs.num_newaxis)
    expanded_shape[axis:axis] = repeat(1, num_newaxis)
    return relay.reshape(data, newshape=expanded_shape)


@tvm.ir.register_op_attr(op_name="strided_slice", attr_key="tim_vx.legalize")
def legal_strided_slice(
    attrs: op_attrs.StridedSliceAttrs,
    inputs: Sequence[Expr],
    types: Sequence[TensorType]
) -> Expr:
    data = inputs[0]
    input_shape = types[0].concrete_shape
    rank = len(input_shape)
    begin = list(repeat(0, rank))
    end = list(input_shape)
    strides = list(repeat(1, rank))

    axes = list(attrs.axes) if attrs.axes else list(range(rank))
    for k, axis in enumerate(axes):
        i = int(axis)
        begin[i] = max(int(attrs.begin[k]), 0)
        end[i] = min(int(attrs.end[k]), input_shape[i])
        strides[i] = int(attrs.strides[k])

    return relay.strided_slice(
        data,
        begin=begin,
        end=end,
        strides=strides
    )


@tvm.ir.register_op_attr(op_name="split", attr_key="tim_vx.legalize")
def legal_split(
    attrs: op_attrs.SplitAttrs,
    inputs: Sequence[Expr],
    types: Sequence[TensorType]
) -> Expr:
    data = inputs[0]
    input_shape = types[0].concrete_shape
    axis = int(attrs.axis)
    indices_or_sections = attrs.indices_or_sections

    if isinstance(indices_or_sections, tvm.ir.Array):
        indices: Tuple[int, ...] = tuple(indices_or_sections)
    else:
        num_sections = int(indices_or_sections)
        len_axis = input_shape[axis]
        section_size = len_axis // num_sections
        indices: Tuple[int, ...] = tuple(
            range(section_size, len_axis, section_size)
        )

    return relay.split(
        data,
        indices_or_sections=indices,
        axis=axis,
    ).tuple_value


@tvm.ir.register_op_attr(op_name="nn.global_avg_pool2d", attr_key="tim_vx.legalize")
def legal_global_avg_pool2d(
    attrs: op_attrs.GlobalPool2DAttrs,
    inputs: Sequence[Expr],
    types: Sequence[TensorType]
) -> Expr:
    data = inputs[0]
    return relay.mean(data, axis=[2, 3], keepdims=True)


@tvm.ir.register_op_attr(op_name="nn.adaptive_avg_pool2d", attr_key="tim_vx.legalize")
def legal_adaptive_avg_pool2d(
    attrs: op_attrs.AdaptivePool2DAttrs,
    inputs: Sequence[Expr],
    types: Sequence[TensorType]
) -> Expr:
    data = inputs[0]
    output_size: Tuple[int, int] = tuple(attrs.output_size)
    if output_size == (1, 1):
        # Global AvgPool2d.
        return relay.mean(data, axis=[2, 3], keepdims=True)
    else:
        # Calculate stride and kernel size.
        hi, wi = types[0].concrete_shape
        ho, wo = output_size
        strides = (hi // ho, wi // wo)
        pool_size = ((ho - 1) * strides[0], (wo - 1) * strides[1])
        return relay.nn.avg_pool2d(
            data,
            pool_size=pool_size,
            strides=strides
        )


def _legal_reduce(
    reduce_op: Callable[..., Expr],
    attrs: op_attrs.ReduceAttrs,
    inputs: Sequence[Expr],
    types: Sequence[TensorType]
) -> Expr:
    input_shape = types[0].concrete_shape
    rank = len(input_shape)

    if attrs.axis:
        reduce_axes = []
        for axis in attrs.axis:
            # Map negative axis into [0, rank) range.
            reduce_axes.append(axis if axis >= 0 else axis + rank)
        # Make sure the in axes are in ascending order.
        reduce_axes.sort()
    else:
        # Global reduction.
        reduce_axes = list(range(rank))

    if attrs.exclude:
        # Handle exclusion.
        excluded_axes = reduce_axes.copy()
        reduce_axes.clear()

        for axis in range(rank):
            if axis in excluded_axes:
                continue
            reduce_axes.append(axis)

    data = inputs[0]
    return reduce_op(data, reduce_axes, attrs.keepdims)


@tvm.ir.register_op_attr(op_name="sum", attr_key="tim_vx.legalize")
def legal_sum(
    attrs: op_attrs.ReduceAttrs,
    inputs: Sequence[Expr],
    types: Sequence[TensorType]
) -> Expr:
    return _legal_reduce(relay.sum, attrs, inputs, types)


@tvm.ir.register_op_attr(op_name="prod", attr_key="tim_vx.legalize")
def legal_prod(
    attrs: op_attrs.ReduceAttrs,
    inputs: Sequence[Expr],
    types: Sequence[TensorType]
) -> Expr:
    return _legal_reduce(relay.prod, attrs, inputs, types)


@tvm.ir.register_op_attr(op_name="max", attr_key="tim_vx.legalize")
def legal_max(
    attrs: op_attrs.ReduceAttrs,
    inputs: Sequence[Expr],
    types: Sequence[TensorType]
) -> Expr:
    return _legal_reduce(relay.max, attrs, inputs, types)


@tvm.ir.register_op_attr(op_name="min", attr_key="tim_vx.legalize")
def legal_min(
    attrs: op_attrs.ReduceAttrs,
    inputs: Sequence[Expr],
    types: Sequence[TensorType]
) -> Expr:
    return _legal_reduce(relay.min, attrs, inputs, types)


@tvm.ir.register_op_attr(op_name="mean", attr_key="tim_vx.legalize")
def legal_mean(
    attrs: op_attrs.ReduceAttrs,
    inputs: Sequence[Expr],
    types: Sequence[TensorType]
) -> Expr:
    return _legal_reduce(relay.mean, attrs, inputs, types)


@tvm.ir.register_op_attr(op_name="argmax", attr_key="tim_vx.legalize")
def legal_argmax(
    attrs: op_attrs.ReduceAttrs,
    inputs: Sequence[Expr],
    types: Sequence[TensorType]
) -> Expr:
    return _legal_reduce(relay.argmax, attrs, inputs, types)


@tvm.ir.register_op_attr(op_name="argmin", attr_key="tim_vx.legalize")
def legal_argmin(
    attrs: op_attrs.ReduceAttrs,
    inputs: Sequence[Expr],
    types: Sequence[TensorType]
) -> Expr:
    return _legal_reduce(relay.argmin, attrs, inputs, types)


@tvm.ir.register_op_attr(op_name="any", attr_key="tim_vx.legalize")
def legal_any(
    attrs: op_attrs.ReduceAttrs,
    inputs: Sequence[Expr],
    types: Sequence[TensorType]
) -> Expr:
    return _legal_reduce(relay.any, attrs, inputs, types)


@tvm.ir.register_op_attr(op_name="all", attr_key="tim_vx.legalize")
def legal_all(
    attrs: op_attrs.ReduceAttrs,
    inputs: Sequence[Expr],
    types: Sequence[TensorType]
) -> Expr:
    return _legal_reduce(relay.all, attrs, inputs, types)


@tvm.ir.register_op_attr(op_name="nn.upsampling", attr_key="tim_vx.legalize")
def legal_nn_upsampling(
    attrs: op_attrs.UpSamplingAttrs,
    inputs: Sequence[Expr],
    types: Sequence[TensorType]
) -> Expr:
    data = inputs[0]
    output_shape = types[1].concrete_shape
    method = str(attrs.method)
    align_corners = bool(attrs.align_corners)
    if method == "bilinear":
        method = "linear"
    return relay.image.resize2d(
        data,
        size=output_shape[2:],
        layout="NCHW",
        method=method,
        coordinate_transformation_mode="align_corners" if align_corners else "half_pixel"
    )


@tvm.ir.register_op_attr(op_name="stack", attr_key="tim_vx.legalize")
def legal_stack(
    attrs: op_attrs.StackAttrs,
    inputs: Sequence[Expr],
    types: Sequence[TensorType]
) -> Expr:
    components: Tuple[relay.Var, ...] = inputs[0]
    num_components = len(components)
    component_shape = types[0].fields[0].concrete_shape
    component_rank = len(component_shape)
    axis = int(attrs.axis)
    if axis < 0:
        axis += (component_rank + 1)

    if axis < component_rank:
        # If axis is not the right-most axis of the output tensor,
        # we can concatenate first then reshape.
        output_shape = list(component_shape)
        output_shape.insert(axis, num_components)
        data = relay.concatenate(components, axis=axis)
        return relay.reshape(data, output_shape)
    else:
        # Otherwise, we need to expand the input tensors then concatenate.
        components_expanded: List[Expr] = []
        for i in range(num_components):
            components_expanded.append(relay.reshape(
                components[i], (*component_shape, 1)
            ))

        return relay.concatenate(components_expanded, axis=axis)


@register_pattern_table("tim_vx")
def tim_vx_pattern_table():
    """Get the TIM-VX pattern table."""

    def nn_conv2d_pattern():
        data = wildcard()
        weight = wildcard()
        bias = wildcard()

        pattern = is_op("nn.conv2d")(data, weight)
        pattern = pattern.optional(
            lambda x:
            is_op("nn.bias_add")(x, bias) |
            is_op("add")(x, bias)
        )
        return pattern

    def qnn_conv2d_pattern():
        data = wildcard()
        weight = wildcard()
        bias = wildcard()

        pattern = is_op("qnn.conv2d")(
            data, weight, is_constant(), is_constant(), is_constant(), is_constant()
        )
        pattern = pattern.optional(
            lambda x:
            is_op("nn.bias_add")(x, bias) |
            is_op("add")(x, bias) |
            is_op("qnn.add")(x, bias, is_constant(), is_constant(),
                             is_constant(), is_constant(), is_constant(), is_constant())
        )
        pattern = is_op("qnn.requantize")(
            pattern, is_constant(), is_constant(), is_constant(), is_constant()
        )
        return pattern

    def nn_conv2d_transpose_pattern():
        data = wildcard()
        weight = wildcard()
        pattern = is_op("nn.conv2d_transpose")(data, weight)
        return pattern

    def qnn_conv2d_transpose_pattern():
        data = wildcard()
        weight = wildcard()
        pattern = is_op("qnn.conv2d_transpose")(
            data, weight, is_constant(), is_constant(), is_constant(), is_constant()
        )
        pattern = is_op("qnn.requantize")(
            pattern, is_constant(), is_constant(), is_constant(), is_constant()
        )
        return pattern

    def nn_dense_pattern():
        data = wildcard()
        weight = wildcard()
        bias = wildcard()
        pattern = is_op("nn.dense")(data, weight)
        pattern = is_op("nn.bias_add")(
            pattern, bias) | is_op("add")(pattern, bias)

        return pattern

    def qnn_dense_pattern():
        data = wildcard()
        weight = wildcard()
        bias = wildcard()

        pattern = is_op("qnn.dense")(
            data, weight, is_constant(), is_constant(), is_constant(), is_constant()
        )
        pattern = pattern.optional(
            lambda x:
            is_op("nn.bias_add")(x, bias) |
            is_op("add")(x, bias) |
            is_op("qnn.add")(x, bias, is_constant(), is_constant(),
                             is_constant(), is_constant(), is_constant(), is_constant())
        )
        pattern = is_op("qnn.requantize")(
            pattern, is_constant(), is_constant(), is_constant(), is_constant()
        )
        return pattern

    def square_pattern():
        data = wildcard()
        pattern = is_op("multiply")(data, data)
        return pattern

    def reciprocal_pattern():
        data = wildcard()
        one = is_constant()
        pattern = is_op("divide")(one, data)
        return pattern

    def check_reciprocal(extract: Call) -> bool:
        one = extract.args[0]
        if isinstance(one, relay.Constant):
            return np.all(one.data.numpy() == 1).item()
        return False

    def mean_variance_pattern():
        data = wildcard()
        mean = is_op("mean")(data)
        variance = is_op("variance")(data, mean)
        mean = is_op("squeeze")(mean) | mean
        pattern = is_tuple([mean, variance])
        return pattern

    def space_to_depth_2x2_pattern():
        data = wildcard()
        slices = tuple(
            is_op("strided_slice")(data).has_attr({
                "strides": (1, 1, 2, 2)  # type: ignore
            }) for _ in range(4)
        )
        slices = is_tuple(slices)
        pattern = is_op("concatenate")(slices).has_attr({
            "axis": 1
        })
        return pattern

    def check_space_to_depth_2x2(extract: Call) -> bool:
        spatial_begins = ((0, 0), (0, 1), (1, 0), (1, 1))
        for slice, spatial_begin in zip(extract.args[0], spatial_begins):
            input_shape: Tuple[int, ...] = tuple(
                slice.args[0].checked_type.shape
            )
            begin: Tuple[int, ...] = tuple(slice.attrs.begin)
            end: Tuple[int, ...] = tuple(slice.attrs.end)
            if begin[:2] != (0, 0) or begin[2:] != spatial_begin or end != input_shape:
                return False
        return True

    def qnn_swish_pattern():
        x = wildcard()
        sigmoid = is_op("qnn.sigmoid")(
            x,
            is_constant(), is_constant(), is_constant(), is_constant()
        )
        sigmoid = sigmoid | is_op("qnn.requantize")(
            sigmoid,
            is_constant(), is_constant(), is_constant(), is_constant()
        )
        pattern = is_op("qnn.mul")(
            x, sigmoid,
            is_constant(), is_constant(), is_constant(), is_constant(),
            is_constant(), is_constant()
        )
        return pattern

    def nn_swish_pattern():
        x = wildcard()
        sigmoid = is_op("sigmoid")(x)
        return x * sigmoid

    def qnn_avg_pool2d_pattern():
        data = wildcard()
        pattern = is_op("cast")(data).has_attr(
            {"dtype": "int32"})  # type: ignore
        pattern = is_op("nn.avg_pool2d")(pattern)
        pattern = is_op("cast")(pattern)
        pattern = pattern | is_op("qnn.requantize")(
            pattern, is_constant(), is_constant(), is_constant(), is_constant()
        )
        return pattern

    def qnn_mean_pattern():
        data = wildcard()
        pattern = \
            is_op("cast")(data).has_attr({"dtype": "int32"}) | \
            is_op("qnn.requantize")(
                data, is_constant(), is_constant(), is_constant(), is_constant()
            ).has_attr({"out_dtype": "int32"})

        pattern = is_op("mean")(pattern)

        pattern = \
            is_op("cast")(pattern) | \
            is_op("qnn.requantize")(
                pattern, is_constant(), is_constant(), is_constant(), is_constant()
            )
        return pattern

    def qnn_sum_pattern():
        data = wildcard()
        pattern = is_op("qnn.dequantize")(data, is_constant(), is_constant())
        pattern = is_op("sum")(pattern)
        pattern = is_op("qnn.quantize")(pattern, is_constant(), is_constant())
        return pattern

    def qnn_prod_pattern():
        data = wildcard()
        pattern = is_op("qnn.dequantize")(data, is_constant(), is_constant())
        pattern = is_op("prod")(pattern)
        pattern = is_op("qnn.quantize")(pattern, is_constant(), is_constant())
        return pattern

    # def qnn_mean_pattern():
    #     data = wildcard()
    #     pattern = is_op("qnn.dequantize")(data, is_constant(), is_constant())
    #     pattern = is_op("mean")(pattern)
    #     pattern = is_op("qnn.quantize")(pattern, is_constant(), is_constant())
    #     return pattern

    tim_vx_patterns = [
        ("tim_vx.nn.conv2d", nn_conv2d_pattern()),
        ("tim_vx.qnn.conv2d", qnn_conv2d_pattern()),
        ("tim_vx.nn.conv2d_transpose", nn_conv2d_transpose_pattern()),
        ("tim_vx.qnn.conv2d_transpose", qnn_conv2d_transpose_pattern()),
        ("tim_vx.nn.dense", nn_dense_pattern()),
        ("tim_vx.qnn.dense", qnn_dense_pattern()),
        ("tim_vx.square", square_pattern()),
        ("tim_vx.reciprocal", reciprocal_pattern(), check_reciprocal),
        ("tim_vx.mean_variance", mean_variance_pattern()),
        ("tim_vx.nn.space_to_depth_2x2",
         space_to_depth_2x2_pattern(), check_space_to_depth_2x2),
        ("tim_vx.nn.swish", nn_swish_pattern()),
        ("tim_vx.qnn.swish", qnn_swish_pattern()),
        ("tim_vx.qnn.avg_pool2d", qnn_avg_pool2d_pattern()),
        ("tim_vx.qnn.sum", qnn_sum_pattern()),
        ("tim_vx.qnn.prod", qnn_prod_pattern()),
        ("tim_vx.qnn.mean", qnn_mean_pattern()),
    ]
    return tim_vx_patterns


# TODO: Detailed checks for each op.
def _register_external_op_helper(op_name: str, supported: bool = True):
    @tvm.ir.register_op_attr(op_name, "target.tim_vx")
    def _func_wrapper(args) -> bool:
        return supported

    return _func_wrapper


# Activation functions.
_register_external_op_helper("nn.relu")
_register_external_op_helper("nn.leaky_relu")
_register_external_op_helper("qnn.leaky_relu")
_register_external_op_helper("sigmoid")
_register_external_op_helper("qnn.sigmoid")
_register_external_op_helper("tanh")
_register_external_op_helper("qnn.tanh")
_register_external_op_helper("qnn.hardswish")
_register_external_op_helper("erf")
_register_external_op_helper("qnn.erf")

# Elementwise unary ops.
_register_external_op_helper("negative")
_register_external_op_helper("abs")
_register_external_op_helper("qnn.abs")
_register_external_op_helper("exp")
_register_external_op_helper("qnn.exp")
_register_external_op_helper("log")
_register_external_op_helper("qnn.log")
_register_external_op_helper("sqrt")
_register_external_op_helper("qnn.sqrt")
_register_external_op_helper("rsqrt")
_register_external_op_helper("qnn.rsqrt")
_register_external_op_helper("square")
_register_external_op_helper("sin")
_register_external_op_helper("round")
_register_external_op_helper("floor")
_register_external_op_helper("ceil")
_register_external_op_helper("logical_not")
_register_external_op_helper("clip")

# Elementwise binary ops.
_register_external_op_helper("add")
_register_external_op_helper("qnn.add")
_register_external_op_helper("subtract")
_register_external_op_helper("qnn.subtract")
_register_external_op_helper("multiply")
_register_external_op_helper("qnn.mul")
_register_external_op_helper("divide")
_register_external_op_helper("floor_divide")
_register_external_op_helper("power")
_register_external_op_helper("equal")
_register_external_op_helper("not_equal")
_register_external_op_helper("less")
_register_external_op_helper("less_equal")
_register_external_op_helper("greater")
_register_external_op_helper("greater_equal")
_register_external_op_helper("minimum")
_register_external_op_helper("maximum")
_register_external_op_helper("logical_and")
_register_external_op_helper("logical_or")

# Reduce ops.
_register_external_op_helper("sum")
_register_external_op_helper("prod")
_register_external_op_helper("mean")
_register_external_op_helper("min")
_register_external_op_helper("max")
_register_external_op_helper("all")
_register_external_op_helper("any")
_register_external_op_helper("argmin")
_register_external_op_helper("argmax")

# Neighborhood ops.
_register_external_op_helper("nn.max_pool2d")
_register_external_op_helper("nn.avg_pool2d")
_register_external_op_helper("image.resize2d")

# Tensor data rearrangements.
_register_external_op_helper("reshape")
_register_external_op_helper("squeeze")
_register_external_op_helper("transpose")
_register_external_op_helper("nn.space_to_depth")
_register_external_op_helper("concatenate")
_register_external_op_helper("qnn.concatenate")
_register_external_op_helper("split")
_register_external_op_helper("strided_slice")

# Gather-Scatter ops.
_register_external_op_helper("where")
_register_external_op_helper("take")
_register_external_op_helper("gather")

# Datatype conversions.
_register_external_op_helper("cast")
_register_external_op_helper("qnn.quantize")
_register_external_op_helper("qnn.dequantize")
_register_external_op_helper("qnn.requantize")


def partition_for_tim_vx(
    mod: tvm.IRModule,
    params: Optional[Dict[str, tvm.nd.NDArray]] = None,
    **opts
) -> tvm.IRModule:
    """Partition the graph greedily offloading supported
    operators to TIM-VX runtime.

    Parameters
    ----------
    mod : Module
        The module to run passes on.
    params : Optional[Dict[str, NDArray]]
        Constant input parameters.

    Returns
    -------
    ret : annotated and partitioned module.
    """
    if params:
        mod["main"] = bind_params_by_name(mod["main"], params)

    seq = tvm.transform.Sequential(
        [
            transform.InferType(),
            transform.FakeQuantizationToInteger(),
            transform.FoldConstant(fold_qnn=True),
            transform.Legalize("tim_vx.legalize"),
            transform.MergeComposite(get_pattern_table("tim_vx")),
            transform.AnnotateTarget("tim_vx"),
            transform.MergeCompilerRegions(),
            transform.PartitionGraph(),
        ]
    )

    return seq(mod)


def is_tim_vx_runtime_enabled() -> bool:
    """Check if TIM-VX runtime is enabled.

    Returns
    -------
    ret: bool
        True if enabled, False if not.
    """
    check_func: Optional[Callable[..., bool]] = tvm.get_global_func(
        "relay.op.is_tim_vx_runtime_enabled", True
    )  # type: ignore
    if check_func:
        return check_func()
    return False
