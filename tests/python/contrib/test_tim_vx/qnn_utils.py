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

"""QNN utility passes for TIM-VX tests."""

from typing import Tuple
import tvm
from tvm import relay
from tvm.relay.op import op_attrs


@relay.transform.function_pass(opt_level=0, required=["InferType"])
class InsertImagePreProcess(relay.ExprMutator):
    """A pass that add image preprocess to the model graph
    so that the model can accept u8 images directly.
    """

    def __init__(self, nhwc_to_nchw: bool = False):
        super().__init__()
        self.nhwc_to_nchw = nhwc_to_nchw

    def transform_function(
        self,
        func: relay.Function,
        mod: tvm.runtime.Module,
        ctx: tvm.transform.PassContext
    ) -> relay.Function:
        self.input_var: relay.Var = func.params[0]
        self.ret_type: relay.Type = func.ret_type

        new_body = self.visit(func.body)

        return relay.Function(
            params=(self.input_var, ),
            body=new_body,
            ret_type=func.ret_type,
            type_params=func.type_params,
            attrs=func.attrs
        )

    def visit_call(self, call: relay.Call) -> relay.Expr:
        arg0 = call.args[0]
        if arg0 == self.input_var and isinstance(call.attrs, op_attrs.QuantizeAttrs):
            input_type: relay.TensorType = arg0.checked_type
            input_shape: Tuple[int, int, int, int] = input_type.shape
            input_dtype: str = call.attrs.out_dtype
            n, c, h, w = input_shape
            if self.nhwc_to_nchw:
                self.input_var = relay.var(
                    name_hint=arg0.name_hint,
                    shape=(n, h, w, c),
                    dtype=input_dtype
                )
                return relay.transpose(self.input_var, (0, 3, 1, 2))
            else:
                self.input_var = relay.var(
                    name_hint=arg0.name_hint,
                    shape=input_shape,
                    dtype=input_dtype
                )
                return self.input_var

        return super().visit_call(call)


@relay.transform.function_pass(opt_level=0, required=["InferType"])
class InsertClassificationPostProcess:
    """A pass that remove qnn.dequantize and insert topk at the
    output of the classification model graph.
    """

    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def transform_function(
        self,
        func: relay.Function,
        mod: tvm.runtime.Module,
        ctx: tvm.transform.PassContext
    ) -> relay.Function:
        scores: relay.Call = func.body
        # Skip dequantize.
        if isinstance(scores.attrs, op_attrs.DequantizeAttrs):
            scores = scores.args[0]

        scores_type: relay.TensorType = scores.checked_type
        scores_dtype: str = scores_type.dtype
        scores_shape: Tuple[int, int] = tuple(scores_type.shape)
        bs, _ = scores_shape

        # Insert topk.
        topk_scores = relay.topk(scores, self.k).tuple_value
        ret_type = relay.TupleType([
            relay.TensorType(shape=(bs, self.k), dtype=scores_dtype),
            relay.TensorType(shape=(bs, self.k), dtype="int32"),
        ])

        return relay.Function(
            params=func.params,
            body=topk_scores,
            ret_type=ret_type,
            type_params=func.type_params,
            attrs=func.attrs
        )


@relay.transform.function_pass(opt_level=0, required=["InferType"])
class FoldQnnBinaryOpExplicitRequantize(relay.ExprMutator):
    """A pass that folds explicit qnn.requantize to qnn binary ops that does not alter dtype. For instance:

      lhs(u8)    rhs(i32)  lhs(u8)   rhs(i32)
       |          |         |         |
       rq(u8)    rq(u8)     |        rq(u8)
       |          |         |         |
        \        /     to    \       /
       qnn.binary_op        qnn.binary_op
             |                    |
             y                    y

    """

    def transform_function(
        self,
        func: relay.Function,
        mod: tvm.runtime.Module,
        ctx: tvm.transform.PassContext
    ) -> relay.Function:
        return self.visit_function(func)

    def visit_call(self, call: relay.Call) -> relay.Expr:
        if isinstance(call.attrs, op_attrs.BroadcastAttrs):
            lhs = call.args[0]
            rhs = call.args[1]
            lhs_scale = call.args[2]
            lhs_zero_point = call.args[3]
            rhs_scale = call.args[4]
            rhs_zero_point = call.args[5]
            output_scale = call.args[6]
            output_zero_point = call.args[7]

            if isinstance(lhs, relay.Call) and \
               isinstance(lhs.attrs, op_attrs.RequantizeAttrs) and \
               lhs.args[0].checked_type.dtype == lhs.attrs.out_dtype:
                lhs_scale = lhs.args[1]
                lhs_zero_point = lhs.args[2]
                lhs = super().visit(lhs.args[0])
            else:
                lhs = super().visit(lhs)

            if isinstance(rhs, relay.Call) and \
               isinstance(rhs.attrs, op_attrs.RequantizeAttrs) and \
               rhs.args[0].checked_type.dtype == rhs.attrs.out_dtype:
                rhs_scale = rhs.args[1]
                rhs_zero_point = rhs.args[2]
                rhs = super().visit(rhs.args[0])
            else:
                rhs = super().visit(rhs)

            return relay.Call(
                op=call.op,
                args=(
                    lhs, rhs,
                    lhs_scale, lhs_zero_point,
                    rhs_scale, rhs_zero_point,
                    output_scale, output_zero_point
                ),
                attrs=call.attrs,
                type_args=call.type_args
            )

        return super().visit_call(call)
