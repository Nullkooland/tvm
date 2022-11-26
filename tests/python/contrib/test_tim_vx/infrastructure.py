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
"""Infrastructure for TIM-VX tests."""

from tvm.relay.op.contrib.tim_vx import partition_for_tim_vx
from tvm.contrib import graph_executor, utils
from tvm.relay.backend.executor_factory import ExecutorFactoryModule
from tvm.target import Target
import tvm
from tvm import relay, rpc
import tvm.testing
import pytest
import os
from typing import Tuple, List, Dict
ValueRange = Tuple[int, int] | Tuple[float, float]


def get_num_cpu_ops(ir_mod: tvm.IRModule) -> int:
    """Get number of ops offloaded to TVM TIR."""

    class CpuOpCounter(relay.ExprVisitor):
        def __init__(self):
            super().__init__()
            self.num_cpu_ops = 0

        def visit_call(self, call):
            if isinstance(call.op, tvm.ir.Op):
                self.num_cpu_ops += 1

            super().visit_call(call)

    counter = CpuOpCounter()
    counter.visit(ir_mod["main"])
    return counter.num_cpu_ops


def get_num_tim_vx_subgraphs(ir_mod: tvm.IRModule) -> int:
    """Get number of subgraphs partitioned for TIM-VX."""
    num_tim_vx_subgraphs = 0
    for global_var in ir_mod.get_global_vars():
        if "tim_vx" in global_var.name_hint:
            num_tim_vx_subgraphs += 1
    return num_tim_vx_subgraphs


def build_module(
    expr_or_ir_mod: relay.Call | relay.Tuple | tvm.IRModule,
    params: Dict[str, tvm.nd.NDArray],
    target: Target,
    build_for_tim_vx: bool = True,
    expected_num_cpu_ops: int = 0,
    expected_num_tim_vx_subgraphs: int = 1
) -> ExecutorFactoryModule:
    """Build module with option to build for TIM-VX."""
    if isinstance(expr_or_ir_mod, relay.Call) or isinstance(expr_or_ir_mod, relay.Tuple):
        # Get Relay IRModule from Expr.
        vars = relay.analysis.free_vars(expr_or_ir_mod)
        func = relay.Function(vars, expr_or_ir_mod)
        if params:
            relay.build_module.bind_params_by_name(func, params)
        ir_mod = tvm.IRModule.from_expr(func)
    else:
        ir_mod = expr_or_ir_mod

    npu_target = os.environ.get("VSIMULATOR_CONFIG", "")
    with tvm.transform.PassContext(
        opt_level=3,
        config={
            "relay.ext.tim_vx.options": {
                "npu_target": npu_target,
                "relax_mode": True,
                "debug_level": 1,
            }
        }
    ):
        if build_for_tim_vx:
            ir_mod = partition_for_tim_vx(ir_mod, params)

            num_cpu_ops = get_num_cpu_ops(ir_mod)
            assert num_cpu_ops == expected_num_cpu_ops, f"Number of ops offloaded to CPU is not as expected (expected: {expected_num_cpu_ops}, actual: {num_cpu_ops})."

            num_tim_vx_subgraphs = get_num_tim_vx_subgraphs(ir_mod)
            assert num_tim_vx_subgraphs == expected_num_tim_vx_subgraphs, f"Number of subgraphs partitioned for TIM-VX backend is not as expected ( expected: {expected_num_tim_vx_subgraphs}, actual: {num_tim_vx_subgraphs})."

        return relay.build(ir_mod, target, params=params)


def build_and_run(
    expr_or_ir_mod: relay.Call | relay.Tuple | tvm.IRModule,
    inputs: Dict[str, tvm.nd.NDArray],
    params: Dict[str, tvm.nd.NDArray] = {},
    build_for_tim_vx: bool = False,
    expected_num_cpu_ops: int = 0,
    expected_num_tim_vx_subgraphs: int = 1,
    remote: rpc.RPCSession = rpc.LocalSession(),
) -> List[tvm.nd.NDArray]:
    if isinstance(remote, rpc.LocalSession) or not build_for_tim_vx:
        target = tvm.target.Target("llvm")
    else:
        target = tvm.target.Target(os.environ.get("TARGET"))

    lib = build_module(
        expr_or_ir_mod,
        params,
        target,
        build_for_tim_vx,
        expected_num_cpu_ops,
        expected_num_tim_vx_subgraphs
    )

    if build_for_tim_vx:
        tmp_path = utils.tempdir()
        lib_name = "mod.so"
        lib_path = tmp_path.relpath(lib_name)

        if isinstance(remote, rpc.LocalSession):
            lib.export_library(lib_path)  # type: ignore
        else:
            sysroot = os.environ.get("SYSROOT")
            cross_cc = os.environ.get("CROSS_CC")
            cross_cc_flags = os.environ.get("CROSS_CC_FLAGS")
            cc_flags = cross_cc_flags.split() if cross_cc_flags else []
            options = [
                f"--sysroot={sysroot}",
                *cc_flags,
            ]
            lib.export_library(lib_path, cc=cross_cc, options=options)  # type: ignore

        remote.upload(lib_path)
        lib = remote.load_module(lib_name)
        device = remote.cpu()
    else:
        device = tvm.cpu()

    rt_mod = graph_executor.GraphModule(lib["default"](device))
    num_outputs = rt_mod.get_num_outputs()

    rt_mod.run(**inputs)
    outputs = [rt_mod.get_output(i) for i in range(num_outputs)]
    return outputs


def verify(
    tim_vx_outputs: List[tvm.nd.NDArray],
    ref_outputs: List[tvm.nd.NDArray],
    atol: float = 1e-7,
    rtol: float = 1e-7,
):
    """Compare inference outputs by TIM-VX runtime and reference (CPU)."""
    for tim_vx_output, ref_output in zip(tim_vx_outputs, ref_outputs):
        tvm.testing.assert_allclose(
            actual=tim_vx_output.numpy(),
            desired=ref_output.numpy(),
            rtol=rtol,
            atol=atol
        )
