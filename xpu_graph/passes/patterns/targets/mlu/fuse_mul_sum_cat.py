from typing import Optional
import torch
from torch import nn, fx
import torch_mlu
from xpu_graph.config import OptLevel
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.utils import logger
from ...utils.check_ops import (
    check_cat_op,
    check_sum_op,
    check_meta_2d,
    check_mul_op,
)
from torch.fx.subgraph_rewriter import replace_pattern
from xpu_graph.fx_utils import FxStage

from .triton_kernel.fused_mul_sum_cat import fused_mul_sum_cat_2inp

class FusedMulSumCat2InpReplacement(nn.Module):
    def forward(
        self,
        mul1,
        mul2,
        mul3,
        mul4
    ):
        mul1, mul2, mul3, mul4 = [x.contiguous() if x is not None and not x.is_contiguous() else x
            for x in [mul1, mul2, mul3, mul4]]
        output = fused_mul_sum_cat_2inp(mul1, mul2, mul3, mul4)
        return output

def find_mul_sum_pattern(gm):
    # Dictionary to store mapping from source nodes to sum nodes
    sum_dict = {}
    candidates = [
        node
        for node in gm.graph.nodes
        if node.op == "call_function"
        and node.target == torch.ops.aten.sum.dim_IntList
    ]
    for node in candidates:
        ### Identify sum operation: input 3D, output 2D ###
        if not check_meta_2d(node):
            continue
        # check sum dim
        if node.args[1] != [1]:
            continue
        # don't keep dim
        if len(node.args) > 2:
            continue
        if len(node.users) != 1:
            continue

        mul_node = node.args[0]
        if not check_mul_op(mul_node):
            continue
        if len(mul_node.users) != 1:
            continue

        if node not in sum_dict:
            sum_dict[node] = [mul_node.args[0], mul_node.args[1]]
    return sum_dict

class FusedMulSumCat(Pattern):
    _opt_level = OptLevel.level2

    def process(self, graph_module: fx.GraphModule) -> bool:
        changed = False
        graph_module.add_submodule("mlu_triton_mul_sum_cat_replacement", FusedMulSumCat2InpReplacement())

        for node in graph_module.graph.nodes:
            is_cat, cat_axis = check_cat_op(node)
            if not is_cat:
                continue
            sum_dict = find_mul_sum_pattern(graph_module)
            if (len(node.args[0]) == 2) and (node.args[0][0] in sum_dict) and (node.args[0][1]) in sum_dict:
                with graph_module.graph.inserting_before(node):
                    new_node = graph_module.graph.call_module(
                        "mlu_triton_mul_sum_cat_replacement",
                        args=(tuple(sum_dict[node.args[0][0]] + sum_dict[node.args[0][1]])),
                    )
                node.replace_all_uses_with(new_node)
                graph_module.graph.erase_node(node)
                changed = True
            else:
                continue

        graph_module.graph.lint()
        graph_module.recompile()
        return changed
