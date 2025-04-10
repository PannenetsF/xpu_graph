from typing import Optional, Union, Tuple

import torch
from torch import nn, fx
import torch.nn.functional as F
from typing import Callable, Optional, List
from xpu_graph.config import OptLevel

from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.utils import logger
from xpu_graph.fx_utils import FxStage

from ..utils.check_ops import (
    check_add_op,
    check_mm_op,
    check_view,
)

def _is_mm_view_add(
    node: fx.Node,
) -> tuple[bool, Optional[fx.Node]]:
    is_add = check_add_op(node)
    if not is_add:
        return False, ()

    view_node = node.args[0]
    bias_node = node.args[1]
    if not check_view(view_node):
        bias_node = node.args[0]
        view_node = node.args[1]
        if not check_view(view_node):
            return False, ()

    mm_node = view_node.args[0]
    view_param = view_node.args[1]
    is_mm, input_node, weight_node = check_mm_op(mm_node)
    if not is_mm:
        return False, ()
    if len(mm_node.users) != 1:
        return False, ()
    if len(view_node.users) != 1:
        return False, ()
    if isinstance(bias_node, float) or isinstance(bias_node, int) or len(bias_node.meta["tensor_meta"].shape) > 2:
        return False, ()

    return True, (bias_node, input_node, weight_node, mm_node, view_node, view_param)

def _is_mm_add(
    node: fx.Node,
) -> tuple[bool, Optional[fx.Node]]:
    if not check_add_op(node):
        return False, ()
    mm_node = node.args[0]
    bias_node = node.args[1]
    # Note: This pattern does not fuse residuals
    is_mm, input_node, weight_node = check_mm_op(mm_node)
    if not is_mm:
        mm_node = node.args[1]
        bias_node = node.args[0]
        is_mm, input_node, weight_node = check_mm_op(mm_node)
        if not is_mm:
            return False, ()
    if len(mm_node.users) != 1:
        return False, ()
    if isinstance(bias_node, float) or isinstance(bias_node, int):
        return False, ()

    return True, (bias_node, input_node, weight_node)

class FusedAddMM(Pattern):
    _opt_level = OptLevel.level2
    _stages = [FxStage.forward, FxStage.pregrad]

    def process(self, graph_module: fx.GraphModule) -> bool:
        changed = False

        for node in reversed(graph_module.graph.nodes):
            is_mm_add, mm_inputs = _is_mm_add(node)
            if is_mm_add:
                with graph_module.graph.inserting_before(node):
                    addmm_node = graph_module.graph.create_node(
                        op="call_function",
                        target=torch.ops.aten.addmm.default,
                        args=(mm_inputs),
                        name=node.name + "_replacement",
                    )
                node.replace_all_uses_with(addmm_node)
                graph_module.graph.erase_node(node)
                changed = True

            is_mm_view_add, mm_inputs = _is_mm_view_add(node)
            if is_mm_view_add:
                with graph_module.graph.inserting_before(node):
                    addmm_node = graph_module.graph.create_node(
                        op="call_function",
                        target=torch.ops.aten.addmm.default,
                        args=(mm_inputs[:-3]),
                        name=node.name + "_replacement",
                    )
                    unsafe_view_node = graph_module.graph.create_node(
                        op="call_function",
                        target=mm_inputs[-2].target,
                        args=(addmm_node, mm_inputs[-1]),
                        name=node.name + "_view_replacement",
                    )
                node.replace_all_uses_with(unsafe_view_node)
                graph_module.graph.erase_node(node)
                graph_module.graph.erase_node(mm_inputs[-2])
                graph_module.graph.erase_node(mm_inputs[-3])

                changed = True

        graph_module.graph.lint()
        graph_module.recompile()
        return changed
