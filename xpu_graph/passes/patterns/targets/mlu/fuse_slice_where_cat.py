from typing import Optional

import torch
from torch import nn, fx
import torch_mlu
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.utils import logger
from xpu_graph.config import OptLevel
from ...utils.check_ops import (
    check_cat_op,
    check_where_op,
    check_zeros_op,
    check_slice_op,
    check_stack_op
)
from .triton_kernel.fused_slice_where_cat import fuse_slice_where_cat


class FusedSliceWhereCatReplacement(nn.Module):
    def forward(
        self,
        where_input,
        slice_input,
        zeros_param,
        slice_params
    ):
        slice_param_tensor = torch.tensor(slice_params, dtype=torch.int32, device=where_input.device)
        slice_num = int(len(slice_params))
        cat_matched = fuse_slice_where_cat(where_input, slice_input, slice_param_tensor, zeros_param[1], slice_num)

        return cat_matched

def flatten_recursive(lst):
    for item in lst:
        if isinstance(item, (list, tuple)):
            yield from flatten_recursive(item)
        else:
            yield item

def get_key(node, is_change_stack):
    if (
        not hasattr(node, "args")
        or len(node.args) != 3
        or (len(node.users) > 1) and not (len(node.users) == 2 and is_change_stack)
        or not check_where_op(node)
    ):
        return None

    cond, x, y = node.args
    if (
        not check_zeros_op(x)
        or not check_slice_op(y)
        or (y.args[1] != 1 and y.args[1] != -1)
    ):
        return None

    if (
        not hasattr(cond, "meta") or "tensor_meta" not in cond.meta
        or not hasattr(y, "meta") or "tensor_meta" not in y.meta
    ):
        return None

    cond_shape = cond.meta["tensor_meta"].shape
    y_shape = y.meta["tensor_meta"].shape
    if len(cond_shape) != 2 or cond_shape[1] != 1 or len(y_shape) != 2:
        return None

    return (cond, x.args[0], y.args[0]), y.args[2]

def group_similar_where_nodes(cat_node, is_change_stack):
    cat_inputs = cat_node.args[0]
    groups = []
    match_group, slice_params = [], []
    unmatched_group = []
    last_key = None

    def flush_group():
        nonlocal match_group, slice_params
        if match_group:
            groups.append((match_group, slice_params))
            match_group, slice_params = [], []

    def flush_unmatched():
        nonlocal unmatched_group
        if unmatched_group:
            groups.append((unmatched_group,))
            unmatched_group = []

    for node in cat_inputs:
        result = get_key(node, is_change_stack)
        if result is None:
            flush_group()
            unmatched_group.append(node)
            continue

        key, param = result
        if key != last_key:
            flush_group()
        flush_unmatched()

        match_group.append(node)
        slice_params.append(param)
        last_key = key

    flush_group()
    flush_unmatched()
    return groups

def _is_slice_where_cat(
    node: fx.Node,
    is_change_stack: bool,
) -> tuple[bool, Optional[fx.Node], Optional[fx.Node], Optional[fx.Node]]:
    is_cat, cat_dim = check_cat_op(node)
    if not is_cat:
        return False, (), ()
    if cat_dim != 1 and cat_dim != -1:
        return False, (), ()

    groups = group_similar_where_nodes(node, is_change_stack)
    group_index = []
    for i in range(len(groups)):
        if len(groups[i]) > 1:
            if len(groups[i][0]) > 1:
                group_index.append(i)
            else:
                groups[i] = groups[i][0]
    if len(group_index) < 1:
        return False, (), ()
    return True, groups, group_index

def _is_stack_to_cat(
    node: fx.Node,
) -> tuple[bool, Optional[fx.Node], Optional[fx.Node], Optional[fx.Node]]:
    is_cat, cat_dim = check_cat_op(node)
    if not is_cat:
        return False, ()
    cat_inputs = node.args[0]
    last_input = cat_inputs[-1]
    stack_inputs = []
    if len(last_input.users) == 1:
        return False, ()
    for key, _ in last_input.users.items():
        if check_stack_op(key) and len(key.args) == 1 and key.args[0] == cat_inputs:
            shape = last_input.meta["tensor_meta"].shape
            input_nums = len(cat_inputs)
            return True, (key, cat_inputs, shape, input_nums)
    return False, ()

class FusedSliceWhereCat(Pattern):
    _opt_level = OptLevel.level2

    def process(self, graph_module: fx.GraphModule) -> bool:
        is_modified = False
        graph_module.add_submodule("mlu_triton_slice_where_cat_replacement", FusedSliceWhereCatReplacement())
        for node in reversed(graph_module.graph.nodes):
            is_change_stack, stack_params = _is_stack_to_cat(node)
            is_match, groups, group_index = _is_slice_where_cat(node, is_change_stack)
            if is_match:
                if is_change_stack:
                    with graph_module.graph.inserting_before(stack_params[0]):
                        reshape_node = graph_module.graph.call_function(
                            torch.ops.aten.view.default,
                            args=(node, [stack_params[2][0], stack_params[3], stack_params[2][1]]),
                            kwargs={},
                        )
                        permute_node = graph_module.graph.call_function(
                            torch.ops.aten.permute.default,
                            args=(reshape_node, [1, 0, 2]),
                            kwargs={},
                        )
                        stack_params[0].replace_all_uses_with(permute_node)
                        graph_module.graph.erase_node(stack_params[0])

                for i in range(len(group_index)):
                    cond, x, y = groups[group_index[i]][0][0].args
                    slice_params = groups[group_index[i]][1]
                    with graph_module.graph.inserting_before(node):
                        new_node = graph_module.graph.call_module(
                            "mlu_triton_slice_where_cat_replacement",
                            args=(
                                cond,
                                y.args[0],
                                x.args[0],
                                slice_params
                            ),
                        )
                        groups[group_index[i]] = new_node
                flattened = list(flatten_recursive(groups))
                with graph_module.graph.inserting_before(node):
                    cat_node = graph_module.graph.call_function(
                        torch.ops.aten.cat.default,
                        args=(flattened, -1),
                    )
                node.replace_all_uses_with(cat_node)
                graph_module.graph.erase_node(node)
                is_modified = True

        graph_module.graph.lint()
        graph_module.recompile()
        return is_modified
