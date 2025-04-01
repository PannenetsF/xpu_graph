from typing import Optional, Tuple, List
import torch
import sys
from .fused_slice import mlu_triton_slice_low_kernel


# Causal Conv1D Forward Function
@torch.library.custom_op(
    "torch_mlu_triton::fused_split_fwd",
    mutates_args=(),
    device_types="mlu",
)
def fused_split_fwd(
    x: torch.Tensor,
    split_size: int,
    dim: int,
) -> torch.Tensor:
    n_rows = x.shape[0]
    input_stride = x.stride(0)
    slice_len = split_size
    block_size_r = n_rows
    block_size_c = slice_len
    size_of_dtype = 2
    if x.dtype == torch.float32:
        size_of_dtype = 4
    nram_limit = 384 * 1024
    if block_size_r * block_size_c * size_of_dtype > nram_limit:
        block_size_r = nram_limit // size_of_dtype // block_size_c

    num_slices = len(start_indices)
    output_tensors = torch.empty(
        (num_slices * x.shape[0], slice_len),
        device=x.device,
        dtype=x.dtype,
        requires_grad=x.requires_grad,
    )
    processor_count = torch.mlu.get_device_properties(
        torch.mlu.current_device()
    ).multi_processor_count
    grid = (processor_count, 1, 1)
    mlu_triton_slice_low_kernel[grid](
        x,
        output_tensors,
        start_indices,
        num_slices,
        slice_len,
        n_rows,
        input_stride,
        block_size_r,
        block_size_c,
    )

    return output_tensors.view(num_slices, x.shape[0], slice_len).unbind(0)


def fused_split(x: torch.Tensor, split_size: int, dim: int = -1):
    return FusedSplitFunction.apply(x, split_size, dim)
