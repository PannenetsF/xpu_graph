import torch
import torch_mlu
import triton
import triton.language as tl
from typing import List

@triton.jit
def single_mul_sum_cat(
    mul0,
    mul1,
    output,
    mul_stride0,
    mul_stride1,
    output_stride,
    input0_row,
    input1_row,
    slice_len,
    row_start_idx,
    output_start_offset,
    is_input0_multi_dim: tl.constexpr,
    is_input1_multi_dim: tl.constexpr,
    BLOCK_SIZE_R: tl.constexpr = 16,
    BLOCK_SIZE_C: tl.constexpr = 16,
    BLOCK_SIZE_S1: tl.constexpr = 128,
    BLOCK_SIZE_S2: tl.constexpr = 128,
):
    if is_input0_multi_dim:
        input_block_ptr0 = tl.make_block_ptr(
            base=mul0,
            shape=(input0_row, slice_len),
            strides=(mul_stride0, 1),
            offsets=(row_start_idx, 0),
            block_shape=(BLOCK_SIZE_R, BLOCK_SIZE_C),
            order=(1, 0),
        )
    else:
        input_block_ptr0 = tl.make_block_ptr(
            base=mul0,
            shape=(input0_row, slice_len),
            strides=(mul_stride0, 1),
            offsets=(0, 0),
            block_shape=(1, BLOCK_SIZE_C),
            order=(1, 0),
        )
    if is_input1_multi_dim:
        input_block_ptr1 = tl.make_block_ptr(
            base=mul1,
            shape=(input1_row, slice_len),
            strides=(mul_stride1, 1),
            offsets=(row_start_idx, 0),
            block_shape=(BLOCK_SIZE_R, BLOCK_SIZE_C),
            order=(1, 0),
        )
    else:
        input_block_ptr1 = tl.make_block_ptr(
            base=mul1,
            shape=(input1_row, slice_len),
            strides=(mul_stride1, 1),
            offsets=(0, 0),
            block_shape=(1, BLOCK_SIZE_C),
            order=(1, 0),
        )

    value0 = tl.load(input_block_ptr0, boundary_check=(0,), padding_option=0)
    value1 = tl.load(input_block_ptr1, boundary_check=(0,), padding_option=0)
    value0 = value0 * value1
    value0 = tl.reshape(value0, [BLOCK_SIZE_R, BLOCK_SIZE_S1, BLOCK_SIZE_S2])
    value0 = tl.sum(value0, axis=1)
    output_block_ptr = tl.make_block_ptr(
        base=output,
        shape=(input1_row, BLOCK_SIZE_S2 * 2),
        strides=(output_stride, 1),
        offsets=(row_start_idx, output_start_offset),
        block_shape=(BLOCK_SIZE_R, BLOCK_SIZE_S2),
        order=(1, 0),
    )
    tl.store(output_block_ptr, value0, boundary_check=(0,))

@triton.jit
def mlu_triton_mul_sum_cat_kernel(
    mul0,
    mul1,
    mul2,
    mul3,
    output,
    mul_stride0,
    mul_stride1,
    mul_stride2,
    mul_stride3,
    output_stride,
    total_jobs,
    input0_row,
    input1_row,
    input2_row,
    input3_row,
    slice_len,
    is_input0_multi_dim: tl.constexpr,
    is_input1_multi_dim: tl.constexpr,
    is_input2_multi_dim: tl.constexpr,
    is_input3_multi_dim: tl.constexpr,
    BLOCK_SIZE_R: tl.constexpr = 16,
    BLOCK_SIZE_C: tl.constexpr = 16,
    BLOCK_SIZE_S1: tl.constexpr = 128,
    BLOCK_SIZE_S2: tl.constexpr = 128,
):
    program_dim = tl.num_programs(axis=0)
    program_id = tl.program_id(0)
    block_jobs = total_jobs // program_dim
    remainder = total_jobs % program_dim
    # by row(batch)
    if program_id < remainder:
        block_jobs_r = block_jobs + 1
        offset = program_id * (block_jobs + 1)
    else:
        block_jobs_r = block_jobs
        offset = remainder * (block_jobs + 1) + (program_id - remainder) * block_jobs

    loop = (block_jobs_r + BLOCK_SIZE_R - 1) // BLOCK_SIZE_R

    for l in range(loop):
        row_start_idx = offset + l * BLOCK_SIZE_R
        single_mul_sum_cat(
            mul0,
            mul1,
            output,
            mul_stride0,
            mul_stride1,
            output_stride,
            input0_row,
            input1_row,
            slice_len,
            row_start_idx,
            0,
            is_input0_multi_dim,
            is_input1_multi_dim,
            BLOCK_SIZE_R,
            BLOCK_SIZE_C,
            BLOCK_SIZE_S1,
            BLOCK_SIZE_S2,
        )
        single_mul_sum_cat(
            mul2,
            mul3,
            output,
            mul_stride0,
            mul_stride1,
            output_stride,
            input2_row,
            input3_row,
            slice_len,
            row_start_idx,
            BLOCK_SIZE_S2,
            is_input2_multi_dim,
            is_input3_multi_dim,
            BLOCK_SIZE_R,
            BLOCK_SIZE_C,
            BLOCK_SIZE_S1,
            BLOCK_SIZE_S2,
        )


@torch.library.custom_op("torch_mlu_triton::fused_mul_sum_cat", mutates_args=())
def fused_mul_sum_cat_2inp(
    mul1: torch.Tensor,
    mul2: torch.Tensor,
    mul3: torch.Tensor,
    mul4: torch.Tensor,
) -> torch.Tensor:
    src_tensor = mul1
    mul1_s0, mul1_s1, mul1_s2 = mul1.shape
    mul2_s0, mul2_s1, mul2_s2 = mul2.shape
    mul3_s0, mul3_s1, mul3_s2 = mul3.shape
    mul4_s0, mul4_s1, mul4_s2 = mul4.shape

    s0 = mul1_s0
    s1 = mul1_s1
    s2 = mul1_s2
    block_size_r = mul2_s0 if s0 == 1 else s0
    block_size_c = s1 * s2
    size_of_dtype = 2
    if src_tensor.dtype == torch.float32:
        size_of_dtype = 4
    nram_limit = 384 * 1024
    if block_size_r * block_size_c * size_of_dtype * 4 > nram_limit:
        block_size_r = nram_limit // (size_of_dtype * block_size_c * 4)
    #block_size_r 3 128 128 16384

    input_row = mul2.shape[0] if s0 == 1 else s0
    output_tensors = torch.zeros(
        (input_row, s2 * 2),
        device=src_tensor.device,
        dtype=src_tensor.dtype,
    )

    total_jobs = mul2.shape[0] if s0 == 1 else s0
    processor_count = torch.mlu.get_device_properties(
        torch.mlu.current_device()
    ).multi_processor_count
    grid = (processor_count, 1, 1)
    mlu_triton_mul_sum_cat_kernel[grid](
        mul1.view(mul1.shape[0], -1),
        mul2.view(mul2.shape[0], -1),
        mul3.view(mul3.shape[0], -1),
        mul4.view(mul4.shape[0], -1),
        output_tensors,
        mul1.stride(0),
        mul2.stride(0),
        mul3.stride(0),
        mul4.stride(0),
        output_tensors.stride(0),
        total_jobs,
        mul1_s0,
        mul2_s0,
        mul3_s0,
        mul4_s0,
        s1 * s2,  # slice_len,
        1 if mul1_s0 > 1 else 0,
        1 if mul2_s0 > 1 else 0,
        1 if mul3_s0 > 1 else 0,
        1 if mul4_s0 > 1 else 0,
        block_size_r,
        s1 * s2,
        s1,
        s2,
    )

    return output_tensors

@fused_mul_sum_cat_2inp.register_fake
def fused_mul_sum_cat_2inp_fake(
    mul1: torch.Tensor,
    mul2: torch.Tensor,
    mul3: torch.Tensor,
    mul4: torch.Tensor,
    ) -> torch.Tensor:
    s0, s1, s2 = mul1.shape
    input_row = mul2.shape[0] if s0 == 1 else s0
    output_tensors = torch.zeros(
        (input_row, s2 * 2),
        device=mul1.device,
        dtype=mul1.dtype,
    )
    return output_tensors

