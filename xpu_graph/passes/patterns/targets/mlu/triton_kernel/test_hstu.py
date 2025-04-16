import pytest
import torch
import torch_mlu

import triton
import numpy as np
import triton.language as tl
from typing import Sequence
from typing import Union
from triton.backends.mlu.driver import BangDriver

from triton.language.extra.mlu.libdevice import ultra_silu_mul_float2half
from triton.language.extra.mlu.libdevice import ultra_silu_mul_float2bfloat16
from triton.language.extra.mlu.libdevice import ultra_silubp


def rand_strided(
    size: Sequence[int],
    stride: Sequence[int],
    dtype: torch.dtype = torch.float32,
    device: Union[str, torch.device] = "cpu",
    extra_size: int = 0,
):
    needed_size = (sum((shape - 1) * stride
                       for shape, stride in zip(size, stride)) + extra_size)
    assert dtype == torch.float32 or dtype == torch.float16 or dtype == torch.bfloat16
    buffer = torch.empty(size=[needed_size], dtype=dtype,
                         device=device).normal_(mean=0.0, std=0.1)
    return torch.as_strided(buffer, size, stride)


def is_divisible(a, b):
    if b == 0:
        raise ValueError("Divisor cannot be 0")
    return a % b == 0


@triton.jit
def silu_forward_precision(x, N_CTX: tl.constexpr, IS_Y: tl.constexpr):
    log2e = -1.442695041
    y = None
    if IS_Y:
        # For y, use tanh
        y = 0.5 * x
        t = tl.tanh(y)
        y = (t + 1) * y
    else:
        # No instruction fusion
        y = 1.0 / (1.0 + tl.exp2(x * log2e)) * x * (1.0 / N_CTX)
    return y


@triton.jit
def silu_forward_perf(x, TYPE, N_CTX: tl.constexpr, IS_Y: tl.constexpr):
    y = None
    if TYPE == tl.float16:
        y = ultra_silu_mul_float2half(x, (1.0 / N_CTX))
    elif TYPE == tl.bfloat16:
        y = ultra_silu_mul_float2bfloat16(x, (1.0 / N_CTX))
    else:
        # Fallback to precision implementation
        y = silu_forward_precision(x, N_CTX, IS_Y)
    return y


@triton.jit
def silu_forward(x, TYPE, PERF_OPT: tl.constexpr, N_CTX: tl.constexpr,
                 IS_Y: tl.constexpr):
    if PERF_OPT == 0:
        return silu_forward_perf(x, TYPE, N_CTX, IS_Y)
    else:
        return silu_forward_precision(x, N_CTX, IS_Y)  #


@triton.jit
def silu_backward_precision(x, N_CTX: tl.constexpr, IS_Y: tl.constexpr):
    y = None
    if IS_Y:
        y = 0.5 * x
        t = tl.tanh(y)
        sig = (1 + t) * 0.5
        y = sig * (1 + x * (1 - sig)) * (1.0 / N_CTX)
    else:
        log2e = -1.442695041
        sig = 1.0 / (1.0 + tl.exp2(x * log2e))
        y = sig * (1 + x * (1 - sig)) * (1.0 / N_CTX)
    return y


@triton.jit
def silu_backward_perf(x, TYPE, N_CTX: tl.constexpr, IS_Y: tl.constexpr):
    y = None
    if TYPE == tl.float32:
        # Fallback to precision implementation
        y = silu_backward_precision(x, N_CTX, IS_Y)
    else:
        y = ultra_silubp(x) * (1.0 / N_CTX)
    return y


@triton.jit
def silu_backward(x, TYPE, PERF_OPT: tl.constexpr, N_CTX: tl.constexpr,
                  IS_Y: tl.constexpr):
    if PERF_OPT == 0:
        return silu_backward_perf(x, TYPE, N_CTX, IS_Y)
    else:
        return silu_backward_precision(x, N_CTX, IS_Y)


@triton.jit
def _attn_fwd_inner(off_z, acc, q, BIAS, K_block_ptr, V_block_ptr, start_m,
                    bias_stride_z, bias_stride_m, bias_stride_n,
                    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
                    BLOCK_N: tl.constexpr, CAUSAL: tl.constexpr,
                    offs_m: tl.constexpr, offs_n: tl.constexpr,
                    N_CTX: tl.constexpr, IS_DIVISIBLE: tl.constexpr,
                    HAS_BIAS: tl.constexpr, EXPANDED_BIAS: tl.constexpr,
                    PERF_OPT: tl.constexpr, IS_Y: tl.constexpr):
    # range of values handled by this stage
    if CAUSAL:
        lo, hi = 0, (start_m + 1) * BLOCK_M
    else:
        lo, hi = 0, N_CTX
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        if IS_DIVISIBLE:
            k = tl.load(K_block_ptr)
        else:
            k = tl.load(K_block_ptr,
                        boundary_check=(0, 1),
                        padding_option="zero")
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if HAS_BIAS:
            bias = None
            if EXPANDED_BIAS:
                if IS_DIVISIBLE:
                    bias = tl.load(BIAS + off_z * bias_stride_z +
                                   offs_m[:, None] * bias_stride_m +
                                   (start_n + offs_n) * bias_stride_n)
                else:
                    bias = tl.load(BIAS + off_z * bias_stride_z +
                                   offs_m[:, None] * bias_stride_m +
                                   (start_n + offs_n) * bias_stride_n,
                                   mask=(offs_m < N_CTX)[:, None] &
                                   (start_n + offs_n < N_CTX)[None, :],
                                   other=0.)
            else:
                if IS_DIVISIBLE:
                    bias = tl.load(BIAS + off_z * bias_stride_z +
                                   (start_n + offs_n) * bias_stride_n)[None, :]
                else:
                    bias = tl.load(BIAS + off_z * bias_stride_z +
                                   (start_n + offs_n) * bias_stride_n,
                                   start_n + offs_n < N_CTX,
                                   other=0.)[None, :]
            qk += tl.dot(q, k)
            qk = qk + bias
        else:
            qk += tl.dot(q, k)
        if CAUSAL and start_n >= start_m * BLOCK_M:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk + tl.where(mask, 0, -1.0e6)
        p = silu_forward(qk, q.type.element_ty, PERF_OPT, N_CTX, IS_Y)
        if IS_DIVISIBLE:
            v = tl.load(V_block_ptr)
        else:
            v = tl.load(V_block_ptr,
                        boundary_check=(0, 1),
                        padding_option="zero")
        qk = p.to(v.type.element_ty)
        qkv = tl.dot(qk, v)
        # -- update output accumulator --
        acc += qkv

        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))

    return acc


@triton.autotune(configs=[
    triton.Config({
        'BLOCK_M': m,
        'BLOCK_N': n
    }, num_stages=s, num_warps=1) for m in [128] for n in [64, 128]
    for s in [3]
],
                 key=[
                     'N_CTX', 'BLOCK_DMODEL', 'PERF_OPT', 'HAS_BIAS', 'CAUSAL',
                     'EXPANDED_BIAS'
                 ])
@triton.heuristics({
    'IS_DIVISIBLE':
    lambda args: args['N_CTX'] % args['BLOCK_M'] == 0 and args['N_CTX'] % args[
        'BLOCK_N'] == 0,
})
@triton.jit
def _attn_fwd(Q, K, V, BIAS, Out, stride_qz, stride_qh, stride_qm, stride_qk,
              stride_kz, stride_kh, stride_kn, stride_kk, stride_vz, stride_vh,
              stride_vk, stride_vn, stride_oz, stride_oh, stride_om, stride_on,
              bias_stride_z, bias_stride_m, bias_stride_n, Z, H,
              N_CTX: tl.constexpr, BLOCK_M: tl.constexpr,
              BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
              CAUSAL: tl.constexpr, IS_DIVISIBLE: tl.constexpr,
              HAS_BIAS: tl.constexpr, EXPANDED_BIAS: tl.constexpr,
              PERF_OPT: tl.constexpr, IS_Y: tl.constexpr):
    program_id = tl.program_id(0)
    program_dim = tl.num_programs(axis=0)
    context_num = triton.cdiv(N_CTX, BLOCK_M)
    total_heads = Z * H * context_num
    deal_num_per_core = total_heads // program_dim
    extra_deal_core_num = total_heads - deal_num_per_core * program_dim
    deal_num_per_core += 1
    core_head_begin = program_id * deal_num_per_core
    if program_id >= extra_deal_core_num:
        deal_num_per_core -= 1
        core_head_begin = program_id * deal_num_per_core + extra_deal_core_num
    if deal_num_per_core <= 0:
        return
    head_begin = core_head_begin
    head_end = head_begin + deal_num_per_core

    for head_idx in range(head_begin, head_end):
        start_m = head_idx % context_num
        off_hz = head_idx // context_num
        off_z = off_hz // H
        off_h = off_hz % H
        q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(
            tl.int64) * stride_qh
        v_offset = off_z.to(tl.int64) * stride_vz + off_h.to(
            tl.int64) * stride_vh
        k_offset = off_z.to(tl.int64) * stride_kz + off_h.to(
            tl.int64) * stride_kh
        o_offset = off_z.to(tl.int64) * stride_oz + off_h.to(
            tl.int64) * stride_oh
        Q_block_ptr = tl.make_block_ptr(
            base=Q + q_offset,
            shape=(N_CTX, BLOCK_DMODEL),
            strides=(stride_qm, stride_qk),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_DMODEL),
            order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            base=V + v_offset,
            shape=(N_CTX, BLOCK_DMODEL),
            strides=(stride_vk, stride_vn),
            offsets=(0, 0),
            block_shape=(BLOCK_N, BLOCK_DMODEL),
            order=(1, 0),
        )
        K_block_ptr = tl.make_block_ptr(
            base=K + k_offset,
            shape=(BLOCK_DMODEL, N_CTX),
            strides=(stride_kk, stride_kn),
            offsets=(0, 0),
            block_shape=(BLOCK_DMODEL, BLOCK_N),
            order=(0, 1),
        )
        O_block_ptr = tl.make_block_ptr(
            base=Out + o_offset,
            shape=(N_CTX, BLOCK_DMODEL),
            strides=(stride_om, stride_on),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_DMODEL),
            order=(1, 0),
        )
        # initialize offsets
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        # load q: it will stay in SRAM throughout
        if IS_DIVISIBLE:
            q = tl.load(Q_block_ptr)
        else:
            q = tl.load(Q_block_ptr,
                        boundary_check=(0, 1),
                        padding_option="zero")
        acc = _attn_fwd_inner(
            off_z,
            acc,
            q,
            BIAS,
            K_block_ptr,
            V_block_ptr,  #
            start_m,
            bias_stride_z,
            bias_stride_m,
            bias_stride_n,
            BLOCK_M,
            BLOCK_DMODEL,
            BLOCK_N,  #
            CAUSAL,
            offs_m,
            offs_n,
            N_CTX,  #
            IS_DIVISIBLE,
            HAS_BIAS,
            EXPANDED_BIAS,
            PERF_OPT,
            IS_Y)

        if IS_DIVISIBLE:
            tl.store(O_block_ptr, acc.to(Out.type.element_ty))
        else:
            tl.store(O_block_ptr,
                     acc.to(Out.type.element_ty),
                     boundary_check=(0, 1))


@triton.autotune(configs=[
    triton.Config({
        'BLOCK_M': m,
        'BLOCK_N': n
    }, num_stages=s, num_warps=1) for m in [64, 128] for n in [64, 128]
    if m >= n for s in [0, 4]
],
                 key=[
                     'N_CTX', 'BLOCK_DMODEL', 'PERF_OPT', 'HAS_BIAS', 'CAUSAL',
                     'EXPANDED_BIAS'
                 ],
                 reset_to_zero=["DQ_ptr"])
@triton.heuristics({
    'IS_DIVISIBLE':
    lambda args: args['N_CTX'] % args['BLOCK_M'] == 0 and args['N_CTX'] % args[
        'BLOCK_N'] == 0,
})
@triton.jit
def _attn_bwd(Q_ptr, K_ptr, V_ptr, DO_ptr, DQ_ptr, DK_ptr, DV_ptr, BIAS,
              stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh,
              stride_kn, stride_kk, stride_vz, stride_vh, stride_vn, stride_vk,
              stride_doz, stride_doh, stride_dom, stride_dok, stride_dqz,
              stride_dqh, stride_dqm, stride_dqk, stride_dkz, stride_dkh,
              stride_dkn, stride_dkk, stride_dvz, stride_dvh, stride_dvn,
              stride_dvk, bias_stride_z, bias_stride_m, bias_stride_n, Z, H,
              N_CTX: tl.constexpr, BLOCK_M: tl.constexpr,
              BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
              CAUSAL: tl.constexpr, IS_DIVISIBLE: tl.constexpr,
              HAS_BIAS: tl.constexpr, EXPANDED_BIAS: tl.constexpr,
              PERF_OPT: tl.constexpr, IS_Y: tl.constexpr):
    tl.static_assert(BLOCK_M >= BLOCK_N)
    program_id = tl.program_id(0)
    program_dim = tl.num_programs(axis=0)
    context_num = triton.cdiv(N_CTX, BLOCK_M)
    total_heads = Z * H * context_num
    deal_num_per_core = total_heads // program_dim
    extra_deal_core_num = total_heads - deal_num_per_core * program_dim
    deal_num_per_core += 1
    core_head_begin = program_id * deal_num_per_core
    if program_id >= extra_deal_core_num:
        deal_num_per_core -= 1
        core_head_begin = program_id * deal_num_per_core + extra_deal_core_num
    if deal_num_per_core <= 0:
        return
    head_begin = core_head_begin
    head_end = head_begin + deal_num_per_core
    for head_idx in range(head_begin, head_end):
        start_m = head_idx % context_num
        off_hz = head_idx // context_num
        off_z = off_hz // H
        off_h = off_hz % H

        Q = Q_ptr + off_z * stride_qz + off_h * stride_qh
        K = K_ptr + off_z * stride_kz + off_h * stride_kh
        V = V_ptr + off_z * stride_vz + off_h * stride_vh
        DO = DO_ptr + off_z * stride_doz + off_h * stride_doh
        DQ = DQ_ptr + off_z * stride_dqz + off_h * stride_dqh
        DK = DK_ptr + off_z * stride_dkz + off_h * stride_dkh
        DV = DV_ptr + off_z * stride_dvz + off_h * stride_dvh

        if CAUSAL:
            lo = start_m * BLOCK_M
        else:
            lo = 0
        offs_q = lo + tl.arange(0, BLOCK_N)
        offs_kv = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_DMODEL)
        # initialize pointers to value-like data
        q_ptrs = Q + (offs_q[:, None] * stride_qm +
                      offs_k[None, :] * stride_qk)
        k_ptrs = K + (offs_kv[:, None] * stride_kn +
                      offs_k[None, :] * stride_kk)
        v_ptrs = V + (offs_kv[:, None] * stride_vn +
                      offs_k[None, :] * stride_vk)
        do_ptrs = DO + (offs_q[:, None] * stride_dom +
                        offs_k[None, :] * stride_dok)
        dq_ptrs = DQ + (offs_q[:, None] * stride_dqm +
                        offs_k[None, :] * stride_dqk)

        dv = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        dk = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        if IS_DIVISIBLE:
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
        else:
            k = tl.load(k_ptrs, mask=(offs_kv < N_CTX)[:, None], other=0.)
            v = tl.load(v_ptrs, mask=(offs_kv < N_CTX)[:, None], other=0.)
        # loop over rows
        for start_n in range(lo, context_num * BLOCK_M, BLOCK_N):
            curr_offs = start_n + tl.arange(0, BLOCK_N)
            # load q, k, v, do on-chip
            if IS_DIVISIBLE:
                q = tl.load(q_ptrs)
            else:
                q = tl.load(q_ptrs,
                            mask=(curr_offs < N_CTX)[:, None],
                            other=0.)
            qk_trans = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            if HAS_BIAS:
                bias = None
                if EXPANDED_BIAS:
                    if IS_DIVISIBLE:
                        bias = tl.load(BIAS + off_z * bias_stride_z +
                                       curr_offs[:, None] * bias_stride_m +
                                       offs_kv * bias_stride_n)
                    else:
                        bias = tl.load(BIAS + off_z * bias_stride_z +
                                       curr_offs[:, None] * bias_stride_m +
                                       offs_kv * bias_stride_n,
                                       mask=(curr_offs < N_CTX)[:, None] &
                                       (offs_kv < N_CTX)[None, :],
                                       other=0.)
                    bias = tl.trans(bias)
                else:
                    if IS_DIVISIBLE:
                        bias = tl.load(BIAS + off_z * bias_stride_z +
                                       offs_kv * bias_stride_n)[:, None]
                    else:
                        bias = tl.load(BIAS + off_z * bias_stride_z +
                                       offs_kv * bias_stride_n,
                                       offs_kv < N_CTX,
                                       other=0.)[:, None]
                qk_trans += tl.dot(k, tl.trans(q))
                qk_trans += bias
            else:
                qk_trans += tl.dot(k, tl.trans(q))

            silu_trans = silu_forward(qk_trans, q.type.element_ty, PERF_OPT,
                                      N_CTX, IS_Y)
            silu_trans = silu_trans.to(Q.dtype.element_ty)
            zero = float("0.")
            zero = zero.to(Q.dtype.element_ty)
            if CAUSAL:
                silu_trans = tl.where(curr_offs[None, :] >= (offs_kv[:, None]),
                                      silu_trans, zero)

            # compute dv
            if IS_DIVISIBLE:
                do = tl.load(do_ptrs)
            else:
                do = tl.load(do_ptrs,
                             mask=(curr_offs < N_CTX)[:, None],
                             other=0.)
            dv += tl.dot(silu_trans, do)

            ds_trans = tl.dot(v, tl.trans(do))
            ds_trans = ds_trans * silu_backward(qk_trans, q.type.element_ty,
                                                PERF_OPT, N_CTX, IS_Y)
            ds_trans = ds_trans.to(Q.dtype.element_ty)
            if CAUSAL:
                ds_trans = tl.where(curr_offs[None, :] >= (offs_kv[:, None]),
                                    ds_trans, zero)
            # compute dk = dot(ds.T, q)
            dk += tl.dot(ds_trans, q)
            # compute dq
            if IS_DIVISIBLE:
                tl.atomic_add(dq_ptrs, tl.dot(tl.trans(ds_trans), k))
            else:
                tl.atomic_add(dq_ptrs,
                              tl.dot(tl.trans(ds_trans), k),
                              mask=(curr_offs < N_CTX)[:, None])
            # increment pointers
            dq_ptrs += BLOCK_N * stride_dqm
            q_ptrs += BLOCK_N * stride_qm
            do_ptrs += BLOCK_N * stride_dom
        # write-back
        dv_ptrs = DV + (offs_kv[:, None] * stride_dvn +
                        offs_k[None, :] * stride_dvk)
        dk_ptrs = DK + (offs_kv[:, None] * stride_dkn +
                        offs_k[None, :] * stride_dkk)
        if IS_DIVISIBLE:
            tl.store(dv_ptrs, dv)
            tl.store(dk_ptrs, dk)
        else:
            tl.store(dv_ptrs, dv, mask=(offs_kv < N_CTX)[:, None])
            tl.store(dk_ptrs, dk, mask=(offs_kv < N_CTX)[:, None])


empty = torch.empty(128, device="mlu")


class _attention(torch.autograd.Function):
    attn_fwd_kernel = _attn_fwd
    attn_bwd_kernel = _attn_bwd

    @staticmethod
    def forward(ctx,
                q,
                k,
                v,
                bias,
                causal,
                has_bias,
                expanded_bias,
                perf_opt,
                provider="triton"):
        # Get capability.
        bang_driver = BangDriver()
        capability = bang_driver.get_current_target().arch // 100
        if capability < 6:
            IS_Y = False
        else:
            IS_Y = True

        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        o = torch.empty_like(q)

        processor_count = torch.mlu.get_device_properties(
            torch.mlu.current_device()).multi_processor_count
        grid = (processor_count, 1, 1)

        N_CTX = q.shape[2]
        bias_stride_z = 1
        bias_stride_m = 1
        bias_stride_n = 1
        if has_bias:
            if not expanded_bias:
                bias = bias.unsqueeze(1)
            bias_stride_z = bias.stride(0)
            bias_stride_m = bias.stride(1)
            bias_stride_n = bias.stride(2)
        _attn_fwd[grid](q,
                        k,
                        v,
                        bias,
                        o,
                        q.stride(0),
                        q.stride(1),
                        q.stride(2),
                        q.stride(3),
                        k.stride(0),
                        k.stride(1),
                        k.stride(2),
                        k.stride(3),
                        v.stride(0),
                        v.stride(1),
                        v.stride(2),
                        v.stride(3),
                        o.stride(0),
                        o.stride(1),
                        o.stride(2),
                        o.stride(3),
                        bias_stride_z,
                        bias_stride_m,
                        bias_stride_n,
                        q.shape[0],
                        q.shape[1],
                        N_CTX,
                        BLOCK_DMODEL=Lk,
                        CAUSAL=causal,
                        HAS_BIAS=has_bias,
                        EXPANDED_BIAS=expanded_bias,
                        PERF_OPT=perf_opt,
                        IS_Y=IS_Y)
        ctx.save_for_backward(q, k, v, o, bias)
        ctx.grid = grid
        ctx.BLOCK_DMODEL = Lk
        ctx.causal = causal
        ctx.PERF_OPT = perf_opt
        ctx.HAS_BIAS = has_bias
        ctx.EXPANDED_BIAS = expanded_bias
        ctx.IS_Y = IS_Y
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, bias = ctx.saved_tensors
        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        processor_count = torch.mlu.get_device_properties(
            torch.mlu.current_device()).multi_processor_count
        grid = (processor_count, 1, 1)

        bias_stride_z = 1
        bias_stride_m = 1
        bias_stride_n = 1
        if ctx.HAS_BIAS:
            bias_stride_z = bias.stride(0)
            bias_stride_m = bias.stride(1)
            bias_stride_n = bias.stride(2)

        _attn_bwd[grid](
            q,
            k,
            v,
            do,
            dq,
            dk,
            dv,
            bias,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            do.stride(0),
            do.stride(1),
            do.stride(2),
            do.stride(3),
            dq.stride(0),
            dq.stride(1),
            dq.stride(2),
            dq.stride(3),
            dk.stride(0),
            dk.stride(1),
            dk.stride(2),
            dk.stride(3),
            dv.stride(0),
            dv.stride(1),
            dv.stride(2),
            dv.stride(3),
            bias_stride_z,
            bias_stride_m,
            bias_stride_n,
            q.shape[0],
            q.shape[1],
            q.shape[2],
            BLOCK_DMODEL=ctx.BLOCK_DMODEL,
            CAUSAL=ctx.causal,
            HAS_BIAS=ctx.HAS_BIAS,
            EXPANDED_BIAS=ctx.EXPANDED_BIAS,
            PERF_OPT=ctx.PERF_OPT,
            IS_Y=ctx.IS_Y,
        )

        return dq, dk, dv, None, None, None, None, None, None


attention = _attention.apply


def hstu(q, k, v, bias, causal, has_bias, expanded_bias):
    q = q.contiguous()
    N_CTX = q.shape[-2]
    M = torch.tril(torch.ones((N_CTX, N_CTX), device="mlu"))

    p_before_silu = torch.matmul(q, k.transpose(2, 3)).float()  # [B, H, N, N]

    if has_bias:
        if not expanded_bias:
            bias = bias.unsqueeze(1).repeat(1, q.shape[2], 1)
        bias = bias.unsqueeze(1).repeat(1, q.shape[1], 1, 1)
        p_before_silu += bias

    if causal:
        p_before_silu[:, :, M == 0] = float("-1e6")

    p_after_silu = torch.nn.functional.silu(p_before_silu) * (1.0 / N_CTX)
    p = p_after_silu.to(q.dtype)

    ref_out = torch.matmul(p, v)

    return ref_out


@pytest.mark.parametrize("Z, H, N_CTX, D_HEAD", [(1, 1, 64, 64),
                                                 (2, 8, 4081, 64),
                                                 (2, 8, 4096, 96),
                                                 (2, 8, 4096, 128)])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("has_bias", [True, False])
# bias supports 2d or 4d layout
@pytest.mark.parametrize("expanded_bias", [True, False])
@pytest.mark.parametrize("perf_opt", [0, 1])
@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_op(Z, H, N_CTX, D_HEAD, causal, has_bias, expanded_bias, perf_opt,
            contiguous, dtype):
    torch.manual_seed(20)
    qkv_shape = [Z, H, N_CTX, D_HEAD]
    qkv_stride = [H * N_CTX * D_HEAD, N_CTX * D_HEAD, D_HEAD, 1]
    bias = None
    bias_shape = None
    bias_stride = None
    if has_bias:
        if expanded_bias:
            bias_shape = [Z, N_CTX, N_CTX]
            bias_stride = [N_CTX * N_CTX, N_CTX, 1]
        else:
            bias_shape = [Z, N_CTX]
            bias_stride = [N_CTX, 1]
    if not contiguous:
        qkv_stride = [stride + 1
                      for stride in qkv_stride[:-1]] + [qkv_stride[-1]]
        if has_bias:
            bias_stride = [stride + 1
                           for stride in bias_stride[:-1]] + [bias_stride[-1]]
    q = rand_strided(qkv_shape, qkv_stride, dtype, 'mlu',
                     Z * H * N_CTX * D_HEAD).requires_grad_()
    k = rand_strided(qkv_shape, qkv_stride, dtype, 'mlu',
                     Z * H * N_CTX * D_HEAD).requires_grad_()
    v = rand_strided(qkv_shape, qkv_stride, dtype, 'mlu',
                     Z * H * N_CTX * D_HEAD).requires_grad_()
    if has_bias:
        bias = rand_strided(bias_shape, bias_stride, dtype, 'mlu',
                            Z * N_CTX * N_CTX).requires_grad_()
    if has_bias:
        bias = torch.bernoulli(bias)
        bias = torch.where(bias == 1, 0.0, -1e8)
    # reference implementation
    dout = torch.randn_like(q)
    ref_out = hstu(q, k, v, bias, causal, has_bias, expanded_bias)
    ref_out.backward(dout)
    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, q.grad = q.grad.clone(), None
    # triton implementation
    # use add instead of mul, fix bias value
    _attention.attn_fwd_kernel.configs = [
        triton.Config({
            'BLOCK_M': 64,
            'BLOCK_N': 32
        },
                      num_stages=4,
                      num_warps=1)
    ]
    _attention.attn_bwd_kernel.configs = [
        triton.Config({
            'BLOCK_M': 64,
            'BLOCK_N': 32
        },
                      num_stages=4,
                      num_warps=1)
    ]
    tri_out = attention(q, k, v, bias, causal, has_bias, expanded_bias,
                        perf_opt)
    tri_out.backward(dout)
    tri_dv, v.grad = v.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dq, q.grad = q.grad.clone(), None
    ## compare
    torch.testing.assert_close(ref_out, tri_out, atol=1e-2, rtol=0)
    torch.testing.assert_close(ref_dv, tri_dv, atol=1e-2, rtol=0)
    torch.testing.assert_close(ref_dk, tri_dk, atol=1e-2, rtol=0)
    torch.testing.assert_close(ref_dq, tri_dq, atol=2e-2, rtol=0)


BATCH, N_HEADS, N_CTX, D_HEAD = 2, 8, 1024, 128
# vary seq length for fixed head and batch=4
configs = []
for mode in ["fwd", "bwd"]:
    for causal in [True]:
        for has_bias in [False, True]:
            for expanded_bias in [False, True]:
                for perf_opt in [0, 1]:
                    for D_HEAD in [64, 96, 128, 256]:
                        configs.append(
                            triton.testing.Benchmark(
                                x_names=["N_CTX"],
                                x_vals=[1024, 2048, 4081, 4096],
                                line_arg="provider",
                                line_vals=["triton"] + ["naive"],
                                line_names=["Triton"] + ["Naive"],
                                styles=[("red", "-"), ("blue", "-")],
                                ylabel="ms",
                                plot_name=
                                f"linear-attention-batch{BATCH}-head{N_HEADS}-d{D_HEAD}-{mode}-causal={causal}-has_bias={has_bias}-perf_opt={perf_opt}",
                                args={
                                    "H": N_HEADS,
                                    "BATCH": BATCH,
                                    "D_HEAD": D_HEAD,
                                    "dtype": torch.bfloat16,
                                    "mode": mode,
                                    "causal": causal,
                                    "has_bias": has_bias,
                                    "expanded_bias": expanded_bias,
                                    "perf_opt": perf_opt,
                                },
                            ))


@triton.testing.perf_report(configs)
def bench_hstu(BATCH,
               H,
               N_CTX,
               D_HEAD,
               mode,
               causal,
               has_bias,
               expanded_bias,
               perf_opt,
               provider,
               dtype=torch.bfloat16,
               device="mlu"):
    warmup = 5
    rep = 8
    q = torch.randn((BATCH, H, N_CTX, D_HEAD),
                    dtype=dtype,
                    device="mlu",
                    requires_grad=True)
    k = torch.randn((BATCH, H, N_CTX, D_HEAD),
                    dtype=dtype,
                    device="mlu",
                    requires_grad=True)
    v = torch.randn((BATCH, H, N_CTX, D_HEAD),
                    dtype=dtype,
                    device="mlu",
                    requires_grad=True)
    bias = None
    if has_bias:
        if expanded_bias:
            bias = torch.rand(size=(BATCH, N_CTX, N_CTX),
                              dtype=torch.float32,
                              device='mlu')
        else:
            bias = torch.rand(size=(BATCH, N_CTX),
                              dtype=torch.float32,
                              device='mlu')

    if provider == "triton":
        fn = lambda: attention(q, k, v, bias, causal, has_bias, expanded_bias,
                               perf_opt)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    if provider == "naive":
        fn = lambda: hstu(q, k, v, bias, causal, has_bias, expanded_bias)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * D_HEAD
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    if mode == "bwd":
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    return total_flops / ms * 1e-9
    #return ms


if __name__ == "__main__":
    bench_hstu.run(print_data=True)

