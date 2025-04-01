import torch
import torch.fx as fx
import torch_mlu

from .triton_kernel.fused_slice import (
    fused_slice_low,
)

from .triton_kernel.fused_slice_cat import (
    fused_slice_cat,
)


class RMSNormModule(torch.nn.Module):
    def forward(self, inputs, weights, epsilon):
        import torch_mlu_ops

        return torch_mlu_ops.fused_rms_norm(
            inputs, None, weights, None, None, epsilon, False
        )


class FuseSliceModule(torch.nn.Module):
    def forward(self, input_tensor, slices_index, slice_len):
        if len(input_tensor.shape) != 2:
            raise NotImplementedError("input must be 2d")
        if slice_len > input_tensor.shape[-1]:
            raise NotImplementedError(
                f"inputshape {input_tensor.shape} don't support slice_len:{slice_len}"
            )
        slices_index = torch.tensor(
            slices_index, dtype=torch.int32, device=input_tensor.device
        )
        output = fused_slice_low(
            input_tensor,
            slices_index,
            slice_len,
        )
        return output


class FuseSplitModule(torch.nn.Module):
    def forward(self, x: torch.Tensor, split_size: int, dim: int):
        if len(x.shape) != 2:
            raise NotImplementedError("input must be 2d")
        split_num = x.shape[1] // split_size
        if split_num * split_size != x.shape[1]:
            raise NotImplementedError("fused split don't support.")
        slices_index = list(range(0, x.shape[1], split_size))
        start_indices = torch.tensor(slices_index, device=x.device, dtype=torch.int32)
        output = fused_slice_low(
            x,
            start_indices,
            split_size,
        )
        return output.unbind(0)


class FuseSliceCatSameInputModule(torch.nn.Module):
    def forward(self, input_tensor, slices):
        if len(input_tensor.shape) != 2:
            raise NotImplementedError("input must be 2d")
        indices = [i for start, end in slices for i in range(start, end)]
        rows, _ = input_tensor.shape
        indices_tensor = torch.tensor(
            indices, dtype=torch.int32, device=input_tensor.device
        )
        return fused_slice_cat(
            input_tensor,
            indices_tensor,
            rows,
            len(indices),
            input_tensor.stride(0),
            16384,  # blocksize
        )


def get_structure_replacements():
    return {
        "FusedRMSNorm": RMSNormModule,
        "FusedSlice": FuseSliceModule,
        "FusedSplit": FuseSplitModule,
        "FusedCatSlice": FuseSliceCatSameInputModule,
        "FusedMultipleSliceCat": FuseSliceCatSameInputModule,
    }
