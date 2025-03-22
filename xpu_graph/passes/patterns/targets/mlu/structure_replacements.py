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
        is_3d = False
        s1 = s2 = 0
        if len(input_tensor.shape) == 3:
            is_3d = True
            s1, s2, s3 = input_tensor.shape
            input_tensor = input_tensor.view(-1, s3)

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
            input_tensor.shape[0],
            input_tensor.stride(0),
        )
        if is_3d:
            return output.view(len(slices_index), s1, s2, slice_len)
        else:
            return output.view(len(slices_index), input_tensor.shape[0], slice_len)


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
        "FusedCatSlice": FuseSliceCatSameInputModule,
        "FusedMultipleSliceCat": FuseSliceCatSameInputModule,
    }
