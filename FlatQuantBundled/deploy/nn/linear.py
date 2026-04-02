import torch
import torch.nn.functional as F

import deploy
from deploy.nn.fp8_utils import (
    dequant_fp8_to_float,
    dequant_int4_packed_activations,
    int4_weight_to_float,
    is_fp8_dtype,
)


class ShapeHandler:
    def __init__(self, x: torch.Tensor):
        self.size_excl_last = x.numel()//x.shape[-1]
        self.shape_excl_last = tuple(x.shape[:-1])

    # Keep the last dim unchanged, flatten all previous dims
    def flatten(self, x: torch.Tensor):
        return x.view(self.size_excl_last, -1)

    # Recover back to the original shape.
    def unflatten(self, x: torch.Tensor):
        return x.view(self.shape_excl_last + (-1,))

    def unflatten_scale(self, x: torch.Tensor):
        return x.view(self.shape_excl_last)


class Linear4bit(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=False, dtype=torch.float16):
        '''
        Symmetric 4-bit Linear Layer.
        '''
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer('weight_scales',
                             torch.zeros((self.out_features, 1), requires_grad=False))
        self.register_buffer('weight', (torch.randint(1, 7, (self.out_features, self.in_features // 2),
                                                             # SubByte weight
                                                             dtype=torch.uint8, requires_grad=False)))
        if bias:                                                        
            self.register_buffer('bias', torch.zeros((self.out_features), dtype=dtype))
        else:
            self.bias = None
        
    def forward(self, x):
        #if torch.cuda.current_device() != x.device:
        #    torch.cuda.set_device(x.device)
        
        assert type(x) == deploy.PackedQuantizedTensor #Quantized input is given
        if is_fp8_dtype(x.quantized_x.dtype):
            return self._forward_fp8_pytorch(x)
        if x.quantized_x.dtype == torch.uint8 and deploy._CUDA is None:
            return self._forward_int4_pytorch(x)
        x, scales_x = x.quantized_x, x.scales_x
        #shape_handler = ShapeHandler(quantized_x)
        #quantized_x = shape_handler.flatten(quantized_x)
        x = deploy.matmul(x, self.weight)
        #out = shape_handler.unflatten(
        #    deploy.sym_dequant(int_result, scales_x, self.weight_scales))
        if self.bias is not None:
            return deploy.sym_dequant(x, scales_x, self.weight_scales) + self.bias
        else:
            return deploy.sym_dequant(x, scales_x, self.weight_scales)

    def _forward_int4_pytorch(self, x) -> torch.Tensor:
        """INT4 activations (Quantizer) without CUDA int4 matmul."""
        out_dtype = self.weight_scales.dtype
        x_f = dequant_int4_packed_activations(x, out_dtype=out_dtype)
        w_f = int4_weight_to_float(self.weight, self.weight_scales, out_dtype=out_dtype)
        return F.linear(x_f, w_f, self.bias)

    def _forward_fp8_pytorch(self, x) -> torch.Tensor:
        """
        Trainium / non-CUDA path: FP8 activations + INT4 weights, computed in BF16/FP16
        via PyTorch `linear`. Preserves activation quantization from the kernel; weights
        stay INT4 in memory and are expanded only for the op.
        """
        out_dtype = self.weight_scales.dtype
        x_f = dequant_fp8_to_float(x, out_dtype=out_dtype)
        w_f = int4_weight_to_float(self.weight, self.weight_scales, out_dtype=out_dtype)
        return F.linear(x_f, w_f, self.bias)

    @staticmethod
    def from_float(module: torch.nn.Linear, weight_scales=None,):
        '''
        Generate a new Linear4bit module from a FP16 Linear module.
        The weight matrix should have the same shape as the weight matrix of the FP16 Linear module and rounded using torch.round()
        routine. We will convert it to subByte representation and save it in the int_weight buffer.
        '''
        weight_matrix = module.weight.data
        device = weight_matrix.device

        int_module = Linear4bit(module.in_features, module.out_features, bias=module.bias is not None, dtype=weight_matrix.dtype).to(weight_matrix.dtype)
        if weight_scales is not None:
            assert weight_scales.shape == (module.out_features, 1), 'weight_scales should have shape (out_features, 1)'
            weight_matrix = weight_matrix.to(device)
            int_module.weight_scales.copy_(weight_scales.to(weight_matrix.dtype))
            int_rounded_weight = (weight_matrix / weight_scales.to(device)).round()
            int_module.weight.copy_(deploy.functional.pack_i4(int_rounded_weight.to(torch.int8)).cpu())

            if module.bias is not None:
                int_module.bias.copy_(module.bias)

        return int_module
