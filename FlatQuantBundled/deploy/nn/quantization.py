import deploy
import torch
from deploy.nn.fp8_utils import is_fp8_dtype


# Prefer FP8 when CUDA INT4 matmul is unavailable (Trainium / CPU).
_USE_FP8 = deploy._CUDA is None and hasattr(torch, "float8_e4m3fn")


class Quantizer(torch.nn.Module):
    def __init__(self, input_clip_ratio=1.0, lac=False):
        super().__init__()
        self.input_clip_ratio = input_clip_ratio
        self.lac = lac
        self.register_buffer("clip_factor_a_max", torch.tensor(4.0))
        self.register_buffer("clip_factor_a_min", torch.tensor(4.0))

    def _compute_scales(self, x: torch.Tensor) -> torch.Tensor:
        if self.lac:
            reshaped_x = x.reshape((-1, x.shape[-1]))
            xmax = reshaped_x.amax(1, keepdim=True)
            xmin = reshaped_x.amin(1, keepdim=True)
            tmp = torch.zeros_like(xmax)
            xmax = torch.maximum(xmax, tmp)
            xmin = torch.minimum(xmin, tmp)

            xmax = xmax * torch.sigmoid(self.clip_factor_a_max.to(x.device))
            xmin = xmin * torch.sigmoid(self.clip_factor_a_min.to(x.device))

            xmax = torch.maximum(torch.abs(xmin), xmax)
            scales_x = xmax / 7
            scales_x = torch.where(xmax == 0, torch.ones_like(scales_x), scales_x)
            return scales_x.to(torch.float16)
        else:
            return (torch.max(torch.abs(x), dim=-1)[0].unsqueeze(1) / 7).to(torch.float16) * self.input_clip_ratio

    def forward(self, x):
        if isinstance(x, deploy.PackedQuantizedTensor):
            return x

        scales_x = self._compute_scales(x)

        if _USE_FP8:
            return self._quantize_fp8(x, scales_x)

        quantized_x = deploy.sym_quant(x, scales_x)
        return deploy.PackedQuantizedTensor(quantized_x, scales_x)

    @staticmethod
    def _quantize_fp8(x: torch.Tensor, scales_x: torch.Tensor):
        """FP8 activation quantization for Trainium / non-CUDA backends."""
        orig_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])
        s_2d = scales_x.view(-1, 1)
        x_scaled = x_2d / s_2d.to(x_2d.dtype)
        x_fp8 = x_scaled.to(torch.float8_e4m3fn)
        return deploy.PackedQuantizedTensor(
            x_fp8.view(orig_shape), scales_x
        )
