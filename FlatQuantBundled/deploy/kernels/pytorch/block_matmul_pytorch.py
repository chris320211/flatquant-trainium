"""
Pure PyTorch implementation of block matrix multiplication with FP8 quantization.
Compatible with AWS Trainium and other PyTorch backends.
"""

import torch
import deploy


def quantize_fp8(x, clip_factor_max, clip_factor_min):
    """
    Quantize tensor to FP8 format with learned clipping.

    Args:
        x: Input tensor [B, M, N]
        clip_factor_max: Learnable max clipping factor
        clip_factor_min: Learnable min clipping factor

    Returns:
        PackedQuantizedTensor with fp8 quantized values and scales
    """
    # Apply learned clipping with sigmoid activation
    sigmoid_max = 1.0 / (1.0 + torch.exp(-clip_factor_max))
    sigmoid_min = 1.0 / (1.0 + torch.exp(-clip_factor_min))

    # Compute per-batch scale (one scale per batch element)
    xmax = x.amax(dim=(-2, -1), keepdim=True)  # [B, 1, 1]
    xmin = x.amin(dim=(-2, -1), keepdim=True)  # [B, 1, 1]

    # Apply learned clipping factors
    xmax = xmax * sigmoid_max
    xmin = xmin * sigmoid_min

    # Compute scale based on max absolute value
    abs_xmin = torch.abs(xmin)
    max_val = torch.maximum(abs_xmin, xmax)

    # Avoid division by zero
    scale = torch.where(max_val == 0.0, torch.ones_like(max_val), max_val)

    # Quantize to FP8
    # Using float8_e4m3fn which is optimized for activations
    x_scaled = x / scale
    x_fp8 = x_scaled.to(torch.float8_e4m3fn)

    # Flatten for PackedQuantizedTensor format [B, -1]
    B = x.shape[0]
    x_fp8_flat = x_fp8.reshape(B, -1)
    scale_flat = scale.squeeze(-1).to(torch.float16)  # [B, 1]

    return deploy.PackedQuantizedTensor(x_fp8_flat, scale_flat)


def block_matmul_pytorch(b, c, seq_len, clip_factor_a_max, clip_factor_a_min, just_quantize=False):
    """
    Performs block matrix multiplication: b @ c with optional FP8 quantization.

    Pure PyTorch implementation compatible with Trainium and optimized for performance.

    Args:
        b: [B, M, N] batch of matrices - input activations
        c: [N, N] matrix - transformation matrix
        seq_len: Sequence length for batch processing
        clip_factor_a_max: Learnable max clipping factor for quantization
        clip_factor_a_min: Learnable min clipping factor for quantization
        just_quantize: If True, return unquantized result as flattened tensor

    Returns:
        If just_quantize=True: Tensor of shape [B, M*N] (unquantized, fp16)
        If just_quantize=False: PackedQuantizedTensor with FP8 quantized result

    Shape:
        - b: (B, M, N) where B = batch_size * seq_len
        - c: (N, N)
        - output: PackedQuantizedTensor with quantized_x of shape (B, M*N) and scales of shape (B, 1)
                  or Tensor of shape (B, M*N) if just_quantize=True
    """
    # Validate inputs
    assert c.shape[0] == c.shape[1], "Matrix C must be square"
    assert b.shape[2] == c.shape[0], f"Incompatible dimensions: b.shape[2]={b.shape[2]}, c.shape[0]={c.shape[0]}"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    assert c.is_contiguous(), "Matrix C must be contiguous"

    B, M, N = b.shape

    # Compute b @ c using PyTorch batched matrix multiplication
    # [B, M, N] @ [N, N] -> [B, M, N]
    bc = torch.matmul(b, c)  # [B, M, N]

    if just_quantize:
        # Return unquantized result, flattened
        return bc.view(B, -1)
    else:
        # Quantize result to FP8 with learned clipping
        return quantize_fp8(bc, clip_factor_a_max, clip_factor_a_min)


def benchmark(B, M, N, S, provider):
    """
    Benchmark function for performance testing (compatible with original interface).

    Args:
        B: Batch size * sequence length
        M: Matrix dimension M
        N: Matrix dimension N
        S: Sequence length
        provider: 'pytorch' for this implementation

    Returns:
        Tuple of (perf_ms, perf_max_ms, perf_min_ms, ms, max_ms, min_ms)
    """
    # Create random test data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    b = torch.randn((B, M, N), device=device, dtype=torch.float16)
    c = torch.randn((N, N), device=device, dtype=torch.float16)

    # Dummy clip factors
    clip_factor_a_max = torch.tensor(1.0, device=device)
    clip_factor_a_min = torch.tensor(1.0, device=device)

    if provider == 'pytorch':
        # Simple timing (can be replaced with more sophisticated benchmarking)
        import time

        # Warmup
        for _ in range(10):
            _ = block_matmul_pytorch(b, c, S, clip_factor_a_max, clip_factor_a_min)

        # Benchmark
        times = []
        for _ in range(100):
            if device == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = block_matmul_pytorch(b, c, S, clip_factor_a_max, clip_factor_a_min)
            if device == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

        times = torch.tensor(times)
        ms = times.median().item()
        min_ms = times.quantile(0.2).item()
        max_ms = times.quantile(0.8).item()

        # Compute TFLOPS
        # Operations: b @ c is 2 * B * M * N * N FLOPs
        total_flops = 2 * B * M * N * N
        perf = lambda t: total_flops * 1e-12 / (t * 1e-3)

        return perf(ms), perf(max_ms), perf(min_ms), ms, max_ms, min_ms
    else:
        raise ValueError(f"Unknown provider: {provider}")
