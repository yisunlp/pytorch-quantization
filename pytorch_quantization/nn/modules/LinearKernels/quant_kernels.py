from typing import Tuple
import torch
import triton
import triton.language as tl
import torch.nn.functional as F

# ==============================================================================
# Triton Kernel
# ==============================================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
    ],
    key=['N'],
)

@triton.jit
def _quant_by_row_kernel(
    x_ptr, output_ptr, scale_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_outm, stride_outn,
    stride_scalem,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel to quantize a matrix row-by-row to INT8.
    Internal calculations are done in float32 for higher precision.
    """
    row_idx = tl.program_id(axis=0)
    if row_idx >= M:
        return
    
    row_start_ptr = x_ptr + row_idx * stride_xm
    row_amax = 0.0

    # Pass 1: Find the absolute maximum value in the row using float32.
    for k_offset in range(0, tl.cdiv(N, BLOCK_SIZE)):
        offset = k_offset * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offset < N
        current_chunk = tl.load(row_start_ptr + offset * stride_xn, mask=mask, other=0.0)
        row_amax = tl.maximum(row_amax, tl.max(tl.abs(current_chunk)))

    # Calculate scale in float32 and store it.
    scale = row_amax.to(tl.float32) / 127.0
    scale = tl.where(scale == 0, 1e-6, scale)
    inv_scale = 1.0 / scale
    tl.store(scale_ptr + row_idx * stride_scalem, scale)

    # Pass 2: Quantize the row to INT8 using the high-precision scale.
    for k_offset in range(0, tl.cdiv(N, BLOCK_SIZE)):
        offset = k_offset * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offset < N
        current_chunk = tl.load(row_start_ptr + offset * stride_xn, mask=mask, other=0.0)
        quantized_chunk = tl.clamp(current_chunk.to(tl.float32) * inv_scale, -127.0, 127.0)
        quantized_chunk = tl.extra.cuda.libdevice.round(quantized_chunk).to(tl.int8)
        output_chunk_ptrs = output_ptr + row_idx * stride_outm + offset * stride_outn
        tl.store(output_chunk_ptrs, quantized_chunk, mask=mask)

# ==============================================================================
# Python Interface
# ==============================================================================

def quant_by_row(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies per-row symmetric INT8 quantization using a Triton kernel."""
    assert x.is_cuda and x.dim() == 2, "Input must be a 2D CUDA tensor."
    x = x.contiguous()
    M, N = x.shape
    quantized_x = torch.empty_like(x, dtype=torch.int8)
    # The scales are stored in float32 to maintain precision.
    scales = torch.empty((M,), device=x.device, dtype=torch.float32)
    grid = lambda meta: (M,)
    
    _quant_by_row_kernel[grid](
        x, quantized_x, scales, M, N,
        x.stride(0), x.stride(1),
        quantized_x.stride(0), quantized_x.stride(1),
        scales.stride(0),
    )
    return quantized_x, scales.unsqueeze(-1)

# ==============================================================================
# Testing and Benchmarking
# ==============================================================================

if __name__ == "__main__":
    M, N = 16*2048, 4096
    DTYPE = torch.float16
    DEVICE = "cuda"

    input_tensor = torch.randn((M, N), device=DEVICE, dtype=DTYPE)

    def quant_by_row_torch(x: torch.Tensor):
        # Use .float() for a fair comparison of precision-aware quantization.
        scales_fp32 = x.float().abs().max(dim=-1).values / 127.0
        scales = scales_fp32
        scales[scales == 0] = 1e-6
        quantized = torch.round(torch.clamp((x / scales[:, None]),-127.0,127.0)).to(torch.int8)
        # Return scales in fp32 to match the triton function's output.
        return quantized, scales.float()

    print(f"Evaluating INT8 quantization for a ({M}, {N}) tensor of type {DTYPE}")
    print("-" * 60)

    # Evaluate the Triton kernel's output.
    q_triton, s_triton = quant_by_row(input_tensor.clone())
    dequant_triton = q_triton.to(DTYPE) * s_triton.to(DTYPE)[:, None]
    loss_triton = F.mse_loss(input_tensor, dequant_triton)
    print(f"Triton Kernel MSE: {loss_triton.item():.8f}")
    print(f"Output shapes: Quantized={q_triton.shape}, Scales={s_triton.shape}")

    # Benchmark performance against a PyTorch equivalent.
    print("\nBenchmarking performance (median of 1000 runs)...")
    ms_triton = triton.testing.do_bench(lambda: quant_by_row(input_tensor), rep=1000, return_mode="median")
    ms_torch = triton.testing.do_bench(lambda: quant_by_row_torch(input_tensor), rep=1000, return_mode="median")
    
    print(f"  - Triton Kernel: {ms_triton:.4f} ms")
    print(f"  - PyTorch Eager: {ms_torch:.4f} ms")
    print(f"  - Speedup: {ms_torch / ms_triton:.2f}x")
    print("-" * 60)