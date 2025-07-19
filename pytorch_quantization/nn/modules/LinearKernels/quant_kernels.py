import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 4096}, num_warps=16),
    ],
    key=['N'],
)
@triton.jit
def _quant_by_row_kernel(
    x_ptr,
    output_ptr,
    scale_ptr,
    x_row_stride,
    output_row_stride,
    N,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Triton Kernel for per-row INT8 quantization.
    Each program in the grid processes one row of the input tensor.
    """
    row_idx = tl.program_id(axis=0)

    row_x_ptr = x_ptr + row_idx * x_row_stride
    row_output_ptr = output_ptr + row_idx * output_row_stride

    # Pass 1: Find the absolute maximum value (amax) for the row.
    # A float32 accumulator is used to maintain precision and prevent overflow.
    row_abs_max = tl.zeros((), dtype=tl.float32)
    for col_offset in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        offsets = col_offset * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask = offsets < N
        x = tl.load(row_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        block_abs_max = tl.max(tl.abs(x))
        row_abs_max = tl.maximum(row_abs_max, block_abs_max)

    # Pass 2: Calculate scale, then quantize and store the row.
    # If amax is 0 (row is all zeros), scale is set to 1.0 to prevent division by zero.
    scale = row_abs_max / 127.0
    scale = tl.where(scale == 0.0, 1e-3, scale)
    tl.store(scale_ptr + row_idx, scale)

    for col_offset in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        offsets = col_offset * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask = offsets < N
        x = tl.load(row_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        quantized_x = x / scale
        quantized_x = tl.extra.cuda.libdevice.round(quantized_x)
        quantized_x = tl.clamp(quantized_x, -127.0, 127.0)
        
        quantized_x_int8 = quantized_x.to(tl.int8)
        tl.store(row_output_ptr + offsets, quantized_x_int8, mask=mask)


def quant_by_row(x: torch.Tensor):
    """
    Performs per-row symmetric quantization on the last dimension of a tensor.

    Args:
        x (torch.Tensor): The input tensor. Must be a 2D or higher-dimensional float tensor.

    Returns:
        A tuple containing:
        - output (torch.Tensor): The INT8 quantized tensor with the same shape as the input.
        - scale (torch.Tensor): The float32 scale factors, with shape `x.shape[:-1]`.
    """
    if not x.is_contiguous():
        x = x.contiguous()
    
    assert x.dtype in [torch.float16, torch.bfloat16, torch.float32], \
        "Input tensor must be of type float16, bfloat16, or float32."
    assert x.dim() >= 2, "Input tensor must be at least 2D."

    input_shape = x.shape
    last_dim = input_shape[-1]
    
    x_2d = x.view(-1, last_dim)
    M, N = x_2d.shape

    output = torch.empty_like(x_2d, dtype=torch.int8)
    scale = torch.empty((M,), dtype=torch.float32, device=x.device)

    # The grid size is the number of rows, as each program handles one row.
    grid = (M, )
    
    _quant_by_row_kernel[grid](
        x_2d,
        output,
        scale,
        x_2d.stride(0),
        output.stride(0),
        N,
    )

    output = output.view(input_shape)
    scale = scale.view(input_shape[:-1] + (1, ))
    
    return output, scale

if __name__ == "__main__":

    def torch_reference_quant_by_row(x: torch.Tensor):
        """
        A reference implementation using pure PyTorch, with logic equivalent to the Triton kernel.
        """
        # 1. To maintain numerical consistency with the Triton kernel, all computations are done in float32.
        x_f32 = x.to(torch.float32)

        # 2. Calculate the absolute maximum value (amax) along the last dimension.
        abs_max = torch.max(torch.abs(x_f32), dim=-1, keepdim=True)[0]

        # 3. Calculate the scaling factor.
        scale = abs_max / 127.0
        # Handle the case of all-zero rows to match the Triton kernel's logic.
        scale[scale == 0] = 1e-3

        # 4. Quantize the tensor.
        # The calculation is done in float32, then clamped and cast to int8.
        quantized_x = torch.round(x_f32 / scale).clamp(-127, 127).to(torch.int8)

        # 5. Return the quantized result and the correctly shaped scale tensor.
        return quantized_x, scale.squeeze(-1)

    # --- Test Parameters ---
    BATCH_SIZE = 16
    SEQ_LEN = 2048
    HIDDEN_DIM = 4096

    print("=" * 60)
    print("🚀 Starting Test for Per-Row Quantization")
    print(f"Tensor Shape (BS, SEQ_LEN, HIDDEN_DIM): ({BATCH_SIZE}, {SEQ_LEN}, {HIDDEN_DIM})")
    print("-" * 60)

    # --- Environment Check ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("⚠️ Warning: No CUDA device detected. Triton cannot run. Please test in a GPU environment.")
        exit()
        
    # --- Data Preparation ---
    test_input = torch.randn(
        (BATCH_SIZE, SEQ_LEN, HIDDEN_DIM),
        device=device,
        dtype=torch.float16
    )
    # Create an all-zero row to test the special handling of the scale factor.
    test_input[0, 0, :] = 0

    # --- 1. Accuracy Verification ---
    print("🔎 Verifying numerical accuracy...")
    triton_quant_out, triton_scale_out = quant_by_row(test_input)
    triton_out = (triton_quant_out * triton_scale_out).to(test_input.dtype)
    is_scale_correct = torch.allclose(triton_out, test_input, atol=1e-1, rtol=1e-2)
    mae = (triton_out - test_input).abs().mean()
    rel_mae = mae / test_input.abs().mean()
    print(f"Accuracy check passed: {is_scale_correct}, MAE: {mae:.6f}, Relative MAE: {rel_mae:.6f}")

    print("-" * 60)

    # --- 2. Performance Benchmarking ---
    print("\n⚡️ Running performance benchmarks...")
    # Use triton.testing.do_bench for accurate measurements.
    ms_triton = triton.testing.do_bench(lambda: quant_by_row(test_input))
    ms_torch = triton.testing.do_bench(lambda: torch_reference_quant_by_row(test_input))
    
    print(f"\nTriton implementation average time: {ms_triton:.4f} ms")
    print(f"PyTorch reference average time:   {ms_torch:.4f} ms")
    
    speedup = ms_torch / ms_triton
    print(f"\n📈 Speedup (Triton vs. PyTorch): {speedup:.2f}x")

    print("=" * 60)