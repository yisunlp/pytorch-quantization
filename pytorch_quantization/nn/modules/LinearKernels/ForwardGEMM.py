import torch
import triton
import triton.language as tl
from quant_kernels import quant_by_row

# ==============================================================================
# 1. Fused GEMM + Dequantization Kernel
# ==============================================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def forward_gemm_kernel(
    x_ptr, x_scale_ptr,
    w_ptr, w_scale_ptr,
    output_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wm, stride_wk,
    stride_outm, stride_outn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Triton kernel for fused INT8 GEMM and dequantization, computing `output = (x @ w.T) * x_scale * w_scale`."""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # Pointers to the first block of x and w.T
    x_block_ptr = tl.make_block_ptr(base=x_ptr, shape=(M, K), strides=(stride_xm, stride_xk),
                                    offsets=(pid_m * BLOCK_SIZE_M, 0), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
                                    order=(1, 0))
    # By swapping the strides of w, we effectively treat it as its transpose, enabling a fused operation.
    w_block_ptr = tl.make_block_ptr(base=w_ptr, shape=(K, N), strides=(stride_wk, stride_wm),
                                    offsets=(0, pid_n * BLOCK_SIZE_N), block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
                                    order=(1, 0))

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(x_block_ptr, boundary_check=(0, 1))
        b = tl.load(w_block_ptr, boundary_check=(0, 1))
        accumulator = tl.dot(a, b, accumulator)
        x_block_ptr = tl.advance(x_block_ptr, (0, BLOCK_SIZE_K))
        w_block_ptr = tl.advance(w_block_ptr, (BLOCK_SIZE_K, 0))

    # Dequantize
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    scale_x = tl.load(x_scale_ptr + offs_m[:, None], mask=offs_m[:, None] < M)
    scale_w = tl.load(w_scale_ptr + offs_n[None, :], mask=offs_n[None, :] < N)
    output = accumulator.to(tl.float32) * scale_x * scale_w

    # Store the final result
    output_ptrs = output_ptr + stride_outm * offs_m[:, None] + stride_outn * offs_n[None, :]
    output_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(output_ptrs, output.to(output_ptr.dtype.element_ty), mask=output_mask)


def ForwardGEMM(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    output_dtype: torch.dtype
) -> torch.Tensor:
    """
    Performs a fused INT8 GEMM with dequantization.

    Args:
        x (torch.Tensor): Input tensor of shape [bs, seq_len, input_dim] and dtype INT8.
        x_scale (torch.Tensor): Scale for x, shape [bs, seq_len, 1] and dtype float32.
        w (torch.Tensor): Weight tensor of shape [output_dim, input_dim] and dtype INT8.
        w_scale (torch.Tensor): Scale for w, shape [output_dim, 1] and dtype float32.
        output_dtype (torch.dtype): The desired data type for the output tensor.

    Returns:
        torch.Tensor: The output tensor of shape [bs, seq_len, output_dim] and specified dtype.
    """
    assert x.dim() == 3 and x_scale.dim() == 3, "Input x and x_scale must be 3D"
    assert w.dim() == 2 and w_scale.dim() == 2, "Weight w and w_scale must be 2D"
    assert x.is_cuda and w.is_cuda, "All tensors must be on a CUDA device"
    assert x.dtype == torch.int8 and w.dtype == torch.int8, "x and w must be INT8"
    
    bs, seq_len, input_dim = x.shape
    output_dim, _ = w.shape
    
    x_2d = x.view(-1, input_dim)
    x_scale_2d = x_scale.view(-1, 1)

    M, K = x_2d.shape
    N, _ = w.shape
    
    assert K == w.shape[1], f"Dimension mismatch: x.shape[-1] ({K}) != w.shape[1] ({w.shape[1]})"
    
    output = torch.empty((M, N), device=x.device, dtype=output_dtype)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)

    forward_gemm_kernel[grid](
        x_2d, x_scale_2d,
        w, w_scale,
        output,
        M, N, K,
        x_2d.stride(0), x_2d.stride(1),
        w.stride(0), w.stride(1),
        output.stride(0), output.stride(1),
    )
    
    return output.view(bs, seq_len, output_dim)

# ==============================================================================
# 2. Example Usage and Verification
# ==============================================================================

if __name__ == "__main__":
    
    # Test Parameters
    BS = 16
    SEQ_LEN = 1024
    INPUT_DIM = 4096
    OUTPUT_DIM = 2048
    DTYPE = torch.float16

    print("=" * 60)
    print("🚀 Running Fused Forward GEMM Test")
    print(f"Shapes: X=({BS}, {SEQ_LEN}, {INPUT_DIM}), W=({OUTPUT_DIM}, {INPUT_DIM})")
    print(f"Output dtype: {DTYPE}")
    print("-" * 60)
    
    # Setup Environment
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("🛑 CUDA device not found. Aborting.")
        exit()

    # Prepare Data
    x_fp = torch.randn((BS, SEQ_LEN, INPUT_DIM), device=device, dtype=DTYPE)
    # Create a non-contiguous (column-major) weight tensor to test the optimized pipeline
    w_fp = torch.randn((INPUT_DIM, OUTPUT_DIM), device=device, dtype=DTYPE).T

    # Quantize inputs using the optimized, flexible quantization function
    x_int8, x_scale = quant_by_row(x_fp)
    x_int8 = x_int8.view(BS, SEQ_LEN, INPUT_DIM) # Reshape back
    w_int8, w_scale = quant_by_row(w_fp)
    
    # 1. Reference Calculation (PyTorch)
    print("🔎 Calculating reference output with PyTorch...")
    x_dequant = x_int8.to(torch.float32) * x_scale.to(torch.float32)
    w_dequant = w_int8.to(torch.float32) * w_scale.to(torch.float32)
    torch_output = torch.matmul(x_dequant, w_dequant.T).to(DTYPE)

    # 2. Triton Kernel Execution
    print("🚀 Executing Triton kernel...")
    triton_output = ForwardGEMM(x_int8, x_scale, w_int8, w_scale, DTYPE)

    # 3. Verification
    print("🔎 Verifying correctness...")
    is_correct = torch.allclose(triton_output, torch_output, atol=1e-1, rtol=1e-2)
    max_abs_err = (triton_output - torch_output).abs().max().item()
    print(f"Correctness check passed: {is_correct}")
    print(f"Max absolute error: {max_abs_err:.6f}")
    
    # 4. Performance Benchmarking
    print("\n⚡️ Running performance benchmark...")
    ms_torch = triton.testing.do_bench(lambda: torch.matmul((x_int8.to(DTYPE) * x_scale), (w_int8.to(DTYPE) * w_scale).T))
    ms_triton = triton.testing.do_bench(lambda: ForwardGEMM(x_int8, x_scale, w_int8, w_scale, DTYPE))
    
    speedup = ms_torch / ms_triton
    
    print(f"\nPyTorch (dequant + matmul) average time: {ms_torch:.4f} ms")
    print(f"Triton (fused kernel) average time:      {ms_triton:.4f} ms")
    print(f"\n📈 Speedup: {speedup:.2f}x")
    print("=" * 60)