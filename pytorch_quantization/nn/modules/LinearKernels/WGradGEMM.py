import torch
import triton
import triton.language as tl
from quant_kernels import quant_by_row

# ==============================================================================
# 1. WGradGEMM Kernel
# ==============================================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def wgrad_gemm_kernel(
    # Matrix Pointers
    a_ptr, b_ptr,          # A = grad_output, B = x
    a_scale_ptr, b_scale_ptr,
    c_ptr,                 # C = grad_w
    # Matrix Dimensions
    M, N, K,               # M=out_dim, N=in_dim, K=bs*seq_len
    # Strides
    stride_ak, stride_am,  # Strides for A (grad_output)
    stride_bk, stride_bn,  # Strides for B (x)
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    DTYPE: tl.constexpr
):
    """
    Computes C = dequant(A.T) @ dequant(B) for the WGrad pass.
    The transposition of A is handled by the block pointer configuration,
    avoiding the need for `trans_a=True` in `tl.dot`.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # --- Block Pointers Setup ---
    # ** THE FIX IS HERE **
    # Load A (grad_output) with a transposed layout to compute A.T @ B.
    # We describe an [M, K] shape using the strides of a [K, M] matrix.
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(0, 1)
    )
    # Load B (x) normally.
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
        order=(1, 0)
    )

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Main loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load INT8 data blocks
        a_quant = tl.load(a_block_ptr) # Shape: [BLOCK_SIZE_M, BLOCK_SIZE_K]
        b_quant = tl.load(b_block_ptr) # Shape: [BLOCK_SIZE_K, BLOCK_SIZE_N]

        # Load scales for the current K-slice
        offs_k = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        a_scale = tl.load(a_scale_ptr + offs_k[None, :], mask=offs_k[None, :] < K, other=0.0)
        b_scale = tl.load(b_scale_ptr + offs_k[:, None], mask=offs_k[:, None] < K, other=0.0)

        # Dequantize and cast
        # `a_quant` is [M, K], so `a_scale` must be broadcast from [1, K].
        a_dequant = a_quant.to(tl.float32) * a_scale
        # `b_quant` is [K, N], so `b_scale` must be broadcast from [K, 1].
        b_dequant = b_quant.to(tl.float32) * b_scale

        a_casted = a_dequant.to(DTYPE)
        b_casted = b_dequant.to(DTYPE)

        # A standard dot product is now equivalent to A.T @ B
        accumulator = tl.dot(a_casted, b_casted, accumulator)

        # Advance block pointers
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K, 0))

    # Store final result
    c = accumulator.to(DTYPE)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

# ==============================================================================
# 2. Python Wrapper for WGradGEMM
# ==============================================================================

def WGradGEMM(
    grad_output_quant: torch.Tensor,
    grad_output_scale: torch.Tensor,
    x_quant: torch.Tensor,
    x_scale: torch.Tensor,
    output_dtype: torch.dtype
) -> torch.Tensor:
    """
    Performs a fused dequantization and WGrad GEMM (grad_w = grad_output.T @ x).

    Args:
        grad_output_quant: Quantized grad_output [bs*seq_len, out_dim], INT8.
        x_quant: Quantized x [bs*seq_len, in_dim], INT8.
        grad_output_scale: Scale for grad_output [bs*seq_len, 1], float32.
        x_scale: Scale for x [bs*seq_len, 1], float32.
        output_dtype: The torch.dtype for the output gradient tensor.

    Returns:
        The gradient tensor `grad_w` [out_dim, in_dim] with specified dtype.
    """
    # Get dimensions
    K, M = grad_output_quant.shape  # K = bs*seq_len, M = out_dim
    _, N = x_quant.shape           # N = in_dim
    assert K == x_quant.shape[0], f"Dimension mismatch: A.shape[0]({K}) != B.shape[0]({x_quant.shape[0]})"
    
    # Prepare output tensor
    grad_w = torch.empty((M, N), device=grad_output_quant.device, dtype=output_dtype)
    
    # Map to Triton dtype
    DTYPE_MAP = {torch.float16: tl.float16, torch.bfloat16: tl.bfloat16, torch.float32: tl.float32}
    if output_dtype not in DTYPE_MAP:
        raise NotImplementedError(f"Dtype {output_dtype} is not supported.")
    DTYPE_TL = DTYPE_MAP[output_dtype]

    # Launch Triton kernel
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)

    wgrad_gemm_kernel[grid](
        grad_output_quant, x_quant,
        grad_output_scale, x_scale,
        grad_w,
        M, N, K,
        grad_output_quant.stride(0), grad_output_quant.stride(1),
        x_quant.stride(0), x_quant.stride(1),
        grad_w.stride(0), grad_w.stride(1),
        DTYPE=DTYPE_TL
    )
    
    return grad_w

# ==============================================================================
# 3. Example Usage and Verification
# ==============================================================================

if __name__ == "__main__":
    # Test Parameters
    BS = 8
    SEQ_LEN = 1024
    INPUT_DIM = 2048
    OUTPUT_DIM = 4096
    DTYPE = torch.float16

    # --- Test Setup ---
    print("=" * 60)
    print("🚀 Running Fused WGradGEMM Test")
    print(f"Shapes: X=({BS*SEQ_LEN}, {INPUT_DIM}), dY=({BS*SEQ_LEN}, {OUTPUT_DIM})")
    print(f"Result grad_w=({OUTPUT_DIM}, {INPUT_DIM})")
    print(f"Output dtype: {DTYPE}")
    print("-" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("🛑 CUDA device not found. Aborting.")
        exit()

    # --- Prepare Data ---
    x_fp = torch.randn((BS * SEQ_LEN, INPUT_DIM), device=device, dtype=DTYPE)
    grad_output_fp = torch.randn((BS * SEQ_LEN, OUTPUT_DIM), device=device, dtype=DTYPE)

    x_quant, x_scale = quant_by_row(x_fp)
    grad_output_quant, grad_output_scale = quant_by_row(grad_output_fp)

    # 1. Reference Calculation (PyTorch)
    print("🔎 Calculating reference output with PyTorch...")
    x_dequant = x_quant.to(torch.float32) * x_scale
    go_dequant = grad_output_quant.to(torch.float32) * grad_output_scale
    torch_output = torch.matmul(go_dequant.T, x_dequant).to(DTYPE)

    # 2. Triton Kernel Execution
    print("🚀 Executing Triton kernel...")
    triton_output = WGradGEMM(grad_output_quant, grad_output_scale, x_quant, x_scale, DTYPE)

    # 3. Verification
    print("🔎 Verifying correctness...")
    is_correct = torch.allclose(triton_output, torch_output, atol=1e-1, rtol=1e-2)

    max_abs_err = (triton_output - torch_output).abs().max().item()
    print(f"Correctness check passed: {is_correct}")
    print(f"Max absolute error: {max_abs_err:.6f}")
    if not is_correct:
        print("‼️ Correctness check failed.")

    # 4. Performance Benchmarking
    print("\n⚡️ Running performance benchmark...")
    ms_torch = triton.testing.do_bench(lambda: torch.matmul((grad_output_quant.to(torch.float32) * grad_output_scale).T, (x_quant.to(torch.float32) * x_scale)).to(DTYPE))
    ms_triton = triton.testing.do_bench(lambda: WGradGEMM(grad_output_quant, grad_output_scale, x_quant, x_scale, DTYPE))
    speedup = ms_torch / ms_triton

    print(f"\nPyTorch (dequant + matmul) average time: {ms_torch:.4f} ms")
    print(f"Triton (fused kernel) time:           {ms_triton:.4f} ms")
    print(f"\n📈 Speedup: {speedup:.2f}x")
    print("=" * 60)