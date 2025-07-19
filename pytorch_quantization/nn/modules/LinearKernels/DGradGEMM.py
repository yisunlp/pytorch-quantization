import torch
import triton
import triton.language as tl
from quant_kernels import quant_by_row

# ==============================================================================
# 1. DGradGEMM Kernel
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
def dgrad_gemm_kernel(
    # Matrix Pointers
    a_ptr, a_scale_ptr,
    b_ptr, b_scale_ptr,
    c_ptr,
    # Matrix Dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    DTYPE: tl.constexpr
):
    """
    Computes C = dequant(A) @ dequant(B) for the DGrad pass.
    
    This kernel uses an access pattern optimal for A@B.T style GEMMs to ensure
    coalesced memory access on matrix A (`grad_output`), which is beneficial
    when the M dimension is large. Dequantization is performed inside the loop.
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

    # Block pointers
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0)
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
        order=(0, 1) # Use column-major-like access for matrix B
    )

    # Initialize accumulator and load scales for A
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    a_scale = tl.load(a_scale_ptr + offs_m[:, None], mask=offs_m[:, None] < M)

    # Main loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load INT8 data blocks
        a_quant = tl.load(a_block_ptr)
        b_quant = tl.load(b_block_ptr)

        # Load scales for the current block of B
        offs_k = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        b_scale = tl.load(b_scale_ptr + offs_k[:, None], mask=offs_k[:, None] < K, other=0.0)

        # Dequantize, cast, and compute dot product
        a_dequant = a_quant.to(tl.float32) * a_scale
        b_dequant = b_quant.to(tl.float32) * b_scale # Broadcasting (K,N) * (K,1)

        a_casted = a_dequant.to(DTYPE)
        b_casted = b_dequant.to(DTYPE)
        
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
# 2. Python Wrapper for DGradGEMM
# ==============================================================================

def DGradGEMM(
    grad_output_quant: torch.Tensor,
    grad_output_scale: torch.Tensor,
    w_quant: torch.Tensor,
    w_scale: torch.Tensor,
    output_dtype: torch.dtype
) -> torch.Tensor:
    """
    Performs a fused dequantization and DGrad GEMM.

    Args:
        grad_output_quant: Quantized grad_output [bs, seq_len, out_dim], INT8.
        grad_output_scale: Scale for grad_output [bs, seq_len, 1], float32.
        w_quant: Quantized weight [out_dim, in_dim], INT8.
        w_scale: Scale for weight [out_dim, 1], float32.
        output_dtype: The torch.dtype for the output tensor.

    Returns:
        The output tensor `grad_x` [bs, seq_len, in_dim] with specified dtype.
    """
    # Input validation
    assert grad_output_quant.dim() == 3 and grad_output_scale.dim() == 3
    assert w_quant.dim() == 2 and w_scale.dim() == 2
    assert grad_output_quant.is_cuda and w_quant.is_cuda
    assert grad_output_quant.dtype == torch.int8 and w_quant.dtype == torch.int8

    # Reshape tensors for 2D matrix multiplication
    bs, seq_len, output_dim = grad_output_quant.shape
    _, input_dim = w_quant.shape

    a_2d = grad_output_quant.view(-1, output_dim)
    a_scale_2d = grad_output_scale.view(-1, 1)

    M, K = a_2d.shape
    _, N = w_quant.shape
    assert K == w_quant.shape[0], f"Dimension mismatch: A.shape[1]({K}) != B.shape[0]({w_quant.shape[0]})"

    # Prepare output tensor
    c = torch.empty((M, N), device=a_2d.device, dtype=output_dtype)
    
    # Map to Triton dtype
    DTYPE_MAP = {torch.float16: tl.float16, torch.bfloat16: tl.bfloat16, torch.float32: tl.float32}
    if output_dtype not in DTYPE_MAP:
        raise NotImplementedError(f"Dtype {output_dtype} is not supported.")
    DTYPE_TL = DTYPE_MAP[output_dtype]

    # Launch Triton kernel
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)

    dgrad_gemm_kernel[grid](
        a_2d, a_scale_2d,
        w_quant, w_scale,
        c,
        M, N, K,
        a_2d.stride(0), a_2d.stride(1),
        w_quant.stride(0), w_quant.stride(1),
        c.stride(0), c.stride(1),
        DTYPE=DTYPE_TL
    )

    return c.view(bs, seq_len, input_dim)

# ==============================================================================
# 3. Example Usage and Verification
# ==============================================================================

if __name__ == "__main__":
    # Test Parameters
    BS = 16
    SEQ_LEN = 4096
    INPUT_DIM = 4096
    OUTPUT_DIM = 2048
    DTYPE = torch.float16

    # --- Test Setup ---
    print("=" * 60)
    print("🚀 Running Fused DGradGEMM Test (Optimized Kernel)")
    print(f"Shapes: grad_output=({BS}, {SEQ_LEN}, {OUTPUT_DIM}), W=({OUTPUT_DIM}, {INPUT_DIM})")
    print(f"M={BS*SEQ_LEN}, N={INPUT_DIM}, K={OUTPUT_DIM}")
    print(f"Output dtype: {DTYPE}")
    print("-" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("🛑 CUDA device not found. Aborting.")
        exit()

    # --- Prepare Data ---
    grad_output_fp = torch.randn((BS, SEQ_LEN, OUTPUT_DIM), device=device, dtype=DTYPE)
    w_fp = torch.randn((OUTPUT_DIM, INPUT_DIM), device=device, dtype=DTYPE)

    grad_output_quant, grad_output_scale = quant_by_row(grad_output_fp)
    w_quant, w_scale = quant_by_row(w_fp)

    # 1. Reference Calculation (PyTorch)
    print("🔎 Calculating reference output with PyTorch...")
    go_dequant = grad_output_quant.to(torch.float32) * grad_output_scale
    w_dequant = w_quant.to(torch.float32) * w_scale
    torch_output = torch.matmul(go_dequant, w_dequant).to(DTYPE)

    # 2. Triton Kernel Execution
    print("🚀 Executing Triton kernel...")
    triton_output = DGradGEMM(grad_output_quant, grad_output_scale, w_quant, w_scale, DTYPE)

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
    ms_torch = triton.testing.do_bench(lambda: torch.matmul((grad_output_quant.to(torch.float32) * grad_output_scale), (w_quant.to(torch.float32) * w_scale)).to(DTYPE))
    ms_triton = triton.testing.do_bench(lambda: DGradGEMM(grad_output_quant, grad_output_scale, w_quant, w_scale, DTYPE))
    speedup = ms_torch / ms_triton

    print(f"\nPyTorch (dequant + matmul) average time: {ms_torch:.4f} ms")
    print(f"Triton (fused optimized kernel) time:   {ms_triton:.4f} ms")
    print(f"\n📈 Speedup: {speedup:.2f}x")
    print("=" * 60)