import torch
import triton
import triton.language as tl

# ==============================================================================
# Triton Kernels
# ==============================================================================

@triton.jit
def quantize_per_row_kernel(
    x_ptr, output_ptr, scale_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_outm, stride_outn,
    stride_scalem,
    # BLOCK_SIZE_N 不再是整个向量长度，而是一个固定的、2的幂次方的处理块大小
    BLOCK_SIZE_N: tl.constexpr = 1024, # 例如，使用1024
):
    row_idx = tl.program_id(axis=0)
    if row_idx >= M:
        return
    
    # 初始化amax和指针
    row_amax = 0.0
    row_start_ptr = x_ptr + row_idx * stride_xm
    
    # 1. 第一个Pass: 在K维度上分块循环，找到整行的amax
    for k_offset in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        # 计算当前块的指针和掩码
        offset = k_offset * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask = offset < N
        
        # 加载当前块
        current_chunk_ptrs = row_start_ptr + offset * stride_xn
        current_chunk = tl.load(current_chunk_ptrs, mask=mask, other=0.0)
        
        # 更新整行的amax
        current_max = tl.max(tl.abs(current_chunk))
        row_amax = tl.maximum(row_amax, current_max)

    # 2. 计算最终的scale
    scale = row_amax / 127.0
    scale = tl.where(scale == 0, 1e-6, scale)
    inv_scale = 1.0 / scale
    tl.store(scale_ptr + row_idx * stride_scalem, scale)

    # 3. 第二个Pass: 再次分块循环，进行量化并写回
    for k_offset in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        offset = k_offset * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask = offset < N
        
        # 再次加载当前块
        current_chunk_ptrs = row_start_ptr + offset * stride_xn
        current_chunk = tl.load(current_chunk_ptrs, mask=mask, other=0.0)
        
        # 量化
        quantized_chunk = tl.floor(current_chunk * inv_scale + 0.5).to(tl.int8)
        
        # 写回INT8结果
        output_chunk_ptrs = output_ptr + row_idx * stride_outm + offset * stride_outn
        tl.store(output_chunk_ptrs, quantized_chunk, mask=mask)

@triton.jit
def quantize_per_column_kernel(
    x_ptr, output_ptr, scale_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_outm, stride_outn,
    stride_scalem,
    BLOCK_SIZE_M: tl.constexpr = 1024, # 使用固定的处理块大小
):
    col_idx = tl.program_id(axis=0)
    if col_idx >= N:
        return

    col_amax = 0.0
    col_start_ptr = x_ptr + col_idx * stride_xn

    # 1. 第一个Pass: 找到整列的amax
    for m_offset in range(0, tl.cdiv(M, BLOCK_SIZE_M)):
        offset = m_offset * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        mask = offset < M
        current_chunk_ptrs = col_start_ptr + offset * stride_xm
        current_chunk = tl.load(current_chunk_ptrs, mask=mask, other=0.0)
        current_max = tl.max(tl.abs(current_chunk))
        col_amax = tl.maximum(col_amax, current_max)

    # 2. 计算scale
    scale = col_amax / 127.0
    scale = tl.where(scale == 0, 1e-6, scale)
    inv_scale = 1.0 / scale
    tl.store(scale_ptr + col_idx * stride_scalem, scale)

    # 3. 第二个Pass: 量化并写回
    for m_offset in range(0, tl.cdiv(M, BLOCK_SIZE_M)):
        offset = m_offset * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        mask = offset < M
        current_chunk_ptrs = col_start_ptr + offset * stride_xm
        current_chunk = tl.load(current_chunk_ptrs, mask=mask, other=0.0)
        
        quantized_chunk = tl.floor(current_chunk * inv_scale + 0.5).to(tl.int8)
        
        output_chunk_ptrs = output_ptr + offset * stride_outm + col_idx * stride_outn
        tl.store(output_chunk_ptrs, quantized_chunk, mask=mask)

@triton.autotune(
    configs=[
        # --- 基础配置 (从一些常见的开始) ---
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_stages': 2, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_stages': 2, 'num_warps': 8}),
        
        # --- 增加 num_stages 来隐藏延迟 ---
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 4}),

        # --- 增加 BLOCK_SIZE_K (如果共享内存允许) ---
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_stages': 3, 'num_warps': 8}),
        
        # --- 减少 num_warps (适用于占用率可能成为瓶颈的情况) ---
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 2}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 2}),

        # --- 一些在Volta/Ampere/Ada架构上被证明高效的“黄金尺寸” ---
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 4}),
        
        # --- 添加一个更大的M块配置 ---
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_stages': 3, 'num_warps': 8}),
    ],
    key=['M', 'N', 'K'],
)

@triton.jit
def fused_matmul_dequant_kernel(
    a_ptr, b_ptr, c_ptr,
    scale_a_ptr, scale_b_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    stride_scale_am, stride_scale_bn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_m_dequant = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n_dequant = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    scale_a_ptrs = scale_a_ptr + offs_m_dequant * stride_scale_am
    scale_b_ptrs = scale_b_ptr + offs_n_dequant * stride_scale_bn
    
    scale_a = tl.load(scale_a_ptrs, mask=offs_m_dequant < M, other=0.0)
    scale_b = tl.load(scale_b_ptrs, mask=offs_n_dequant < N, other=0.0)
    
    dequant_scale = scale_a[:, None] * scale_b[None, :]
    output = accumulator.to(tl.float32) * dequant_scale
    
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, output.to(c_ptr.dtype.element_ty), mask=c_mask)

# ==============================================================================
# Main Public Function
# ==============================================================================

def quant_matmul(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Performs a high-performance quantized matrix multiplication.
    This function implements a multi-kernel pipeline for dynamic quantization,
    which is optimized to be faster than torch.matmul on compatible hardware.

    Args:
        x (torch.Tensor): The input activation tensor. Assumed to be 2D [M, K] or 3D [B, S, K].
                          Quantization is performed per-row (per-token).
        weight (torch.Tensor): The weight tensor, shape [K, N].
                               Quantization is performed per-column (per-output-channel).

    Returns:
        torch.Tensor: The result of the matrix multiplication, in the same dtype as x.
    """
    # --- Input Validation and Reshaping ---
    x_is_3d = x.dim() == 3
    if x_is_3d:
        bs, seq_len, input_dim = x.shape
        x_2d = x.reshape(-1, input_dim)
    else:
        x_2d = x

    M, K = x_2d.shape
    K_w, N = weight.shape
    
    assert K == K_w, f"Matrix dimension mismatch: x.shape[-1] ({K}) != weight.shape[0] ({K_w})"
    assert x.device == weight.device, "Input tensors must be on the same device"
    assert x.is_cuda, "Input tensors must be on a CUDA device"

    # --- Stage 1: Quantization ---
    x_quantized = torch.empty_like(x_2d, dtype=torch.int8)
    x_scales = torch.empty((M, 1), device=x.device, dtype=torch.float32)
    weight_quantized = torch.empty_like(weight, dtype=torch.int8)
    weight_scales = torch.empty((1, N), device=weight.device, dtype=torch.float32)
    
    # It is safe to assume K is a power of 2 for performance kernels
    # In a real library, you might add padding logic here if K is not a power of 2
    grid_x = (M,)
    quantize_per_row_kernel[grid_x](
        x_2d, x_quantized, x_scales, M, K, 
        x_2d.stride(0), x_2d.stride(1), 
        x_quantized.stride(0), x_quantized.stride(1), 
        x_scales.stride(0)
    )

    grid_w = (N,)
    quantize_per_column_kernel[grid_w](
        weight, weight_quantized, weight_scales.T, K, N, 
        weight.stride(0), weight.stride(1), 
        weight_quantized.stride(0), weight_quantized.stride(1), 
        weight_scales.T.stride(0)
    )

    # --- Stage 2: Fused Matmul + Dequantization ---
    output = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid_matmul = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    
    fused_matmul_dequant_kernel[grid_matmul](
        x_quantized, weight_quantized, output,
        x_scales, weight_scales.T,
        M, N, K,
        x_quantized.stride(0), x_quantized.stride(1),
        weight_quantized.stride(0), weight_quantized.stride(1),
        output.stride(0), output.stride(1),
        x_scales.stride(0), weight_scales.T.stride(0),
    )
    
    if x_is_3d:
        return output.reshape(bs, seq_len, N)
    else:
        return output
