import torch
from torch.autograd import Function
from .quant_kernels import quant_by_row
from .ForwardGEMM import ForwardGEMM
from .DGradGEMM import DGradGEMM
from .WGradGEMM import WGradGEMM

class QuantizedLinearFunction(Function):
    """
    Custom autograd Function for a linear layer with quantized forward and backward passes.
    This simulates the behavior of quantized operations on hardware for both passes.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        dtype = x.dtype
        x_quant, x_scale = quant_by_row(x)
        w_quant, w_scale = quant_by_row(weight)
    
        output = ForwardGEMM(x_quant, x_scale, w_quant, w_scale, dtype)
        if bias is not None:
            output += bias
        
        ctx.save_for_backward(x_quant, x_scale, w_quant, w_scale)
        ctx.bias_is_present = bias is not None
        ctx.dtype = dtype
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        grad_x = grad_w = grad_b = None
        x_quant, x_scale, w_quant, w_scale = ctx.saved_tensors
        dtype = ctx.dtype
        grad_output_quant, grad_output_scale = quant_by_row(grad_output)

        if ctx.needs_input_grad[0]:
            grad_x = DGradGEMM(grad_output_quant, grad_output_scale, w_quant, w_scale, dtype)

        if ctx.needs_input_grad[1]:
            bs, seq_len, input_dim = x_quant.shape
            output_dim = w_quant.shape[0]
            
            x_quant_reshaped = x_quant.reshape(bs * seq_len, input_dim)
            x_scale_reshaped = x_scale.reshape(bs * seq_len, 1)
            grad_output_quant_reshaped = grad_output_quant.reshape(bs * seq_len, output_dim)
            grad_output_scale_reshaped = grad_output_scale.reshape(bs * seq_len, 1)
            
            grad_w = WGradGEMM(grad_output_quant_reshaped, grad_output_scale_reshaped, x_quant_reshaped, x_scale_reshaped, dtype)
        
        if ctx.bias_is_present and ctx.needs_input_grad[2]:
            grad_b = grad_output.sum(dim=0)
        
        return grad_x, grad_w, grad_b


class QuantizedLinearFunctionWithFullBackward(Function):
    """
    Custom autograd Function for a linear layer with a quantized forward pass 
    and a full-precision backward pass. This is often used for stable
    Quantization-Aware Training (QAT).
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        x_quant, x_scale = quant_by_row(x)
        w_quant, w_scale = quant_by_row(weight)
        dtype = x.dtype
        output = ForwardGEMM(x_quant, x_scale, w_quant, w_scale, dtype)
        if bias is not None:
            output += bias
        
        ctx.save_for_backward(x, weight)
        ctx.bias_is_present = bias is not None

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        grad_x = grad_w = grad_b = None
        x, weight = ctx.saved_tensors

        if ctx.needs_input_grad[0]:
            grad_x = torch.matmul(grad_output, weight)

        if ctx.needs_input_grad[1]:
            x_reshaped = x.reshape(-1, x.shape[-1])
            grad_output_reshaped = grad_output.reshape(-1, grad_output.shape[-1])
            grad_w = torch.matmul(x_reshaped.T, grad_output_reshaped).T
        
        if ctx.bias_is_present and ctx.needs_input_grad[2]:
            grad_b = grad_output_reshaped.sum(dim=0)
        
        return grad_x, grad_w, grad_b
    
if __name__ == '__main__':
    # 定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 定义输入
    batch_size, seq_length, input_dim, output_dim = 64, 256, 4096, 4096
    x = torch.randn(batch_size, seq_length, input_dim, requires_grad=True, device=device, dtype=torch.bfloat16)
    weight = torch.randn(output_dim, input_dim, requires_grad=True, device=device, dtype=torch.bfloat16)
    bias = torch.randn(output_dim, requires_grad=True, device=device, dtype=torch.bfloat16)

    # 1. 测试 QuantizedLinearFunction (量化反向)
    print("--- Testing QuantizedLinearFunction (Quantized Backward) ---")
    output_quant_bwd = QuantizedLinearFunction.apply(x, weight, bias)
    loss_quant_bwd = output_quant_bwd.sum()
    loss_quant_bwd.backward()
    print("Shape of x.grad:", x.grad.shape if x.grad is not None else "None")
    print("Shape of weight.grad:", weight.grad.shape if weight.grad is not None else "None")
    print("Shape of bias.grad:", bias.grad.shape if bias.grad is not None else "None")

    # 清零梯度
    x.grad.zero_()
    weight.grad.zero_()
    bias.grad.zero_()

    print("\n" + "="*50 + "\n")

    # 2. 测试 QuantizedLinearFunctionFullBackward (全精度反向)
    print("--- Testing QuantizedLinearFunctionFullBackward (Full-Precision Backward) ---")
    output_full_bwd = QuantizedLinearFunctionWithFullBackward.apply(x, weight, bias)
    loss_full_bwd = output_full_bwd.sum()
    loss_full_bwd.backward()
    print("Shape of x.grad:", x.grad.shape if x.grad is not None else "None")
    print("Shape of weight.grad:", weight.grad.shape if weight.grad is not None else "None")
    print("Shape of bias.grad:", bias.grad.shape if bias.grad is not None else "None")
