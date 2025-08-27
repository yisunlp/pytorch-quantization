import torch
from .quant_kernels import quant_by_row
from .ForwardGEMM import ForwardGEMM

class QuantLinearFunction(torch.autograd.Function):
    """
    Defines a custom linear operation for Quantization-Aware Training (QAT).

    The forward pass simulates INT8 quantization on the input and weight tensors
    before matrix multiplication. This helps the model adapt to precision loss.

    The backward pass implements a Straight-Through Estimator (STE). It calculates
    gradients using the original full-precision tensors, allowing for stable
    training by ignoring the non-differentiable quantization step.
    """
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        
        bs, sq, in_dim = input.shape
        original_dtype = input.dtype
        input_reshaped = input.reshape(bs * sq, in_dim)

        x_quant, x_scale = quant_by_row(input_reshaped)
        w_quant, w_scale = quant_by_row(weight)

        output = ForwardGEMM(x_quant, x_scale, w_quant, w_scale, original_dtype)
        
        if bias is not None:
            output += bias
        
        out_features = weight.shape[0]
        return output.reshape(bs, sq, out_features)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        bs, sq, out_features = grad_output.shape
        in_dim = input.shape[-1]
        
        grad_output_reshaped = grad_output.reshape(bs * sq, out_features)
        input_reshaped = input.reshape(bs * sq, in_dim)

        if ctx.needs_input_grad[0]:
            grad_input = grad_output_reshaped.matmul(weight)
            grad_input = grad_input.reshape(bs, sq, in_dim)
            
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output_reshaped.t().matmul(input_reshaped)
            
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output_reshaped.sum(dim=0)

        return grad_input, grad_weight, grad_bias