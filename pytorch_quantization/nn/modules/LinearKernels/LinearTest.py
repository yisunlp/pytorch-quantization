# ==============================================================================
# 文件名: LinearFunction.py
#
# 重要提示：
# 这是一个占位文件 (placeholder)。
# 请用您自己包含真实自定义 Linear 层实现的文件替换此文件。
# 为了使基准测试脚本能够运行，请确保您的类名与 `benchmark_linear.py`
# 中的导入名称一致，并且它们具有与 torch.nn.Linear 相似的接口。
#
# ==============================================================================

import torch
import torch.nn as nn
from LinearFunction import QuantizedLinearFunction, QuantizedLinearFunctionWithFullBackward
# 假设这是您的第一个自定义 Linear 实现
# 它可能是一个使用 Triton 内核的量化 Linear 层，例如 W8A16
class CustomLinearA(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        # 这里的实现应该替换为您自己的代码
        # 为了演示，我们暂时只包装一个标准的 nn.Linear
        self.linear = nn.Linear(in_features, out_features, bias)
        print("警告: 正在使用 CustomLinearA 的占位符实现。")

    def forward(self, x):
        # 您的前向传播函数，可能调用一个自定义的 aotriton 内核
        return QuantizedLinearFunction.apply(x, self.linear.weight, self.linear.bias)

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

# 假设这是您的第二个自定义 Linear 实现
class CustomLinearB(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        # 这里的实现应该替换为您自己的代码
        self.linear = nn.Linear(in_features, out_features, bias)
        print("警告: 正在使用 CustomLinearB 的占位符实现。")

    def forward(self, x):
        return QuantizedLinearFunctionWithFullBackward.apply(x, self.linear.weight,self.linear.bias)

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias
    
# ==============================================================================
# 文件名: benchmark_linear.py
#
# 描述:
#   此脚本用于比较 torch.nn.Linear 与 `LinearFunction.py` 中定义的两个
#   自定义 Linear 层的速度和精度。
#
#   - 精度测试: 比较前向输出和反向传播梯度的差异。
#   - 速度测试: 使用 triton.testing.do_bench 测量前向和反向的耗时。
#
# ==============================================================================

import torch
import torch.nn as nn
import triton
import triton.language as tl
import warnings


# --- 测试参数 ---
BATCH_SIZE = 128
SEQ_LEN = 1024
INPUT_DIM = 4096
OUTPUT_DIM = 4096
# 使用 float16 进行测试，因为大多数自定义 Triton 内核都是为它优化的
DTYPE = torch.float16
DEVICE = 'cuda'


def print_header(title):
    print("\n" + "="*80)
    print(f"| {title:^76} |")
    print("="*80)

def compare_tensors(ref_tensor, other_tensor, name):
    """比较两个张量的精度并打印结果"""
    max_abs_err = (ref_tensor - other_tensor).abs().max().item()
    mse = (ref_tensor - other_tensor).pow(2).mean().item()
    
    # torch.allclose 对于不同的 PyTorch 版本可能有不同的默认 atol/rtol
    # 这里我们使用一个相对宽松的容忍度，因为自定义内核（特别是量化内核）会有数值差异
    is_close = torch.allclose(ref_tensor, other_tensor, atol=1e-2, rtol=1e-2)
    
    print(f"{name:<25} | Allclose: {str(is_close):<10} | Max Abs Error: {max_abs_err:<18.6e} | MSE: {mse:<18.6e}")

def run_benchmark():
    """执行精度和速度的基准测试"""

    if not torch.cuda.is_available():
        print("错误: CUDA 设备不可用。此基准测试需要 GPU。")
        return

    # --- 1. 初始化模型和数据 ---
    print_header("初始化模型和数据")
    print(f"测试配置: BS={BATCH_SIZE}, SeqLen={SEQ_LEN}, In={INPUT_DIM}, Out={OUTPUT_DIM}, Dtype={DTYPE}")

    # 创建输入数据
    input_tensor = torch.randn(
        (BATCH_SIZE, SEQ_LEN, INPUT_DIM),
        device=DEVICE,
        dtype=DTYPE
    )
    # 用于计算梯度的上游梯度
    grad_output = torch.randn(
        (BATCH_SIZE, SEQ_LEN, OUTPUT_DIM),
        device=DEVICE,
        dtype=DTYPE
    )

    # 初始化 PyTorch 原生 Linear
    torch_linear = nn.Linear(INPUT_DIM, OUTPUT_DIM, bias=False).to(DEVICE, dtype=DTYPE)

    # 初始化自定义 Linear 层
    try:
        custom_linear_a = CustomLinearA(INPUT_DIM, OUTPUT_DIM, bias=False).to(DEVICE, dtype=DTYPE)
        custom_linear_b = CustomLinearB(INPUT_DIM, OUTPUT_DIM, bias=False).to(DEVICE, dtype=DTYPE)
    except Exception as e:
        print(f"初始化自定义层时出错: {e}")
        print("请检查您的 CustomLinearA 和 CustomLinearB 实现。")
        return

    # 关键步骤：确保所有模型使用完全相同的权重，以便公平比较
    with torch.no_grad():
        # 如果您的自定义层使用不同的权重格式（例如量化），您需要修改这里
        # 假设它们有一个 `.weight` 属性可以被直接赋值
        if hasattr(custom_linear_a, 'weight'):
            custom_linear_a.weight.copy_(torch_linear.weight)
        else:
             warnings.warn("CustomLinearA 没有 .weight 属性，无法同步权重。")

        if hasattr(custom_linear_b, 'weight'):
            custom_linear_b.weight.copy_(torch_linear.weight)
        else:
             warnings.warn("CustomLinearB 没有 .weight 属性，无法同步权重。")

    models = {
        "torch.nn.Linear": torch_linear,
        "CustomLinearA": custom_linear_a,
        "CustomLinearB": custom_linear_b,
    }

    # --- 2. 精度测试 ---
    print_header("精度测试 (Precision Test)")
    
    # 准备存储结果
    outputs = {}
    input_grads = {}
    weight_grads = {}

    for name, model in models.items():
        # 为每个模型创建新的输入张量，以避免梯度污染
        input_clone = input_tensor.clone().requires_grad_()
        
        # 前向传播
        output = model(input_clone)
        outputs[name] = output
        
        # 反向传播
        output.backward(gradient=grad_output)
        input_grads[name] = input_clone.grad
        weight_grads[name] = model.weight.grad
    
    # 比较结果
    ref_name = "torch.nn.Linear"
    print(f"{'Comparison Target':<25} | {'Status':<10} | {'Max Absolute Error':<20} | {'Mean Squared Error':<20}")
    print("-" * 80)
    for name in models:
        if name == ref_name:
            continue
        print(f"--- 比较: {name} ---")
        compare_tensors(outputs[ref_name], outputs[name], "Forward Output")
        compare_tensors(input_grads[ref_name], input_grads[name], "Input Gradient (d_input)")
        compare_tensors(weight_grads[ref_name], weight_grads[name], "Weight Gradient (d_weight)")

    # --- 3. 速度测试 ---
    print_header("速度测试 (Performance Benchmark)")
    
    forward_times = {}
    backward_times = {}

    for name, model in models.items():
        # 准备用于 benchmark 的函数
        input_fresh = torch.randn_like(input_tensor)

        # Benchmark 前向传播
        fw_fn = lambda: model(input_fresh)
        ms_forward = triton.testing.do_bench(fw_fn)
        forward_times[name] = ms_forward

        # Benchmark 前向+反向传播
        bw_fn = lambda: model(input_fresh).backward(gradient=grad_output, retain_graph=True)
        ms_backward = triton.testing.do_bench(bw_fn)
        backward_times[name] = ms_backward

    # 打印速度结果
    ref_time_fw = forward_times[ref_name]
    ref_time_bw = backward_times[ref_name]

    print(f"{'Model Name':<25} | {'Forward (ms)':<15} | {'Speedup vs Torch':<20} | {'Fw+Bw (ms)':<15} | {'Speedup vs Torch':<20}")
    print("-" * 100)
    for name in models:
        fw_time = forward_times[name]
        bw_time = backward_times[name]
        fw_speedup = ref_time_fw / fw_time if fw_time > 0 else float('inf')
        bw_speedup = ref_time_bw / bw_time if bw_time > 0 else float('inf')
        print(f"{name:<25} | {fw_time:<15.4f} | {f'{fw_speedup:.2f}x':<20} | {bw_time:<15.4f} | {f'{bw_speedup:.2f}x':<20}")

    print("="*100)


if __name__ == "__main__":
    run_benchmark()