#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


"""Quantized Linear"""
import torch
from torch import nn
from torch.nn import functional as F

from pytorch_quantization import tensor_quant
from .LinearKernels.LinearFunction import QuantizedLinearFunctionWithFullBackward

from . import _utils

__all__ = ["Linear", "QuantLinear"]

class QuantLinear(nn.Linear, _utils.QuantMixin):
    """Quantized version of nn.Linear

    Apply quantized linear to the incoming data, y = dequant(quant(x)quant(A)^T + b).

    Keep Module name "Linear" instead of "QuantLinear" so that it can be easily dropped into preexisting model and load
    pretrained weights. An alias "QuantLinear" is defined below. The base code is a copy of nn.Linear, see detailed
    comment of original arguments there.

    Quantization descriptors are passed in in kwargs. If not presents, default_quant_desc_input and
    default_quant_desc_weight are used.

    Keyword Arguments:
        quant_desc_input: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`.
            Quantization descriptor of input.
        quant_desc_wegiht: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`.
            Quantization descriptor of weight.

    Raises:
        ValueError: If unsupported arguments are passed in.
        KeyError: If unsupported kwargs are passed in.

    Readonly properties:
        - input_quantizer:
        - weight_quantizer:

    Static methods:
        - set_default_quant_desc_input: Set default_quant_desc_input
        - set_default_quant_desc_weight: Set default_quant_desc_weight
    """

    default_quant_desc_input = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_LINEAR_WEIGHT_PER_ROW

    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super(QuantLinear, self).__init__(in_features, out_features, bias)
        quant_desc_input, quant_desc_weight = _utils.pop_quant_desc_in_kwargs(self.__class__, **kwargs)
        self.dynamic_input = quant_desc_input.dynamic_input

        self.init_quantizer(quant_desc_input, quant_desc_weight)

    def forward(self, input):
        if not self.training and self.dynamic_input:
            input_abs_max = torch.max(torch.abs(input), dim=-1, keepdim=True)[0]
            input_scale = (input_abs_max / 127.0).clamp(min=1e-8)
            input = input / input_scale
            # Scale the weight to match the input scale, now default True, need to set dynamic_input=True for Linear
            if True: # Now weight need to scale dynamically because the weight btq and TRT refit operation
                weight_abs_max = torch.max(torch.abs(self.weight), dim=-1, keepdim=True)[0]
                weight_scale = (weight_abs_max / 127.0).clamp(min=1e-8)
                weight = self.weight / weight_scale

            quant_input = self._input_quantizer(input)
            quant_weight = self._weight_quantizer(weight)

            output = F.linear(quant_input, quant_weight)
            output = output * input_scale
            if self.bias is not None:
                output += self.bias.unsqueeze(0)
        else:
            # Defultly, use our true quantization and Linear function to accelerate the training
            # Instead of using the torch.nn.functional.linear with bf16/fp16, we use FP8/INT8 kernel for GEMM
            # Initial code:
            # quant_input = self._input_quantizer(input)
            # quant_weight = self._weight_quantizer(self.weight)
            # output = F.linear(quant_input, quant_weight, bias=self.bias)
            
            # During inference, please specify dynamic_input=True because we have deleted the code for the initial static quantization
            
            # New code:
            output = QuantizedLinearFunctionWithFullBackward.apply(input, self.weight, self.bias)

        return output


Linear = QuantLinear
