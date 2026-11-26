"""Equivalent small-channel Conv1d implemented through explicit windows and GEMM.

Useful on older GPUs where deterministic dilated cuDNN convolutions are slow.
Only the stride-one, groups-one, zero-padding convolutions in Leo's CNN apply.
"""

import torch.nn.functional as F
from torch import nn


class LinearConv1d(nn.Conv1d):
    def forward(self, x):
        if self.stride != (1,) or self.groups != 1 or self.padding_mode != "zeros":
            raise ValueError("Only stride=1, groups=1 and zero padding are supported")
        kernel, dilation, padding = (
            self.kernel_size[0],
            self.dilation[0],
            self.padding[0],
        )
        if padding:
            x = F.pad(x, (padding, padding))
        windows = x.unfold(2, dilation * (kernel - 1) + 1, 1)[..., ::dilation]
        windows = windows.permute(0, 2, 1, 3).reshape(x.shape[0], windows.shape[2], -1)
        y = F.linear(windows, self.weight.reshape(self.out_channels, -1), self.bias)
        return y.transpose(1, 2).contiguous()


def use_linear_convolutions(model):
    for name, layer in list(model.named_children()):
        if isinstance(layer, nn.Conv1d):
            replacement = LinearConv1d(
                layer.in_channels,
                layer.out_channels,
                layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
                dilation=layer.dilation,
                groups=layer.groups,
                bias=layer.bias is not None,
                device=layer.weight.device,
                dtype=layer.weight.dtype,
            )
            replacement.load_state_dict(layer.state_dict())
            setattr(model, name, replacement)
        else:
            use_linear_convolutions(layer)
    return model
