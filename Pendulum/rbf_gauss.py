import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor
from torch.nn.parameter import Parameter


class RBF_gaussian(nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    centers: Tensor
    log_sigmas: Tensor

    def __init__(
        self, in_features: int, out_features: int, device=None, dtype=None
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centers = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        self.log_sigmas = Parameter(torch.empty(out_features, **factory_kwargs))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.centers, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.centers)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.log_sigmas, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        size_exp = (input.size(0), self.out_features, self.in_features)
        x = input.unsqueeze(1).expand(size_exp)
        c = self.centers.unsqueeze(0).expand(size_exp)
        distances = (x - c).pow(2).sum(-1).pow(0.5) / torch.exp(
            self.log_sigmas
        ).unsqueeze(0)
        out = torch.exp(-1 * (distances.pow(2)))
        return out

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"
