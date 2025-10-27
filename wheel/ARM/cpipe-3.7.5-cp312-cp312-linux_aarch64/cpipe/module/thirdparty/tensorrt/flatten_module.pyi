import torch.nn as nn
from _typeshed import Incomplete

class Unflatten(nn.Module):
    module: Incomplete
    input_flattener: Incomplete
    output_flattener: Incomplete
    def __init__(self, module, input_flattener=None, output_flattener=None) -> None: ...
    def forward(self, *args): ...

class Flatten(nn.Module):
    module: Incomplete
    input_flattener: Incomplete
    output_flattener: Incomplete
    def __init__(self, module, input_flattener=None, output_flattener=None) -> None: ...
    def forward(self, *args): ...
