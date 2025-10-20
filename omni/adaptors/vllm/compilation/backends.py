from typing import Callable

import torch.fx as fx

class OmniBackend:
    def __init__(self, vllm_config):
        self.vllm_config = vllm_config

    def __call__(self, graph: fx.GraphModule, example_inputs) -> Callable:
        raise NotImplementedError("current not supported")