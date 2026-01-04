import random
import os
import torch
import numpy as np

class Model:
    def __init__(self, config, device):
        # Allow flexible config structure (support both dict access and attribute)
        if isinstance(config, dict):
             self.provider = config.get("model_info", {}).get("provider", "unknown")
             self.name = config.get("model_info", {}).get("name", "unknown")
             params = config.get("params", {})
             self.seed = int(params.get("seed", 42))
             self.temperature = float(params.get("temperature", 0.0))
             self.gpus = [str(g) for g in params.get("gpus", [])]
        else:
            # Fallback or different config object
             self.provider = "unknown"
             self.name = "unknown"
             self.seed = 42
             self.temperature = 0.0
             self.gpus = []

        self.device = device

    def print_model_info(self):
        print(f"{'-'*len(f'| Model name: {self.name}')}\n| Provider: {self.provider}\n| Model name: {self.name}\n{'-'*len(f'| Model name: {self.name}')}")

    def set_API_key(self):
        raise NotImplementedError("ERROR: Interface doesn't have the implementation for set_API_key")
    
    def query(self, msg, top_tokens=100000):
        raise NotImplementedError("ERROR: Interface doesn't have the implementation for query")
    
    def initialize_seed(self):
        random.seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # if you are using multi-GPU.
        if len(self.gpus) > 1:
             torch.cuda.manual_seed_all(self.seed)
    
    def initialize_gpus(self):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(self.gpus)
