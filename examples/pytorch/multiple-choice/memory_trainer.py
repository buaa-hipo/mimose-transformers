from typing import DefaultDict
import torch, json
from torch import nn
from tqdm import tqdm
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers import Trainer
from transformers.trainer import logger
from transformers.manager import cast_forward, recover_forward, Manager, PolyPrediction
import time

class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)

def print_shape(inputs):
    for k, v in inputs.items():
        print(f"{k}: {v.shape}")

def add_hook(model, log_dict, trainer):
    old_forward = model.forward
    log_dict['memory'] = []
    
    def forward(*args, **kwargs):
        prev_memory = torch.cuda.memory_allocated()
        ret = old_forward(*args, **kwargs)
        if trainer.state.global_step > 1:
            log_dict['memory'].append(torch.cuda.memory_allocated() - prev_memory)
        return ret
    model.forward = forward
    model.old_forward = old_forward

def remove_hook(model):
    model.forward = model.old_forward
        

class MemoryPredict(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_shape = DefaultDict(int)
        self.predict_func = {}
        self.predict_memory = {'model': {'seq_length': [], 'memory': []}}
        self.real_memory = {'model': {'seq_length': [], 'memory': []}}
        self.real_memory['encoder'] = {}
        add_hook(self.model.bert.encoder.layer[0], self.real_memory['encoder'], self)


    def training_step(self, model: nn.Module, inputs) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        torch.cuda.reset_peak_memory_stats()
        prev_memory = torch.cuda.memory_allocated()
        
        seq_length = inputs['input_ids'].shape[-1]
        step = self.state.global_step
        if step > 1:
            self.real_memory['model']['seq_length'].append(seq_length)

        ret = super().training_step(model, inputs)
        if step > 1:
            self.real_memory['model']['memory'].append(torch.cuda.max_memory_allocated() - prev_memory)
        if step > 31:
            self.predict_func['model'] = PolyPrediction(self.real_memory['model']['seq_length'], self.real_memory['model']['memory'])
        if step > 32:
            self.predict_memory['model']['seq_length'].append(seq_length)
            self.predict_memory['model']['memory'].append(self.predict_func['model'](seq_length))
        # if step > 100:
        #     logger.info("real_memory=" + json.dumps(self.real_memory))
        #     logger.info("predict_memory=" + json.dumps(self.predict_memory))
        #     exit(0)
        return ret


    def train(self, *args, **kwargs):

        ret = super().train(*args, **kwargs)
        remove_hook(self.model.bert.encoder.layer[0])

        logger.info("real_memory=" + json.dumps(self.real_memory))
        logger.info("predict_memory=" + json.dumps(self.predict_memory))
        return ret
    
    
    def _save_checkpoint(self, model, trial, metrics=None):
        pass
