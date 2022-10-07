from typing import DefaultDict
import torch, json
from torch import nn
from tqdm import tqdm
from transformers import Trainer
from transformers.trainer import logger
from transformers.manager import cast_forward, Manager

class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)

def print_shape(inputs):
    for k, v in inputs.items():
        print(f"{k}: {v.shape}")


class CountShape(Trainer):
    def __init__(self, *args, **kwargs):
        training_args = kwargs["args"]
        memory_threshold = training_args.memory_threshold
        if memory_threshold > 3:
                torch.cuda.set_per_process_memory_fraction(memory_threshold * (1024 ** 3) / torch.cuda.get_device_properties(0).total_memory)
        if kwargs["args"].dynamic_checkpoint:
            warmup_iters = training_args.warmup_iters
            self.dc_manager = Manager(warmup_iters=warmup_iters)
            self.dc_manager.set_max_memory_GB(memory_threshold=memory_threshold-0.8)
            self.dc_manager.static_strategy = training_args.static_checkpoint
            self.dc_manager.max_input = training_args.max_input_size
            self.dc_manager.min_input = training_args.min_input_size
            cast_forward(kwargs["model"].bert.encoder, "0", self.dc_manager)
        super().__init__(*args, **kwargs)
        self.input_shape = DefaultDict(int)
        self.memory_collect = {}
        self.shape_order = []
        self.profile_memory = kwargs.get("profile_memory", False)


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
        seq_length = inputs['input_ids'].shape[-1]
        self.input_shape[seq_length] += 1
        self.shape_order.append(seq_length)

        if self.profile_memory:
            torch.cuda.empty_cache()
        torch.cuda.memory.reset_peak_memory_stats()
        if self.args.dynamic_checkpoint:
            self.dc_manager.set_input_size(seq_length)
        ret = super().training_step(model, inputs)

        # if self.profile_memory:
        if seq_length not in self.memory_collect:
            self.memory_collect[seq_length] = []
        self.memory_collect[seq_length].append(torch.cuda.max_memory_allocated())

        if self.args.dynamic_checkpoint:
            self.dc_manager.after_update()
        return ret
    
    def count_input_size(self):
        train_dataloader = self.get_train_dataloader()
        for inputs in tqdm(train_dataloader):
            seq_length = inputs['input_ids'].shape[-1]
            self.input_shape[seq_length] += 1
            self.shape_order.append(seq_length)
        logger.info("shape_count=" + json.dumps(self.input_shape))
        logger.info("memory_count=" + json.dumps(self.memory_collect))
        logger.info("shape_order=" + json.dumps(self.shape_order))
        exit(0)


    def train(self, *args, **kwargs):
        if hasattr(self.args, "only_input_size") and self.args.only_input_size:
            self.count_input_size()

        ret = super().train(*args, **kwargs)
        if len(self.memory_collect) == 0:
            self.memory_collect[-1] = [torch.cuda.max_memory_allcated()]
        logger.info("shape_count=" + json.dumps(self.input_shape))
        logger.info("memory_count=" + json.dumps(self.memory_collect))
        logger.info("shape_order=" + json.dumps(self.shape_order))
        if self.args.dynamic_checkpoint:
            logger.info("strategy: " + json.dumps(self.dc_manager.cached_strategy, cls=SetEncoder))
        return ret
    
    
    def _save_checkpoint(self, model, trial, metrics=None):
        pass
