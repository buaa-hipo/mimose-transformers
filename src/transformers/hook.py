import torch

tensor_map = {}
copy_stream = torch.cuda.Stream()

def copy_tensor_async(dst, src):
    with torch.cuda.stream(copy_stream):
        dst.data = src

# For each tensor in outputs run the forward_funciton and register backward_function as hook
def _apply_forward_and_backward_to_tensors_only(
    module, forward_function, backward_function, outputs
):
    if type(outputs) is tuple:
        touched_outputs = []
        for output in outputs:
            touched_output = _apply_forward_and_backward_to_tensors_only(
                module, forward_function, backward_function, output
            )
            touched_outputs.append(touched_output)
        return tuple(touched_outputs)
    elif type(outputs) is torch.Tensor:
        forward_function(outputs)
        if outputs.requires_grad:
            outputs.register_hook(backward_function)
        else:
            # logger.debug("output dose not require grad {outputs}")
            pass
        return outputs
    else:
        return outputs


# apply torch.autograd.Function that calls a backward_function to tensors in output
def _apply_to_tensors_only(module, functional, backward_function, outputs):
    if type(outputs) is tuple:
        touched_outputs = []
        for output in outputs:
            touched_output = _apply_to_tensors_only(
                module, functional, backward_function, output
            )
            touched_outputs.append(touched_output)
        return tuple(touched_outputs)
    elif type(outputs) is torch.Tensor:
        # logger.debug(f'_apply_to_tensors_only {module}')
        return functional.apply(module, backward_function, outputs)
    else:
        # print('_apply_to_tensors_only', outputs)
        return outputs


class PreBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module, pre_backward_function, outputs):
        ctx.module = module
        ctx.pre_backward_function = pre_backward_function
        module.applied_pre_backward = False
        # logger.debug(f"**After Forward: {ctx.module.__class__.__name__}")
        # print(f"**After Forward: {ctx.module.__class__.__name__}", flush=True)
        # TODO(jiaruifang) Why detach?
        outputs = outputs.detach()
        return outputs

    @staticmethod
    def backward(ctx, *args):
        # logger.debug(f"**Before Backward: {ctx.module.__class__.__name__}")
        # print(f"**Before Backward: {ctx.module.__class__.__name__}", flush=True)
        ctx.pre_backward_function(ctx.module)
        return (None, None) + args


class PostBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module, pre_backward_function, output):
        ctx.module = module
        output = output.detach()
        # logger.debug(f"**PostBackwardFunction forward: {ctx.module.__class__.__name__}")
        # print(f"**PostBackwardFunction forward: {ctx.module.__class__.__name__}", flush=True)
        ctx.pre_backward_function = pre_backward_function
        return output

    @staticmethod
    def backward(ctx, *args):
        """
        Args:
            activation_grad of the next layer.
        Returns:
            grad of the input activation.
        """
        # logger.debug(
        #     f"**PostBackwardFunction backward: {ctx.module.__class__.__name__}"
        # )
        # print(f"**PostBackwardFunction backward: {ctx.module.__class__.__name__}", flush=True)
        ctx.pre_backward_function(ctx.module)
        return (None, None) + args


# Need to be idempotent.
def pre_sub_module_forward_function(sub_module, client, name):
    for sub_name, param in list(sub_module.named_parameters(recurse=False)):
        if param.data_ptr() != 0:
            tensor_map[param] = param.clone()
            # param.data = param.cuda()
            copy_tensor_async(param, param.cuda())
        else:
            # for checkpoint: recompute
            # param.data = getattr(sub_module, sub_name + "_cpu").cuda()
            # param.data = tensor_map[param].cuda()
            copy_tensor_async(param, tensor_map[param].cuda())

            

    for sub_name, buffer in list(sub_module.named_buffers(recurse=False)):
        if buffer.data_ptr() != 0:
            tensor_map[buffer] = buffer.clone()
            # buffer.data = buffer.cuda()
            copy_tensor_async(buffer, buffer.cuda())
        else:
            # buffer.data = getattr(sub_module, sub_name + "_cpu").cuda()
            # buffer.data = tensor_map[buffer].cuda()
            copy_tensor_async(buffer, tensor_map[buffer].cuda())



# release submodule
def post_sub_module_forward_function(sub_module, client, name):
    for sub_name, param in sub_module.named_parameters(recurse=False):
        param.data = torch.tensor([], dtype=param.dtype, device=param.device)
    
    for sub_name, buffer in list(sub_module.named_buffers(recurse=False)):
        buffer.data = torch.tensor([], dtype=buffer.dtype, device=buffer.device)



def pre_sub_module_backward_function(sub_module, client, name):
    for sub_name, param in sub_module.named_parameters(recurse=False):
        # param.data = tensor_map[param].cuda()
        copy_tensor_async(param, tensor_map[param].cuda())
    
    for sub_name, buffer in list(sub_module.named_buffers(recurse=False)):
        # buffer.data = tensor_map[buffer].cuda()
        copy_tensor_async(buffer, tensor_map[buffer].cuda())


def post_sub_module_backward_function(sub_module, client, name):
    # import pdb; pdb.set_trace()
    for sub_name, param in sub_module.named_parameters(recurse=False):
        param.data = tensor_map[param].clone()
        param.grad = param.grad.cpu()
    
    for sub_name, buffer in list(sub_module.named_buffers(recurse=False)):
        buffer.data = torch.tensor([], dtype=buffer.dtype, device=buffer.device)


def _register_hooks_recursively(module, client, name=""):
    r"""Register hook in post order traverse."""

    for child_name, child in module.named_children():
        _register_hooks_recursively(child, client, name + child_name)

    # Early return on modules with no parameters or buffers that
    # are not in their children.
    if (
        len(list(module.named_parameters(recurse=False))) == 0
        and len(list(module.named_buffers(recurse=False))) == 0
    ):
        return

    def _pre_forward_module_hook(module, *args):
        pre_sub_module_forward_function(module, client, name)

    def _post_forward_module_hook(module, *args):
        post_sub_module_forward_function(module, client, name)

    # The hook can modify the output
    def _pre_backward_module_hook(module, inputs, output):
        def _run_before_backward_function(sub_module):
            pre_sub_module_backward_function(sub_module, client, name)

        return _apply_to_tensors_only(
            module, PreBackwardFunction, _run_before_backward_function, output
        )

    def _post_backward_module_hook(module, inputs):
        def _run_after_backward_function(sub_module):
            post_sub_module_backward_function(sub_module, client, name)

        return _apply_to_tensors_only(
            module, PostBackwardFunction, _run_after_backward_function, inputs
        )

    module.register_forward_pre_hook(_pre_forward_module_hook)
    module.register_forward_hook(_post_forward_module_hook)

    module.register_forward_hook(_pre_backward_module_hook)
    module.register_forward_pre_hook(_post_backward_module_hook)


def setup_hooks(module, name=""):
    _register_hooks_recursively(module, None, name)
