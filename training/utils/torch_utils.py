from typing import Dict, Optional, Union

import torch
from accelerate import Accelerator
from diffusers.utils.torch_utils import is_compiled_module


def unwrap_model(accelerator: Accelerator, model):
    """Return the underlying model from an Accelerator wrapper.

    Args:
        accelerator: Accelerator instance managing the model.
        model: Wrapped model instance to unwrap.

    Returns:
        The unwrapped model, with compiled wrappers removed if present.
    """
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def align_device_and_dtype(
    x: Union[torch.Tensor, Dict[str, torch.Tensor]],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    """Move tensors or tensor dicts to a target device/dtype.

    Args:
        x: Tensor or nested dict of tensors to update.
        device: Target device; if None, device is unchanged.
        dtype: Target dtype; if None, dtype is unchanged.

    Returns:
        Tensor or dict with updated device/dtype.
    """
    if isinstance(x, torch.Tensor):
        if device is not None:
            x = x.to(device)
        if dtype is not None:
            x = x.to(dtype)
    elif isinstance(x, dict):
        if device is not None:
            x = {k: align_device_and_dtype(v, device, dtype) for k, v in x.items()}
        if dtype is not None:
            x = {k: align_device_and_dtype(v, device, dtype) for k, v in x.items()}
    return x


def expand_tensor_dims(tensor, ndim):
    """Unsqueeze a tensor until it reaches the requested rank.

    Args:
        tensor: Input tensor to reshape.
        ndim: Desired number of dimensions.

    Returns:
        Tensor with ``ndim`` dimensions.
    """
    while len(tensor.shape) < ndim:
        tensor = tensor.unsqueeze(-1)
    return tensor
