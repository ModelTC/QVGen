import gc
from typing import Any, Dict, Union

import torch
from accelerate.logging import get_logger


logger = get_logger("qvgen")


def get_memory_statistics(precision: int = 3) -> Dict[str, Any]:
    """Collect device memory statistics in GiB where available.

    Args:
        precision: Decimal precision for rounding GiB values.

    Returns:
        Dict with allocated/reserved/max memory values (GiB).
    """
    memory_allocated = None
    memory_reserved = None
    max_memory_allocated = None
    max_memory_reserved = None

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        memory_allocated = torch.cuda.memory_allocated(device)
        memory_reserved = torch.cuda.memory_reserved(device)
        max_memory_allocated = torch.cuda.max_memory_allocated(device)
        max_memory_reserved = torch.cuda.max_memory_reserved(device)

    elif torch.backends.mps.is_available():
        memory_allocated = torch.mps.current_allocated_memory()

    else:
        logger.warning(
            "No CUDA, MPS, or ROCm device found. Memory statistics are not available."
        )

    return {
        "memory_allocated": round(
            bytes_to_gigabytes(memory_allocated), ndigits=precision
        ),
        "memory_reserved": round(
            bytes_to_gigabytes(memory_reserved), ndigits=precision
        ),
        "max_memory_allocated": round(
            bytes_to_gigabytes(max_memory_allocated), ndigits=precision
        ),
        "max_memory_reserved": round(
            bytes_to_gigabytes(max_memory_reserved), ndigits=precision
        ),
    }


def bytes_to_gigabytes(x: int) -> float:
    """Convert bytes to GiB.

    Args:
        x: Size in bytes.

    Returns:
        Size in GiB, or None if input is None.
    """
    if x is not None:
        return x / 1024**3


def free_memory() -> None:
    """Free cached CUDA memory and run garbage collection."""
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    #  handle non-cuda devices


def make_contiguous(
    x: Union[torch.Tensor, Dict[str, torch.Tensor]]
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """Ensure tensors or tensor dicts are contiguous in memory.

    Args:
        x: Tensor or dict of tensors.

    Returns:
        Contiguous tensor or dict of tensors; non-tensors are returned as-is.
    """
    if isinstance(x, torch.Tensor):
        return x.contiguous()
    elif isinstance(x, dict):
        return {k: make_contiguous(v) for k, v in x.items()}
    else:
        return x
