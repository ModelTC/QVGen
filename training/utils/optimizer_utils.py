import inspect

import torch
from accelerate.logging import get_logger


logger = get_logger("qvgen")


def get_optimizer(
    params_to_optimize,
    optimizer_name: str = "adam",
    learning_rate: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.95,
    beta3: float = 0.98,
    epsilon: float = 1e-8,
    weight_decay: float = 1e-4,
    prodigy_decouple: bool = False,
    prodigy_use_bias_correction: bool = False,
    prodigy_safeguard_warmup: bool = False,
    use_8bit: bool = False,
    use_4bit: bool = False,
    use_torchao: bool = False,
    use_deepspeed: bool = False,
    use_cpu_offload_optimizer: bool = False,
    offload_gradients: bool = False,
) -> torch.optim.Optimizer:
    """Create an optimizer with optional low-bit or offloaded variants.

    Args:
        params_to_optimize: Iterable of parameters to optimize.
        optimizer_name: Optimizer name ("adam", "adamw", "prodigy", "came").
        learning_rate: Base learning rate.
        beta1: Adam beta1 coefficient.
        beta2: Adam beta2 coefficient.
        beta3: Third beta term for optimizers that use it.
        epsilon: Numerical stability epsilon.
        weight_decay: Weight decay factor.
        prodigy_decouple: Whether to decouple weight decay for Prodigy.
        prodigy_use_bias_correction: Whether Prodigy uses bias correction.
        prodigy_safeguard_warmup: Whether Prodigy safeguards warmup.
        use_8bit: Whether to use 8-bit optimizer variants.
        use_4bit: Whether to use 4-bit optimizer variants (torchao only).
        use_torchao: Whether to source optimizers from torchao.
        use_deepspeed: Whether to use Accelerate DummyOptim for DeepSpeed.
        use_cpu_offload_optimizer: Whether to offload optimizer states to CPU.
        offload_gradients: Whether to offload gradients when using CPU offload.

    Returns:
        Instantiated optimizer.
    """
    optimizer_name = optimizer_name.lower()

    # Use DeepSpeed optimzer
    if use_deepspeed:
        from accelerate.utils import DummyOptim

        return DummyOptim(
            params_to_optimize,
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=epsilon,
            weight_decay=weight_decay,
        )

    # TODO: consider moving the validation logic to `args.py` when we have torchao.
    if use_8bit and use_4bit:
        raise ValueError("Cannot set both `use_8bit` and `use_4bit` to True.")

    if (use_torchao and (use_8bit or use_4bit)) or use_cpu_offload_optimizer:
        try:
            import torchao  # noqa

        except ImportError:
            raise ImportError(
                "To use optimizers from torchao, please install the torchao library: `USE_CPP=0 pip install torchao`."
            )

    if not use_torchao and use_4bit:
        raise ValueError("4-bit Optimizers are only supported with torchao.")

    # Optimizer creation
    supported_optimizers = ["adam", "adamw", "prodigy", "came"]
    if optimizer_name not in supported_optimizers:
        logger.warning(
            f"Unsupported choice of optimizer: {optimizer_name}. Supported optimizers include {supported_optimizers}. Defaulting to `AdamW`."
        )
        optimizer_name = "adamw"

    if (use_8bit or use_4bit) and optimizer_name not in ["adam", "adamw"]:
        raise ValueError(
            "`use_8bit` and `use_4bit` can only be used with the Adam and AdamW optimizers."
        )

    if use_8bit:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

    if optimizer_name == "adamw":
        if use_torchao:
            from torchao.prototype.low_bit_optim import AdamW4bit, AdamW8bit

            optimizer_class = (
                AdamW8bit if use_8bit else AdamW4bit if use_4bit else torch.optim.AdamW
            )
        else:
            optimizer_class = bnb.optim.AdamW8bit if use_8bit else torch.optim.AdamW

        init_kwargs = {
            "betas": (beta1, beta2),
            "eps": epsilon,
            "weight_decay": weight_decay,
        }

    elif optimizer_name == "adam":
        if use_torchao:
            from torchao.prototype.low_bit_optim import Adam4bit, Adam8bit

            optimizer_class = (
                Adam8bit if use_8bit else Adam4bit if use_4bit else torch.optim.Adam
            )
        else:
            optimizer_class = bnb.optim.Adam8bit if use_8bit else torch.optim.Adam

        init_kwargs = {
            "betas": (beta1, beta2),
            "eps": epsilon,
            "weight_decay": weight_decay,
        }

    elif optimizer_name == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError(
                "To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`"
            )

        optimizer_class = prodigyopt.Prodigy

        if learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        init_kwargs = {
            "lr": learning_rate,
            "betas": (beta1, beta2),
            "beta3": beta3,
            "eps": epsilon,
            "weight_decay": weight_decay,
            "decouple": prodigy_decouple,
            "use_bias_correction": prodigy_use_bias_correction,
            "safeguard_warmup": prodigy_safeguard_warmup,
        }

    elif optimizer_name == "came":
        try:
            import came_pytorch
        except ImportError:
            raise ImportError(
                "To use CAME, please install the came-pytorch library: `pip install came-pytorch`"
            )

        optimizer_class = came_pytorch.CAME

        init_kwargs = {
            "lr": learning_rate,
            "eps": (1e-30, 1e-16),
            "betas": (beta1, beta2, beta3),
            "weight_decay": weight_decay,
        }

    if use_cpu_offload_optimizer:
        from torchao.prototype.low_bit_optim import CPUOffloadOptimizer

        if "fused" in inspect.signature(optimizer_class.__init__).parameters:
            init_kwargs.update({"fused": True})

        optimizer = CPUOffloadOptimizer(
            params_to_optimize,
            optimizer_class=optimizer_class,
            offload_gradients=offload_gradients,
            **init_kwargs,
        )
    else:
        optimizer = optimizer_class(params_to_optimize, **init_kwargs)

    return optimizer


def gradient_norm(parameters):
    """Compute the L2 norm of gradients across parameters.

    Args:
        parameters: Iterable of parameters whose gradients are inspected.

    Returns:
        L2 norm of all gradients.
    """
    norm = 0
    for param in parameters:
        if param.grad is None:
            continue
        local_norm = param.grad.detach().data.norm(2)
        norm += local_norm.item() ** 2
    norm = norm**0.5
    return norm


def max_gradient(parameters):
    """Compute the maximum absolute gradient across parameters.

    Args:
        parameters: Iterable of parameters whose gradients are inspected.

    Returns:
        Maximum absolute gradient value.
    """
    max_grad_value = float("-inf")
    for param in parameters:
        if param.grad is None:
            continue
        local_max_grad = param.grad.detach().data.abs().max()
        max_grad_value = max(max_grad_value, local_max_grad.item())
    return max_grad_value
