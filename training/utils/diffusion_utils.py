import math
from typing import Optional, Union

import torch
from diffusers import CogVideoXDDIMScheduler, FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import compute_loss_weighting_for_sd3


# Default values copied from https://github.com/huggingface/diffusers/blob/8957324363d8b239d82db4909fbf8c0875683e3d/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py#L47
def resolution_dependent_timestep_flow_shift(
    latents: torch.Tensor,
    sigmas: torch.Tensor,
    base_image_seq_len: int = 256,
    max_image_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> torch.Tensor:
    """Apply a resolution-dependent flow shift to sigmas.

    Args:
        latents: Latent tensor of shape [B, C, H, W] or [B, C, F, H, W].
        sigmas: Original sigma schedule.
        base_image_seq_len: Sequence length at which base_shift applies.
        max_image_seq_len: Sequence length at which max_shift applies.
        base_shift: Shift applied at base_image_seq_len.
        max_shift: Shift applied at max_image_seq_len.

    Returns:
        Shifted sigma schedule.
    """
    image_or_video_sequence_length = 0
    if latents.ndim == 4:
        image_or_video_sequence_length = latents.shape[2] * latents.shape[3]
    elif latents.ndim == 5:
        image_or_video_sequence_length = (
            latents.shape[2] * latents.shape[3] * latents.shape[4]
        )
    else:
        raise ValueError(f"Expected 4D or 5D tensor, got {latents.ndim}D tensor")

    m = (max_shift - base_shift) / (max_image_seq_len - base_image_seq_len)
    b = base_shift - m * base_image_seq_len
    mu = m * image_or_video_sequence_length + b
    sigmas = default_flow_shift(latents, sigmas, shift=mu)
    return sigmas


def default_flow_shift(sigmas: torch.Tensor, shift: float = 1.0) -> torch.Tensor:
    """Apply the default sigma shift transformation.

    Args:
        sigmas: Sigma schedule.
        shift: Shift magnitude.

    Returns:
        Shifted sigmas.
    """
    sigmas = (sigmas * shift) / (1 + (shift - 1) * sigmas)
    return sigmas


def compute_density_for_timestep_sampling(
    weighting_scheme: str,
    batch_size: int,
    logit_mean: float = None,
    logit_std: float = None,
    mode_scale: float = None,
    device: torch.device = torch.device("cpu"),
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    r"""
    Compute the density for sampling the timesteps when doing SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "logit_normal":
        # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
        u = torch.normal(
            mean=logit_mean,
            std=logit_std,
            size=(batch_size,),
            device=device,
            generator=generator,
        )
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device=device, generator=generator)
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size,), device=device, generator=generator)
    return u


def get_scheduler_alphas(
    scheduler: Union[CogVideoXDDIMScheduler, FlowMatchEulerDiscreteScheduler]
) -> torch.Tensor:
    """Return alphas for supported schedulers.

    Args:
        scheduler: Scheduler instance.

    Returns:
        Alpha schedule tensor, or None for flow-match scheduler.
    """
    if isinstance(scheduler, FlowMatchEulerDiscreteScheduler):
        return None
    elif isinstance(scheduler, CogVideoXDDIMScheduler):
        return scheduler.alphas_cumprod.clone()
    else:
        raise ValueError(f"Unsupported scheduler type {type(scheduler)}")


def get_scheduler_sigmas(
    scheduler: Union[CogVideoXDDIMScheduler, FlowMatchEulerDiscreteScheduler]
) -> torch.Tensor:
    """Return sigma schedule for supported schedulers.

    Args:
        scheduler: Scheduler instance.

    Returns:
        Sigma schedule tensor.
    """
    if isinstance(scheduler, FlowMatchEulerDiscreteScheduler):
        return scheduler.sigmas.clone()
    elif isinstance(scheduler, CogVideoXDDIMScheduler):
        return scheduler.timesteps.clone().float() / float(
            scheduler.config.num_train_timesteps
        )
    else:
        raise ValueError(f"Unsupported scheduler type {type(scheduler)}")


def prepare_sigmas(
    scheduler: Union[CogVideoXDDIMScheduler, FlowMatchEulerDiscreteScheduler],
    sigmas: torch.Tensor,
    batch_size: int,
    num_train_timesteps: int,
    flow_weighting_scheme: str = "none",
    flow_logit_mean: float = 0.0,
    flow_logit_std: float = 1.0,
    flow_mode_scale: float = 1.29,
    device: torch.device = torch.device("cpu"),
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample sigma values for a batch given a weighting scheme.

    Args:
        scheduler: Scheduler instance.
        sigmas: Full sigma schedule.
        batch_size: Number of samples to draw.
        num_train_timesteps: Number of training timesteps.
        flow_weighting_scheme: Scheme for timestep sampling.
        flow_logit_mean: Mean for logit-normal weighting.
        flow_logit_std: Std for logit-normal weighting.
        flow_mode_scale: Scale for mode weighting.
        device: Torch device for sampling.
        generator: Optional RNG generator.

    Returns:
        Batch of sampled sigmas.
    """
    if isinstance(scheduler, FlowMatchEulerDiscreteScheduler):
        weights = compute_density_for_timestep_sampling(
            weighting_scheme=flow_weighting_scheme,
            batch_size=batch_size,
            logit_mean=flow_logit_mean,
            logit_std=flow_logit_std,
            mode_scale=flow_mode_scale,
            device=device,
            generator=generator,
        )
        indices = (weights * num_train_timesteps).long()
    elif isinstance(scheduler, CogVideoXDDIMScheduler):
        #  Currently, only uniform sampling is supported. Add more sampling schemes.
        weights = torch.rand(size=(batch_size,), device=device, generator=generator)
        indices = (weights * num_train_timesteps).long()
    else:
        raise ValueError(f"Unsupported scheduler type {type(scheduler)}")

    return sigmas[indices]


def prepare_loss_weights(
    scheduler: Union[CogVideoXDDIMScheduler, FlowMatchEulerDiscreteScheduler],
    alphas: Optional[torch.Tensor] = None,
    sigmas: Optional[torch.Tensor] = None,
    flow_weighting_scheme: str = "none",
) -> torch.Tensor:
    """Compute loss weights for a given scheduler type.

    Args:
        scheduler: Scheduler instance.
        alphas: Alpha schedule for DDIM-based schedulers.
        sigmas: Sigma schedule for flow-match schedulers.
        flow_weighting_scheme: Weighting scheme for SD3 loss.

    Returns:
        Loss weights tensor.
    """
    if isinstance(scheduler, FlowMatchEulerDiscreteScheduler):
        return compute_loss_weighting_for_sd3(
            sigmas=sigmas, weighting_scheme=flow_weighting_scheme
        )
    elif isinstance(scheduler, CogVideoXDDIMScheduler):
        # SNR is computed as (alphas / (1 - alphas)), but for some reason CogVideoX uses 1 / (1 - alphas).
        #  Experiment if using alphas / (1 - alphas) gives better results.
        return 1 / (1 - alphas)
    else:
        raise ValueError(f"Unsupported scheduler type {type(scheduler)}")


def prepare_target(
    scheduler: Union[CogVideoXDDIMScheduler, FlowMatchEulerDiscreteScheduler],
    noise: torch.Tensor,
    latents: torch.Tensor,
) -> torch.Tensor:
    """Prepare target prediction based on scheduler type.

    Args:
        scheduler: Scheduler instance.
        noise: Sampled noise tensor.
        latents: Latents tensor.

    Returns:
        Target tensor for loss computation.
    """
    if isinstance(scheduler, FlowMatchEulerDiscreteScheduler):
        target = noise - latents
    elif isinstance(scheduler, CogVideoXDDIMScheduler):
        target = latents
    else:
        raise ValueError(f"Unsupported scheduler type {type(scheduler)}")

    return target
