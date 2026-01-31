import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from accelerate.logging import get_logger

# from accelerate import init_empty_weights
from diffusers import (
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
    # WanImageToVideoPipeline,
    WanPipeline,
    WanTransformer3DModel,
    UniPCMultistepScheduler,
)
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from PIL.Image import Image
from transformers import T5TokenizerFast, AutoTokenizer, UMT5EncoderModel
import ftfy
import html

logger = get_logger("qvgen")  # pylint: disable=invalid-name


def collate_fn_t2v(
    batch: List[List[Dict[str, torch.Tensor]]]
) -> Dict[str, torch.Tensor]:
    """Collate a batch for text-to-video training.

    Args:
        batch: Nested batch produced by the bucket sampler.

    Returns:
        Dict with prompts and stacked video tensors.
    """
    return {
        "prompts": [x["prompt"] for x in batch[0]],
        "videos": torch.stack([x["video"] for x in batch[0]]),
    }


def basic_clean(text):
    """Normalize unicode and HTML entities.

    Args:
        text: Input text string.

    Returns:
        Cleaned string.
    """
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    """Collapse repeated whitespace to single spaces.

    Args:
        text: Input text string.

    Returns:
        Cleaned string.
    """
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    """Apply basic and whitespace cleanup to a prompt.

    Args:
        text: Input prompt string.

    Returns:
        Cleaned prompt string.
    """
    text = whitespace_clean(basic_clean(text))
    return text


def load_condition_models(
    model_id: str = "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    text_encoder_dtype: torch.dtype = torch.bfloat16,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    **kwargs,
):
    """Load tokenizer and text encoder for prompt conditioning.

    Args:
        model_id: Hugging Face model ID.
        text_encoder_dtype: Dtype for text encoder.
        revision: Optional model revision.
        cache_dir: Optional cache directory.
        **kwargs: Unused additional args.

    Returns:
        Dict with ``tokenizer`` and ``text_encoder``.
    """
    tokenizer = T5TokenizerFast.from_pretrained(
        model_id, subfolder="tokenizer", revision=revision, cache_dir=cache_dir
    )
    text_encoder = UMT5EncoderModel.from_pretrained(
        model_id,
        subfolder="text_encoder",
        torch_dtype=text_encoder_dtype,
        revision=revision,
        cache_dir=cache_dir,
    )
    return {"tokenizer": tokenizer, "text_encoder": text_encoder}


def load_latent_models(
    model_id: str = "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    vae_dtype: torch.dtype = torch.float32,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    **kwargs,
):
    """Load the VAE used for latent encoding/decoding.

    Args:
        model_id: Hugging Face model ID.
        vae_dtype: Dtype for the VAE.
        revision: Optional model revision.
        cache_dir: Optional cache directory.
        **kwargs: Unused additional args.

    Returns:
        Dict with ``vae``.
    """
    vae = AutoencoderKLWan.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=vae_dtype,
        revision=revision,
        cache_dir=cache_dir,
    )
    return {"vae": vae}


def load_diffusion_models(
    model_id: str = "THUDM/CogVideoX-5b",
    transformer_dtype: torch.dtype = torch.bfloat16,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    **kwargs,
):
    """Load transformer and scheduler components for diffusion.

    Args:
        model_id: Hugging Face model ID.
        transformer_dtype: Dtype for transformer weights.
        revision: Optional model revision.
        cache_dir: Optional cache directory.
        **kwargs: Unused additional args.

    Returns:
        Dict with ``transformer`` and ``scheduler``.
    """
    transformer = WanTransformer3DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        torch_dtype=transformer_dtype,
        revision=revision,
        cache_dir=cache_dir,
    )
    scheduler = (
        FlowMatchEulerDiscreteScheduler()
    )  # need to switch to UniPCMultistepScheduler
    return {"transformer": transformer, "scheduler": scheduler}


def initialize_pipeline(
    model_id: str = "THUDM/CogVideoX-5b",
    text_encoder_dtype: torch.dtype = torch.bfloat16,
    transformer_dtype: torch.dtype = torch.bfloat16,
    vae_dtype: torch.dtype = torch.bfloat16,
    tokenizer: Optional[AutoTokenizer] = None,
    text_encoder: Optional[UMT5EncoderModel] = None,
    transformer: Optional[WanTransformer3DModel] = None,
    vae: Optional[AutoencoderKLWan] = None,
    scheduler: Optional[FlowMatchEulerDiscreteScheduler] = None,
    device: Optional[torch.device] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    enable_slicing: bool = False,
    enable_tiling: bool = False,
    enable_model_cpu_offload: bool = False,
    is_training: bool = False,
    **kwargs,
) -> WanPipeline:
    """Initialize a WanPipeline with optional component overrides.

    Args:
        model_id: Hugging Face model ID.
        text_encoder_dtype: Dtype for text encoder.
        transformer_dtype: Dtype for transformer.
        vae_dtype: Dtype for VAE.
        tokenizer: Optional tokenizer override.
        text_encoder: Optional text encoder override.
        transformer: Optional transformer override.
        vae: Optional VAE override.
        scheduler: Optional scheduler override.
        device: Device to move the pipeline to.
        revision: Optional model revision.
        cache_dir: Optional cache directory.
        enable_slicing: Whether to enable VAE slicing.
        enable_tiling: Whether to enable VAE tiling.
        enable_model_cpu_offload: Whether to offload weights to CPU.
        is_training: Whether initialization is for training.
        **kwargs: Unused additional args.

    Returns:
        Initialized WanPipeline instance.
    """
    component_name_pairs = [
        ("tokenizer", tokenizer),
        ("text_encoder", text_encoder),
        ("transformer", transformer),
        ("vae", vae),
        ("scheduler", scheduler),
    ]
    components = {}
    for name, component in component_name_pairs:
        if component is not None:
            components[name] = component

    pipe = WanPipeline.from_pretrained(
        model_id, **components, revision=revision, cache_dir=cache_dir
    )
    pipe.text_encoder = pipe.text_encoder.to(dtype=text_encoder_dtype)
    pipe.vae = pipe.vae.to(dtype=vae_dtype)

    # The transformer should already be in the correct dtype when training, so we don't need to cast it here.
    # If we cast, whilst using fp8 layerwise upcasting hooks, it will lead to an error in the training during
    # DDP optimizer step.
    if not is_training:
        pipe.transformer = pipe.transformer.to(dtype=transformer_dtype)

    # if enable_slicing:
    #     pipe.vae.enable_slicing()
    # if enable_tiling:
    #     pipe.vae.enable_tiling()

    if enable_model_cpu_offload:
        pipe.enable_model_cpu_offload(device=device)
    else:
        pipe.to(device=device)

    return pipe


def prepare_conditions(
    tokenizer,
    text_encoder,
    prompt: Union[str, List[str]],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    max_sequence_length: int = 512,  # TODO: this should be configurable
    **kwargs,
):
    """Encode prompts into text embeddings for conditioning.

    Args:
        tokenizer: Tokenizer instance.
        text_encoder: Text encoder instance.
        prompt: Prompt or list of prompts.
        device: Target device.
        dtype: Target dtype.
        max_sequence_length: Maximum token length.
        **kwargs: Unused additional args.

    Returns:
        Dict with ``prompt_embeds`` tensor.
    """
    device = device or text_encoder.device
    dtype = dtype or text_encoder.dtype
    return _get_t5_prompt_embeds(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        prompt=prompt,
        max_sequence_length=max_sequence_length,
        device=device,
        dtype=dtype,
    )


def _normalize_latents(
    latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor
) -> torch.Tensor:
    """Normalize latents using dataset statistics.

    Args:
        latents: Latent tensor to normalize.
        latents_mean: Mean tensor.
        latents_std: Std tensor.

    Returns:
        Normalized latents tensor.
    """
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(device=latents.device)
    latents_std = latents_std.view(1, -1, 1, 1, 1).to(device=latents.device)
    latents = ((latents.float() - latents_mean) * latents_std).to(latents)
    return latents


def prepare_latents(
    vae: AutoencoderKLWan,
    image_or_video: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    generator: Optional[torch.Generator] = None,
    precompute: bool = False,
    # compute_posterior: bool = True,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """Encode input frames into latent samples via the VAE.

    Args:
        vae: VAE model for encoding.
        image_or_video: Input tensor of images or videos.
        device: Target device.
        dtype: Target dtype.
        generator: RNG generator for sampling.
        precompute: Whether called from precompute pipeline.
        **kwargs: Unused additional args.

    Returns:
        Dict with sampled ``latents`` tensor.
    """
    device = device or vae.device
    dtype = dtype or vae.dtype
    if image_or_video.ndim == 4:
        image_or_video = image_or_video.unsqueeze(2)
    assert (
        image_or_video.ndim == 5
    ), f"Expected 5D tensor, got {image_or_video.ndim}D tensor"

    image_or_video = image_or_video.to(device=device, dtype=vae.dtype)
    image_or_video = image_or_video.permute(0, 2, 1, 3, 4)
    moments = vae._encode(image_or_video)
    latents = moments.to(dtype=dtype)

    latents_mean = torch.tensor(vae.config.latents_mean)
    latents_std = 1.0 / torch.tensor(vae.config.latents_std)
    mu, logvar = torch.chunk(latents, 2, dim=1)
    mu = _normalize_latents(mu, latents_mean, latents_std)
    logvar = _normalize_latents(logvar, latents_mean, latents_std)
    latents = torch.cat([mu, logvar], dim=1)

    posterior = DiagonalGaussianDistribution(latents)
    latents = posterior.sample(generator=generator)
    del posterior

    return {"latents": latents}


def forward_pass(
    transformer: WanTransformer3DModel,
    prompt_embeds: torch.Tensor,
    noisy_latents: torch.Tensor,
    timesteps: torch.LongTensor,
    **kwargs,
) -> Tuple[torch.Tensor, ...]:
    """Run the transformer forward pass for denoising.

    Args:
        transformer: Wan transformer model.
        prompt_embeds: Prompt conditioning embeddings.
        noisy_latents: Noisy latent inputs.
        timesteps: Diffusion timesteps.
        **kwargs: Unused additional args.

    Returns:
        Dict with denoised ``latents`` tensor.
    """
    denoised_latent = transformer(
        hidden_states=noisy_latents,
        timestep=timesteps,
        encoder_hidden_states=prompt_embeds,
        return_dict=False,
    )[0]

    return {"latents": denoised_latent}


def validation(
    pipeline: WanPipeline,
    prompt: str,
    image: Optional[Image] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_frames: Optional[int] = None,
    num_inference_steps: int = 50,
    generator: Optional[torch.Generator] = None,
    **kwargs,
):
    """Run a validation inference step and return a video artifact.

    Args:
        pipeline: WanPipeline instance.
        prompt: Text prompt to render.
        image: Optional input image for i2v.
        height: Output height.
        width: Output width.
        num_frames: Number of frames to generate.
        num_inference_steps: Denoising steps.
        generator: Optional RNG generator.
        **kwargs: Unused additional args.

    Returns:
        List of tuples containing ("video", frames).
    """
    generation_kwargs = {
        "prompt": prompt,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "num_inference_steps": num_inference_steps,
        "generator": generator,
        "return_dict": True,
        "output_type": "pil",
    }
    generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}
    # switch schedular to inference
    prev = pipeline.scheduler
    flow_shift = 3.0  # hard code, for 480P
    scheduler = UniPCMultistepScheduler(
        prediction_type="flow_prediction",
        use_flow_sigmas=True,
        num_train_timesteps=1000,
        flow_shift=flow_shift,
    )
    pipeline.scheduler = scheduler
    output = pipeline(**generation_kwargs).frames[0]
    pipeline.scheduler = prev
    return [("video", output)]


def _get_t5_prompt_embeds(
    tokenizer: AutoTokenizer,
    text_encoder: UMT5EncoderModel,
    prompt: Union[str, List[str]] = None,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    """Tokenize prompts and compute T5 embeddings.

    Args:
        tokenizer: Tokenizer instance.
        text_encoder: T5 text encoder.
        prompt: Prompt or list of prompts.
        max_sequence_length: Max token length for padding.
        device: Target device.
        dtype: Target dtype.

    Returns:
        Dict with ``prompt_embeds`` tensor.
    """
    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt = [prompt_clean(u) for u in prompt]

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
    seq_lens = mask.gt(0).sum(dim=1).long()

    prompt_embeds = text_encoder(
        text_input_ids.to(device), mask.to(device)
    ).last_hidden_state
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
    prompt_embeds = torch.stack(
        [
            torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
            for u in prompt_embeds
        ],
        dim=0,
    )

    # duplicate text embeddings for each generation per prompt, using mps friendly metho

    return {"prompt_embeds": prompt_embeds}


WAN_T2V_LORA_CONFIG = {
    "pipeline_cls": WanPipeline,
    "load_condition_models": load_condition_models,
    "load_latent_models": load_latent_models,
    "load_diffusion_models": load_diffusion_models,
    "initialize_pipeline": initialize_pipeline,
    "prepare_conditions": prepare_conditions,
    "prepare_latents": prepare_latents,
    # "post_latent_preparation": post_latent_preparation, for precompute, not support now
    "collate_fn": collate_fn_t2v,
    # "calculate_noisy_latents": calculate_noisy_latents,
    "forward_pass": forward_pass,
    "validation": validation,
}
