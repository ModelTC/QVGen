from typing import Any, Dict, List, Optional, Union

import torch
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDDIMScheduler,
    CogVideoXPipeline,
    CogVideoXTransformer3DModel,
)
from PIL import Image
from transformers import T5EncoderModel, T5Tokenizer

from .utils import prepare_rotary_positional_embeddings


def load_condition_models(
    model_id: str = "THUDM/CogVideoX-5b",
    text_encoder_dtype: torch.dtype = torch.bfloat16,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    **kwargs,
):
    """Load tokenizer and text encoder for CogVideoX prompts.

    Args:
        model_id: Hugging Face model ID.
        text_encoder_dtype: Dtype for text encoder weights.
        revision: Optional model revision.
        cache_dir: Optional cache directory.
        **kwargs: Unused additional args.

    Returns:
        Dict with ``tokenizer`` and ``text_encoder``.
    """
    tokenizer = T5Tokenizer.from_pretrained(
        model_id, subfolder="tokenizer", revision=revision, cache_dir=cache_dir
    )
    text_encoder = T5EncoderModel.from_pretrained(
        model_id,
        subfolder="text_encoder",
        torch_dtype=text_encoder_dtype,
        revision=revision,
        cache_dir=cache_dir,
    )
    return {"tokenizer": tokenizer, "text_encoder": text_encoder}


def load_latent_models(
    model_id: str = "THUDM/CogVideoX-5b",
    vae_dtype: torch.dtype = torch.bfloat16,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    **kwargs,
):
    """Load the VAE for CogVideoX latents.

    Args:
        model_id: Hugging Face model ID.
        vae_dtype: Dtype for VAE weights.
        revision: Optional model revision.
        cache_dir: Optional cache directory.
        **kwargs: Unused additional args.

    Returns:
        Dict with ``vae``.
    """
    vae = AutoencoderKLCogVideoX.from_pretrained(
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
    """Load transformer and scheduler for CogVideoX diffusion.

    Args:
        model_id: Hugging Face model ID.
        transformer_dtype: Dtype for transformer weights.
        revision: Optional model revision.
        cache_dir: Optional cache directory.
        **kwargs: Unused additional args.

    Returns:
        Dict with ``transformer`` and ``scheduler``.
    """
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        torch_dtype=transformer_dtype,
        revision=revision,
        cache_dir=cache_dir,
    )
    scheduler = CogVideoXDDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    return {"transformer": transformer, "scheduler": scheduler}


def initialize_pipeline(
    model_id: str = "THUDM/CogVideoX-5b",
    text_encoder_dtype: torch.dtype = torch.bfloat16,
    transformer_dtype: torch.dtype = torch.bfloat16,
    vae_dtype: torch.dtype = torch.bfloat16,
    tokenizer: Optional[T5Tokenizer] = None,
    text_encoder: Optional[T5EncoderModel] = None,
    transformer: Optional[CogVideoXTransformer3DModel] = None,
    vae: Optional[AutoencoderKLCogVideoX] = None,
    scheduler: Optional[CogVideoXDDIMScheduler] = None,
    device: Optional[torch.device] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    enable_slicing: bool = False,
    enable_tiling: bool = False,
    enable_model_cpu_offload: bool = False,
    is_training: bool = False,
    **kwargs,
) -> CogVideoXPipeline:
    """Initialize a CogVideoX pipeline with optional components.

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
        device: Target device.
        revision: Optional model revision.
        cache_dir: Optional cache directory.
        enable_slicing: Whether to enable VAE slicing.
        enable_tiling: Whether to enable VAE tiling.
        enable_model_cpu_offload: Whether to offload weights to CPU.
        is_training: Whether initialization is for training.
        **kwargs: Unused additional args.

    Returns:
        Initialized CogVideoXPipeline instance.
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

    pipe = CogVideoXPipeline.from_pretrained(
        model_id, **components, revision=revision, cache_dir=cache_dir
    )
    pipe.text_encoder = pipe.text_encoder.to(dtype=text_encoder_dtype)
    pipe.vae = pipe.vae.to(dtype=vae_dtype)

    # The transformer should already be in the correct dtype when training, so we don't need to cast it here.
    # If we cast, whilst using fp8 layerwise upcasting hooks, it will lead to an error in the training during
    # DDP optimizer step.
    if not is_training:
        pipe.transformer = pipe.transformer.to(dtype=transformer_dtype)

    if enable_slicing:
        pipe.vae.enable_slicing()
    if enable_tiling:
        pipe.vae.enable_tiling()

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
    max_sequence_length: int = 226,  # TODO: this should be configurable
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
        Dict with ``prompt_embeds``.
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


def prepare_latents(
    vae: AutoencoderKLCogVideoX,
    image_or_video: torch.Tensor,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    generator: Optional[torch.Generator] = None,
    precompute: bool = False,
    **kwargs,
) -> torch.Tensor:
    """Encode input frames into CogVideoX latents.

    Args:
        vae: VAE encoder.
        image_or_video: Input tensor of images or videos.
        device: Target device.
        dtype: Target dtype.
        generator: RNG generator.
        precompute: Whether running in precompute mode.
        **kwargs: Unused additional args.

    Returns:
        Dict with latents tensor.
    """
    device = device or vae.device
    dtype = dtype or vae.dtype

    if image_or_video.ndim == 4:
        image_or_video = image_or_video.unsqueeze(2)
    assert (
        image_or_video.ndim == 5
    ), f"Expected 5D tensor, got {image_or_video.ndim}D tensor"

    image_or_video = image_or_video.to(device=device, dtype=vae.dtype)
    image_or_video = image_or_video.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
    if not precompute:
        latents = vae.encode(image_or_video).latent_dist.sample(generator=generator)
        if not vae.config.invert_scale_latents:
            latents = latents * vae.config.scaling_factor
        # For training Cog 1.5, we don't need to handle the scaling factor here.
        # The CogVideoX team forgot to multiply here, so we should not do it too. Invert scale latents
        # is probably only needed for image-to-video training.
        #  investigate this
        # else:
        #     latents = 1 / vae.config.scaling_factor * latents
        latents = latents.to(dtype=dtype)
        return {
            "latents": latents.permute(0, 2, 1, 3, 4)
        }  # TODO: fix for CogVideoX [B, F, C, H, W]
    else:
        # handle vae scaling in the `train()` method directly.
        if vae.use_slicing and image_or_video.shape[0] > 1:
            encoded_slices = [
                vae._encode(x_slice) for x_slice in image_or_video.split(1)
            ]
            h = torch.cat(encoded_slices)
        else:
            h = vae._encode(image_or_video)
        return {"latents": h}


def post_latent_preparation(
    vae_config: Dict[str, Any],
    latents: torch.Tensor,
    patch_size_t: Optional[int] = None,
    **kwargs,
) -> torch.Tensor:
    """Apply VAE scaling and optional padding to latents.

    Args:
        vae_config: VAE config with scaling factor flags.
        latents: Latents tensor.
        patch_size_t: Optional temporal patch size for padding.
        **kwargs: Unused additional args.

    Returns:
        Dict with processed latents.
    """
    if not vae_config.invert_scale_latents:
        latents = latents * vae_config.scaling_factor
    # For training Cog 1.5, we don't need to handle the scaling factor here.
    # The CogVideoX team forgot to multiply here, so we should not do it too. Invert scale latents
    # is probably only needed for image-to-video training.
    #  investigate this
    # else:
    #     latents = 1 / vae_config.scaling_factor * latents
    latents = _pad_frames(latents, patch_size_t)
    latents = latents.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
    return {"latents": latents}


def collate_fn_t2v(
    batch: List[List[Dict[str, torch.Tensor]]]
) -> Dict[str, torch.Tensor]:
    """Collate a batch for text-to-video training.

    Args:
        batch: Nested batch from bucket sampler.

    Returns:
        Dict with prompts and stacked videos.
    """
    return {
        "prompts": [x["prompt"] for x in batch[0]],
        "videos": torch.stack([x["video"] for x in batch[0]]),
    }


def calculate_noisy_latents(
    scheduler: CogVideoXDDIMScheduler,
    noise: torch.Tensor,
    latents: torch.Tensor,
    timesteps: torch.LongTensor,
) -> torch.Tensor:
    """Add noise to latents using the scheduler.

    Args:
        scheduler: DDIM scheduler instance.
        noise: Noise tensor.
        latents: Clean latents tensor.
        timesteps: Diffusion timesteps.

    Returns:
        Noisy latents tensor.
    """
    noisy_latents = scheduler.add_noise(latents, noise, timesteps)
    return noisy_latents


def forward_pass(
    transformer: CogVideoXTransformer3DModel,
    scheduler: CogVideoXDDIMScheduler,
    prompt_embeds: torch.Tensor,
    latents: torch.Tensor,
    noisy_latents: torch.Tensor,
    timesteps: torch.LongTensor,
    ofs_emb: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """Run the CogVideoX transformer forward pass.

    Args:
        transformer: Transformer model.
        scheduler: DDIM scheduler instance.
        prompt_embeds: Prompt embeddings.
        latents: Latents tensor.
        noisy_latents: Noisy latents tensor.
        timesteps: Diffusion timesteps.
        ofs_emb: Optional ofs embedding override.
        **kwargs: Unused additional args.

    Returns:
        Dict with denoised latents.
    """
    # Just hardcode for now. In Diffusers, we will refactor such that RoPE would be handled within the model itself.
    VAE_SPATIAL_SCALE_FACTOR = 8
    transformer_config = (
        transformer.module.config
        if hasattr(transformer, "module")
        else transformer.config
    )
    batch_size, num_frames, num_channels, height, width = noisy_latents.shape
    rope_base_height = transformer_config.sample_height * VAE_SPATIAL_SCALE_FACTOR
    rope_base_width = transformer_config.sample_width * VAE_SPATIAL_SCALE_FACTOR

    image_rotary_emb = (
        prepare_rotary_positional_embeddings(
            height=height * VAE_SPATIAL_SCALE_FACTOR,
            width=width * VAE_SPATIAL_SCALE_FACTOR,
            num_frames=num_frames,
            vae_scale_factor_spatial=VAE_SPATIAL_SCALE_FACTOR,
            patch_size=transformer_config.patch_size,
            patch_size_t=transformer_config.patch_size_t
            if hasattr(transformer_config, "patch_size_t")
            else None,
            attention_head_dim=transformer_config.attention_head_dim,
            device=transformer.device,
            base_height=rope_base_height,
            base_width=rope_base_width,
        )
        if transformer_config.use_rotary_positional_embeddings
        else None
    )
    ofs_emb = (
        None
        if transformer_config.ofs_embed_dim is None
        else latents.new_full((batch_size,), fill_value=2.0)
    )

    velocity = transformer(
        hidden_states=noisy_latents,
        timestep=timesteps,
        encoder_hidden_states=prompt_embeds,
        ofs=ofs_emb,
        image_rotary_emb=image_rotary_emb,
        return_dict=False,
    )[0]
    # For CogVideoX, the transformer predicts the velocity. The denoised output is calculated by applying the same
    # code paths as scheduler.get_velocity(), which can be confusing to understand.
    denoised_latents = scheduler.get_velocity(velocity, noisy_latents, timesteps)

    return {"latents": denoised_latents}


def validation(
    pipeline: CogVideoXPipeline,
    prompt: str,
    image: Optional[Image.Image] = None,
    video: Optional[List[Image.Image]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_frames: Optional[int] = None,
    num_videos_per_prompt: int = 1,
    generator: Optional[torch.Generator] = None,
    **kwargs,
):
    """Run a validation inference step and return video output.

    Args:
        pipeline: CogVideoXPipeline instance.
        prompt: Text prompt to render.
        image: Optional image input.
        video: Optional video input.
        height: Output height.
        width: Output width.
        num_frames: Number of frames to generate.
        num_videos_per_prompt: Samples per prompt.
        generator: RNG generator.
        **kwargs: Unused additional args.

    Returns:
        List of tuples containing ("video", frames).
    """
    generation_kwargs = {
        "prompt": prompt,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "num_videos_per_prompt": num_videos_per_prompt,
        "generator": generator,
        "return_dict": True,
        "output_type": "pil",
    }
    generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}
    output = pipeline(**generation_kwargs).frames[0]
    return [("video", output)]


def _get_t5_prompt_embeds(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]] = None,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    """Tokenize prompts and compute T5 embeddings.

    Args:
        tokenizer: Tokenizer instance.
        text_encoder: T5 encoder.
        prompt: Prompt or list of prompts.
        max_sequence_length: Max token length for padding.
        device: Target device.
        dtype: Target dtype.

    Returns:
        Dict with ``prompt_embeds`` tensor.
    """
    prompt = [prompt] if isinstance(prompt, str) else prompt

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    return {"prompt_embeds": prompt_embeds}


def _pad_frames(latents: torch.Tensor, patch_size_t: int):
    """Pad latents so temporal length is divisible by patch_size_t.

    Args:
        latents: Latents tensor of shape [B, F, C, H, W].
        patch_size_t: Temporal patch size.

    Returns:
        Padded latents tensor.
    """
    if patch_size_t is None or patch_size_t == 1:
        return latents

    # `latents` should be of the following format: [B, C, F, H, W].
    # For CogVideoX 1.5, the latent frames should be padded to make it divisible by patch_size_t
    latent_num_frames = latents.shape[2]
    additional_frames = patch_size_t - latent_num_frames % patch_size_t

    if additional_frames > 0:
        last_frame = latents[:, :, -1:, :, :]
        padding_frames = last_frame.repeat(1, 1, additional_frames, 1, 1)
        latents = torch.cat([latents, padding_frames], dim=2)

    return latents


#  refactor into model specs for better re-use
COGVIDEOX_T2V_LORA_CONFIG = {
    "pipeline_cls": CogVideoXPipeline,
    "load_condition_models": load_condition_models,
    "load_latent_models": load_latent_models,
    "load_diffusion_models": load_diffusion_models,
    "initialize_pipeline": initialize_pipeline,
    "prepare_conditions": prepare_conditions,
    "prepare_latents": prepare_latents,
    "post_latent_preparation": post_latent_preparation,
    "collate_fn": collate_fn_t2v,
    "calculate_noisy_latents": calculate_noisy_latents,
    "forward_pass": forward_pass,
    "validation": validation,
}
