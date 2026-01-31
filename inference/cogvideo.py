import argparse
import logging
from typing import Literal, Optional
from loguru import logger
import safetensors
import torch

from diffusers import (
    CogVideoXDPMScheduler,
    CogVideoXDDIMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXPipeline,
    CogVideoXVideoToVideoPipeline,
)
from diffusers.utils import export_to_video, load_image, load_video
import sys
import os
from pathlib import Path

import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from training.quant.quant import replace_linear, QuantLinear, QuantLoRALinear


logging.basicConfig(level=logging.INFO)


RESOLUTION_MAP = {
    # cogvideox1.5-*
    "cogvideox1.5-5b-i2v": (768, 1360),
    "cogvideox1.5-5b": (768, 1360),
    # cogvideox-*
    "cogvideox-5b-i2v": (480, 720),
    "cogvideox-5b": (480, 720),
    "cogvideox-2b": (480, 720),
}


def generate_video(
    args,
    prompt: str,
    model_path: str,
    lora_path: str = None,
    lora_rank: int = 128,
    num_frames: int = 81,
    width: Optional[int] = None,
    height: Optional[int] = None,
    output_path: str = "./output.mp4",
    image_or_video_path: str = "",
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    generate_type: str = Literal[
        "t2v", "i2v", "v2v"
    ],  # i2v: image to video, v2v: video to video
    seed: int = 42,
    fps: int = 16,
):
    """Generate a CogVideoX sample and save it to disk.

    Args:
        args: CLI arguments namespace with quantization options.
        prompt: Text prompt describing the video.
        model_path: Path to the base diffusion model.
        lora_path: Optional LoRA weights path for adapter fusion.
        lora_rank: Rank used when fusing LoRA weights.
        num_frames: Number of frames to generate.
        width: Output width; defaults to model-recommended resolution.
        height: Output height; defaults to model-recommended resolution.
        output_path: Path to write the output video.
        image_or_video_path: Input image/video path for i2v/v2v modes.
        num_inference_steps: Number of denoising steps.
        guidance_scale: Classifier-free guidance scale.
        num_videos_per_prompt: Number of samples per prompt.
        dtype: Computation dtype for the pipeline.
        generate_type: One of "t2v", "i2v", or "v2v".
        seed: Random seed for reproducibility.
        fps: Output video frame rate.

    Returns:
        None. The generated video is saved to ``output_path``.
    """

    # 1.  Load the pre-trained CogVideoX pipeline with the specified precision (bfloat16).
    # add device_map="balanced" in the from_pretrained function and remove the enable_model_cpu_offload()
    # function to use Multi GPUs.

    image = None
    video = None

    model_name = model_path.split("/")[-1].lower()
    desired_resolution = RESOLUTION_MAP[model_name]
    if width is None or height is None:
        height, width = desired_resolution
        logging.info(
            f"\033[1mUsing default resolution {desired_resolution} for {model_name}\033[0m"
        )
    elif (height, width) != desired_resolution:
        if generate_type == "i2v":
            # For i2v models, use user-defined width and height
            logging.warning(
                f"\033[1;31mThe width({width}) and height({height}) are not recommended for {model_name}. The best resolution is {desired_resolution}.\033[0m"
            )
        else:
            # Otherwise, use the recommended width and height
            logging.warning(
                f"\033[1;31m{model_name} is not supported for custom resolution. Setting back to default resolution {desired_resolution}.\033[0m"
            )
            height, width = desired_resolution

    if generate_type == "i2v":
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(
            model_path, torch_dtype=dtype
        )
        image = load_image(image=image_or_video_path)
    elif generate_type == "t2v":
        pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=dtype)
        if args.quant:
            quantizer_type = {
                "w": args.w_quantizer,
                "act": args.act_quantizer,
            }
            q_params = {
                "w": {
                    "bit": args.w_bit,
                    "sym": False,
                    "granularity": "per_channel",
                    "cali": "minmax",
                    "group_size": -1,
                    "round_zero": True,
                    "use_grad_scaling": True,
                },
                "act": {
                    "bit": args.act_bit,
                    "sym": False,
                    "granularity": "per_token",
                    "cali": "minmax",
                    "round_zero": True,
                    "clip_ratio": 0.95,
                },
            }
            quant_state_dict = safetensors.torch.load_file(
                os.path.join(
                    args.quant_model_path, "diffusion_pytorch_model.safetensors"
                )
            )
            random_list = []
            idx = 0
            lora = False
            for k in quant_state_dict.keys():
                if "lora" in k:
                    lora = True
                    break
            if lora:
                remove = []
                for k, v in quant_state_dict.items():
                    if k.endswith(".w"):
                        weight = v
                        loraA = quant_state_dict[f"{k[:-2]}.loraA.weight"]
                        loraB = quant_state_dict[f"{k[:-2]}.loraB.weight"]
                        lora_w = loraB @ loraA
                        quant_state_dict[k] = lora_w + weight  # alpha = 64 and r = 64
                        remove.append(f"{k[:-2]}.loraA.weight")
                        remove.append(f"{k[:-2]}.loraB.weight")
                for k in remove:
                    del quant_state_dict[k]
            for n, m in pipe.transformer.named_modules():
                if isinstance(m, nn.Linear):
                    if quant_state_dict.get(f"{n}.wquantizer.scale", None) is None:
                        random_list.append(idx)
                        logger.info(f"Skip Layer: {n}")
                    idx += 1
            replace_linear(
                pipe.transformer, quantizer_type, q_params, random_list=random_list
            )
            # initialize quantization parameters
            for _, module in pipe.transformer.named_modules():
                if isinstance(module, QuantLinear):
                    module.wquantizer(module.w.detach())
                    module.wquantizer.build()
                    module.aquantizer.build()
                    module.set_quant_state(True, args.use_aq)
            pipe.transformer.load_state_dict(quant_state_dict, strict=True)

    else:
        pipe = CogVideoXVideoToVideoPipeline.from_pretrained(
            model_path, torch_dtype=dtype
        )
        video = load_video(image_or_video_path)

    # If you're using with lora, add this code
    if lora_path:
        pipe.load_lora_weights(
            lora_path,
            weight_name="pytorch_lora_weights.safetensors",
            adapter_name="test_1",
        )
        pipe.fuse_lora(components=["transformer"], lora_scale=1 / lora_rank)

    # 2. Set Scheduler.
    # Can be changed to `CogVideoXDPMScheduler` or `CogVideoXDDIMScheduler`.
    # We recommend using `CogVideoXDDIMScheduler` for CogVideoX-2B.
    # using `CogVideoXDPMScheduler` for CogVideoX-5B / CogVideoX-5B-I2V.

    pipe.scheduler = CogVideoXDDIMScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )
    # pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    # 3. Enable CPU offload for the model.
    # turn off if you have multiple GPUs or enough GPU memory(such as H100) and it will cost less time in inference
    # and enable to("cuda")
    pipe.to("cuda")

    # pipe.enable_model_cpu_offload()
    # pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    # 4. Generate the video frames based on the prompt.
    # `num_frames` is the Number of frames to generate.
    if generate_type == "i2v":
        video_generate = pipe(
            height=height,
            width=width,
            prompt=prompt,
            image=image,
            # The path of the image, the resolution of video will be the same as the image for CogVideoX1.5-5B-I2V, otherwise it will be 720 * 480
            num_videos_per_prompt=num_videos_per_prompt,  # Number of videos to generate per prompt
            num_inference_steps=num_inference_steps,  # Number of inference steps
            num_frames=num_frames,  # Number of frames to generate
            use_dynamic_cfg=True,  # This id used for DPM scheduler, for DDIM scheduler, it should be False
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(
                seed
            ),  # Set the seed for reproducibility
        ).frames[0]
    elif generate_type == "t2v":
        if args.act_quantizer == "learnable_clipped_dynamic" and args.fix:
            for n, m in pipe.transformer.named_modules():
                if isinstance(m, (QuantLinear, QuantLoRALinear)):
                    m.aquantizer.set_auto(True)
        video_generate = pipe(
            height=height,
            width=width,
            prompt=prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            use_dynamic_cfg=True,
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),
        ).frames[0]
    else:
        video_generate = pipe(
            height=height,
            width=width,
            prompt=prompt,
            video=video,  # The path of the video to be used as the background of the video
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            use_dynamic_cfg=True,
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(
                seed
            ),  # Set the seed for reproducibility
        ).frames[0]
    export_to_video(video_generate, output_path, fps=fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a video from a text prompt using CogVideoX"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="The description of the video to be generated",
    )
    parser.add_argument(
        "--image_or_video_path",
        type=str,
        default=None,
        help="The path of the image to be used as the background of the video",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="THUDM/CogVideoX1.5-5B",
        help="Path of the pre-trained model use",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="The path of the LoRA weights to be used",
    )
    parser.add_argument(
        "--lora_rank", type=int, default=128, help="The rank of the LoRA weights"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./output.mp4",
        help="The path save generated video",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=6.0,
        help="The scale for classifier-free guidance",
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=50, help="Inference steps"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="Number of steps for the inference process",
    )
    parser.add_argument(
        "--width", type=int, default=None, help="The width of the generated video"
    )
    parser.add_argument(
        "--height", type=int, default=None, help="The height of the generated video"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=16,
        help="The frames per second for the generated video",
    )
    parser.add_argument(
        "--num_videos_per_prompt",
        type=int,
        default=1,
        help="Number of videos to generate per prompt",
    )
    parser.add_argument(
        "--generate_type", type=str, default="t2v", help="The type of video generation"
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", help="The data type for computation"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="The seed for reproducibility"
    )

    # for quantized model
    parser.add_argument(
        "--quant", action="store_true", help="whether to load quantized model."
    )
    parser.add_argument(
        "--quant_model_path",
        type=str,
        default="",
        help="Path of the quantized model use",
    )
    parser.add_argument(
        "--w_bit",
        type=int,
        default=4,
        help="Number of bits for weights.",
        choices=[2, 3, 4, 5, 6, 7, 8],
    )
    parser.add_argument(
        "--w_granularity",
        type=str,
        default="per_channel",
        help="Granularity of weight quantization.",
        choices=["per_group", "per_channel", "per_tensor"],
    )
    parser.add_argument(
        "--w_group_size",
        type=int,
        default=-1,
        help="Number of groups, which is force to -1 for non-group quantization, for per_group weight quantization.",
    )
    parser.add_argument(
        "--act_quantizer",
        type=str,
        default="dynamic",
        help="Activation quantizer.",
        choices=["dynamic", "learnable_clipped_dynamic", "lsq"],
    )
    parser.add_argument(
        "--act_bit",
        type=int,
        default=8,
        help="Number of bits for activations.",
        choices=[2, 3, 4, 5, 6, 7, 8],
    )
    parser.add_argument(
        "--use_aq",
        action="store_true",
        help="Use activation quantization.",
    )
    parser.add_argument(
        "--clip_group_num",
        type=int,
        default=50,
        help="Activation clipping group number.",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Whether to fix the bug.",
    )
    parser.add_argument(
        "--w_quantizer",
        type=str,
        default="lsq+",
        help="Weight quantizer.",
        choices=["dynamic", "learnable_clipped_dynamic", "lsq", "lsq+"],
    )

    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    generate_video(
        args,
        prompt=args.prompt,
        model_path=args.model_path,
        lora_path=args.lora_path,
        lora_rank=args.lora_rank,
        output_path=args.output_path,
        num_frames=args.num_frames,
        width=args.width,
        height=args.height,
        image_or_video_path=args.image_or_video_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_videos_per_prompt=args.num_videos_per_prompt,
        dtype=dtype,
        generate_type=args.generate_type,
        seed=args.seed,
        fps=args.fps,
    )
