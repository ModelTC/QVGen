import argparse
import datetime
import gc
import math
import random
from loguru import logger
import os
from typing import Literal, Optional
import numpy as np
import torch.nn as nn
import torch

from diffusers import (
    WanPipeline,
    AutoencoderKLWan,
    UniPCMultistepScheduler,
)
from diffusers.utils import export_to_video, load_image, load_video
import safetensors.torch
from tqdm import tqdm

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from training.quant.quant import replace_linear, QuantLinear, QuantLoRALinear

# Recommended resolution for each model (width, height)
RESOLUTION_MAP = {
    "wan2.1-t2v-1.3b-diffusers": (480, 832),
    "wan2.1-t2v-14b-diffusers": (720, 1280),
}


def seed_everything(seed: int):
    """Seed RNGs for reproducible distributed sampling.

    Args:
        seed: Base random seed to apply across RNG backends.

    Returns:
        None.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def generate_video(
    args,
    model_path: str,
    num_frames: int = 81,
    width: Optional[int] = None,
    height: Optional[int] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
    fps: int = 16,
):
    """Run DDP inference for WAN text-to-video and save outputs.

    Args:
        args: CLI arguments namespace with quantization and IO options.
        model_path: Path to the base diffusion model.
        num_frames: Number of frames to generate.
        width: Output width; defaults to model-recommended resolution.
        height: Output height; defaults to model-recommended resolution.
        num_inference_steps: Number of denoising steps.
        guidance_scale: Classifier-free guidance scale.
        dtype: Computation dtype for the pipeline.
        seed: Base random seed (rank-adjusted internally).
        fps: Output video frame rate.

    Returns:
        None. Videos are saved under ``args.save_dir`` by rank 0.
    """
    # 1.  Load the pre-trained CogVideoX pipeline with the specified precision (bfloat16).
    # add device_map="balanced" in the from_pretrained function and remove the enable_model_cpu_offload()
    # function to use Multi GPUs.
    assert (
        torch.cuda.is_available()
    ), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)
    dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=7200))
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.seed * dist.get_world_size() + rank
    seed_everything(seed)
    torch.cuda.set_device(device)
    if rank != 0:
        list_save_dir = [None]
    else:
        if args.resume_dir != "":
            args.save_dir = args.resume_dir
            if not os.path.exists(args.save_dir):
                raise ValueError(f"Resume directory {args.save_dir} does not exist.")
            logger.add(os.path.join(args.save_dir, "log.txt"))
            logger.info(args)
            list_save_dir = [args.save_dir]
        else:
            args.save_dir = os.path.join(
                args.save_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            )
            os.makedirs(args.save_dir)
            logger.add(os.path.join(args.save_dir, "log.txt"))
            logger.info(args)
            list_save_dir = [args.save_dir]
    dist.barrier()
    # get the latest save_dir
    dist.broadcast_object_list(list_save_dir, src=0)
    args.save_dir = list_save_dir[0]
    logger.info(
        f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}. (save_dir={args.save_dir})"
    )

    model_name = model_path.split("/")[-1].lower()
    desired_resolution = RESOLUTION_MAP[model_name]
    if width is None or height is None:
        height, width = desired_resolution
        logger.info(
            f"\033[1mUsing default resolution {desired_resolution} for {model_name}\033[0m"
        )
    elif (height, width) != desired_resolution:
        logger.warning(
            f"\033[1;31m{model_name} is not supported for custom resolution. Setting back to default resolution {desired_resolution}.\033[0m"
        )
        height, width = desired_resolution

    vae = AutoencoderKLWan.from_pretrained(
        model_path, subfolder="vae", torch_dtype=torch.float32
    )
    if desired_resolution[0] == 480:
        flow_shift = 3.0
    else:
        flow_shift = 5.0
    scheduler = UniPCMultistepScheduler(
        prediction_type="flow_prediction",
        use_flow_sigmas=True,
        num_train_timesteps=1000,
        flow_shift=flow_shift,
    )
    pipe = WanPipeline.from_pretrained(model_path, torch_dtype=dtype, vae=vae)
    pipe.scheduler = scheduler
    if args.quant:
        quantizer_type = {
            "w": args.w_quantizer,
            "act": args.act_quantizer,
        }
        q_params = {
            "w": {
                "bit": args.w_bit,
                "sym": False,
                "granularity": args.w_granularity,
                "cali": "minmax",
                "group_size": args.w_group_size,
                "round_zero": True,
                "use_grad_scaling": True,
            },
            "act": {
                "bit": args.act_bit,
                "sym": False,
                "granularity": "per_token",
                "cali": "minmax",
                "round_zero": True,
                "clip_ratio": args.clip_ratio,
            },
        }
        quant_state_dict = safetensors.torch.load_file(os.path.join(args.quant_model_path, "diffusion_pytorch_model.safetensors"))
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
                    if rank == 0:
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
        del quant_state_dict
        gc.collect()
        torch.cuda.empty_cache()

    pipe.to(device)

    prompts_dir = args.prompts_dir
    prompts_list = []
    count = 0
    assert (
        args.vbench_2 != "" or args.aug_prompts_dir == ""
    ), "aug_prompts can only be used when employing vbench-2 test."
    if args.vbench_2:
        required_dimensions = [
            "Human_Anatomy",
            "Human_Identity",
            "Human_Clothes",
            "Diversity",
            "Composition",
            "Dynamic_Spatial_Relationship",
            "Dynamic_Attribute",
            "Motion_Order_Understanding",
            "Human_Interaction",
            "Complex_Landscape",
            "Complex_Plot",
            "Camera_Motion",
            "Motion_Rationality",
            "Instance_Preservation",
            "Mechanics",
            "Thermotics",
            "Material",
            "Multi-View_Consistency",
        ]
        if args.aug_prompts_dir != "":
            prompts_dir = args.aug_prompts_dir
            original_prompts_dir = args.prompts_dir
    else:
        required_dimensions = ["overall_consistency", "subject_consistency", "scene"]
    for file_path in os.listdir(prompts_dir):
        file_name = file_path[:-4]
        if file_name not in required_dimensions:
            continue
        save_dir = os.path.join(args.save_dir, file_name)
        if rank == 0:
            os.makedirs(save_dir, exist_ok=True)
        dist.barrier()
        if args.aug_prompts_dir == "":
            file = open(os.path.join(prompts_dir, file_path))
            iters = iter(file)
            for prompt in iters:
                # remove the '\n' at the end of the prompt
                prompt = prompt[:-1] if prompt[-1] == "\n" else prompt
                count += 1
                if args.vbench_2:
                    repeat = 3 if file_name != "Diversity" else 20
                else:
                    repeat = 5 if file_name != "temporal_flickering" else 25
                for idx in range(repeat):
                    prompts_list.append([prompt, save_dir, idx])
        else:
            file = open(os.path.join(prompts_dir, file_path))
            original_file = open(os.path.join(original_prompts_dir, file_path))
            iters = iter(file)
            original_iters = iter(original_file)
            for original_prompt, prompt in zip(original_iters, iters):
                # remove the '\n' at the end of the prompt
                prompt = prompt[:-1] if prompt[-1] == "\n" else prompt
                original_prompt = original_prompt.strip()
                count += 1
                if args.vbench_2:
                    repeat = 3 if file_name != "Diversity" else 20
                else:
                    repeat = 5 if file_name != "temporal_flickering" else 25
                for idx in range(repeat):
                    prompts_list.append([prompt, save_dir, idx, original_prompt])

    n = args.batch_size
    total_samples = len(prompts_list)
    if rank == 0:
        logger.info(
            f"Total number of videos that will be sampled: {total_samples} (Number of Prompts: {count})"
        )
    samples_needed_per_gpu = [
        total_samples // dist.get_world_size() for _ in range(dist.get_world_size())
    ]
    for i in range(total_samples % dist.get_world_size()):
        samples_needed_per_gpu[i] += 1
    start = sum(samples_needed_per_gpu[:rank])
    num = samples_needed_per_gpu[rank]
    logger.info(f"start={start}, num={num}, rank={rank}")
    iterations = math.ceil(num / n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar

    # 4. Generate the video frames based on the prompt.
    # `num_frames` is the Number of frames to generate.
    head, tail = -1, -1
    for i in pbar:
        head = start + i * n
        tail = head + n if head + n <= start + num else start + num
        # logger.info(f"head={head}, tail={tail}, rank={rank}")
        prompts = [p[0] for p in prompts_list[head:tail]]
        save_dirs = [p[1] for p in prompts_list[head:tail]]
        idexes = [p[2] for p in prompts_list[head:tail]]
        if args.aug_prompts_dir != "":
            original_prompts = [p[3] for p in prompts_list[head:tail]]
        if args.act_quantizer == "learnable_clipped_dynamic" and args.fix:
            for _, m in pipe.transformer.named_modules():
                if isinstance(m, (QuantLinear, QuantLoRALinear)):
                    m.aquantizer.set_auto(True)
        # check existence of the video (may be generated last time)
        if args.resume_dir != "":
            if args.aug_prompts_dir != "":
                videos = [
                    os.path.join(
                        save_dirs[j], f"{original_prompts[j][:180]}-{idexes[j]}.mp4"
                    )
                    for j in range(len(prompts))
                ]
                if all(os.path.exists(v) for v in videos):
                    logger.info(
                        f"Video from {head}-{tail} already exists, skipping... (rank={rank})"
                    )
                    continue
            else:
                videos = [
                    os.path.join(
                        save_dirs[j],
                        f"{prompts[j] if args.vbench_2 == '' else prompts[j][:180]}-{idexes[j]}.mp4",
                    )
                    for j in range(len(prompts))
                ]
                if all(os.path.exists(v) for v in videos):
                    logger.info(
                        f"Video from {head}-{tail} already exists, skipping... (rank={rank})"
                    )
                    continue
        video_generate = pipe(
            height=height,
            width=width,
            prompt=prompts,
            num_videos_per_prompt=1,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            # use_dynamic_cfg=True,
            guidance_scale=guidance_scale,
        ).frames
        for j, v in enumerate(video_generate):
            if args.aug_prompts_dir != "":
                export_to_video(
                    v,
                    os.path.join(
                        save_dirs[j], f"{original_prompts[j][:180]}-{idexes[j]}.mp4"
                    ),
                    fps=fps,
                )
            else:
                export_to_video(
                    v,
                    os.path.join(
                        save_dirs[j],
                        f"{prompts[j] if args.vbench_2 == '' else prompts[j][:180]}-{idexes[j]}.mp4",
                    ),
                    fps=fps,
                )

    dist.monitored_barrier(timeout=datetime.timedelta(seconds=7200))
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a video from a text prompt using CogVideoX"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/Wan2.1-T2V-1.3B-Diffusers",
        help="Path of the pre-trained model use",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=5.0,
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
        "--w_quantizer",
        type=str,
        default="lsq",
        help="Activation quantizer.",
        choices=["lsq+", "lsq"],
    )
    parser.add_argument(
        "--act_quantizer",
        type=str,
        default="dynamic",
        help="Activation quantizer.",
        choices=["dynamic", "learnable_clipped_dynamic"],
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
    # for evaluation
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Whether to fix the bug.",
    )
    parser.add_argument(
        "--prompts_dir",
        type=str,
        default="VBench/prompts/prompts_per_dimension",
        help="Prompts for video generation",
    )
    parser.add_argument(
        "--clip_ratio",
        type=float,
        default=0.95,
        help="Clip ratio for activation quantization.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for video generation.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="VideoQAT/outputs/",
        help="Directory for video generation.",
    )
    parser.add_argument(
        "--vbench_2",
        action="store_true",
        help="Whether to use vbench2.",
    )
    parser.add_argument(
        "--resume_dir",
        type=str,
        default="",
        help="Directory for resume video generation.",
    )
    parser.add_argument(
        "--aug_prompts_dir",
        type=str,
        default="",
        help="The directory of augumentated prompts.",
    )
    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    generate_video(
        args,
        model_path=args.model_path,
        num_frames=args.num_frames,
        width=args.width,
        height=args.height,
        guidance_scale=args.guidance_scale,
        dtype=dtype,
        seed=args.seed,
        fps=args.fps,
    )
