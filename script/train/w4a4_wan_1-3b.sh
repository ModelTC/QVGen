#!/bin/bash
export WANDB_MODE="online"
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export FINETRAINERS_LOG_LEVEL=DEBUG

GPU_IDS="0,1,2,3,4,5,6,7"

DATA_ROOT="dataset/OpenVid-49x480x832"
CAPTION_COLUMN="prompts.txt"
VIDEO_COLUMN="videos.txt"
OUTPUT_DIR="path/to/your/output_path" # Employ BF16 Here

# Model arguments
model_cmd="--model_name wan \
    --use_teacher \
    --pretrained_model_name_or_path models/Wan2.1-T2V-1.3B-Diffusers"

# Dataset arguments
dataset_cmd="--data_root $DATA_ROOT \
  --video_column $VIDEO_COLUMN \
  --caption_column $CAPTION_COLUMN \
  --video_resolution_buckets 49x480x832 \
  --caption_dropout_p 0.05"

# Dataloader arguments
dataloader_cmd="--dataloader_num_workers 16"

# Training arguments
training_cmd="--training_type full-finetune \
  --seed 42 \
  --batch_size 6 \
  --train_steps 2667 \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing \
  --checkpointing_steps 200 \
  --checkpointing_limit 2 \
  --resume_from_checkpoint=latest"

# Optimizer arguments
optimizer_cmd="--optimizer adamw \
  --lr 3e-5 \
  --lr_scheduler cosine_with_restarts \
  --lr_warmup_steps 267 \
  --lr_num_cycles 1 \
  --beta1 0.9 \
  --beta2 0.95 \
  --weight_decay 1e-4 \
  --epsilon 1e-8 \
  --max_grad_norm 1.0"

validation_cmd="--validation_prompt \"A detailed wooden toy ship with intricately carved masts and sails is seen gliding smoothly over a plush, blue carpet that mimics the waves of the sea. The ship's hull is painted a rich brown, with tiny windows. The carpet, soft and textured, provides a perfect backdrop, resembling an oceanic expanse. Surrounding the ship are various other toys and children's items, hinting at a playful environment. The scene captures the innocence and imagination of childhood, with the toy ship's journey symbolizing endless adventures in a whimsical, indoor setting.@@@81x480x832:::The camera follows behind a white vintage SUV with a black roof rack as it speeds up a steep dirt road surrounded by pine trees on a steep mountain slope, dust kicks up from its tires, the sunlight shines on the SUV as it speeds along the dirt road, casting a warm glow over the scene. The dirt road curves gently into the distance, with no other cars or vehicles in sight. The trees on either side of the road are redwoods, with patches of greenery scattered throughout. The car is seen from the rear following the curve with ease, making it seem as if it is on a rugged drive through the rugged terrain. The dirt road itself is surrounded by steep hills and mountains, with a clear blue sky above with wispy clouds.@@@81x480x832:::A street artist, clad in a worn-out denim jacket and a colorful bandana, stands before a vast concrete wall in the heart, holding a can of spray paint, spray-painting a colorful bird on a mottled wall.@@@81x480x832:::In the haunting backdrop of a war-torn city, where ruins and crumbled walls tell a story of devastation, a poignant close-up frames a young girl. Her face is smudged with ash, a silent testament to the chaos around her. Her eyes glistening with a mix of sorrow and resilience, capturing the raw emotion of a world that has lost its innocence to the ravages of conflict.@@@81x480x832\" \
          --validation_separator ::: \
          --num_validation_videos 1 \
          --validation_steps 200 \
          --validation_frame_rate 16"

# Miscellaneous arguments
miscellaneous_cmd="--tracker_name qvgen \
  --output_dir $OUTPUT_DIR \
  --nccl_timeout 1800 \
  --report_to none \
  --allow_tf32 \
  --report_to wandb"

quantization_cmd="--w_quantizer lsq+ \
  --w_bit 4 \
  --w_granularity per_channel \
  --w_use_grad_scaling \
  --w_cali mse \
  --act_quantizer learnable_clipped_dynamic \
  --act_lr 1e-2 \
  --act_bit 4 \
  --act_clip_ratio 0.95 \
  --w_lr 3e-5 \
  --use_aq \
  --clip_group_num 50 \
  --progress_rank 32 \
  --progress_iter 1 \
  --progress_cur_alpha_strategy cosine \
  --progress_alpha_T 444 \
  --progress_warm_up 0 \
  --progress_training \
  --progress_lr 3e-5"

cmd="accelerate launch --main_process_port 13880 --config_file accelerate_configs/deepspeed.yaml --gpu_ids $GPU_IDS train.py \
  $model_cmd \
  $dataset_cmd \
  $dataloader_cmd \
  $training_cmd \
  $optimizer_cmd \
  $validation_cmd \
  $miscellaneous_cmd \
  $quantization_cmd"

echo "Running command: $cmd"
eval $cmd
echo -ne "-------------------- Finished executing script --------------------\n\n"