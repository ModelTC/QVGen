#!/bin/bash

MODEL_ID="models/CogVideoX-2b"

NUM_GPUS=8


DATA_ROOT="dataset/video"
DATASET_FILE="dataset/OpenVid-1M/data/train/OpenVidHD.csv"
CAPTION_COLUMN="caption"
VIDEO_COLUMN="video"
OUTPUT_DIR="dataset/OpenVid-49x480x720"
HEIGHT_BUCKETS="480"
WIDTH_BUCKETS="720"
FRAME_BUCKETS="49"
MAX_NUM_FRAMES="49"
MAX_SEQUENCE_LENGTH=512
TARGET_FPS=16
BATCH_SIZE=10
DTYPE=fp32

CMD_WITHOUT_PRE_ENCODING="\
  torchrun --nproc_per_node=$NUM_GPUS \
    prepare_dataset/prepare_dataset.py \
      --model_id $MODEL_ID \
      --data_root $DATA_ROOT \
      --caption_column $CAPTION_COLUMN \
      --dataset_file $DATASET_FILE \
      --video_column $VIDEO_COLUMN \
      --output_dir $OUTPUT_DIR \
      --height_buckets $HEIGHT_BUCKETS \
      --width_buckets $WIDTH_BUCKETS \
      --frame_buckets $FRAME_BUCKETS \
      --max_num_frames $MAX_NUM_FRAMES \
      --max_sequence_length $MAX_SEQUENCE_LENGTH \
      --target_fps $TARGET_FPS \
      --batch_size $BATCH_SIZE \
      --dataloader_num_worker 8 \
      --dtype $DTYPE
"

CMD_WITH_PRE_ENCODING="$CMD_WITHOUT_PRE_ENCODING"

CMD=$CMD_WITH_PRE_ENCODING

echo "===== Running \`$CMD\` ====="
eval $CMD
echo -ne "===== Finished running script =====\n"
