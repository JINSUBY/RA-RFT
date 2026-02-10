#!/bin/bash
# Preprocess test split for ActivityNet and save to the same directory as train
# This will add test videos to the existing preprocessed directory without duplicating train videos

GPU_ID=${1:-0}  # Default to GPU 0, or pass as first argument

MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
DATASET="tvgbench"
TEST_DATA="dataset/anno/tvgbench_riq.json"  # Using test.json for preprocessing
VIDEO_FOLDER="dataset/videos/tvgbench"
MAX_PIX=3584
MIN_PIX=16
NUM_WORKERS=4
# Use the same output directory as train so train and test are in one place
OUTPUT_DIR=dataset/preprocessed_video/tvgbench

echo "Preprocessing test split for tvgbench..."
echo "Output directory: $OUTPUT_DIR"
echo "Using GPU: $GPU_ID"
echo "This will add test videos to the existing preprocessed data (train videos will be skipped if already exist)"

CUDA_VISIBLE_DEVICES=$GPU_ID python src/utils/preprocess_dataset.py \
  --model_name $MODEL_PATH \
  --dataset $DATASET \
  --train_data_path $TEST_DATA \
  --video_folder $VIDEO_FOLDER \
  --max_pix_size $MAX_PIX \
  --min_pix_size $MIN_PIX \
  --num_workers $NUM_WORKERS \
  --output_dir $OUTPUT_DIR

echo "Test data preprocessing complete!"
