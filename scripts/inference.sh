#!/bin/bash

# Multi-GPU inference script for RIQ (Relevance-Irrelevance Query) task

GPU_LIST="0,1,2,3"
MODEL_PATH="./ckpts/your_model_checkpoint"
OUTPUT_DIR="./inference_output"
TEST_DATA_PATH="./dataset/anno/test_data.json"
PREPROCESSED_DATA_PATH="./dataset/preprocessed_data"
LOG_DIR="$OUTPUT_DIR/gpu_logs"

# Create output and log directories
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

echo "================================================================================"
echo "Multi-GPU RIQ Inference"
echo "================================================================================"
echo "Model: $MODEL_PATH"
echo "Test data: $TEST_DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "Preprocessed data: $PREPROCESSED_DATA_PATH"
echo "GPUs: $GPU_LIST"
echo "Logs: $LOG_DIR"
echo ""

IFS=',' read -ra gpus <<< "$GPU_LIST"
num_gpus=${#gpus[@]}

echo "Running inference on $num_gpus GPUs..."

for ((i=0; i<num_gpus; i++)); do
    gpu=${gpus[i]}
    log_file="$LOG_DIR/gpu${gpu}.log"
    echo "Starting GPU $gpu (index $i/$num_gpus)... (log: $log_file)"

    CUDA_VISIBLE_DEVICES=$gpu python inference.py \
        --model_path $MODEL_PATH \
        --test_data_path $TEST_DATA_PATH \
        --output_dir $OUTPUT_DIR \
        --max_pixels 2809856 \
        --min_pixels 12544 \
        --max_new_tokens 512 \
        --batch_size 8 \
        --preprocessed_data_path $PREPROCESSED_DATA_PATH \
        --curr_idx $i \
        --total_idx $num_gpus \
        --pipeline_parallel_size 1 \
        --dtype half > "$log_file" 2>&1 &
done

wait

echo ""
echo "================================================================================"
echo "All GPU processes completed!"
echo "================================================================================"
echo ""

echo "Aggregating results from all GPUs..."
python scripts/aggregate_inference_results.py $OUTPUT_DIR

echo ""
echo "================================================================================"
echo "Inference completed!"
echo "================================================================================"
echo "Results saved to: $OUTPUT_DIR/inference_results_merged.json"
echo "GPU logs saved to: $LOG_DIR/"
echo "================================================================================"
