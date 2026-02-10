#!/bin/bash

# Multi-GPU evaluation script for RIQ (Relevance-Irrelevance Query) task

GPU_LIST="0,1,2,3"
MODEL_PATH="./ckpts/your_model_checkpoint"
OUTPUT_DIR="./logs/evaluation_output"
TEST_DATA_PATH="./dataset/anno/test_data.json"
PREPROCESSED_DATA_PATH="./dataset/preprocessed_data"

echo "================================================================================"
echo "Multi-GPU RIQ Evaluation"
echo "================================================================================"
echo "Model: $MODEL_PATH"
echo "Test data: $TEST_DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "Preprocessed data: $PREPROCESSED_DATA_PATH"
echo "GPUs: $GPU_LIST"
echo ""

IFS=',' read -ra gpus <<< "$GPU_LIST"
num_gpus=${#gpus[@]}

echo "Running evaluation on $num_gpus GPUs..."

for ((i=0; i<num_gpus; i++)); do
    gpu=${gpus[i]}
    echo "Starting GPU $gpu (index $i/$num_gpus)..."

    CUDA_VISIBLE_DEVICES=$gpu python evaluate_riq.py \
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
        --pipeline_parallel_size 1 &
done

wait

echo ""
echo "================================================================================"
echo "All GPU processes completed!"
echo "================================================================================"
echo ""

echo "Aggregating results from all GPUs..."
python scripts/eval/aggregate_eval_results.py $OUTPUT_DIR

echo ""
echo "================================================================================"
echo "Evaluation completed!"
echo "Results saved to: $OUTPUT_DIR/evaluation_results_merged.json"
echo "================================================================================"
