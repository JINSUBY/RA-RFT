#!/bin/bash
# Training RA-RFT model with both answerable and refusable queries

export WANDB_PROJECT=Refuse-VTG
export EXP_NAME=rarft/qwen_7b_rarft_final

export PYTHONPATH=".:$PYTHONPATH"
export DEBUG_MODE="true"
export LOG_PATH="/data/jinsuby/time-r1/logs/$EXP_NAME/$EXP_NAME.txt"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# NCCL timeout settings to prevent timeout during reward computation
export NCCL_TIMEOUT=3600
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

OUTDIR=/data/jinsuby/time-r1/logs/$EXP_NAME
BASE_MODEL_NAME_OR_PATH="Boshenxx/Time-R1-7B"

# Create output and log directories
mkdir -p $OUTDIR
mkdir -p $(dirname $LOG_PATH)

# Auto-resume from latest checkpoint
RESUME_FLAG=""
if [ -d "$OUTDIR" ]; then
    LATEST_CHECKPOINT=$(ls -td $OUTDIR/checkpoint-* 2>/dev/null | head -1)
    if [ -n "$LATEST_CHECKPOINT" ]; then
        echo "=========================================="
        echo "✅ CHECKPOINT FOUND: $LATEST_CHECKPOINT"
        echo "Resuming training with:"
        echo "  - Model weights"
        echo "  - Optimizer state (AdamW momentum/variance)"
        echo "  - LR Scheduler state"
        echo "  - RNG state (for reproducibility)"
        echo "  - Global step & epoch"
        echo "=========================================="
        RESUME_FLAG="--resume_from_checkpoint $LATEST_CHECKPOINT"
    else
        echo "=========================================="
        echo "No checkpoint found. Starting training from scratch..."
        echo "=========================================="
    fi
else
    echo "=========================================="
    echo "Output directory does not exist. Starting training from scratch..."
    echo "=========================================="
fi

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12406" \
    main_rarft.py \
    --deepspeed scripts/configs/zero3_offload.json \
    --output_dir $OUTDIR \
    --model_name_or_path $BASE_MODEL_NAME_OR_PATH \
    --train_data_path ./dataset/annotations/hi_vtg_train.json \
    --dataset_name rarft_dataset \
    --max_prompt_length 8192 \
    --max_completion_length 200 \
    --num_generations 8 \
    --generation_batch_size 8 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 true \
    --data_seed 42 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --fix_vit true \
    --slide_window false \
    --num_train_epochs 3 \
    --run_name $EXP_NAME \
    --report_to wandb \
    --reward_funcs format refuse_iou explain_correction \
    --temperature 1.0 \
    --prompt_type rarft \
    --is_curriculum_learning false \
    --logging_dir $OUTDIR \
    --save_steps 50 \
    --save_only_model false \
    --save_total_limit 3 \
    $RESUME_FLAG
