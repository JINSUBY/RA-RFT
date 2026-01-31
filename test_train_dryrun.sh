#!/bin/bash

# Dry-run 테스트: 1 스텝만 실행
export CUDA_VISIBLE_DEVICES=0

python main_rarft.py \
  --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
  --train_data_path dataset/annotations/train_riq_onlyquery.json \
  --output_dir /tmp/rarft_dryrun \
  --num_train_epochs 0.001 \
  --save_steps 999999 \
  --logging_steps 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 5e-7 \
  --num_generations 2 \
  --temperature 1.0 \
  --beta 0.05 \
  --max_completion_length 256 \
  --reward_funcs format_v2 conditioned_iou_v2 refusal_v1_correction_v1 \
  --prompt_type riq_v2 \
  --bf16 true \
  --use_grpo true \
  --fix_vit false \
  --max_steps 2 \
  --report_to none 2>&1 | tee /tmp/rarft_dryrun.log

echo ""
echo "========================================="
echo "Dry-run 로그 요약:"
tail -50 /tmp/rarft_dryrun.log
