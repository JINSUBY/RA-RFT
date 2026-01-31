# Training Guide

This guide provides detailed information about training RA-RFT models, including reward functions, hyperparameters, and customization options.

## Quick Start

```bash
# Activate environment
conda activate rarft

# Start training with default settings
bash scripts/train_rarft.sh
```

## Training Architecture

RA-RFT uses **Group Relative Policy Optimization (GRPO)** with a multi-reward system designed for refusal-aware temporal grounding.

### Training Pipeline

```
Video + Query → Qwen2.5-VL-7B → Generate 4 Completions
                                       ↓
                    Compute Rewards (format + IoU + refusal)
                                       ↓
                    GRPO Loss + KL Divergence → Update Model
```

## Reward Functions

RA-RFT combines three complementary reward functions:

### 1. Format Reward

**Purpose**: Ensures outputs follow the required RA-RFT format.

**Required Format**:
```xml
<think>thought process</think>
<answer>answer content</answer>
<correction>correction or NIL</correction>
```

**Scoring**:
- ✅ 1.0: All three tags present in correct order
- ❌ 0.0: Missing tags or incorrect format

**Implementation**:
```python
pattern = r"<think>.*?</think>\s*<answer>.*?</answer>\s*<correction>.*?</correction>"
reward = 1.0 if re.fullmatch(pattern, output) else 0.0
```

### 2. Refuse-IoU Reward

**Purpose**: Task-aware temporal localization reward.

**Logic Table**:

| Task Type | Has Timestamp | Reward |
|-----------|---------------|--------|
| Answerable | ✅ Yes | IoU × timestamp_accuracy |
| Answerable | ❌ No | 0.0 (penalty) |
| Refusable | ✅ Yes | 0.0 (penalty) |
| Refusable | ❌ No | 1.0 (correct refusal) |

**IoU Calculation**:
```python
intersection = max(0, min(pred_end, gt_end) - max(pred_start, gt_start))
union = max(pred_end, gt_end) - min(pred_start, gt_start)
iou = intersection / union if union > 0 else 0.0
```

**Timestamp Accuracy**:
```python
# Normalize by video duration for fair comparison
gt_start_norm = gt_start / duration
gt_end_norm = gt_end / duration
pred_start_norm = pred_start / duration
pred_end_norm = pred_end / duration

# Penalize deviation from ground truth
start_accuracy = 1 - abs(gt_start_norm - pred_start_norm)
end_accuracy = 1 - abs(gt_end_norm - pred_end_norm)

# Final reward
reward = iou * start_accuracy * end_accuracy
```

**Example**:
- Video duration: 100s
- GT: [20s, 40s], Pred: [22s, 38s]
- IoU: 0.9, Accuracy: 0.98 × 0.98 = 0.96
- **Final Reward**: 0.9 × 0.96 = **0.864**

### 3. Explain + Correction Reward

**Purpose**: Trains refusal detection and query correction capabilities.

**Components**:

**A. Explain Reward (0.0 - 1.0)**
```python
has_timestamp = bool(re.search(r'\d+\.\d+\s+to\s+\d+\.\d+', answer))

if task_type == "answerable":
    refusal_score = 1.0 if has_timestamp else 0.0
else:  # refusable
    refusal_score = 0.0 if has_timestamp else 1.0
```

**B. Correction Reward (0.0 - 1.0)**

For refusable queries only, measures how well the correction matches expected answers:

```python
from sentence_transformers import SentenceTransformer

# Extract correction from output
correction = extract_correction_content(output)

# Encode using Sentence-BERT
correction_emb = sbert_model.encode(correction)
reference_embs = [sbert_model.encode(ref) for ref in reference_queries]

# Find best match
similarities = [cosine_similarity(correction_emb, ref_emb)
                for ref_emb in reference_embs]
best_similarity = max(similarities)

# Bonus reward with threshold
correction_bonus = max(0, best_similarity - 0.5)  # Threshold = 0.5
```

**Total Reward**:
```python
total_reward = refusal_score + correction_bonus
# Range: [0.0, 1.5] for refusable queries
# Range: [0.0, 1.0] for answerable queries
```

**Example - Refusable Query**:
- Query: "a person is playing basketball"
- Video content: cooking scene
- Model output: refuses (no timestamp) ✅ → 1.0
- Correction: "a person is cooking pasta"
- Best reference: "a person is cooking in the kitchen"
- Similarity: 0.85
- **Correction Bonus**: 0.85 - 0.5 = 0.35
- **Total Reward**: 1.0 + 0.35 = **1.35**

## Hyperparameters

### Core Training Parameters

```bash
# Model
--model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct  # Base model
--prompt_type rarft                               # Prompt template

# GRPO Settings
--num_generations 4          # Samples per prompt (G in paper)
--temperature 1.0            # Sampling temperature (higher = diverse)
--beta 0.05                  # KL divergence coefficient
--use_grpo true              # Use GRPO (vs PPO)

# Optimization
--learning_rate 5e-7         # Learning rate for AdamW
--num_train_epochs 1         # Number of epochs
--per_device_train_batch_size 1       # Batch size per GPU
--gradient_accumulation_steps 1       # Gradient accumulation
--max_grad_norm 1.0          # Gradient clipping

# Generation
--max_completion_length 512  # Max tokens for completion
--max_prompt_length 2048     # Max tokens for prompt

# Hardware
--bf16 true                  # Use bfloat16 precision
--gradient_checkpointing true         # Save memory
--deepspeed scripts/configs/zero3_offload.json  # DeepSpeed config
```

### Reward Weights

All rewards are summed with equal weight (1.0 each):

```python
total_reward = format_reward + iou_reward + refusal_correction_reward
```

To adjust weights, modify the reward functions in `main_rarft.py`:

```python
def custom_weighted_reward(prompts, completions, **kwargs):
    r1 = format_reward(completions)
    r2 = refuse_iou_reward(completions, **kwargs)
    r3 = explain_correction_reward(completions, **kwargs)

    # Custom weights
    return [0.5*a + 1.0*b + 1.5*c for a, b, c in zip(r1, r2, r3)]
```

### DeepSpeed Configuration

The default config (`scripts/configs/zero3_offload.json`) uses:

- **ZeRO Stage 3**: Full parameter sharding
- **Offloading**: Parameters + optimizer states to CPU
- **Memory**: ~40GB per GPU for 7B model

**Key Settings**:
```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    }
  },
  "bf16": {
    "enabled": true
  },
  "gradient_accumulation_steps": 1,
  "gradient_clipping": 1.0
}
```

## Training Workflow

### 1. Data Preparation

Ensure your data is in RIQ format:

```bash
# Verify data format
python -c "
import json
data = json.load(open('dataset/annotations/hi_vtg_train.json'))
print(f'Total samples: {len(data)}')
print(f'First sample keys: {list(data[0].keys())}')
"
```

### 2. Configure Training Script

Edit `scripts/train_rarft.sh`:

```bash
# Experiment name (affects checkpoint save path)
EXP_NAME=rarft/qwen_7b_final

# Dataset path
TRAIN_DATA_PATH=./dataset/annotations/hi_vtg_train.json

# Reward functions (space-separated)
REWARD_FUNCS="format refuse_iou explain_correction"
```

### 3. Launch Training

```bash
# Single-node, 8 GPUs
bash scripts/train_rarft.sh

# Multi-node (adjust hostfile)
deepspeed --hostfile=hostfile scripts/train_rarft.sh
```

### 4. Monitor Training

**Weights & Biases**:
- Automatically logs to W&B if `wandb` is installed
- Project name: "time_r1"
- Run name: based on `EXP_NAME`

**Key Metrics**:
- `reward`: Total reward (sum of all reward functions)
- `rewards/format_reward`: Format compliance
- `rewards/refuse_iou_reward`: Temporal IoU
- `rewards/explain_correction_reward`: Explain + correction
- `completion_length`: Average generation length
- `generation_entropy`: Output diversity
- `kl`: KL divergence from reference model

**Console Output**:
```
Query: a person is cooking...
Task Type: answerable
Generated Completions (4):
  [1] 15.2 to 45.8...
  [2] 14.5 to 46.0...
  [3] No relevant segment found...
  [4] 15.8 to 44.2...
================================================================================
reward_format: [1.0, 1.0, 1.0, 1.0]
reward_refuse_iou: [0.92, 0.88, 0.0, 0.85]
reward_explain: [0.65, 0.72, -0.15, 0.55]
reward_correction: [1.0, 1.0, 0.0, 0.82]
```

### 5. Checkpoint Management

**Checkpoints are saved to**:
```
checkpoints/{EXP_NAME}/
├── checkpoint-100/
├── checkpoint-200/
└── checkpoint-final/
```

**Resume from checkpoint**:
```bash
# In train_rarft.sh, uncomment:
--resume_from_checkpoint checkpoints/rarft/qwen_7b_final/checkpoint-200
```

## Advanced Customization

### Custom Reward Functions

Add new reward functions in `main_rarft.py`:

```python
def my_custom_reward(completions, **kwargs):
    """
    Custom reward function example.

    Args:
        completions: List of generated text completions
        **kwargs: Additional fields from dataset (video_path, task_type, etc.)

    Returns:
        List of rewards (float), one per completion
    """
    rewards = []
    for completion in completions:
        # Your custom logic here
        reward = compute_my_metric(completion)
        rewards.append(reward)

    return rewards

# Then in train script:
--reward_funcs format refuse_iou my_custom_reward
```

### Curriculum Learning

Enable curriculum learning for refusable queries:

In `src/time_r1/rl/timer1_trainer_rarft.py`, uncomment the curriculum section:

```python
# Curriculum: gradually increase difficulty
if self.state.epoch < 1:
    selected_idx = 2  # Easy samples (high similarity)
elif self.state.epoch < 2:
    selected_idx = rng.randint(1, pool_size - 1)  # Medium
else:
    selected_idx = rng.randint(0, pool_size - 1)  # All samples
```

### Model Architecture Changes

**Use larger model**:
```bash
--model_name_or_path Qwen/Qwen2.5-VL-72B-Instruct
```

**Freeze vision encoder**:
```bash
--fix_vit true  # Freeze ViT, only train merger
```

**Use LoRA**:
```bash
--use_peft true
--lora_r 64
--lora_alpha 16
--lora_target_modules q_proj k_proj v_proj o_proj
```

## Troubleshooting

### Issue: OOM (Out of Memory)

**Solutions**:
1. Reduce batch size: `--per_device_train_batch_size 1`
2. Enable gradient checkpointing: `--gradient_checkpointing true`
3. Reduce generation count: `--num_generations 2`
4. Reduce max length: `--max_completion_length 256`
5. Use ZeRO Stage 3 offload (already default)

### Issue: Slow Training

**Solutions**:
1. Use FlashAttention: `--attn_implementation flash_attention_2`
2. Reduce video resolution: `--max_pixels 6422528` (half of default)
3. Increase batch size with gradient accumulation:
   ```bash
   --per_device_train_batch_size 1
   --gradient_accumulation_steps 8
   ```

### Issue: Low Reward Scores

**Debugging**:
1. Check format reward first (should be 1.0 quickly)
2. Verify data format matches expected RIQ structure
3. Inspect generated completions in console output
4. Reduce temperature for more deterministic outputs: `--temperature 0.7`

### Issue: NCCL Timeout

**Solutions**:
1. Set environment variables:
   ```bash
   export NCCL_TIMEOUT=3600000  # 1 hour
   export NCCL_DEBUG=INFO
   ```
2. Check network connectivity between GPUs
3. Verify all GPUs use same random seed (handled automatically)

## Performance Tips

### Optimal Settings for Different Hardware

**8x A100 (80GB)**:
```bash
--per_device_train_batch_size 2
--gradient_accumulation_steps 1
--num_generations 4
--bf16 true
--attn_implementation flash_attention_2
```

**8x A100 (40GB)**:
```bash
--per_device_train_batch_size 1
--gradient_accumulation_steps 2
--num_generations 4
--gradient_checkpointing true
--deepspeed scripts/configs/zero3_offload.json
```

**4x A100 (80GB)**:
```bash
--per_device_train_batch_size 1
--gradient_accumulation_steps 4
--num_generations 4
```

## Expected Training Time

| Configuration | Time per Epoch | Total Time (1 epoch) |
|--------------|----------------|----------------------|
| 8x A100 80GB | ~8 hours | ~8 hours |
| 8x H100 80GB | ~6 hours | ~6 hours |
| 4x A100 80GB | ~16 hours | ~16 hours |

*Based on 10,000 training samples with default settings*

## Evaluation During Training

To evaluate on validation set during training:

```bash
--evaluation_strategy steps
--eval_steps 500
--eval_data_path dataset/annotations/val_riq.json
```

## Next Steps

After training:
1. Evaluate on test sets: See main [README.md](../README.md#evaluation)
2. Convert to vLLM format for faster inference
3. Fine-tune on domain-specific data
4. Experiment with different reward weights

## References

- GRPO Paper: [DeepSeekMath](https://arxiv.org/abs/2402.03300)
- Qwen2.5-VL: [Model Card](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- DeepSpeed: [ZeRO Documentation](https://www.deepspeed.ai/tutorials/zero/)
