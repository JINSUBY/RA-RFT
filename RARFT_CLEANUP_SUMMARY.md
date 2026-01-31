# RA-RFT Repository Cleanup Summary

## Overview
This document summarizes the cleanup performed to prepare a publication-ready RA-RFT repository from the original time-r1 codebase.

## Files Created

### 1. main_rarft.py
**Location**: `/data/jinsuby/video_relevance/time-r1-github/main_rarft.py`

**Source**: Cleaned from `/data/jinsuby/video_relevance/time-r1/main_grpo_relevance.py`

**Changes Made**:
- **Kept only 3 reward functions**:
  - `format_reward_v2` - Validates RIQ_v2 format (<think><answer><correction>)
  - `conditioned_iou_timestamp_reward_v2` - Task-aware IoU reward
  - `refusal_v1_correction_v1_reward` - Combined refusal and correction reward

- **Removed functions**:
  - All other reward variants (v2-v7)
  - Judge-based reward variants
  - GTE model initialization
  - Qwen judge model initialization
  - compute_similarity_score function (unused in kept functions)
  - relevance_reward variants (v1-v7)
  - relevance_judge_model variants (v1-v2)
  - iou_timestamp_reward (basic version)
  - Other correction variants

- **Terminology changes applied**:
  - `relevance` → `refusal`
  - `is_relevant` → `should_refuse`
  - `irrelevant` → `refusable`
  - `gt_answers_contrast` → `refusable_queries`
  - `relevant_query` → `answerable_query`
  - `relevance_reward` → `refusal_reward`
  - `grpo_relevance` → `rarft_dataset`
  - `load_json_dataset_riq` → `load_json_dataset_rarft`

- **Updated prompts in docstrings**:
  - "relevant samples" → "answerable samples"
  - "irrelevant samples" → "refusable samples"
  - Task types changed from "relevant"/"irrelevant" to "answerable"/"refusable"

- **Removed Korean comments**: None found in the original file

- **Kept essential imports and license header**: Apache 2.0 license preserved

### 2. scripts/train_rarft.sh
**Location**: `/data/jinsuby/video_relevance/time-r1-github/scripts/train_rarft.sh`

**Source**: Cleaned from `/data/jinsuby/video_relevance/time-r1/scripts/grpo/train_grpo_relevance_cvpr_rarft.sh`

**Changes Made**:
- Updated `EXP_NAME` to `rarft/qwen_7b_rarft_final`
- Removed all commented experiment names
- Updated script path to reference `main_rarft.py`
- Updated dataset_name to `rarft_dataset`
- Simplified checkpoint resume logic (already clean in original)
- Kept only the final working configuration
- Updated reward functions to match cleaned set: `format_v2 conditioned_iou_v2 refusal_v1_correction_v1`

## Terminology Mapping

| Original Term | Publication Term | Context |
|--------------|------------------|---------|
| relevance | refusal | General concept |
| is_relevant | should_refuse | Boolean checks |
| irrelevant | refusable | Queries that should be refused |
| relevant | answerable | Queries that can be answered |
| gt_answers_contrast | refusable_queries | Contrastive answers |
| relevant_query | answerable_query | Reference query for correction |
| relevance_reward | refusal_reward | Reward function name |
| grpo_relevance | rarft_dataset | Dataset identifier |
| GRPO_Relevance | RA-RFT | Method name |

## Reward Functions Explanation

### format_reward_v2
Validates that the model output contains all three required tags in correct order:
- `<think>...</think>` - Reasoning process
- `<answer>...</answer>` - Final answer
- `<correction>...</correction>` - Correction or NIL

Returns: 1.0 if valid format, 0.0 otherwise

### conditioned_iou_timestamp_reward_v2
Task-aware temporal IoU reward that handles both answerable and refusable queries:

**For answerable queries**:
- With timestamp ("XX to XX"): Compute IoU with ground truth
- Without timestamp: 0.0 (should have timestamps)

**For refusable queries**:
- With timestamp: 0.0 (should NOT output timestamps)
- Without timestamp: 1.0 (correct behavior)

### refusal_v1_correction_v1_reward
Combined reward function (sum of two components):

**1. Refusal Reward (contrastive similarity)**:
- Compares predicted answer against:
  - Ground truth answer (positive)
  - Refusable query answer (negative)
- Reward = similarity(pred, gt) - similarity(pred, refusable)

**2. Correction Reward**:
- For answerable queries: Must output `<correction>NIL</correction>`
- For refusable queries: Must output corrected query (similarity with answerable query)

## Additional Files Needed

To make the repository fully functional, the following files from the original repository need to be copied/created:

### Required from time-r1:
1. `src/time_r1/rl/timer1_trainer_grpo_relevance.py` → Needs terminology updates
2. `src/time_r1/rl/__init__.py` → Needs export updates
3. `src/time_r1/__init__.py` → Import updates
4. `src/utils/process_data.py` → For `process_vision_info_v3`
5. `scripts/zero3_offload.json` → DeepSpeed config

### Terminology Updates Needed in Trainer File

In `timer1_trainer_grpo_relevance.py`, update these occurrences:

**Line 61-79**: Update prompt templates
- "relevant to the query" → "answerable for the query"
- "relevant segment exists" → "segment is answerable"
- "relevant segment" → "answerable segment"
- "relevant query to the video" → "answerable query for the video"

**Line 568-600**: Update dynamic sampling logic
- `'irrelevant'` → `'refusable'` (as string literal in code)
- `'irrelevant_pool_json'` → `'refusable_pool_json'`
- "Deserialize irrelevant query pool" → "Deserialize refusable query pool"

Note: The term "irrelevant" on line 449 ("this check is irrelevant") is a general English word, not domain terminology, so it should remain unchanged.

## Key Design Decisions

1. **Kept SBERT model initialization**: Essential for similarity computation in all reward functions
2. **Removed GTE/Qwen judge**: Not used in the three core reward functions
3. **Preserved batch encoding**: Critical performance optimization
4. **Kept full dataset loading logic**: Includes curriculum learning and dynamic sampling
5. **Maintained checkpoint resume logic**: Production-ready training recovery

## Testing Recommendations

Before publication, verify:
1. All imports resolve correctly
2. Reward functions compute without errors
3. Training script executes with minimal data
4. Terminology is consistent across all files
5. No references to removed reward variants remain
6. Dataset loading handles both answerable/refusable samples correctly

## Publication Readiness Checklist

- [x] Removed experimental reward variants
- [x] Applied systematic terminology changes
- [x] Cleaned training script
- [x] Preserved Apache 2.0 license
- [x] Documented reward functions
- [ ] Update trainer file with terminology changes
- [ ] Copy required utility files
- [ ] Update __init__.py exports
- [ ] Add README with usage instructions
- [ ] Test end-to-end training pipeline
- [ ] Verify all paths and imports work in new location

## Notes

- The original codebase had 11 different reward function variants
- Final version uses only 3 core functions (27% of original)
- Code reduction improves maintainability and reproducibility
- All Korean comments were already absent from the source files
- Curriculum learning logic is preserved but can be toggled via flag
