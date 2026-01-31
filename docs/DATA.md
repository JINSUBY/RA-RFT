# Dataset Format Specification

This document describes the data format used for RA-RFT training and evaluation.

## Overview

RA-RFT uses the **RIQ (Refusal-aware Instance-level Query)** dataset format, which extends standard video temporal grounding datasets with refusal detection capabilities.

## Dataset Structure

```
dataset/
└── annotations/
    └── hi_vtg_train.json  # Training data
```

The annotation file is a JSON array where each element represents one training sample.

## Data Fields

### Common Fields (All Samples)

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `video` | string | Unique video identifier | `"v_abc123"` |
| `video_path` | string | Absolute or relative path to video file | `"/data/videos/v_abc123.mp4"` |
| `duration` | float | Video duration in seconds | `120.5` |
| `problem` | string | Query text describing the temporal event | `"a person is walking"` |
| `task_type` | string | Query type: `"answerable"` or `"refusable"` | `"answerable"` |
| `gt_answers` | array | Ground truth annotations | See below |

### Answerable Query Fields

For queries that can be answered (relevant segment exists):

```json
{
  "video": "v_example_001",
  "video_path": "/data/activitynet/videos/v_example_001.mp4",
  "duration": 120.5,
  "problem": "a person is cooking pasta",
  "task_type": "answerable",
  "gt_answers": [
    {
      "answer": [15.2, 45.8]
    }
  ]
}
```

**Ground Truth Format:**
- `answer`: `[start_time, end_time]` in seconds
- Multiple ground truth segments can be provided for ambiguous cases

### Refusable Query Fields

For queries that should be refused (no relevant segment exists):

```json
{
  "video": "v_example_002",
  "video_path": "/data/activitynet/videos/v_example_002.mp4",
  "duration": 85.3,
  "problem": "a person is playing basketball",
  "task_type": "refusable",
  "refusable_queries": [
    {
      "problem": "a person is cooking in the kitchen",
      "gt_answers": [{"answer": [10.5, 30.2]}]
    },
    {
      "problem": "a person is washing dishes",
      "gt_answers": [{"answer": [35.0, 55.8]}]
    },
    {
      "problem": "a person is setting the table",
      "gt_answers": [{"answer": [60.1, 75.4]}]
    }
  ],
  "gt_answers": [
    {
      "answer": [-1, -1]
    }
  ]
}
```

**Refusable Query Specific Fields:**
- `refusable_queries`: Array of alternative queries that **would** be answerable for this video
  - Each alternative has its own `problem` and `gt_answers`
  - Used for training the query correction capability
- `gt_answers`: Always `[{"answer": [-1, -1]}]` for refusable queries
  - The sentinel value `[-1, -1]` indicates no valid temporal segment

**Dynamic Sampling:**
During training, one alternative query is randomly sampled from `refusable_queries` for each forward pass. This curriculum strategy helps the model learn refusal patterns across diverse query types.

## Model Output Format

Models trained with RA-RFT produce structured outputs in XML-style tags:

### Answerable Query Output

```xml
<think>
The video shows a kitchen scene. At <timestep>15.2 to 45.8</timestep>, I can see a person cooking pasta. They are boiling water, adding pasta, and stirring.
</think>
<answer>15.2 to 45.8</answer>
<correction>NIL</correction>
```

### Refusable Query Output

```xml
<think>
The video shows a kitchen scene with cooking activities. However, there is no basketball playing in this video. The query mentions basketball, which is NOT present. Instead, the video shows cooking activities.
</think>
<answer>
The video does not contain basketball playing. The query is unanswerable because the video shows cooking activities in a kitchen, not sports activities.
</answer>
<correction>a person is cooking in the kitchen</correction>
```

## Format Validation

The `format_reward` function validates the following:

1. ✅ **All outputs must have**: `<think>...</think>` + `<answer>...</answer>` + `<correction>...</correction>`
2. ✅ **Tag order**: `<think>` → `<answer>` → `<correction>`
3. ✅ **No extra content**: Only the three specified tags

**Reward:**
- 1.0 if format is valid
- 0.0 if format is invalid

## Refuse-IoU Reward

The `refuse_iou_reward` function checks both task type and timestamp format:

| Task Type | Has Timestamp | Reward Calculation |
|-----------|---------------|-------------------|
| Answerable | Yes | IoU(predicted, ground_truth) × timestamp_accuracy |
| Answerable | No | 0.0 (should output timestamps) |
| Refusable | Yes | 0.0 (should NOT output timestamps) |
| Refusable | No | 1.0 (correct behavior) |

**Timestamp Accuracy:**
```python
timestamp_accuracy = (1 - abs(gt_start_norm - pred_start_norm)) * (1 - abs(gt_end_norm - pred_end_norm))
```

Where `*_norm` values are normalized by video duration to handle videos of different lengths.

## Explain + Correction Reward

The `explain_correction_reward` function:

1. **Detects task type from output**:
   - Presence of timestamp pattern (`XX.XX to XX.XX`) → predicted as answerable
   - Absence of timestamp → predicted as refusable

2. **For answerable queries**:
   - Correct if output contains timestamps
   - Penalized if output refuses

3. **For refusable queries**:
   - Correct if output refuses (no timestamps)
   - **Bonus reward** if correction quality is high (measured by sentence similarity)

**Correction Quality:**
```python
similarity = cosine_similarity(correction_embedding, reference_query_embedding)
correction_bonus = max(0, similarity - threshold)
```

## Data Statistics

The provided `hi_vtg_train.json` contains:

| Split | Total Samples | Answerable | Refusable | Avg Duration |
|-------|--------------|------------|-----------|--------------|
| Train | ~10,000 | ~7,000 | ~3,000 | 120s |

**Refusable Query Sources:**
- 30% from ActivityNet (different event categories)
- 70% from Charades-STA (different activities)

## Creating Your Own Dataset

To create a custom RIQ dataset:

1. **Start with a standard VTG dataset** (e.g., ActivityNet, Charades-STA)

2. **Generate refusable queries**:
   ```python
   # For each video, find queries from other videos
   # that are semantically different but plausible
   refusable_queries = find_semantically_different_queries(
       current_video_embedding,
       all_other_videos,
       min_distance=0.5  # Cosine distance threshold
   )
   ```

3. **Format as RIQ**:
   ```python
   {
       "video": video_id,
       "video_path": video_path,
       "duration": duration,
       "problem": unanswerable_query,
       "task_type": "refusable",
       "refusable_queries": [
           {"problem": alternative_1, "gt_answers": [...]},
           {"problem": alternative_2, "gt_answers": [...]}
       ],
       "gt_answers": [{"answer": [-1, -1]}]
   }
   ```

4. **Balance the dataset**:
   - Target ratio: 70% answerable, 30% refusable
   - Ensure refusable queries cover diverse failure modes

## Data Loading

The trainer automatically handles data loading. Key features:

- **Dynamic Sampling**: For refusable queries, one alternative is randomly selected each iteration
- **Caching**: Video frames are cached for faster training
- **Preprocessing**: Videos are preprocessed to extract features

## Data Augmentation

RA-RFT supports optional data augmentation:

```python
# In training script
--augmentation_prob 0.5  # 50% chance to apply augmentation
```

**Augmentation strategies:**
- Temporal cropping (random video segments)
- Frame sampling rate variation
- Query paraphrasing (for data diversity)

## Evaluation Data

For evaluation, use standard VTG datasets:

- **ActivityNet Captions**: Test set with temporal annotations
- **Charades-STA**: Test set with activity localization

Convert to RIQ format by setting all `task_type` to `"answerable"`.

## Next Steps

After preparing your data:

1. Verify data format: `python scripts/verify_data.py --data_path dataset/annotations/hi_vtg_train.json`
2. Configure training: Edit `scripts/train_rarft.sh`
3. Start training: `bash scripts/train_rarft.sh`

## References

- ActivityNet Captions: http://activity-net.org/
- Charades-STA: https://prior.allenai.org/projects/charades
