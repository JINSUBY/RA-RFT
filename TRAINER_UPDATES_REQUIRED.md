# Required Updates for timer1_trainer_grpo_relevance.py

## Overview
This document specifies the exact terminology changes needed in the trainer file to align with RA-RFT publication terminology.

## String Replacements Required

Apply these replacements systematically throughout the file:

### In Prompt Templates (Lines 61-92)

**Pattern 1**: "relevant to the query"
```python
# BEFORE:
"determine whether the video contains a segment that is relevant to the query"

# AFTER:
"determine whether the video contains a segment that is answerable for the query"
```

**Pattern 2**: "relevant segment exists"
```python
# BEFORE:
"**If a relevant segment exists**"

# AFTER:
"**If an answerable segment exists**"
```

**Pattern 3**: "no relevant segment exists"
```python
# BEFORE:
"**If no relevant segment exists**"

# AFTER:
"**If no answerable segment exists**"
```

**Pattern 4**: "relevant segment"
```python
# BEFORE:
"does not contain a relevant segment"

# AFTER:
"does not contain an answerable segment"
```

**Pattern 5**: "relevant query to the video"
```python
# BEFORE:
"output a corrected query that is likely to be relevant query to the video"

# AFTER:
"output a corrected query that is likely to be answerable for the video"
```

### In Dynamic Sampling Logic (Lines 568-600)

**Pattern 6**: task_type string literal
```python
# BEFORE (line 570):
if inputs[0].get('task_type') == 'irrelevant' and 'irrelevant_pool_json' in inputs[0]:

# AFTER:
if inputs[0].get('task_type') == 'refusable' and 'refusable_pool_json' in inputs[0]:
```

**Pattern 7**: pool JSON key
```python
# BEFORE (line 575):
pool = json.loads(inputs[0]['irrelevant_pool_json'])

# AFTER:
pool = json.loads(inputs[0]['refusable_pool_json'])
```

**Pattern 8**: Comment
```python
# BEFORE (line 574):
# Deserialize irrelevant query pool

# AFTER:
# Deserialize refusable query pool
```

## DO NOT Change

**Line 449**: Keep this unchanged (general English usage)
```python
# CORRECT (keep as-is):
# model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant.
```

This is a general English word meaning "not applicable", not domain terminology.

## Systematic Search-and-Replace Approach

If using automated tools, apply these replacements IN ORDER:

1. `"relevant to the query"` → `"answerable for the query"`
2. `"a relevant segment"` → `"an answerable segment"`
3. `"no relevant segment"` → `"no answerable segment"`
4. `"relevant query to the video"` → `"answerable query for the video"`
5. `'irrelevant'` (as string literal) → `'refusable'` (ONLY in task_type checks)
6. `'irrelevant_pool_json'` → `'refusable_pool_json'`
7. `"irrelevant query pool"` → `"refusable query pool"` (in comments)

## Verification

After applying changes, verify:
- All 17 occurrences of domain-specific "relevant/irrelevant" are updated
- General English usage remains unchanged (line 449)
- String literals in code match dataset keys ('refusable', 'refusable_pool_json')
- Task type checks use 'answerable' and 'refusable' consistently

## Quick Sed Commands (Optional)

For automated replacement:
```bash
# Backup first
cp timer1_trainer_grpo_relevance.py timer1_trainer_grpo_relevance.py.backup

# Apply replacements (in this exact order)
sed -i 's/relevant to the query/answerable for the query/g' timer1_trainer_grpo_relevance.py
sed -i 's/a relevant segment/an answerable segment/g' timer1_trainer_grpo_relevance.py
sed -i 's/no relevant segment/no answerable segment/g' timer1_trainer_grpo_relevance.py
sed -i 's/relevant query to the video/answerable query for the video/g' timer1_trainer_grpo_relevance.py
sed -i "s/'irrelevant'/'refusable'/g" timer1_trainer_grpo_relevance.py
sed -i "s/'irrelevant_pool_json'/'refusable_pool_json'/g" timer1_trainer_grpo_relevance.py
sed -i 's/irrelevant query pool/refusable query pool/g' timer1_trainer_grpo_relevance.py

# Note: Line 449 should remain unchanged as it's general English usage
```

## Expected Result

Total changes: 17 occurrences updated across prompt templates and dynamic sampling logic.
