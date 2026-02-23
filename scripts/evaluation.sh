#!/bin/bash
# LLM-based evaluation of model responses for the RIQ task

# -----------------------------------------------------------------------
# Configuration — edit these paths before running
# -----------------------------------------------------------------------
DATA_PATH="./inference_output/time-r1_rarft_test2/inference_results_merged.json"
OUTPUT_DIR="./evaluation_results/time-r1_rarft_test2"

NUM_SPLITS=27
NUM_WORKERS=27

# -----------------------------------------------------------------------
# Derived paths
# -----------------------------------------------------------------------
OUT_METRICS="$OUTPUT_DIR/metrics.json"
OUT_DATA="$OUTPUT_DIR/metrics_items.json"

mkdir -p "$OUTPUT_DIR"

echo "================================================================================"
echo "LLM Evaluation"
echo "================================================================================"
echo "Data:        $DATA_PATH"
echo "Metrics out: $OUT_METRICS"
echo "Items out:   $OUT_DATA"
echo "Splits:      $NUM_SPLITS"
echo "Workers:     $NUM_WORKERS"
echo "================================================================================"
echo ""

# evaluate.py imports from src.llm_eval, so we run it from the project root
cd "$(dirname "$0")/.." || exit 1

python evaluate.py \
    --data "$DATA_PATH" \
    --out "$OUT_METRICS" \
    --out_data "$OUT_DATA" \
    --num_splits "$NUM_SPLITS" \
    --num_workers "$NUM_WORKERS"

echo ""
echo "================================================================================"
echo "Evaluation completed!"
echo "================================================================================"
echo "Metrics: $OUT_METRICS"
echo "Items:   $OUT_DATA"
echo "================================================================================"
