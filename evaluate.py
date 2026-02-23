#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parallel evaluation of model responses with concurrent processing.

This script splits the input data into multiple chunks and evaluates them
in parallel to significantly speed up evaluation when API calls are the bottleneck.

Metrics:
1) overall relevance accuracy
2) per-class precision, recall, F1, within-class accuracy
3) Mean temporal IoU (all)
4) Mean category Jaccard
5) Macro-F1
6) Micro-F1
7) Label-wise F1
"""

import argparse
import json
import multiprocessing as mp
import time
import warnings
from pathlib import Path

from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

from src.llm_eval.processor import (
    _init_worker,
    merge_metrics,
    process_split,
    process_split_with_progress,
    split_data,
)


MAX_RECOMMENDED_WORKERS = 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="Path to JSON dataset")
    ap.add_argument("--out", type=str, required=True, help="Path to save metrics JSON")
    ap.add_argument("--out_data", type=str, default=None,
                    help="Path to save input data with classification results added (optional)")
    ap.add_argument("--num_splits", type=int, default=4,
                    help="Number of parallel splits (default: 4)")
    ap.add_argument("--num_workers", type=int, default=None,
                    help="Number of worker processes (default: num_splits)")
    ap.add_argument("--verbose", action="store_true", help="Show progress information")
    args = ap.parse_args()

    # Load data
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"File not found: {data_path}")
    data = json.loads(data_path.read_text())

    num_splits = max(1, args.num_splits)
    num_workers = args.num_workers if args.num_workers is not None else num_splits

    if num_workers > MAX_RECOMMENDED_WORKERS:
        print(f"\n{'=' * 80}")
        print(f"WARNING: {num_workers} workers is very high and may cause API rate limits.")
        print(f"Recommended maximum: {MAX_RECOMMENDED_WORKERS} workers")
        print(f"{'=' * 80}\n")

    if num_workers == 1:
        num_splits = 1

    print(f"\n{'=' * 80}")
    print(f"Starting parallel evaluation")
    print(f"Total items: {len(data)}")
    print(f"Number of splits: {num_splits}")
    print(f"Number of workers: {num_workers}")
    print(f"{'=' * 80}\n")

    splits = split_data(data, num_splits)
    print(f"Split sizes: {[len(s) for s in splits]}\n")

    if num_workers == 1:
        metrics_list = []
        for idx, split in enumerate(splits):
            metrics_list.append(process_split(split, idx, verbose=args.verbose))
    else:
        print("Starting parallel processing...\n")
        progress_counter = mp.Value('i', 0)
        total_items = len(data)

        with mp.Pool(
            processes=num_workers,
            initializer=_init_worker,
            initargs=(progress_counter,),
        ) as pool:
            async_results = [
                pool.apply_async(process_split_with_progress, (split, idx))
                for idx, split in enumerate(splits)
            ]

            with tqdm(total=total_items, desc="Processing items", ncols=100) as pbar:
                last_value = 0
                while any(not r.ready() for r in async_results):
                    current_value = progress_counter.value
                    if current_value > last_value:
                        pbar.update(current_value - last_value)
                        last_value = current_value
                    time.sleep(0.1)

                current_value = progress_counter.value
                if current_value > last_value:
                    pbar.update(current_value - last_value)

            metrics_list = [r.get() for r in async_results]

    # Merge and print summary
    print(f"\n{'=' * 80}")
    print("Merging results from all splits...")
    merged = merge_metrics(metrics_list)

    print(f"\n{'=' * 80}")
    print(f"Summary ({merged['num_items']} items)")
    print(f"[Relevance] overall acc: {merged['relevance']['overall_accuracy']:.4f}")
    print(f"[RA-IoU] mean(all): {merged['RA-IoU']['mean_all']:.4f}")
    print(
        f"[R@k] R@0.3: {merged['RA-IoU']['R@0.3']:.2f}% | "
        f"R@0.5: {merged['RA-IoU']['R@0.5']:.2f}% | "
        f"R@0.7: {merged['RA-IoU']['R@0.7']:.2f}%"
    )
    print(f"[RT-IoU] {merged['RT-IoU']:.4f}")
    print(
        f"[Irrelevant Samples] mean reasoning score: "
        f"{merged['irrelevant_samples']['mean_reasoning_score']:.4f} | "
        f"count: {merged['irrelevant_samples']['count']}"
    )
    print(
        f"[Irrelevant Samples] mean SBERT similarity: "
        f"{merged['irrelevant_samples']['mean_sbert_similarity']:.4f} | "
        f"SBERT count: {merged['irrelevant_samples']['sbert_count']}"
    )
    print(
        f"[True Negative Samples] mean reasoning score: "
        f"{merged['true_negative_samples']['mean_reasoning_score']:.4f} | "
        f"count: {merged['true_negative_samples']['count']}"
    )
    print(
        f"[True Negative Samples] mean SBERT similarity: "
        f"{merged['true_negative_samples']['mean_sbert_similarity']:.4f} | "
        f"SBERT count: {merged['true_negative_samples']['sbert_count']}"
    )

    # Save metrics JSON (excluding updated_items)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_to_save = {k: v for k, v in merged.items() if k != "updated_items"}
    out_path.write_text(json.dumps(metrics_to_save, ensure_ascii=False, indent=2))
    print(f"\nSaved metrics to: {out_path}")

    # Save updated items with classification results
    updated_items = merged.get("updated_items", [])
    if updated_items:
        if args.out_data:
            out_data_path = Path(args.out_data)
        else:
            out_data_path = out_path.parent / f"{out_path.stem}_items{out_path.suffix}"

        out_data_path.parent.mkdir(parents=True, exist_ok=True)
        out_data_path.write_text(json.dumps(updated_items, ensure_ascii=False, indent=2))
        print(f"Saved data with classification results to: {out_data_path}")
        print(f"   Total items: {len(updated_items)}")
    else:
        print("No updated items to save")

    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
