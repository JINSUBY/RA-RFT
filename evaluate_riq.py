"""
Evaluation script for RIQ (Relevance-Irrelevance Query) task with vLLM.
"""

import argparse
import json
import os
from typing import List

import torch
from tqdm import tqdm
from transformers import AutoProcessor

from src.vllm_inference.vllm_infer import vllmWrapper
from src.utils import process_vision_info_v3

SYSTEM_PROMPT = "You are a video analysis expert."

QUESTION_TEMPLATE = """Given a video and a query "[EVENT]", determine whether the video contains a segment that is relevant to the query.

**If a relevant segment exists**, output your thought process within the <think> </think> tags, including analysis with specific time ranges (xx.xx to xx.xx) in <timestep> </timestep> tags.
Then, provide the start and end times (in seconds, precise to two decimal places) within the <answer> </answer> tags.
For example: <answer>12.54 to 17.83</answer>. Finally, provide "<correction>NIL</correction>".

**If no relevant segment exists**, output your thought process within the <think> </think> tags, including analysis with specific time ranges (xx.xx to xx.xx) in <timestep> </timestep> tags.
Then, state explicitly that the video does not contain a relevant segment and explain why you think so within the <answer> </answer> tags.
Finally, based on the given video and query, output a corrected query that is likely to be relevant query to the video within the <correction> </correction> tags.
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model on RIQ task with vLLM")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to test data JSON file")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results", help="Directory to save results")
    parser.add_argument("--preprocessed_data_path", type=str, default=None, help="Path to preprocessed video features")
    parser.add_argument("--max_pixels", type=int, default=2809856, help="Maximum pixels for video processing")
    parser.add_argument("--min_pixels", type=int, default=12544, help="Minimum pixels for video processing")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum new tokens to generate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--curr_idx", type=int, default=0, help="Current GPU index for multi-GPU evaluation")
    parser.add_argument("--total_idx", type=int, default=1, help="Total number of GPUs")
    parser.add_argument("--pipeline_parallel_size", type=int, default=1, help="Pipeline parallel size for vLLM")
    return parser.parse_args()


class ModelArgs:
    """Arguments wrapper for vllmWrapper."""
    def __init__(self, args):
        self.model_base = args.model_path
        self.pipeline_parallel_size = args.pipeline_parallel_size
        self.total_pixels = args.max_pixels
        self.max_new_tokens = args.max_new_tokens


def load_model_and_processor(args):
    """Load vLLM model and processor."""
    print(f"Loading model from {args.model_path}")

    model = vllmWrapper(ModelArgs(args))
    processor = AutoProcessor.from_pretrained(args.model_path)
    processor.tokenizer.padding_side = "left"
    processor.image_processor.max_pixels = args.max_pixels
    processor.image_processor.min_pixels = args.min_pixels

    return model, processor


def load_video_features(video_path: str, preprocessed_data_path: str = None):
    """Load video features, using cached preprocessed features if available."""
    if preprocessed_data_path:
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        preprocessed_dir = os.path.join(preprocessed_data_path, video_id)
        video_inputs_file = os.path.join(preprocessed_dir, "video_inputs.pt")
        video_kwargs_file = os.path.join(preprocessed_dir, "video_kwargs.json")

        if os.path.exists(video_inputs_file) and os.path.exists(video_kwargs_file):
            video_inputs = torch.load(video_inputs_file)
            with open(video_kwargs_file, 'r') as f:
                utils = json.load(f)
            return video_inputs, utils["fps"]

    # Fallback: process video directly
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": video_path, "total_pixels": 3584 * 28 * 28, "min_pixels": 16 * 28 * 28},
            {"type": "text", "text": "dummy"},
        ],
    }]
    _, video_inputs, utils = process_vision_info_v3([messages], return_video_kwargs=True)
    return video_inputs, utils["fps"]


def prepare_vllm_inputs(processor, queries: List[str], video_inputs, fps_inputs):
    """Prepare inputs for vLLM inference."""
    fps_value = fps_inputs[0] if isinstance(fps_inputs, list) else fps_inputs
    vllm_inputs = []

    for query in queries:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "video"},
                {"type": "text", "text": QUESTION_TEMPLATE.replace("[EVENT]", query)},
            ]},
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        vllm_inputs.append({
            "raw_prompt_ids": processor.tokenizer.encode(text, add_special_tokens=False),
            "multi_modal_data": {"video": video_inputs},
            "mm_processor_kwargs": {"fps": fps_value},
        })

    return vllm_inputs


def inference_batch(model, processor, queries: List[str], video_inputs, fps_inputs, max_new_tokens: int) -> List[str]:
    """Run vLLM inference on a batch of queries."""
    vllm_inputs = prepare_vllm_inputs(processor, queries, video_inputs, fps_inputs)
    batch_inputs = {
        "raw_prompt_ids": [inp["raw_prompt_ids"] for inp in vllm_inputs],
        "multi_modal_data": [inp["multi_modal_data"] for inp in vllm_inputs],
        "mm_processor_kwargs": [inp["mm_processor_kwargs"] for inp in vllm_inputs],
    }
    return model.generate(batch_inputs, max_new_tokens=max_new_tokens)


def evaluate(args):
    """Main evaluation function."""
    model, processor = load_model_and_processor(args)

    print(f"Loading test data from {args.test_data_path}")
    with open(args.test_data_path, 'r') as f:
        test_data = json.load(f)

    # Split data for multi-GPU evaluation
    if args.total_idx > 1:
        total_samples = len(test_data)
        samples_per_gpu = (total_samples + args.total_idx - 1) // args.total_idx
        start_idx = args.curr_idx * samples_per_gpu
        end_idx = min((args.curr_idx + 1) * samples_per_gpu, total_samples)
        test_data = test_data[start_idx:end_idx]
        print(f"GPU {args.curr_idx}/{args.total_idx}: Processing samples {start_idx}-{end_idx-1} ({len(test_data)} samples)")
    else:
        print(f"Total test samples: {len(test_data)}")

    all_results = []

    for sample in tqdm(test_data, desc="Evaluating"):
        video_path = sample['video']
        qa_items = sample['QA']
        video_inputs, fps_inputs = load_video_features(video_path, args.preprocessed_data_path)

        for batch_start in range(0, len(qa_items), args.batch_size):
            batch_items = qa_items[batch_start:batch_start + args.batch_size]
            batch_queries = [item['q'] for item in batch_items]
            batch_responses = inference_batch(model, processor, batch_queries, video_inputs, fps_inputs, args.max_new_tokens)

            for item, response in zip(batch_items, batch_responses):
                gt_timestamp = sample["timestamp"] if item['r'] else None
                all_results.append({
                    'video': video_path,
                    'query': item['q'],
                    'model_output': response,
                    'gt_relevant': item['r'],
                    'gt_timestamp': gt_timestamp,
                    'gt_output': item['a'],
                    'irrelevance_level': item.get('irrelevance_level', ""),
                    'gt_categories': item.get('c', [])
                })

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_filename = f"evaluation_results_gpu{args.curr_idx}.json" if args.total_idx > 1 else "evaluation_results.json"
    results_path = os.path.join(args.output_dir, results_filename)

    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nProcessed {len(test_data)} samples")
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
