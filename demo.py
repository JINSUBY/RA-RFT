#!/usr/bin/env python3
"""
Demo script for RA-RFT inference.

Usage:
    python demo.py \
        --model_path checkpoints/rarft_qwen_7b/checkpoint-final \
        --video_path path/to/video.mp4 \
        --query "a person is cooking pasta"
"""

import argparse
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="RA-RFT Demo Inference")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained RA-RFT model checkpoint",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Path to input video file",
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Query describing the temporal event",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (lower = more deterministic)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on",
    )
    return parser.parse_args()


QUESTION_TEMPLATE_RARFT = """Given a video and a query "[EVENT]", determine whether the video contains a segment that is relevant to the query.

**If the query is answerable** (relevant segment exists), output your thought process within the <think> </think> tags, including analysis with specific time ranges (xx.xx to xx.xx) in <timestep> </timestep> tags.
Then, provide the start and end times (in seconds, precise to two decimal places) within the <answer> </answer> tags.
For example: <answer>12.54 to 17.83</answer>. Finally, provide "<correction>NIL</correction>".

**If the query should be refused** (no relevant segment exists), output your thought process within the <think> </think> tags, including analysis with specific time ranges (xx.xx to xx.xx) in <timestep> </timestep> tags.
Then, state explicitly that the video does not contain a relevant segment and explain why you think so within the <answer> </answer> tags.
Finally, based on the given video and query, output a corrected query that is likely to be answerable for the video within the <correction> </correction> tags.
"""


def main():
    args = parse_args()

    print("=" * 80)
    print("RA-RFT Demo Inference")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Video: {args.video_path}")
    print(f"Query: {args.query}")
    print(f"Device: {args.device}")
    print("=" * 80)

    # Load model and processor
    print("\n[1/4] Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(args.model_path)

    print("[2/4] Preparing input...")
    # Prepare prompt
    prompt_text = QUESTION_TEMPLATE_RARFT.replace("[EVENT]", args.query)

    # Prepare conversation with video
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {
                    "type": "video",
                    "video": args.video_path,
                    "total_pixels": 3584 * 28 * 28,
                    "min_pixels": 16 * 28 * 28,
                },
            ],
        }
    ]

    # Apply chat template
    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Process inputs
    inputs = processor(
        text=[text_prompt],
        videos=[args.video_path],
        padding=True,
        return_tensors="pt",
    )

    # Move to device
    inputs = {k: v.to(args.device) for k, v in inputs.items()}

    print("[3/4] Generating response...")
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=True,
        )

    # Decode
    response = processor.batch_decode(
        outputs[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )[0]

    print("[4/4] Parsing response...")
    print("\n" + "=" * 80)
    print("RESULT")
    print("=" * 80)
    print(response)
    print("=" * 80)

    # Parse structured output
    import re

    think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    correction_match = re.search(r"<correction>(.*?)</correction>", response, re.DOTALL)

    if think_match and answer_match and correction_match:
        print("\n📝 STRUCTURED OUTPUT:")
        print("-" * 80)

        print("\n💭 Thought Process:")
        print(think_match.group(1).strip())

        print("\n✅ Answer:")
        answer_text = answer_match.group(1).strip()
        print(answer_text)

        # Check if timestamp is present
        timestamp_match = re.search(r"(\d+\.?\d*)\s+to\s+(\d+\.?\d*)", answer_text)
        if timestamp_match:
            start_time = float(timestamp_match.group(1))
            end_time = float(timestamp_match.group(2))
            print(f"\n⏱️  Temporal Segment: [{start_time:.2f}s - {end_time:.2f}s]")
            print(f"   Duration: {end_time - start_time:.2f}s")
        else:
            print("\n🚫 Query Refused (no relevant segment found)")

        print("\n🔧 Correction:")
        correction_text = correction_match.group(1).strip()
        if correction_text.upper() == "NIL":
            print("NIL (query is answerable)")
        else:
            print(f'"{correction_text}"')
            print("   (suggested corrected query)")
    else:
        print("\n⚠️  Warning: Response does not follow expected format")
        print("   Expected: <think>...</think> <answer>...</answer> <correction>...</correction>")

    print("\n" + "=" * 80)
    print("Demo completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
