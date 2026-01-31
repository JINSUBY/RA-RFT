# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import math
import os
import random
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import numpy as np
import torch
from datasets import Dataset
from deepspeed.runtime.fp16.loss_scaler import LossScaler
from deepspeed.runtime.zero.config import ZeroStageEnum
from rouge_score import rouge_scorer
from src.time_r1 import TimeR1_Trainer_RARFT
from tqdm import tqdm
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from trl import GRPOConfig, ModelConfig, ScriptArguments, TrlParser, get_peft_config


@dataclass
class MY_GRPOConfig(GRPOConfig):
    fix_vit: bool = field(
        default=False,
        metadata={"help": "Whether to fix the ViT model"},
    )

    slide_window: bool = field(
        default=False,
        metadata={"help": "Whether to use slide window"},
    )
    max_window_layers: int = field(
        default=2, metadata={"help": "sliding window layers bottom"}
    )
    sliding_window_length: int = field(
        default=4096, metadata={"help": "sliding window length"}
    )

    prompt_type: str = field(
        default="rarft",
        metadata={"help": "Prompt type for RA-RFT training."},
    )

    use_grpo: bool = field(
        default=False,
        metadata={"help": "Whether to use GRPO"},
    )


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for RA-RFT training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'format', 'refuse_iou', 'correction', 'explain_correction'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["format", "refuse_iou", "explain_correction"],
        metadata={"help": "List of reward functions. Possible values: 'format', 'refuse_iou', 'explain_correction'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )

    train_data_path: str = field(
        default="./dataset/annotations/hi_vtg_train.json",
        metadata={"help": "Path to the training data JSON file."},
    )

    eval_data_path: str = field(
        default="./dataset/finetune/charades/Charades/charades_annotation/val.json",
        metadata={"help": "Path to the evaluation data JSON file."},
    )

    video_folder: str = field(
        default="./dataset/finetune/charades/Charades/Charades_v1",
        metadata={"help": "Path to the folder containing video files."},
    )

    is_curriculum_learning: bool = field(
        default=False,
        metadata={"help": "Whether to use curriculum learning."},
    )

    is_early_stopping: bool = field(
        default=False,
        metadata={"help": "Whether to use early stopping"},
    )


def parse_timestamp_output(output_string):
    """Parses timestamp output from model completions."""
    answer_matches = re.findall(r"<answer>(.*?)</answer>", output_string, re.DOTALL)

    if not answer_matches:
        return None

    last_answer_content = answer_matches[-1]
    print("last_answer_content:", last_answer_content)

    matches = re.findall(
        r"(\d+\.?\d*) (to|and) (\d+\.?\d*)", last_answer_content, re.IGNORECASE
    )
    if not matches:
        return None
    last_match = matches[-1]
    start_time = float(last_match[0])
    end_time = float(last_match[2])
    return start_time, end_time


def extract_answer_content(completion: str) -> Optional[str]:
    """Extract content from <answer> tags"""
    answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    matches = answer_pattern.findall(completion)
    if matches:
        return matches[-1].strip()
    return None


def extract_correction_content(completion: str) -> Optional[str]:
    """Extract content from <correction> tags"""
    correction_pattern = re.compile(r"<correction>(.*?)</correction>", re.DOTALL)
    matches = correction_pattern.findall(completion)
    if matches:
        return matches[-1].strip()
    return None


# Per-rank SBERT model instances for parallel processing
_sbert_models = {}

def init_sbert_model(accelerator=None):
    """Initialize per-rank Sentence-BERT model for parallel processing

    Each GPU (rank) gets its own SBERT instance to enable true parallel processing.
    Without this, all GPUs share one model instance causing sequential bottleneck.

    Reference: https://github.com/UKPLab/sentence-transformers/issues/3023
    DeepSpeed Zero3 partitions model parameters, causing 'weight' must be 2-D error.
    Solution: Load model with deepspeed.zero.Init(enabled=False) and use
    deepspeed.zero.GatheredParameters during inference.
    """
    global _sbert_models

    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        rank = 0
        world_size = 1

    if rank not in _sbert_models:
        from sentence_transformers import SentenceTransformer
        import deepspeed

        with deepspeed.zero.Init(enabled=False):
            model = SentenceTransformer('all-MiniLM-L6-v2')
            model.eval()

            if torch.cuda.is_available():
                device = f'cuda:{rank % torch.cuda.device_count()}'
                model = model.to(device)

            _sbert_models[rank] = model

    return _sbert_models[rank]


def format_reward(completions, **kwargs):
    """
    Format Reward: Validates RA-RFT output structure.

    Required format: <think>...</think> <answer>...</answer> <correction>...</correction>

    Returns:
        1.0 if all three tags are present in correct order, 0.0 otherwise
    """
    pattern = re.compile(
        r"<think>.*?</think>\s*<answer>.*?</answer>\s*<correction>.*?</correction>",
        re.DOTALL
    )
    matches = [re.fullmatch(pattern, content.strip()) for content in completions]
    rewards = [1.0 if match else 0.0 for match in matches]
    print(f"reward_format: {rewards}")
    return rewards


def refuse_iou_reward(completions, solution=None, task_type=None, **kwargs):
    """
    Refuse-IoU Reward: Task-aware temporal localization reward.

    Logic:
        - Answerable + has timestamp → IoU with ground truth
        - Answerable + no timestamp → 0.0 (penalty)
        - Refusable + has timestamp → 0.0 (penalty, should refuse)
        - Refusable + no timestamp → 1.0 (correct refusal)

    Args:
        completions: Model-generated completions
        solution: Ground truth timestamps for answerable queries
        task_type: Task types ("answerable" or "refusable")
        durations: Video durations (passed in kwargs)

    Returns:
        List of IoU rewards conditioned on task type
    """
    pred_answers = [extract_answer_content(c) for c in completions]
    durations = kwargs.get("durations")

    iou_rewards = [0.0] * len(completions)

    timestamp_pattern = re.compile(r'(\d+\.?\d*)\s+to\s+(\d+\.?\d*)', re.IGNORECASE)

    if task_type is None:
        print("Warning: task_type is None, all refuse_iou rewards will be 0")
        return iou_rewards

    for i, pred_answer in enumerate(pred_answers):
        if pred_answer is None:
            iou_rewards[i] = 0.0
            continue

        match = timestamp_pattern.search(pred_answer)

        if i < len(task_type):
            t_type = task_type[i]

            if t_type == "answerable":
                if match:
                    start_time = float(match.group(1))
                    end_time = float(match.group(2))

                    if solution and solution[i] and durations and durations[i]:
                        gt_start, gt_end = solution[i]
                        duration = durations[i]

                        intersection = max(0, min(end_time, gt_end) - max(start_time, gt_start))
                        union = max(end_time, gt_end) - min(start_time, gt_start)

                        if union > 0:
                            iou = intersection / union

                            gt_start_norm = gt_start / duration
                            gt_end_norm = gt_end / duration
                            pred_start_norm = start_time / duration
                            pred_end_norm = end_time / duration

                            iou_rewards[i] = iou * (1 - abs(gt_start_norm - pred_start_norm)) * (1 - abs(gt_end_norm - pred_end_norm))
                        else:
                            iou_rewards[i] = 0.0
                    else:
                        iou_rewards[i] = 0.0
                else:
                    iou_rewards[i] = 0.0

            elif t_type == "refusable":
                if match:
                    iou_rewards[i] = 0.0
                else:
                    iou_rewards[i] = 1.0

    rewards_formatted = [round(r, 2) for r in iou_rewards]
    print(f"reward_refuse_iou: {rewards_formatted}")
    return iou_rewards


def explain_correction_reward(
    completions,
    gt_answers=None,
    refusable_queries=None,
    task_type=None,
    answerable_query=None,
    **kwargs
):
    """
    Explain + Correction Reward: Combined semantic refusal detection and query correction.

    This reward has two components:

    1. Explain Reward (0.0 - 1.0):
       - Measures how well the model identifies answerable vs refusable queries
       - Uses contrastive similarity: prediction vs ground_truth vs refusable_queries

    2. Correction Reward (0.0 - 1.0):
       - For answerable queries: Must output "NIL" in correction tag
       - For refusable queries: Must output corrected query similar to answerable_query

    Total reward: explain_reward + correction_reward

    Args:
        completions: Model-generated completions
        gt_answers: Ground truth answers (for explain component)
        refusable_queries: Contrastive queries (for explain component)
        task_type: Task types ("answerable" or "refusable")
        answerable_query: Reference queries for correction similarity

    Returns:
        List of combined rewards (explain + correction)
    """
    from sentence_transformers import util
    import torch
    import deepspeed

    pred_answers = [extract_answer_content(c) for c in completions]
    gt_answers_extracted = [extract_answer_content(a) for a in gt_answers]
    reject_answers = [extract_answer_content(a) for a in refusable_queries]
    correction_contents = [extract_correction_content(c) for c in completions]

    explain_rewards = [0.0] * len(completions)
    correction_rewards = [0.0] * len(completions)

    explain_texts = []
    explain_valid_indices = []

    correction_texts = []
    correction_refusable_indices = []
    answerable_query_text = None

    for i, (pred, gt, rej) in enumerate(zip(pred_answers, gt_answers_extracted, reject_answers)):
        if pred is None:
            pred = "None"
            continue
        explain_valid_indices.append(i)
        explain_texts.extend([pred, gt, rej])

    if task_type:
        for i, (correction_content, t_type) in enumerate(zip(correction_contents, task_type)):
            if t_type == "answerable":
                if correction_content is not None and "NIL" in correction_content.strip().upper():
                    correction_rewards[i] = 1.0
                else:
                    correction_rewards[i] = 0.0
            elif t_type == "refusable":
                if correction_content is None:
                    correction_rewards[i] = 0.0
                elif "NIL" in correction_content.strip().upper():
                    correction_rewards[i] = 0.0
                else:
                    if answerable_query and answerable_query[i]:
                        correction_refusable_indices.append(i)
                        correction_texts.append(correction_content)
                        if answerable_query_text is None:
                            answerable_query_text = answerable_query[i]
                    else:
                        correction_rewards[i] = 0.0

    all_texts = explain_texts + correction_texts
    if answerable_query_text:
        all_texts.append(answerable_query_text)

    if all_texts:
        model = init_sbert_model()

        param_list = [p for p in model.parameters()]

        with torch.inference_mode():
            if param_list and hasattr(param_list[0], 'ds_id'):
                with deepspeed.zero.GatheredParameters(param_list, modifier_rank=None):
                    embeddings = model.encode(all_texts, convert_to_tensor=True, show_progress_bar=False, batch_size=32)
            else:
                embeddings = model.encode(all_texts, convert_to_tensor=True, show_progress_bar=False, batch_size=32)

        num_explain_texts = len(explain_texts)
        num_correction_texts = len(correction_texts)

        explain_embs = embeddings[:num_explain_texts]
        correction_embs = embeddings[num_explain_texts:num_explain_texts + num_correction_texts]
        answerable_query_emb = embeddings[-1] if answerable_query_text else None

        for idx, valid_idx in enumerate(explain_valid_indices):
            base = idx * 3
            pred_emb = explain_embs[base]
            gt_emb = explain_embs[base + 1]
            rej_emb = explain_embs[base + 2]

            score_a = util.cos_sim(pred_emb, gt_emb).item()
            score_reject_a = util.cos_sim(pred_emb, rej_emb).item()

            score_a = (score_a + 1.0) / 2.0
            score_reject_a = (score_reject_a + 1.0) / 2.0

            explain_rewards[valid_idx] = float(score_a - score_reject_a)

        if answerable_query_emb is not None:
            for idx, refusable_idx in enumerate(correction_refusable_indices):
                correction_emb = correction_embs[idx]

                score = util.cos_sim(correction_emb, answerable_query_emb).item()
                score = (score + 1.0) / 2.0
                correction_rewards[refusable_idx] = float(score)

    total_rewards = [r + c for r, c in zip(explain_rewards, correction_rewards)]

    explain_formatted = [round(r, 2) for r in explain_rewards]
    correction_formatted = [round(c, 2) for c in correction_rewards]

    print(f"reward_explain: {explain_formatted}")
    print(f"reward_correction: {correction_formatted}")

    return total_rewards


reward_funcs_registry = {
    "format": format_reward,
    "refuse_iou": refuse_iou_reward,
    "explain_correction": explain_correction_reward,
}

metric_funcs_registry = {}


def load_json_dataset_rarft(
    train_data_path, is_curriculum_learning=False, preprocessed_data_path=None
):
    """
    Load RA-RFT dataset from hi_vtg_train.json format.
    Returns individual examples (not paired) - each answerable and refusable query is a separate example.
    """

    def create_dataset_from_json(file_path, split_name):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        examples = []
        
        for item in tqdm(data, desc=f"Processing {split_name} videos"):
            video_path = item.get("video")
            grpo_data = item.get("GRPO", [])
            duration = item.get("duration")
            timestamp = item.get("timestamp")

            answerable_queries = [g for g in grpo_data if g.get("r") == True]
            refusable_queries = [g for g in grpo_data if g.get("r") == False]

            pos_query = answerable_queries[0]
            neg_query = random.choice(refusable_queries)

            pos_answer = pos_query.get("a", "").strip()
            neg_answer = neg_query.get("a", "").strip()

            positive_data = {
                "task_type": "answerable",
                "problem": pos_query.get("q", "").strip(),
                "gt_answers": pos_answer,
                "refusable_queries": neg_answer,
                "answerable_query": None,
                "choices": "",
                "solution": (float(timestamp[0]), float(timestamp[1])),
                "video_path": video_path,
                "durations": duration,
                "preprocessed_path": "",
            }
            examples.append(positive_data)

            refusable_pool = [
                {
                    "problem": refusable_queries[i].get("q", "").strip(),
                    "gt_answers": refusable_queries[i].get("a", "").strip(),
                }
                for i in range(min(3, len(refusable_queries)))
            ]

            negative_data = {
                "task_type": "refusable",
                "refusable_pool_json": json.dumps(refusable_pool),
                "problem": "",
                "gt_answers": "",
                "refusable_queries": pos_answer,
                "answerable_query": pos_query.get("q", "").strip(),
                "choices": "",
                "solution": None,
                "video_path": video_path,
                "durations": duration,
                "preprocessed_path": "",
            }
            examples.append(negative_data)

        if not examples:
            return None

        print("is_curriculum_learning:", is_curriculum_learning)
        if not is_curriculum_learning:
            random.shuffle(examples)

        print(f"Total examples: {len(examples)}")
        answerable_count = sum(1 for ex in examples if ex.get('task_type') == 'answerable')
        refusable_count = sum(1 for ex in examples if ex.get('task_type') == 'refusable')
        print(f"Answerable queries: {answerable_count}")
        print(f"Refusable query pools: {refusable_count} (each pool has 3 queries)")
        print(f"Total refusable queries available: {refusable_count * 3}")
        for i, ex in enumerate(examples[:3]):
            if ex.get('task_type') == 'answerable':
                print(f"sample {i+1} [answerable]: {ex['problem'][:50]}...")
            else:
                pool = json.loads(ex['refusable_pool_json'])
                print(f"sample {i+1} [refusable pool]: {len(pool)} queries available")

        dataset = Dataset.from_list(examples)

        def __getitem__(self, idx):
            example = dataset[idx]
            return example

        from types import MethodType

        dataset.__getitem__ = MethodType(__getitem__, dataset)
        return dataset

    train_dataset = create_dataset_from_json(train_data_path, "train")

    return train_dataset


class SaveEpochEndCallback(TrainerCallback):
    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.is_world_process_zero:
            trainer = kwargs.get("trainer")
            if trainer is None:
                return

            epoch_checkpoint_dir = os.path.join(
                args.output_dir, f"epoch-{int(state.epoch)}"
            )

            print(
                f"\n{'='*20} Callback: Saving model checkpoint at end of epoch {int(state.epoch)} to {epoch_checkpoint_dir} {'='*20}\n"
            )
            trainer.save_model(epoch_checkpoint_dir)


class StopAfterNEpochsCallback(TrainerCallback):
    def __init__(self, num_epochs_to_train=1):
        super().__init__()
        self.num_epochs_to_train = num_epochs_to_train
        print(
            f"Callback initialized: Training will stop after {self.num_epochs_to_train} completed epoch(s)."
        )

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.epoch >= self.num_epochs_to_train:
            print(
                f"Epoch {state.epoch:.0f} completed. Stopping training as per StopAfterNEpochsCallback (target: {self.num_epochs_to_train} epoch(s))."
            )
            control.should_training_stop = True


def set_global_seed(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def main(script_args, training_args, model_args):

    set_global_seed(42)

    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    metric_funcs = list(metric_funcs_registry.values())

    dataset = load_json_dataset_rarft(
        script_args.train_data_path,
        script_args.is_curriculum_learning,
    )

    trainer_cls = TimeR1_Trainer_RARFT
    print("using: ", trainer_cls)

    callbacks_list = []
    if script_args.is_early_stopping:
        callbacks_list.append(StopAfterNEpochsCallback())

    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        metric_funcs=metric_funcs,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        callbacks=callbacks_list,
    )

    if training_args.resume_from_checkpoint is not None:
        trainer_state_path = os.path.join(
            training_args.resume_from_checkpoint, "trainer_state.json"
        )
        if os.path.exists(trainer_state_path):
            print(f"Loading trainer state from: {trainer_state_path}")
            with open(trainer_state_path, "r") as f:
                trainer_state = json.load(f)
            resumed_global_step = trainer_state.get("global_step", 0)

        num_micro_batches_per_epoch_per_gpu = len(trainer.get_train_dataloader())
        total_steps = math.ceil(
            trainer.args.num_train_epochs
            * num_micro_batches_per_epoch_per_gpu
            / trainer.args.gradient_accumulation_steps
        )

        remaining_steps = total_steps - resumed_global_step

        print(f"Resume Info:")
        print(f"  - Resumed from step: {resumed_global_step}")
        print(f"  - Total steps for {trainer.args.num_train_epochs} epochs: {total_steps}")
        print(f"  - Remaining steps: {remaining_steps}")

        print(
            f"Resuming training from checkpoint: {training_args.resume_from_checkpoint}"
        )
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        trainer.train()

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, MY_GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
