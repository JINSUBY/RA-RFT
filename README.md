# RA-RFT: Refusal-Aware Reinforcement Fine-Tuning for Video Temporal Grounding

[![arXiv](https://img.shields.io/badge/arXiv-2511.23151-b31b1b.svg)](https://arxiv.org/abs/2511.23151)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

Official implementation of **RA-RFT** (Refusal-Aware Reinforcement Fine-Tuning), a novel training framework for video temporal grounding that enables models to both accurately locate temporal segments and appropriately refuse unanswerable queries.

## 🎯 Key Features

- **Refusal Detection**: Models learn to identify when queries cannot be answered from video content
- **Query Correction**: For refusable queries, models suggest corrected queries that would be answerable
- **GRPO Training**: Group Relative Policy Optimization for efficient reinforcement learning
- **Multi-Reward System**: Combined rewards for format compliance, temporal IoU, and refusal accuracy

## 📋 Table of Contents

- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Model Checkpoints](#model-checkpoints)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## 🚀 Installation

### Requirements

- Python 3.10.12 or higher
- CUDA 12.4 or higher
- 8x NVIDIA A100 (80GB) or H100 GPUs recommended
- 500GB+ disk space for datasets and checkpoints

### Setup

```bash
# Clone the repository
git clone https://github.com/JINSUBY/RA-RFT.git
cd RA-RFT

# Create conda environment
conda create -n rarft python=3.10.12
conda activate rarft

# Install PyTorch with CUDA support
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install other dependencies
pip install -r requirements.txt

# Install flash-attention (requires CUDA)
pip install flash-attn --no-build-isolation
```

For detailed installation instructions, see [docs/INSTALL.md](docs/INSTALL.md).

## 📊 Dataset Preparation

RA-RFT uses the RIQ (Refusal-aware Instance-level Query) dataset format, which contains:
- **Answerable queries**: Standard temporal grounding queries with timestamps
- **Refusable queries**: Unanswerable queries requiring refusal and correction

### Data Format

```json
{
  "video": "video_id",
  "video_path": "/path/to/video.mp4",
  "duration": 120.5,
  "problem": "query text",
  "task_type": "answerable",
  "gt_answers": [
    {"answer": [10.5, 25.3]}
  ]
}
```

For refusable queries:
```json
{
  "video": "video_id",
  "video_path": "/path/to/video.mp4",
  "duration": 120.5,
  "problem": "unanswerable query",
  "task_type": "refusable",
  "refusable_queries": [
    {"problem": "alternative query 1", "gt_answers": [...]},
    {"problem": "alternative query 2", "gt_answers": [...]}
  ],
  "gt_answers": [{"answer": [-1, -1]}]
}
```

See [docs/DATA.md](docs/DATA.md) for complete data format specifications.

## 🏋️ Training

### Quick Start

```bash
# Train RA-RFT on RIQ dataset
bash scripts/train_rarft.sh
```

### Key Training Parameters

```bash
--model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct  # Base model
--dataset_name rarft_dataset                       # Dataset identifier
--reward_funcs format refuse_iou explain_correction
--num_generations 4                                # Samples per prompt
--temperature 1.0                                  # Sampling temperature
--per_device_train_batch_size 1                   # Batch size per GPU
--gradient_accumulation_steps 1                   # Gradient accumulation
--num_train_epochs 1                              # Training epochs
--learning_rate 5e-7                              # Learning rate
--beta 0.05                                       # KL divergence coefficient
```

### Reward Functions

1. **format**: Validates RA-RFT format (`<think>...</think> <answer>...</answer> <correction>...</correction>`)
2. **refuse_iou**: Task-aware temporal IoU (rewards correct timestamps for answerable queries, penalizes timestamps for refusable queries)
3. **explain_correction**: Combined reward for refusal detection and query correction quality

For detailed training instructions, see [docs/TRAINING.md](docs/TRAINING.md).

## 📈 Evaluation

```bash
# Evaluate on ActivityNet test set
python evaluate.py \
  --model_path checkpoints/rarft_qwen_7b \
  --dataset activitynet \
  --split test

# Evaluate on Charades-STA test set
python evaluate.py \
  --model_path checkpoints/rarft_qwen_7b \
  --dataset charades \
  --split test
```

## 🤖 Model Checkpoints

Pre-trained RA-RFT models will be available on Hugging Face Hub:

- [RA-RFT-Qwen2.5-VL-7B](https://huggingface.co/JINSUBY/RA-RFT-Qwen2.5-VL-7B) (Coming soon)

## 📝 Citation

If you find this work useful for your research, please cite:

```bibtex
@article{rarft2025,
  title={RA-RFT: Refusal-Aware Reinforcement Fine-Tuning for Video Temporal Grounding},
  author={[Your Name]},
  journal={arXiv preprint arXiv:2511.23151},
  year={2025}
}
```

## 🙏 Acknowledgments

This work builds upon several excellent projects:

- [Time-R1](https://github.com/xiaomi-research/time-r1): Foundation for video temporal grounding with reinforcement learning
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL): Vision-language backbone model
- [TRL](https://github.com/huggingface/trl): Transformer Reinforcement Learning library
- [DeepSpeed](https://github.com/microsoft/DeepSpeed): Efficient distributed training

## 📄 License

This project is released under the Apache 2.0 License. See [LICENSE](LICENSE) for details.

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact [your email].
