# RA-RFT: Refusal-Aware Reinforcement Fine-Tuning for Video Temporal Grounding

[![arXiv](https://img.shields.io/badge/arXiv-2511.23151-b31b1b.svg)](https://arxiv.org/abs/2511.23151)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

Official implementation of **RA-RFT** (Refusal-Aware Reinforcement Fine-Tuning), a novel training framework for video temporal grounding that enables models to both accurately locate temporal segments and appropriately refuse unanswerable queries.

## 🚀 Installation

### Requirements

- Python 3.10.12 or higher
- CUDA 12.4 or higher
- 8x NVIDIA A100 (80GB) GPUs recommended
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

RA-RFT uses the High-Irrelevant Video Temporal Grounding (HI-VTG) dataset, which contains:
- **Answerable queries**: Standard temporal grounding queries with timestamps
- **Refusable queries**: Unanswerable queries requiring refusal and correction

## 🏋️ Training

### Quick Start

```bash
# Train RA-RFT on RIQ dataset
bash scripts/train_rarft.sh
```

### Reward Functions

1. **format**: Validates RA-RFT format (`<think>...</think> <answer>...</answer> <correction>...</correction>`)
2. **refuse_iou**: Task-aware temporal IoU (rewards correct timestamps for answerable queries, penalizes timestamps for refusable queries)
3. **explain_correction**: Combined reward for refusal explanation and query correction

For detailed training instructions, see [docs/TRAINING.md](docs/TRAINING.md).

## 📈 Inference & Evaluation

The evaluation pipeline consists of two stages:
1. **Inference**: Generate model responses for test queries
2. **Evaluation**: Assess response quality using LLM-based metrics

### Stage 1: Inference (Model Response Generation)

Generate model responses for the HI-VTG test set (Including relevance and irrelevance queries):

```bash
# Single-GPU inference
python inference.py \
    --model_path ./ckpts/your_model \
    --test_data_path ./dataset/anno/test.json \
    --output_dir ./inference_output \
    --preprocessed_data_path ./dataset/preprocessed_video \
    --batch_size 8

# Multi-GPU inference with automatic result aggregation
bash scripts/inference.sh
```

**Key parameters:**
- `--model_path`: Path to trained model checkpoint
- `--test_data_path`: Test dataset JSON file
- `--preprocessed_data_path`: Preprocessed video features (speeds up inference)
- `--batch_size`: Batch size per GPU
- `--curr_idx`, `--total_idx`: For manual multi-GPU data sharding

**Output:** `inference_results_merged.json` containing model responses for all test samples.

### Stage 2: Evaluation (LLM-based Assessment)

Evaluate the generated responses using GPT-4 as a judge:

```bash
# Configure your OpenAI API key
export OPENAI_API_KEY="your-api-key"

# Run LLM evaluation
python evaluate.py \
    --data ./inference_output/inference_results_merged.json \
    --out ./evaluation_results/metrics.json \
    --num_splits 27 \
    --num_workers 27

# Or use the convenience script
bash scripts/evaluation.sh
```

**Evaluation metrics:**
- **Relevance Classification**: Accuracy, precision, recall, F1 for relevant/irrelevant detection
- **RA-IoU** (Relevance-Aware IoU): Temporal localization accuracy with R@0.3, R@0.5, R@0.7
- **RT-IoU** (Reasoning-Tag IoU): Semantic Relevance Category prediction accuracy using Jaccard similarity
- **Reasoning Quality**: LLM-scored quality of refusal explanations (0-5 scale)
- **SBERT Similarity**: Semantic similarity between model response and ground-truth response
- **Hardness-Level Analysis**: Performance breakdown by query difficulty (original/weak/moderate/strong)

**Key parameters:**
- `--data`: Path to inference results (merged JSON)
- `--out`: Output path for evaluation metrics
- `--num_splits`: Number of data splits for parallel processing
- `--num_workers`: Number of parallel worker processes (default: matches num_splits)

**Output files:**
- `metrics.json`: Comprehensive evaluation metrics
- `metrics_items.json`: Per-sample evaluation details with LLM scores

**Note:** The LLM evaluation uses parallel processing to speed up API calls. Adjust `num_workers` based on your API rate limits (recommended: ≤20 to avoid throttling).

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