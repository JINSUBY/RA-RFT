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

## 📈 Evaluation

```
**TO BE UPDATED**
```

## 🤖 Model Checkpoints

Pre-trained RA-RFT models will be available on Hugging Face Hub:
- [RA-RFT-Qwen2.5-VL-7B](https://huggingface.co/JINSUBY/RA-RFT-Qwen2.5-VL-7B) (Coming soon)

## 📝 Citation

If you find this work useful for your research, please cite:

```bibtex
@article{lee2025learning,
  title={Learning to Refuse: Refusal-Aware Reinforcement Fine-Tuning for Hard-Irrelevant Queries in Video Temporal Grounding},
  author={Lee, Jin-Seop and Lee, SungJoon and Jung, SeongJun and Li, Boyang and Lee, Jee-Hyong},
  journal={arXiv preprint arXiv:2511.23151},
  year={2025}
}
```

## 🙏 Acknowledgments

This work builds upon several excellent projects:

- [Time-R1](https://github.com/xiaomi-research/time-r1): Foundation for video temporal grounding with reinforcement learning

## 📄 License

This project is released under the Apache 2.0 License. See [LICENSE](LICENSE) for details.