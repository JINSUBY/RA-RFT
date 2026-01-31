# Installation Guide

This guide provides detailed instructions for setting up the RA-RFT training environment.

## System Requirements

### Hardware Requirements

**Minimum Configuration:**
- GPU: 8x NVIDIA A100 (80GB) or H100
- RAM: 512GB+ system memory
- Storage: 500GB+ free disk space
- CPU: 64+ cores recommended

**Tested Configurations:**
- 8x NVIDIA A100 (80GB) - Used for main experiments
- 8x NVIDIA H100 (80GB) - Alternative configuration

### Software Requirements

- **OS**: Ubuntu 20.04 LTS or later
- **Python**: 3.10.12 (tested version)
- **CUDA**: 12.4 or higher
- **cuDNN**: 8.9.0 or higher

## Installation Steps

### 1. Environment Setup

Create a dedicated conda environment:

```bash
# Create environment
conda create -n rarft python=3.10.12 -y
conda activate rarft

# Verify Python version
python --version  # Should output: Python 3.10.12
```

### 2. PyTorch Installation

Install PyTorch with CUDA 12.4 support:

```bash
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Verify PyTorch installation:

```python
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}')"
```

Expected output:
```
PyTorch: 2.5.1+cu124
CUDA Available: True
CUDA Version: 12.4
```

### 3. Core Dependencies

Install the main dependencies:

```bash
# Hugging Face ecosystem
pip install transformers==4.48.3
pip install accelerate==1.3.0
pip install peft==0.14.0
pip install trl==0.13.0

# DeepSpeed for distributed training
pip install deepspeed==0.16.4

# Qwen2.5-VL utilities
pip install qwen-vl-utils==0.0.8
```

### 4. Video Processing Libraries

```bash
# Video decoding
pip install av==14.0.1
pip install decord==0.6.0

# Computer vision
pip install opencv-python==4.10.0.84
```

### 5. Evaluation Metrics

```bash
# Sentence-BERT for refusal reward computation
pip install sentence-transformers==2.2.2

# ROUGE score for text similarity
pip install rouge-score==0.1.2
```

### 6. Flash Attention (Optional but Recommended)

Flash Attention significantly speeds up training:

```bash
pip install flash-attn --no-build-isolation
```

**Note**: This requires CUDA and may take 5-10 minutes to compile.

If installation fails, you can still train without flash attention by setting:
```bash
--attn_implementation eager
```

### 7. Additional Utilities

```bash
# Data handling
pip install datasets==3.2.0
pip install numpy==1.26.4
pip install pandas==2.0.3

# Experiment tracking
pip install wandb==0.18.7

# Progress bars
pip install tqdm==4.66.1

# Version checking
pip install packaging==23.1
```

### 8. Verification

Verify all critical packages are installed:

```bash
python -c "
import torch
import transformers
import accelerate
import deepspeed
import peft
import trl
import av
import sentence_transformers

print('All core packages imported successfully!')
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'Accelerate: {accelerate.__version__}')
print(f'DeepSpeed: {deepspeed.__version__}')
print(f'PEFT: {peft.__version__}')
print(f'TRL: {trl.__version__}')
"
```

## Alternative: One-Command Installation

For a quick setup, install all dependencies at once:

```bash
conda create -n rarft python=3.10.12 -y
conda activate rarft
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution**: Reduce batch size or enable gradient checkpointing:

```bash
--per_device_train_batch_size 1
--gradient_checkpointing true
```

### Issue: Flash Attention Installation Fails

**Solution**: Skip flash attention and use eager attention:

```bash
# In train_rarft.sh, change:
--attn_implementation eager
```

### Issue: DeepSpeed ZeRO Stage 3 Errors

**Solution**: Verify DeepSpeed config and NCCL environment:

```bash
# Check NCCL version
python -c "import torch; print(torch.cuda.nccl.version())"

# Set NCCL debug mode
export NCCL_DEBUG=INFO
```

### Issue: Sentence-BERT Import Error

**Solution**: Ensure sentence-transformers is correctly installed:

```bash
pip uninstall sentence-transformers -y
pip install sentence-transformers==2.2.2
```

### Issue: Video Decoding Errors

**Solution**: Install video codec dependencies:

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y ffmpeg libavcodec-dev libavformat-dev libavutil-dev

# Then reinstall av
pip uninstall av -y
pip install av==14.0.1
```

## Conda Environment Export

After successful installation, export your environment for reproducibility:

```bash
conda env export > environment.yml
pip freeze > requirements_frozen.txt
```

## Next Steps

Once installation is complete:

1. Prepare your dataset following [DATA.md](DATA.md)
2. Configure training parameters in `scripts/train_rarft.sh`
3. Start training following [TRAINING.md](TRAINING.md)

## Hardware-Specific Notes

### For A100 GPUs

- Enable TF32 for faster training:
  ```bash
  export NVIDIA_TF32_OVERRIDE=1
  ```

### For H100 GPUs

- Use FlashAttention-2 for optimal performance
- Consider enabling FP8 training (experimental):
  ```bash
  --bf16 false
  --fp8 true
  ```

### For Multi-Node Training

Set up NCCL environment variables:

```bash
export NCCL_SOCKET_IFNAME=eth0  # Adjust for your network interface
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_NET_GDR_LEVEL=3
```

## Support

If you encounter issues not covered here:

1. Check the [GitHub Issues](https://github.com/JINSUBY/RA-RFT/issues)
2. Review DeepSpeed documentation: https://www.deepspeed.ai/
3. Check Transformers documentation: https://huggingface.co/docs/transformers
