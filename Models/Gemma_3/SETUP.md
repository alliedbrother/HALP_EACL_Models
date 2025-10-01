# Gemma 3 12B Setup Guide

Complete setup instructions for running Gemma 3 12B VQA extraction.

## Quick Start

```bash
# Navigate to Gemma3_12B directory
cd /root/akhil/HALP_EACL_Models/Models/Gemma3_12B

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate gemma3_12b

# Verify installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"

# Run test
python run_gemma3_extraction.py \
    --vqa-dataset /path/to/your/vqa_test.csv \
    --images-dir /path/to/your/images/ \
    --output-dir ./test_output \
    --test
```

## Environment Setup

### Method 1: Conda (Recommended)

```bash
# Create environment
conda env create -f environment.yml

# Activate
conda activate gemma3_12b

# Verify
conda list | grep -E "torch|transformers"
```

### Method 2: pip + virtualenv

```bash
# Create virtual environment
python3.10 -m venv gemma3_env

# Activate
source gemma3_env/bin/activate

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt

# Verify
pip list | grep -E "torch|transformers"
```

## Pre-flight Checks

### 1. Check CUDA

```bash
nvidia-smi
```

Expected output:
- CUDA Version: 12.1+
- GPU: RTX 3090/4090 or better with 24GB+ VRAM

### 2. Check Python Packages

```bash
python << EOF
import torch
import transformers

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Transformers version: {transformers.__version__}")

# Check bfloat16 support
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"bfloat16 support: {torch.cuda.is_bf16_supported()}")
EOF
```

Expected output:
- PyTorch version: 2.1.0+
- CUDA available: True
- Transformers version: 4.50.0+
- bfloat16 support: True

### 3. Check Disk Space

```bash
df -h .
```

Ensure at least 50GB free for:
- Model cache (~25GB)
- Output embeddings (varies with dataset size)

### 4. Test Model Download

```bash
python << EOF
from transformers import AutoProcessor

print("Testing model download...")
processor = AutoProcessor.from_pretrained(
    "google/gemma-3-12b-it",
    cache_dir="./model_cache",
    trust_remote_code=True
)
print("✓ Processor loaded successfully")
EOF
```

## Data Preparation

### VQA Dataset Format

Your CSV should have these columns:

```csv
question_id,image_id,question,answer
1,image001.jpg,"What is the main object?","cat"
2,image002.jpg,"What color is the sky?","blue"
```

Example Python code to create CSV:

```python
import pandas as pd

data = {
    'question_id': [1, 2, 3],
    'image_id': ['img1.jpg', 'img2.jpg', 'img3.jpg'],
    'question': ['What color?', 'How many?', 'Where is?'],
    'answer': ['red', '3', 'park']
}

df = pd.DataFrame(data)
df.to_csv('vqa_dataset.csv', index=False)
```

### Images Directory Structure

```
images/
├── img1.jpg
├── img2.jpg
├── img3.jpg
└── ...
```

Images should be:
- Format: JPG, PNG, JPEG
- Any size (will be resized to 896x896)
- RGB or grayscale

## Running Extraction

### Step 1: Test Run (3 samples)

```bash
python run_gemma3_extraction.py \
    --vqa-dataset /path/to/vqa.csv \
    --images-dir /path/to/images/ \
    --output-dir ./test_output \
    --test
```

This will:
- Process 3 random samples
- Create `test_output/gemma3_12b_embeddings_part_001.h5`
- Take ~1-2 minutes

### Step 2: Inspect Test Output

```bash
python inspect_h5.py test_output/gemma3_12b_embeddings_part_001.h5
```

Verify:
- ✓ Metadata is correct
- ✓ Embeddings have expected shapes
- ✓ Questions and answers are stored
- ✓ No error messages

### Step 3: Full Run

```bash
# Run with checkpointing every 1000 samples
python run_gemma3_extraction.py \
    --vqa-dataset /path/to/full_vqa.csv \
    --images-dir /path/to/images/ \
    --output-dir ./output \
    --checkpoint-interval 1000
```

### Step 4: Monitor Progress

In another terminal:

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Check output directory
ls -lh output/

# Monitor logs (if redirected)
tail -f extraction.log
```

## Troubleshooting

### Issue: "CUDA out of memory"

**Solution:**
```bash
# Check GPU memory
nvidia-smi

# Kill other processes using GPU
fuser -v /dev/nvidia*

# Reduce checkpoint interval to save more frequently
python run_gemma3_extraction.py ... --checkpoint-interval 500
```

### Issue: "transformers version too old"

**Solution:**
```bash
# Upgrade transformers
pip install --upgrade transformers>=4.50.0

# Verify
python -c "import transformers; print(transformers.__version__)"
```

### Issue: "Model not found or download fails"

**Solution:**
```bash
# Check internet connection
ping huggingface.co

# Try with HuggingFace token (if model is gated)
export HF_TOKEN="your_token_here"

# Or login
huggingface-cli login
```

### Issue: "Image not found" errors

**Solution:**
```python
# Verify image paths
import pandas as pd
import os

df = pd.read_csv('vqa.csv')
images_dir = '/path/to/images'

for idx, row in df.iterrows():
    img_path = os.path.join(images_dir, row['image_id'])
    if not os.path.exists(img_path):
        print(f"Missing: {img_path}")
```

### Issue: Script crashes without error

**Solution:**
```bash
# Run with verbose logging
python -u run_gemma3_extraction.py ... 2>&1 | tee extraction.log

# Enable Python debugging
python -m pdb run_gemma3_extraction.py ...
```

## Performance Optimization

### 1. Use SSD for Cache

```bash
# Move cache to SSD
export HF_HOME="/path/to/ssd/huggingface_cache"
```

### 2. Pre-download Model

```bash
# Download model before running
python << EOF
from transformers import AutoProcessor, AutoModelForVision2Seq

processor = AutoProcessor.from_pretrained(
    "google/gemma-3-12b-it",
    cache_dir="./model_cache",
    trust_remote_code=True
)

model = AutoModelForVision2Seq.from_pretrained(
    "google/gemma-3-12b-it",
    cache_dir="./model_cache",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map='cpu'  # Download without loading to GPU
)
EOF
```

### 3. Monitor System Resources

```bash
# Install monitoring tools
pip install gpustat

# Monitor in real-time
gpustat -i 1
```

## Expected Outputs

For a dataset with 10,000 samples:

```
output/
├── gemma3_12b_embeddings_part_001.h5  (~500MB)
├── gemma3_12b_embeddings_part_002.h5  (~500MB)
├── ...
└── gemma3_12b_embeddings_part_010.h5  (~500MB)

Total: ~5GB for 10,000 samples
```

Each HDF5 file contains:
- Vision representations: 1000 samples × embedding_dim
- Layer representations: 1000 samples × 5 layers × embedding_dim
- Text data: questions, answers, metadata

## Next Steps

After extraction:

1. **Merge HDF5 files** (optional):
   ```python
   # TODO: Create merge script if needed
   ```

2. **Train HALP probes**:
   ```bash
   # Use embeddings for hallucination prediction
   cd ../../  # Back to project root
   python train_halp_probes.py --embeddings output/
   ```

3. **Analyze results**:
   ```python
   # Load and analyze embeddings
   import h5py
   import numpy as np

   with h5py.File('output/gemma3_12b_embeddings_part_001.h5', 'r') as f:
       # Your analysis code
       pass
   ```

## Support

For issues:
- Check [README.md](README.md) for detailed documentation
- Review [Gemma 3 documentation](https://ai.google.dev/gemma/docs/core)
- Open issue in project repository
