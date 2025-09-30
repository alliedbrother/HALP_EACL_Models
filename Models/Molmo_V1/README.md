# Molmo VQA Embedding Extractor for RTX 4090

Optimized extraction script for HALP (Hallucination Prediction via Probing) using Molmo-7B-O-0924 on NVIDIA RTX 4090.

## Overview

This script extracts three types of embeddings before text generation for hallucination prediction:

1. **Vision-only representation**: From vision encoder before image-to-text projection
2. **Vision token representation**: At image token boundary in decoder layers (0, n/4, n/2, 3n/4, n)
3. **Query token representation**: At query token boundary in decoder layers (0, n/4, n/2, 3n/4, n)

## Requirements

- **Hardware**: NVIDIA RTX 4090 (24GB VRAM) or similar GPU
- **CUDA**: Version 11.8 or 12.1+
- **Python**: 3.10+

## Installation

```bash
# Install PyTorch with CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt
```

## Usage

### Test Mode (3 samples)
```bash
python run_molmo_extraction_rtx4090.py \
    --vqa-dataset /path/to/sampled_10k_relational_dataset.csv \
    --images-dir /path/to/all_images \
    --output-dir ./test_output \
    --test
```

### Full Extraction (10k samples)
```bash
python run_molmo_extraction_rtx4090.py \
    --vqa-dataset /path/to/sampled_10k_relational_dataset.csv \
    --images-dir /path/to/all_images \
    --output-dir ./output \
    --checkpoint-interval 1000
```

### Arguments

- `--vqa-dataset`: Path to VQA CSV file (required)
- `--images-dir`: Directory containing images (required)
- `--output-dir`: Output directory for HDF5 files (default: ./output)
- `--model`: Model path (default: allenai/Molmo-7B-O-0924)
- `--cache-dir`: Model cache directory (default: ./model_cache)
- `--checkpoint-interval`: Save checkpoint every N samples (default: 1000)
- `--test`: Enable test mode (only 3 samples)

## Output Format

HDF5 files saved as `molmo_embeddings_part_XXX.h5` with structure:

```
question_id/
├── question (string)
├── image_id (string)
├── answer (string)                              # Model-generated answer
├── ground_truth_answer (string)                 # Ground truth from dataset
├── vision_only_representation (array: 4096)     # Vision encoder output
├── vision_token_representation/
│   ├── layer_0 (array: 4096)
│   ├── layer_7 (array: 4096)
│   ├── layer_15 (array: 4096)
│   ├── layer_23 (array: 4096)
│   └── layer_31 (array: 4096)
└── query_token_representation/
    ├── layer_0 (array: 4096)
    ├── layer_7 (array: 4096)
    ├── layer_15 (array: 4096)
    ├── layer_23 (array: 4096)
    └── layer_31 (array: 4096)
```

## Optimizations for RTX 4090

- **bfloat16 precision**: Native support on RTX 4090 for 2x memory efficiency
- **Autocast**: Automatic mixed precision for optimal performance
- **Device map**: Automatic model distribution across GPU
- **Checkpointing**: Saves every 1000 samples to prevent data loss

## Inspecting Output

Use the included `inspect_h5.py` script to view HDF5 contents:

```bash
python inspect_h5.py output/molmo_embeddings_part_001.h5
```

## Expected Performance

- **Memory usage**: ~16-18GB VRAM
- **Speed**: ~1-2 samples/second (depends on image size and text length)
- **Full 10k extraction**: ~2-3 hours

## Troubleshooting

### CUDA Out of Memory
- Reduce `--checkpoint-interval` to save more frequently
- Close other GPU-using applications

### Generation Failures
- Check that CUDA is properly installed: `python -c "import torch; print(torch.cuda.is_available())"`
- Verify GPU has enough VRAM: `nvidia-smi`

### Model Download Issues
- Ensure stable internet connection
- Model downloads to `./model_cache` (~26GB)

## Citation

Based on the HALP paper:
```
@article{halp2024,
  title={Detecting Vision-Language Model Hallucinations Before Generation},
  author={[Authors]},
  year={2024}
}
```

## License

Apache 2.0 (following Molmo model license)