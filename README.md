# HALP - Hallucination Prediction via Probing

Implementation of HALP (Hallucination Prediction via Probing) for detecting Vision-Language Model hallucinations before generation.

## Overview

This repository contains extraction scripts for various Vision-Language Models (VLMs) used in HALP research. The goal is to extract internal representations from VLMs **before** text generation to predict hallucination.

## Repository Structure

```
HALP_EACL_Local/
├── Models/
│   └── Molmo_V1/          # Molmo-7B-O-0924 extraction (optimized for RTX 4090)
└── README.md
```

## Models

### Molmo-7B-O-0924 (Ready ✅)

Optimized extraction script for RTX 4090 with CUDA.

**Location**: `Models/Molmo_V1/`

**Features**:
- Extracts vision-only representations
- Extracts vision token representations (5 decoder layers)
- Extracts query token representations (5 decoder layers)
- Generates model answers for validation
- Checkpointing every 1000 samples
- HDF5 output format

**Quick Start**:
```bash
cd Models/Molmo_V1
pip install -r requirements.txt
python run_molmo_extraction_rtx4090.py --vqa-dataset <path> --images-dir <path> --test
```

See `Models/Molmo_V1/README.md` for detailed instructions.

## Embedding Types

HALP extracts three types of embeddings to predict hallucination:

1. **Vision-only representation**
   - Extracted from vision encoder before image-to-text projection
   - Captures raw visual understanding

2. **Vision token representation**
   - Extracted at image token boundary in decoder layers
   - Captures how model processes visual information in text space

3. **Query token representation**
   - Extracted at query token boundary in decoder layers
   - Captures model's understanding just before generation

## Requirements

### Hardware
- NVIDIA GPU with CUDA (RTX 4090 recommended, 24GB VRAM)
- ~30GB disk space for model cache
- ~5GB disk space for outputs (10k samples)

### Software
- Python 3.10+
- PyTorch 2.0+ with CUDA
- See individual model directories for specific requirements

## Output Format

All extraction scripts output HDF5 files with the following structure:

```
question_id/
├── question (string)
├── image_id (string)
├── answer (string)                              # Model-generated answer
├── ground_truth_answer (string)                 # Ground truth
├── vision_only_representation (array: D)        # Vision encoder output
├── vision_token_representation/
│   ├── layer_0 (array: D)
│   ├── layer_n/4 (array: D)
│   ├── layer_n/2 (array: D)
│   ├── layer_3n/4 (array: D)
│   └── layer_n (array: D)
└── query_token_representation/
    ├── layer_0 (array: D)
    ├── layer_n/4 (array: D)
    ├── layer_n/2 (array: D)
    ├── layer_3n/4 (array: D)
    └── layer_n (array: D)
```

Where `D` is the model's hidden dimension (e.g., 4096 for Molmo-7B).

## Usage

### 1. Extract Embeddings

```bash
cd Models/Molmo_V1
python run_molmo_extraction_rtx4090.py \
    --vqa-dataset /path/to/vqa_dataset.csv \
    --images-dir /path/to/images \
    --output-dir ./output \
    --checkpoint-interval 1000
```

### 2. Inspect Output

```bash
python inspect_h5.py output/molmo_embeddings_part_001.h5
```

### 3. Train HALP Probing Classifier

(Coming soon - use extracted embeddings to train binary classifier for hallucination prediction)

## Research Goal

Predict VLM hallucination **before** text generation by:
1. Extracting internal representations at key boundaries
2. Training probing classifiers on these representations
3. Detecting hallucination patterns in pre-generation states

## Citation

```bibtex
@article{halp2024,
  title={Detecting Vision-Language Model Hallucinations Before Generation},
  author={[Authors]},
  year={2024}
}
```

## License

Apache 2.0 (following model licenses)

## Contributing

This is a research repository. For issues or questions:
- Check individual model README files
- Open an issue with detailed error logs

## Acknowledgments

- AllenAI for Molmo-7B-O-0924
- HALP paper authors for the methodology