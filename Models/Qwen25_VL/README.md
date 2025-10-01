# Qwen2.5-VL-7B-Instruct VQA Embedding Extractor

Multimodal vision-language model for extracting embeddings from VQA datasets for HALP (Hallucination Prediction via Probing).

## Model Information

- **Model**: Qwen/Qwen2.5-VL-7B-Instruct
- **Type**: Multimodal (Vision + Language)
- **Vision Encoder**: ViT (Vision Transformer) with window attention
- **Parameters**: 7 billion
- **Context Window**: 32,768 tokens
- **Precision**: BF16

## Features

Qwen2.5-VL-7B-Instruct is a state-of-the-art multimodal model that:
- Handles both text and image inputs with dynamic resolution
- Uses optimized ViT vision encoder with SwiGLU and RMSNorm
- Supports flexible image and video understanding
- Provides advanced visual reasoning capabilities
- Utilizes mRoPE (modified Rotary Positional Embedding)

## Extracted Embeddings

This script extracts three types of embeddings:

1. **Vision Representation**: Direct output from ViT vision encoder (mean pooled)
2. **Vision Token Representations**: Decoder layer outputs at image token boundary
3. **Query Token Representations**: Decoder layer outputs at query token boundary

Embeddings are extracted from 5 strategic layers: 0, n/4, n/2, 3n/4, n-1

## System Requirements

### Hardware
- **GPU**: NVIDIA GPU with at least 24GB VRAM
  - Recommended: RTX 3090, RTX 4090, A5000, A6000, A100
- **RAM**: 32GB+ system RAM recommended
- **Storage**: 50GB+ free space (for model cache and outputs)

### Software
- **CUDA**: 12.1 or higher
- **Python**: 3.10+
- **PyTorch**: 2.1.0+ with CUDA support
- **Transformers**: 4.37.0+

## Installation

### Option 1: Using pip

```bash
# Create virtual environment
python3.10 -m venv qwen25vl_env
source qwen25vl_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Option 2: Using Conda

```bash
# Create conda environment
conda create -n qwen25vl python=3.10
conda activate qwen25vl

# Install PyTorch with CUDA
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python run_qwen25vl_extraction.py \
    --vqa-dataset /path/to/vqa_dataset.csv \
    --images-dir /path/to/images/ \
    --output-dir ./output
```

### Test Mode (3 samples)

```bash
python run_qwen25vl_extraction.py \
    --vqa-dataset /path/to/vqa_dataset.csv \
    --images-dir /path/to/images/ \
    --output-dir ./test_output \
    --test
```

### Full Options

```bash
python run_qwen25vl_extraction.py \
    --vqa-dataset /path/to/vqa_dataset.csv \
    --images-dir /path/to/images/ \
    --output-dir ./output \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --cache-dir ./model_cache \
    --checkpoint-interval 1000
```

### Command-line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--vqa-dataset` | Yes | - | Path to VQA CSV file (must have: question_id, image_id, question, answer) |
| `--images-dir` | Yes | - | Directory containing images |
| `--output-dir` | No | `./output` | Output directory for HDF5 embeddings |
| `--model` | No | `Qwen/Qwen2.5-VL-7B-Instruct` | Model name or path |
| `--cache-dir` | No | `./model_cache` | Model cache directory |
| `--checkpoint-interval` | No | `1000` | Save checkpoint every N samples |
| `--test` | No | `False` | Test mode (process only 3 samples) |

## Input Data Format

### VQA CSV Format
The input CSV must contain these columns:
```csv
question_id,image_id,question,answer
1,image1.jpg,"What color is the sky?","blue"
2,image2.jpg,"How many people are visible?","3"
```

Alternative column names are also supported:
- `image_name` instead of `image_id`
- `gt_answer` instead of `answer`

### Images Directory
- Images should be accessible via `{images_dir}/{image_id}`
- Supported formats: JPG, PNG, JPEG
- Images will be automatically processed with dynamic resolution

## Output Format

Embeddings are saved as HDF5 files with checkpointing:

```
output/
├── qwen25vl_7b_embeddings_part_001.h5
├── qwen25vl_7b_embeddings_part_002.h5
└── ...
```

### HDF5 Structure

```
qwen25vl_7b_embeddings_part_001.h5
├── [metadata attributes]
│   ├── model_name: "Qwen/Qwen2.5-VL-7B-Instruct"
│   ├── model_type: "qwen2.5-vl-7b-instruct"
│   ├── device: "cuda"
│   ├── dtype: "bfloat16"
│   └── num_samples: 1000
├── question_id_1/
│   ├── question (string)
│   ├── image_id (string)
│   ├── answer (string)
│   ├── ground_truth_answer (string)
│   ├── vision_only_representation (array: [D])
│   ├── vision_token_representation/
│   │   ├── layer_0 (array: [D])
│   │   ├── layer_7 (array: [D])
│   │   ├── layer_14 (array: [D])
│   │   ├── layer_21 (array: [D])
│   │   └── layer_27 (array: [D])
│   ├── query_token_representation/
│   │   └── [same layer structure]
└── ...
```

Where `D` is the hidden dimension (3584 for Qwen2.5-VL-7B).

## Inspecting Results

Use the provided inspection utility:

```bash
python inspect_h5.py output/qwen25vl_7b_embeddings_part_001.h5
```

## Performance Notes

### Memory Usage
- **Model Loading**: ~14-16GB VRAM
- **Inference**: ~18-20GB VRAM peak
- Use bfloat16 for optimal memory efficiency

### Speed Estimates
- RTX 4090: ~2-3 samples/second
- A100: ~3-4 samples/second
- For 10,000 samples: ~1-2 hours

### Optimization Tips
1. Use `--checkpoint-interval` to save progress regularly
2. Monitor GPU memory with `nvidia-smi`
3. Use test mode first to verify setup
4. Ensure images are in supported formats

## Troubleshooting

### OOM (Out of Memory) Errors
- Reduce batch size (currently 1, minimal)
- Use GPU with more VRAM
- Check for other processes using GPU memory
- Clear CUDA cache regularly (handled automatically)

### Model Loading Issues
- Verify transformers>=4.37.0 (required for Qwen2.5-VL)
- Check HuggingFace token if using gated models
- Ensure sufficient disk space for model cache (~14GB)

### CUDA Errors
- Verify CUDA 12.1+ is installed
- Check PyTorch CUDA compatibility: `python -c "import torch; print(torch.cuda.is_available())"`
- Update NVIDIA drivers if needed

### Import Errors (qwen_vl_utils)
- Install qwen-vl-utils: `pip install qwen-vl-utils`
- Ensure transformers is up to date

## Architecture Details

### Vision Encoder
- ViT-based architecture with window attention
- Optimized with SwiGLU activation and RMSNorm
- Dynamic resolution support
- Outputs pooled vision representation

### Language Model
- 28 transformer layers (7B model)
- Hidden size: 3584
- Extracts embeddings at 5 strategic layers (0, 7, 14, 21, 27)

### Token Boundaries
- **Vision tokens**: Detected from `image_grid_thw` (temporal, height, width grid)
- **Query tokens**: Last position before generation

## References

- [Qwen2.5-VL Model Card](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923)
- [HALP Paper](https://arxiv.org/abs/2407.14488) (Hallucination Prediction via Probing)

## License

Qwen2.5-VL models are released under the Apache 2.0 License. See the [model card](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) for details.

## Citation

If you use Qwen2.5-VL in your research, please cite:

```bibtex
@article{qwen2.5-vl,
  title={Qwen2.5-VL Technical Report},
  author={Qwen Team},
  journal={arXiv preprint arXiv:2502.13923},
  year={2025}
}
```

## Acknowledgments

- Qwen Team for the Qwen2.5-VL model
- HALP paper authors for the methodology
- Hugging Face for the transformers library
