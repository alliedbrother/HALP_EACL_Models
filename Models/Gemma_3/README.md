# Gemma 3 12B VQA Embedding Extractor

Multimodal vision-language model for extracting embeddings from VQA datasets for HALP (Hallucination Prediction via Probing).

## Model Information

- **Model**: Google Gemma 3 12B Instruction-Tuned (google/gemma-3-12b-it)
- **Type**: Multimodal (Vision + Language)
- **Vision Encoder**: SigLIP (custom)
- **Parameters**: 12.2 billion
- **Context Window**: 128K tokens
- **Image Resolution**: 896x896 (normalized, encoded to 256 tokens per image)

## Features

Gemma 3 12B is a state-of-the-art multimodal model that:
- Handles both text and image inputs
- Uses SigLIP vision encoder for visual understanding
- Supports 140+ languages
- Provides advanced visual reasoning capabilities

## Extracted Embeddings

This script extracts four types of embeddings:

1. **Vision Representation**: Direct output from SigLIP vision encoder (pooled)
2. **Vision Token Representations**: Decoder layer outputs at image token boundary
3. **Query Token Representations**: Decoder layer outputs at query token boundary
4. **Multimodal Representations**: Combined vision-text representations (mean pooled)

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
- **Transformers**: 4.50.0+ (critical for Gemma 3 support)

## Installation

### Option 1: Using Conda (Recommended)

```bash
# Create environment from file
cd /root/akhil/HALP_EACL_Models/Models/Gemma3_12B
conda env create -f environment.yml

# Activate environment
conda activate gemma3_12b
```

### Option 2: Using pip

```bash
# Create virtual environment
python3.10 -m venv gemma3_env
source gemma3_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Usage

### Basic Usage

```bash
python run_gemma3_extraction.py \
    --vqa-dataset /path/to/vqa_dataset.csv \
    --images-dir /path/to/images/ \
    --output-dir ./output
```

### Test Mode (3 samples)

```bash
python run_gemma3_extraction.py \
    --vqa-dataset /path/to/vqa_dataset.csv \
    --images-dir /path/to/images/ \
    --output-dir ./test_output \
    --test
```

### Full Options

```bash
python run_gemma3_extraction.py \
    --vqa-dataset /path/to/vqa_dataset.csv \
    --images-dir /path/to/images/ \
    --output-dir ./output \
    --model google/gemma-3-12b-it \
    --cache-dir ./model_cache \
    --checkpoint-interval 1000
```

### Command-line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--vqa-dataset` | Yes | - | Path to VQA CSV file (must have: question_id, image_id, question, answer) |
| `--images-dir` | Yes | - | Directory containing images |
| `--output-dir` | No | `./output` | Output directory for HDF5 embeddings |
| `--model` | No | `google/gemma-3-12b-it` | Model name or path |
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

### Images Directory
- Images should be accessible via `{images_dir}/{image_id}`
- Supported formats: JPG, PNG, JPEG
- Images will be automatically resized to 896x896

## Output Format

Embeddings are saved as HDF5 files with checkpointing:

```
output/
├── gemma3_12b_embeddings_part_001.h5
├── gemma3_12b_embeddings_part_002.h5
└── ...
```

### HDF5 Structure

```
gemma3_12b_embeddings_part_001.h5
├── [metadata attributes]
│   ├── model_name: "google/gemma-3-12b-it"
│   ├── model_type: "gemma-3-12b-multimodal"
│   ├── device: "cuda"
│   ├── dtype: "bfloat16"
│   └── num_samples: 1000
├── question_id_1/
│   ├── question (string)
│   ├── image_id (string)
│   ├── answer (string)
│   ├── ground_truth_answer (string)
│   ├── vision_representation (array: [D])
│   ├── vision_token_representation/
│   │   ├── layer_0 (array: [D])
│   │   ├── layer_7 (array: [D])
│   │   ├── layer_14 (array: [D])
│   │   ├── layer_21 (array: [D])
│   │   └── layer_27 (array: [D])
│   ├── query_token_representation/
│   │   └── [same layer structure]
│   └── multimodal_representation/
│       └── [same layer structure]
└── ...
```

## Inspecting Results

Use the provided inspection utility:

```bash
python inspect_h5.py output/gemma3_12b_embeddings_part_001.h5
```

## Performance Notes

### Memory Usage
- **Model Loading**: ~24GB VRAM
- **Inference**: ~26-28GB VRAM peak
- Use bfloat16 for optimal memory efficiency

### Speed Estimates
- RTX 4090: ~2-3 samples/second
- A100: ~3-4 samples/second
- For 10,000 samples: ~1-2 hours

### Optimization Tips
1. Use `--checkpoint-interval` to save progress regularly
2. Monitor GPU memory with `nvidia-smi`
3. Use test mode first to verify setup
4. Ensure images are pre-processed (correct format/size)

## Troubleshooting

### OOM (Out of Memory) Errors
- Reduce batch size (currently 1)
- Use GPU with more VRAM
- Check for other processes using GPU memory

### Model Loading Issues
- Verify transformers>=4.50.0 (required for Gemma 3)
- Check HuggingFace token if using gated models
- Ensure sufficient disk space for model cache

### CUDA Errors
- Verify CUDA 12.1+ is installed
- Check PyTorch CUDA compatibility: `python -c "import torch; print(torch.cuda.is_available())"`
- Update NVIDIA drivers if needed

## References

- [Gemma 3 Blog Post](https://blog.google/technology/developers/gemma-3/)
- [Gemma 3 Model Card](https://huggingface.co/google/gemma-3-12b-it)
- [Gemma 3 Documentation](https://ai.google.dev/gemma/docs/core)
- [HALP Paper](https://arxiv.org/abs/2407.14488) (Hallucination Prediction via Probing)

## License

Gemma 3 models are released under the Gemma Terms of Use. See the [model card](https://huggingface.co/google/gemma-3-12b-it) for details.

## Citation

If you use Gemma 3 in your research, please cite:

```bibtex
@misc{gemma3,
  title={Gemma 3 Technical Report},
  author={Google DeepMind},
  year={2025},
  url={https://arxiv.org/html/2503.19786v1}
}
```
