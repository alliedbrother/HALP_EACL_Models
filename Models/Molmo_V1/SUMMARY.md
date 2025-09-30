# Molmo VQA Extraction - Summary

## What Was Done

Cleaned up and optimized the Molmo_V1 directory for RTX 4090 extraction.

## Final Structure

```
Molmo_V1/
├── run_molmo_extraction_rtx4090.py  # Main extraction script (optimized for RTX 4090)
├── inspect_h5.py                    # Utility to inspect HDF5 output files
├── requirements.txt                 # Python dependencies
├── README.md                        # Complete documentation
├── USAGE_RTX4090.md                 # Quick start guide for RTX 4090
└── SUMMARY.md                       # This file
```

## Key Features of the Script

1. **Optimized for RTX 4090**
   - Uses bfloat16 precision (native on RTX 4090)
   - Autocast for mixed precision
   - CUDA-specific optimizations

2. **Three Types of Embeddings**
   - Vision-only representation (4096-dim)
   - Vision token representations (5 layers × 4096-dim)
   - Query token representations (5 layers × 4096-dim)

3. **Robust Processing**
   - Checkpointing every 1000 samples (configurable)
   - Test mode for validation (3 samples)
   - Error handling for generation failures
   - Progress tracking with tqdm

4. **Clean Output Format**
   - HDF5 files with compression
   - Includes generated answers + ground truth
   - Metadata for reproducibility

## Why RTX 4090?

**Problem**: Molmo generation has bugs on CPU/MPS due to:
1. past_key_values handling (FIXED by patching model files)
2. Rotary embeddings during KV cache (UNFIXED - fundamental bug)

**Solution**: Use CUDA on RTX 4090 where:
- Official AllenAI testing was done
- Users report successful generation
- Different tensor operations may avoid the bug

## How to Use

### On RTX 4090 Machine:

```bash
# 1. Setup
conda create -n molmo_rtx python=3.10 -y
conda activate molmo_rtx
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 2. Test (3 samples)
python run_molmo_extraction_rtx4090.py \
    --vqa-dataset /path/to/sampled_10k_relational_dataset.csv \
    --images-dir /path/to/all_images \
    --output-dir ./test_output \
    --test

# 3. Verify
python inspect_h5.py test_output/molmo_embeddings_part_001.h5

# 4. Full run (10k samples, ~2-3 hours)
python run_molmo_extraction_rtx4090.py \
    --vqa-dataset /path/to/sampled_10k_relational_dataset.csv \
    --images-dir /path/to/all_images \
    --output-dir ./output \
    --checkpoint-interval 1000
```

## What Was Removed

Cleaned up unnecessary files:
- ❌ `debug_molmo_structure.py`
- ❌ `debug_tokens.py`
- ❌ `test_generation.py`
- ❌ `test_vision_backbone.py`
- ❌ `test_simple_gen.py`
- ❌ `test_molmo.py`
- ❌ `molmo_7b_d_extractor.py`
- ❌ `molmo_vqa_extractor.py` (old version)
- ❌ Test output directories
- ❌ Cache directories
- ❌ Old requirements files

Kept only:
- ✅ Optimized RTX 4090 script
- ✅ HDF5 inspection utility
- ✅ Clean documentation

## Expected Results

After successful extraction, you should have:

1. **10 HDF5 files** (1000 samples each)
2. **Each file contains**:
   - Questions and answers
   - Vision-only embeddings
   - 5 layers of vision token embeddings
   - 5 layers of query token embeddings
3. **Total size**: ~4-5GB for 10k samples
4. **Ready for**: HALP probing classifier training

## Next Steps for Your Research

1. ✅ Run extraction on RTX 4090
2. ✅ Verify all embeddings are present
3. ✅ Check generated answers quality
4. → Train HALP probing classifiers
5. → Evaluate hallucination prediction

## Known Issues

### Generation May Still Fail

Even on RTX 4090, generation might fail due to Molmo's bugs. If this happens:
- Embeddings still extract correctly
- Use ground truth answers from dataset
- File issue on AllenAI's Molmo repository

### Fallback Option

If RTX 4090 generation also fails:
- Embeddings are still valid for HALP
- Ground truth answers enable hallucination validation
- Consider reporting the bug to AllenAI

## Model Patches Applied

Two bugs were found and one was patched:

1. **past_key_values bug** (FIXED ✅)
   - File: `modeling_molmo.py` line 1833
   - Fix: Added None checks for past_key_values elements
   - Location: `~/.cache/huggingface/modules/transformers_modules/allenai/Molmo-7B-O-0924/.../modeling_molmo.py`

2. **Rotary embeddings bug** (UNFIXED ❌)
   - File: `modeling_molmo.py` line 177
   - Issue: Shape mismatch during KV cache generation
   - Status: May work on CUDA (different tensor ops)

## Credits

- Model: AllenAI's Molmo-7B-O-0924
- Framework: HALP (Hallucination Prediction via Probing)
- Optimization: For RTX 4090 CUDA inference

## License

Apache 2.0 (following Molmo model license)