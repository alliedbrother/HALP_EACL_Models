# RTX 4090 Extraction Checklist

## Before Running

### 1. Hardware Requirements
- [ ] RTX 4090 or similar GPU (24GB VRAM)
- [ ] CUDA 11.8+ or 12.1+ installed
- [ ] NVIDIA drivers up to date

### 2. Software Requirements
- [ ] Python 3.10+ installed
- [ ] Conda or venv available
- [ ] ~30GB free disk space (for model cache)
- [ ] ~5GB free disk space (for output)

### 3. Data Ready
- [ ] VQA CSV file exists: `sampled_10k_relational_dataset.csv`
- [ ] Images directory exists and contains all images
- [ ] You have the full paths to both

## Installation Steps

```bash
# 1. Create environment
conda create -n molmo_rtx python=3.10 -y
conda activate molmo_rtx

# 2. Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

Expected output: `CUDA: True`

## Test Run

```bash
# Update paths below with your actual paths
python run_molmo_extraction_rtx4090.py \
    --vqa-dataset /path/to/sampled_10k_relational_dataset.csv \
    --images-dir /path/to/all_images \
    --output-dir ./test_output \
    --test
```

### Test Run Checklist
- [ ] Script starts without errors
- [ ] Model downloads successfully (~26GB, first run only)
- [ ] Processes 3 samples
- [ ] Creates `test_output/molmo_embeddings_part_001.h5`
- [ ] No CUDA out of memory errors
- [ ] Completes in ~5-10 minutes

## Verify Test Output

```bash
python inspect_h5.py test_output/molmo_embeddings_part_001.h5
```

### Output Verification
- [ ] File opens without errors
- [ ] Contains 3 question IDs
- [ ] Each has `question`, `image_id`, `answer`, `ground_truth_answer`
- [ ] Has `vision_only_representation` (4096 dims)
- [ ] Has 5 layers in `vision_token_representation`
- [ ] Has 5 layers in `query_token_representation`
- [ ] **CRITICAL**: Check `answer` field - should be actual text, not error message
  - ✅ Good: "A red apple on a table"
  - ❌ Bad: "[Generation failed: ...]"

## If Test Succeeds

### Full Extraction Run

```bash
python run_molmo_extraction_rtx4090.py \
    --vqa-dataset /path/to/sampled_10k_relational_dataset.csv \
    --images-dir /path/to/all_images \
    --output-dir ./output \
    --checkpoint-interval 1000
```

### Monitor Progress
In another terminal:
```bash
watch -n 10 'ls -lh output/*.h5 2>/dev/null | tail -5 && echo "---" && nvidia-smi | grep MiB'
```

### During Extraction
- [ ] GPU memory usage: 16-18GB (check with `nvidia-smi`)
- [ ] GPU temperature < 85°C
- [ ] New HDF5 files appear every ~30-60 minutes
- [ ] Script shows progress bar
- [ ] No repeated errors in log

### After Completion (~2-3 hours)
- [ ] 10 HDF5 files created
- [ ] Total size: ~4-5GB
- [ ] Last file has remaining samples (< 1000)
- [ ] No error logs indicating critical failures

## If Test Fails

### Generation Errors
If `answer` field shows error messages:

1. **Check CUDA is working**:
   ```bash
   python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
   ```

2. **Check GPU memory**:
   ```bash
   nvidia-smi
   ```

3. **Try with fresh cache**:
   ```bash
   rm -rf model_cache
   python run_molmo_extraction_rtx4090.py --test ...
   ```

### Fallback Plan
If generation consistently fails even on RTX 4090:
- [ ] Embeddings still extract correctly
- [ ] Use ground truth answers from dataset
- [ ] Continue with HALP research (embeddings are primary need)
- [ ] Consider reporting bug to AllenAI

## Success Criteria

You can proceed with HALP research if:
- ✅ All 10k samples processed
- ✅ All embeddings present (vision_only, vision_token, query_token)
- ✅ Either: Generated answers work OR ground truth available
- ✅ HDF5 files readable and uncorrupted

## Next Steps After Success

1. Archive the HDF5 files safely
2. Begin HALP probing classifier training
3. Use embeddings to predict hallucination
4. Validate predictions against generated/ground truth answers

## Support

If stuck:
- Check `SUMMARY.md` for overview
- Read `USAGE_RTX4090.md` for details
- Review `README.md` for troubleshooting

Good luck! 🚀