# Quick Start Guide for RTX 4090

## Step 1: Setup Environment

```bash
# On your RTX 4090 machine, create a new conda/venv environment
conda create -n molmo_rtx python=3.10 -y
conda activate molmo_rtx

# Install PyTorch with CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Verify CUDA Setup

```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

Expected output:
```
CUDA Available: True
GPU: NVIDIA GeForce RTX 4090
```

## Step 3: Test Run (3 samples)

```bash
python run_molmo_extraction_rtx4090.py \
    --vqa-dataset /path/to/sampled_10k_relational_dataset.csv \
    --images-dir /path/to/all_images \
    --output-dir ./test_output \
    --test
```

This will:
- Load Molmo-7B-O-0924 (~26GB download on first run)
- Process 3 random samples
- Save to `test_output/molmo_embeddings_part_001.h5`
- Take ~5-10 minutes

## Step 4: Verify Output

```bash
python inspect_h5.py test_output/molmo_embeddings_part_001.h5
```

Check that:
- ✅ Generated answers are present (not error messages)
- ✅ All embeddings have correct shapes (4096 dimensions)
- ✅ Vision token and query token representations for 5 layers

## Step 5: Full Extraction (10k samples)

Once test succeeds:

```bash
python run_molmo_extraction_rtx4090.py \
    --vqa-dataset /path/to/sampled_10k_relational_dataset.csv \
    --images-dir /path/to/all_images \
    --output-dir ./output \
    --checkpoint-interval 1000
```

Expected:
- **Time**: ~2-3 hours for 10k samples
- **Memory**: ~16-18GB VRAM (monitor with `nvidia-smi`)
- **Output**: 10 HDF5 files (1000 samples each)

## Step 6: Monitor Progress

In another terminal:
```bash
watch -n 5 'ls -lh output/*.h5 && nvidia-smi'
```

## Important Notes

### If Generation Still Fails on RTX 4090

The script includes error handling. If CUDA generation fails:
- Check script logs for specific error
- Embeddings will still be extracted correctly
- Answer field will show error message
- You can still use ground truth answers for hallucination validation

### Performance Tips

1. **Close other GPU applications** before running
2. **Use checkpoint-interval wisely**:
   - Larger = faster but more data loss if crash
   - Smaller = safer but more I/O overhead
3. **Monitor temperature**: RTX 4090 can get hot during long runs

### Expected File Sizes

- Each checkpoint file: ~400-500MB (1000 samples)
- Total for 10k: ~4-5GB

## Troubleshooting

### Model Download is Slow
- First run downloads ~26GB to `./model_cache`
- Takes 10-30 minutes depending on connection
- Subsequent runs use cached model

### CUDA Out of Memory
```bash
# Check memory usage
nvidia-smi

# If OOM, try reducing checkpoint interval (saves memory)
python run_molmo_extraction_rtx4090.py --checkpoint-interval 500 ...
```

### Generation Returns Errors
- This might still happen due to Molmo bugs
- Embeddings will still extract correctly
- Use ground truth answers from dataset for validation

## Contact

If you encounter issues specific to the RTX 4090 setup, check:
1. CUDA version compatibility
2. PyTorch installation
3. GPU driver version

Good luck with your HALP research! 🚀