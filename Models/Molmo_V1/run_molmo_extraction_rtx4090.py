#!/usr/bin/env python3
"""
Molmo VQA Embedding Extractor for RTX 4090
Optimized for CUDA inference with bfloat16

Extracts three types of embeddings for HALP (Hallucination Prediction via Probing):
1. Vision-only representation (from vision encoder)
2. Vision token representation (at image token boundary in decoder layers)
3. Query token representation (at query token boundary in decoder layers)
"""

import os
import argparse
import h5py
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from PIL import Image
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from tqdm import tqdm
import logging

# Setup logging - both console and file
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class MolmoExtractorRTX4090:
    """Optimized Molmo extractor for RTX 4090 with CUDA + bfloat16"""

    def __init__(self, model_path: str = "allenai/Molmo-7B-O-0924", cache_dir: str = "./model_cache"):
        """Initialize extractor for RTX 4090"""
        self.model_path = model_path
        self.cache_dir = cache_dir

        # Verify CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This script requires RTX 4090 with CUDA.")

        logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        self._load_model()

    def _load_model(self):
        """Load model optimized for RTX 4090"""
        logger.info(f"Loading Molmo model: {self.model_path}")

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            cache_dir=self.cache_dir,
            trust_remote_code=True
        )

        # Load model with bfloat16 for RTX 4090 (official recommendation)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,  # Use bfloat16 on GPU
            device_map='cuda'
        )
        self.model.eval()

        # Detect transformer layers
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'transformer'):
            if hasattr(self.model.model.transformer, 'blocks'):
                self.num_layers = len(self.model.model.transformer.blocks)
                logger.info(f"Detected {self.num_layers} transformer layers")
            else:
                raise RuntimeError("Cannot find transformer blocks")
        else:
            raise RuntimeError("Unexpected model structure")

    def extract_embeddings(self, image: Image.Image, question: str) -> Dict:
        """Extract all three types of embeddings for a VQA pair"""

        # Calculate target layers (0, n/4, n/2, 3n/4, n)
        target_layers = [
            0,
            self.num_layers // 4,
            self.num_layers // 2,
            3 * self.num_layers // 4,
            self.num_layers - 1
        ]
        logger.info(f"Target layers: {target_layers}")

        # Process inputs
        inputs = self.processor.process(images=[image], text=question)
        inputs = {k: v.to('cuda', dtype=torch.bfloat16 if v.dtype in [torch.float32, torch.float16] else v.dtype).unsqueeze(0) if torch.is_tensor(v) else v for k, v in inputs.items()}

        # Clone inputs for generation (to avoid corruption)
        gen_inputs = {k: v.clone() if torch.is_tensor(v) else v for k, v in inputs.items()}

        # Step 1: Generate answer with autocast (as per official recommendation)
        generated_text = self._generate_answer(gen_inputs)

        # Step 2: Extract vision-only representation
        vision_only_rep = self._extract_vision_only(image)

        # Step 3: Extract decoder embeddings
        vision_token_reps, query_token_reps = self._extract_decoder_embeddings(inputs, target_layers)

        return {
            'vision_only_representation': vision_only_rep,
            'vision_token_representations': vision_token_reps,
            'query_token_representations': query_token_reps,
            'generated_answer': generated_text
        }

    def _generate_answer(self, inputs: Dict) -> str:
        """Generate answer using official recommendation for CUDA"""
        try:
            # Official recommendation: use autocast with bfloat16 on CUDA
            with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                output = self.model.generate_from_batch(
                    inputs,
                    GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
                    tokenizer=self.processor.tokenizer
                )

            # Decode generated text
            generated_tokens = output[0, inputs['input_ids'].size(1):]
            generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            return generated_text

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"[Generation failed: {str(e)}]"

    def _extract_vision_only(self, image: Image.Image) -> Optional[np.ndarray]:
        """Extract vision encoder representation before projection"""
        try:
            # Process image only
            vision_inputs = self.processor.process(images=[image], text="")

            # Move to GPU and add batch dimension with correct dtype
            if isinstance(vision_inputs, dict):
                vision_inputs = {
                    k: v.to('cuda', dtype=torch.bfloat16 if v.dtype in [torch.float32, torch.float16] else v.dtype).unsqueeze(0) if torch.is_tensor(v) and v.dim() in [2, 3]
                    else v.to('cuda', dtype=torch.bfloat16 if v.dtype in [torch.float32, torch.float16] else v.dtype) if torch.is_tensor(v)
                    else v
                    for k, v in vision_inputs.items()
                }

            with torch.no_grad(), torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                if 'images' in vision_inputs and 'image_masks' in vision_inputs:
                    images = vision_inputs['images']
                    masks = vision_inputs['image_masks']

                    # Ensure batch dimension
                    if images.dim() == 3:
                        images = images.unsqueeze(0)
                    if masks.dim() == 2:
                        masks = masks.unsqueeze(0)

                    # Call vision backbone
                    vision_output = self.model.model.vision_backbone(images, masks)

                    # Extract features (returns tuple: image_features, cls_embed)
                    if isinstance(vision_output, tuple) and len(vision_output) >= 2:
                        image_features = vision_output[0]  # [B, T, N, D]
                        # Pool over sequence and patches
                        pooled = image_features.mean(dim=(1, 2)).squeeze(0)
                        return pooled.cpu().float().numpy()

            return None

        except Exception as e:
            logger.error(f"Vision extraction failed: {e}")
            return None

    def _extract_decoder_embeddings(self, inputs: Dict, target_layers: List[int]) -> tuple:
        """Extract embeddings from decoder layers at token boundaries"""

        layer_outputs = {}
        hooks = []

        def create_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    layer_outputs[layer_idx] = output[0].detach()
                else:
                    layer_outputs[layer_idx] = output.detach()
            return hook

        # Register hooks
        for layer_idx in target_layers:
            if layer_idx < self.num_layers:
                hook = self.model.model.transformer.blocks[layer_idx].register_forward_hook(
                    create_hook(layer_idx)
                )
                hooks.append(hook)

        # Forward pass
        try:
            with torch.no_grad():
                _ = self.model(**inputs)

            # Find token boundaries
            image_boundary, query_boundary = self._find_token_boundaries(inputs)

            # Extract embeddings at boundaries
            vision_token_reps = {}
            query_token_reps = {}

            for layer_idx in target_layers:
                if layer_idx in layer_outputs:
                    hidden_states = layer_outputs[layer_idx]  # [batch, seq_len, hidden_dim]

                    # Vision token representation (at image boundary)
                    if image_boundary < hidden_states.shape[1]:
                        vision_token_reps[f'layer_{layer_idx}'] = hidden_states[0, image_boundary, :].cpu().float().numpy()

                    # Query token representation (at query boundary)
                    if query_boundary < hidden_states.shape[1]:
                        query_token_reps[f'layer_{layer_idx}'] = hidden_states[0, query_boundary, :].cpu().float().numpy()

        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()

        return vision_token_reps, query_token_reps

    def _find_token_boundaries(self, inputs: Dict) -> tuple:
        """Find image and query token boundaries"""
        try:
            if 'image_input_idx' in inputs:
                image_input_idx = inputs['image_input_idx']
                if torch.is_tensor(image_input_idx):
                    # Find where image tokens end
                    image_tokens = image_input_idx[image_input_idx >= 0]
                    if len(image_tokens) > 0:
                        image_boundary = int(image_tokens.max().item()) + 1
                        input_ids = inputs['input_ids']
                        seq_len = input_ids.shape[1] if input_ids.dim() > 1 else input_ids.shape[0]
                        return image_boundary, seq_len - 1

            # Fallback: estimate boundaries
            input_ids = inputs['input_ids']
            seq_len = input_ids.shape[1] if input_ids.dim() > 1 else input_ids.shape[0]
            return seq_len // 3, seq_len - 1

        except Exception as e:
            logger.warning(f"Boundary detection failed: {e}, using fallback")
            input_ids = inputs['input_ids']
            seq_len = input_ids.shape[1] if input_ids.dim() > 1 else input_ids.shape[0]
            return seq_len // 3, seq_len - 1

    def process_dataset(self,
                       vqa_csv_path: str,
                       images_dir: str,
                       output_dir: str,
                       checkpoint_interval: int = 1000,
                       test_mode: bool = False):
        """Process entire VQA dataset with checkpointing"""

        # Load dataset
        logger.info(f"Loading VQA dataset: {vqa_csv_path}")
        df = pd.read_csv(vqa_csv_path)

        if test_mode:
            logger.info("TEST MODE: Processing 3 random samples")
            df = df.sample(n=3, random_state=42)

        os.makedirs(output_dir, exist_ok=True)

        # Setup file logging
        log_file = os.path.join(output_dir, 'molmo_extraction.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)

        logger.info("="*60)
        logger.info(f"Starting extraction: {len(df)} samples")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Log file: {log_file}")
        logger.info("="*60)

        # Process samples
        current_batch = {}
        processed_count = 0
        failed_count = 0
        start_time = datetime.now()

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing VQA"):
            try:
                question_id = row['question_id']
                image_id = row['image_name']
                question = row['question']
                gt_answer = row['gt_answer']

                # Load image
                image_path = os.path.join(images_dir, image_id)
                if not os.path.exists(image_path):
                    logger.warning(f"Image not found: {image_path}")
                    continue

                image = Image.open(image_path).convert('RGB')

                # Extract embeddings
                embeddings = self.extract_embeddings(image, question)

                # Store in batch
                current_batch[question_id] = {
                    'question': question,
                    'image_id': image_id,
                    'vision_only_representation': embeddings['vision_only_representation'],
                    'vision_token_representation': embeddings['vision_token_representations'],
                    'query_token_representation': embeddings['query_token_representations'],
                    'answer': embeddings['generated_answer'],
                    'ground_truth_answer': gt_answer
                }

                processed_count += 1

                # Progress logging every 100 samples
                if processed_count % 100 == 0:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    speed = processed_count / elapsed if elapsed > 0 else 0
                    eta_seconds = (len(df) - processed_count) / speed if speed > 0 else 0
                    eta = str(pd.Timedelta(seconds=int(eta_seconds)))
                    logger.info(f"Progress: {processed_count}/{len(df)} | Speed: {speed:.2f} samples/sec | ETA: {eta} | Failed: {failed_count}")

                # GPU memory logging every 500 samples
                if processed_count % 500 == 0:
                    if torch.cuda.is_available():
                        mem_allocated = torch.cuda.memory_allocated(0) / 1e9
                        mem_reserved = torch.cuda.memory_reserved(0) / 1e9
                        logger.info(f"GPU Memory: Allocated={mem_allocated:.2f}GB, Reserved={mem_reserved:.2f}GB")

                # Save checkpoint
                if processed_count % checkpoint_interval == 0:
                    self._save_checkpoint(current_batch, output_dir, processed_count // checkpoint_interval)
                    current_batch = {}
                    logger.info(f"Saved checkpoint at {processed_count} samples")

            except Exception as e:
                failed_count += 1
                logger.error(f"Failed to process {question_id}: {e}")
                continue

        # Save final batch
        if current_batch:
            final_part = (processed_count // checkpoint_interval) + 1
            self._save_checkpoint(current_batch, output_dir, final_part)
            logger.info(f"Saved final checkpoint with {len(current_batch)} samples")

        # Final summary
        total_time = (datetime.now() - start_time).total_seconds()
        avg_speed = processed_count / total_time if total_time > 0 else 0
        logger.info("="*60)
        logger.info(f"✅ Processing completed!")
        logger.info(f"   Total processed: {processed_count} samples")
        logger.info(f"   Failed: {failed_count} samples")
        logger.info(f"   Total time: {pd.Timedelta(seconds=int(total_time))}")
        logger.info(f"   Average speed: {avg_speed:.2f} samples/sec")
        logger.info("="*60)

    def _save_checkpoint(self, batch_data: Dict, output_dir: str, part_num: int):
        """Save batch to HDF5 file"""
        filename = f"molmo_embeddings_part_{part_num:03d}.h5"
        filepath = os.path.join(output_dir, filename)

        with h5py.File(filepath, 'w') as f:
            # Metadata
            f.attrs['model_name'] = self.model_path
            f.attrs['device'] = 'cuda'
            f.attrs['dtype'] = 'bfloat16'
            f.attrs['created_at'] = datetime.now().isoformat()
            f.attrs['num_samples'] = len(batch_data)

            for question_id, data in batch_data.items():
                grp = f.create_group(question_id)

                # Store strings
                grp.create_dataset('question', data=data['question'], dtype=h5py.string_dtype())
                grp.create_dataset('image_id', data=data['image_id'], dtype=h5py.string_dtype())
                grp.create_dataset('answer', data=data['answer'], dtype=h5py.string_dtype())
                grp.create_dataset('ground_truth_answer', data=data['ground_truth_answer'], dtype=h5py.string_dtype())

                # Store vision only representation
                if data['vision_only_representation'] is not None:
                    grp.create_dataset('vision_only_representation',
                                     data=data['vision_only_representation'],
                                     compression='gzip')

                # Store layer representations
                for rep_type in ['vision_token_representation', 'query_token_representation']:
                    if data[rep_type]:
                        rep_grp = grp.create_group(rep_type)
                        for layer_name, embedding in data[rep_type].items():
                            rep_grp.create_dataset(layer_name, data=embedding, compression='gzip')

        logger.info(f"Saved {len(batch_data)} samples to {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Molmo VQA Extractor for RTX 4090")
    parser.add_argument('--vqa-dataset', required=True, help='Path to VQA CSV dataset')
    parser.add_argument('--images-dir', required=True, help='Directory containing images')
    parser.add_argument('--output-dir', default='./output', help='Output directory for embeddings')
    parser.add_argument('--model', default='allenai/Molmo-7B-O-0924', help='Model path')
    parser.add_argument('--cache-dir', default='./model_cache', help='Model cache directory')
    parser.add_argument('--checkpoint-interval', type=int, default=1000, help='Save every N samples')
    parser.add_argument('--test', action='store_true', help='Test mode (3 samples)')

    args = parser.parse_args()

    # Initialize extractor
    logger.info("🚀 Initializing Molmo Extractor for RTX 4090")
    extractor = MolmoExtractorRTX4090(model_path=args.model, cache_dir=args.cache_dir)

    # Process dataset
    logger.info(f"📊 Dataset: {args.vqa_dataset}")
    logger.info(f"🖼️  Images: {args.images_dir}")
    logger.info(f"💾 Output: {args.output_dir}")

    extractor.process_dataset(
        vqa_csv_path=args.vqa_dataset,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        checkpoint_interval=args.checkpoint_interval,
        test_mode=args.test
    )

    logger.info("✅ All done!")


if __name__ == "__main__":
    main()