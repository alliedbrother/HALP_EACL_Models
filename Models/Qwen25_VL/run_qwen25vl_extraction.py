#!/usr/bin/env python3
"""
Qwen2.5-VL-7B-Instruct VQA Embedding Extractor
Multimodal vision-language model with ViT vision encoder

Extracts embeddings for HALP (Hallucination Prediction via Probing):
1. Vision representation (from ViT vision encoder)
2. Vision token representation (at image token boundary in decoder layers)
3. Query token representation (at query token boundary in decoder layers)
"""

import os
import argparse
import h5py
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from PIL import Image
from datetime import datetime

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import logging

# Setup logging with file output
log_file = f'extraction_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Qwen25VLExtractorGPU:
    """Qwen2.5-VL-7B-Instruct extractor optimized for GPU with bfloat16"""

    def __init__(self, model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct", cache_dir: str = "./model_cache"):
        """Initialize Qwen2.5-VL-7B-Instruct extractor"""
        self.model_path = model_path
        self.cache_dir = cache_dir

        # Verify CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This script requires GPU with CUDA.")

        logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        self._load_model()

    def _load_model(self):
        """Load Qwen2.5-VL-7B-Instruct model and processor"""
        logger.info(f"Loading Qwen2.5-VL-7B-Instruct model: {self.model_path}")

        # Load processor (handles both text and image preprocessing)
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            cache_dir=self.cache_dir,
            trust_remote_code=True
        )

        # Load model with bfloat16 for GPU efficiency
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map='cuda'
        )
        self.model.eval()

        # Detect model architecture
        self._detect_architecture()

    def _detect_architecture(self):
        """Detect Qwen2.5-VL architecture components"""
        # Qwen2.5-VL uses a vision encoder (ViT) + language model
        if hasattr(self.model, 'visual'):
            logger.info("Vision encoder detected (ViT)")
            self.has_vision_encoder = True
        else:
            logger.warning("Vision encoder not found in expected location")
            self.has_vision_encoder = False

        # Detect language model layers - Qwen2.5-VL structure is model.model.layers
        if hasattr(self.model, 'model'):
            if hasattr(self.model.model, 'config'):
                config = self.model.model.config
                self.num_layers = getattr(config, 'num_hidden_layers', 28)
                self.hidden_size = getattr(config, 'hidden_size', 3584)
                logger.info(f"Detected {self.num_layers} language model layers")
                logger.info(f"Hidden size: {self.hidden_size}")
            else:
                # Default for Qwen2.5-VL-7B
                self.num_layers = 28
                self.hidden_size = 3584
                logger.warning(f"Cannot find config, using defaults: layers={self.num_layers}, hidden_size={self.hidden_size}")
        else:
            # Default for Qwen2.5-VL-7B
            self.num_layers = 28
            self.hidden_size = 3584
            logger.warning(f"Cannot find model.model, using defaults: layers={self.num_layers}, hidden_size={self.hidden_size}")

    def extract_embeddings(self, image: Image.Image, question: str) -> Dict:
        """Extract all embeddings for a VQA pair

        Returns dictionary with:
            - vision_only_representation: From ViT vision encoder
            - vision_token_representation: At last image token in decoder layers
            - query_token_representation: At last query token in decoder layers
            - answer: Generated answer
        """

        # Calculate target layers (0, n/4, n/2, 3n/4, n-1)
        target_layers = [
            0,
            self.num_layers // 4,
            self.num_layers // 2,
            3 * self.num_layers // 4,
            self.num_layers - 1
        ]
        logger.info(f"Target layers: {target_layers}")

        # Format input with chat template
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question}
                ]
            }
        ]

        # Process inputs
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to('cuda')

        # Step 1: Generate answer
        generated_text = self._generate_answer(inputs)

        # Step 2: Extract vision-only representation
        vision_only_rep = self._extract_vision_representation(image)

        # Step 3: Extract decoder embeddings at token boundaries
        vision_token_reps, query_token_reps = self._extract_decoder_embeddings(
            inputs, target_layers
        )

        return {
            'vision_only_representation': vision_only_rep,
            'vision_token_representation': vision_token_reps,
            'query_token_representation': query_token_reps,
            'answer': generated_text
        }

    def _generate_answer(self, inputs: Dict) -> str:
        """Generate answer using Qwen2.5-VL-7B-Instruct"""
        try:
            with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False,
                    temperature=None,
                    top_p=None
                )

            # Decode generated text (skip input tokens)
            generated_tokens = outputs[0, inputs['input_ids'].size(1):]
            generated_text = self.processor.decode(generated_tokens, skip_special_tokens=True).strip()

            return generated_text

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"[Generation failed: {str(e)}]"

    def _extract_vision_representation(self, image: Image.Image) -> Optional[np.ndarray]:
        """Extract vision encoder (ViT) representation BEFORE projector

        Extracts features from the last ViT transformer block BEFORE the merger/projector.
        This gives raw vision features in ViT dimension space (not language model space).
        """
        try:
            if not hasattr(self.model, 'visual'):
                logger.warning("Vision encoder not accessible")
                return None

            # Format image for vision encoder
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": "Describe the image."}
                    ]
                }
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )

            # Move to GPU
            if 'pixel_values' in inputs and inputs['pixel_values'] is not None:
                pixel_values = inputs['pixel_values'].to('cuda')
                # Get grid_thw if available (required by Qwen2.5-VL)
                grid_thw = inputs.get('image_grid_thw', None)
                if grid_thw is not None:
                    grid_thw = grid_thw.to('cuda')
            else:
                logger.error("No pixel_values in processor output")
                return None

            # Hook to capture output BEFORE the merger/projector
            vision_features_before_merger = {}

            def hook_before_merger(module, input, output):
                # Capture the output of ViT blocks before merger
                vision_features_before_merger['hidden_states'] = output.detach()

            with torch.no_grad(), torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                # Register hook on the last ViT block (before merger)
                # model.visual.blocks[-1] is the last transformer block before merger
                hook_handle = self.model.visual.blocks[-1].register_forward_hook(hook_before_merger)

                try:
                    # Run forward pass through vision encoder
                    # This will trigger our hook to capture features before merger
                    if grid_thw is not None:
                        _ = self.model.visual(pixel_values, grid_thw=grid_thw)
                    else:
                        # Fallback: create default grid_thw
                        batch_size = pixel_values.shape[0]
                        grid_thw = torch.tensor([[1, 24, 24]], device='cuda').expand(batch_size, -1)
                        _ = self.model.visual(pixel_values, grid_thw=grid_thw)

                    # Get the captured features (before merger)
                    if 'hidden_states' in vision_features_before_merger:
                        vision_features = vision_features_before_merger['hidden_states']
                    else:
                        logger.error("Failed to capture vision features before merger")
                        return None

                    # Average pool over sequence dimension to get single vector
                    if vision_features.dim() == 3:
                        # [batch, seq_len, hidden] -> [batch, hidden]
                        vision_rep = vision_features.mean(dim=1)
                    elif vision_features.dim() == 2:
                        # [seq_len, hidden] -> [hidden]
                        vision_rep = vision_features.mean(dim=0, keepdim=True)
                    else:
                        vision_rep = vision_features

                    return vision_rep.squeeze(0).cpu().float().numpy()

                finally:
                    # Remove hook
                    hook_handle.remove()

        except Exception as e:
            logger.error(f"Vision extraction failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _extract_decoder_embeddings(self, inputs: Dict, target_layers: List[int]) -> Tuple:
        """Extract embeddings from decoder layers at token boundaries

        Returns:
            vision_token_reps: Representations at last image token
            query_token_reps: Representations at last query token
        """

        layer_outputs = {}
        hooks = []

        def create_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    layer_outputs[layer_idx] = output[0].detach()
                else:
                    layer_outputs[layer_idx] = output.detach()
            return hook

        # Register hooks on language model layers
        # Qwen2.5-VL: layers are at model.model.language_model.layers
        if (not hasattr(self.model, 'model') or
            not hasattr(self.model.model, 'language_model') or
            not hasattr(self.model.model.language_model, 'layers')):
            logger.error("Cannot access model.model.language_model.layers")
            return {}, {}

        language_layers = self.model.model.language_model.layers

        for layer_idx in target_layers:
            if layer_idx < self.num_layers and layer_idx < len(language_layers):
                hook = language_layers[layer_idx].register_forward_hook(
                    create_hook(layer_idx)
                )
                hooks.append(hook)

        # Forward pass
        try:
            with torch.no_grad():
                _ = self.model(**inputs)

            # Find token boundaries
            vision_token_boundary, query_token_boundary = self._find_token_boundaries(inputs)

            # Extract embeddings at boundaries
            vision_token_reps = {}
            query_token_reps = {}

            for layer_idx in target_layers:
                if layer_idx in layer_outputs:
                    hidden_states = layer_outputs[layer_idx]  # [batch, seq_len, hidden_dim]

                    # Vision token representation (at last image token, just before query starts)
                    if vision_token_boundary < hidden_states.shape[1]:
                        vision_token_reps[f'layer_{layer_idx}'] = (
                            hidden_states[0, vision_token_boundary, :].cpu().float().numpy()
                        )

                    # Query token representation (at last query token)
                    if query_token_boundary < hidden_states.shape[1]:
                        query_token_reps[f'layer_{layer_idx}'] = (
                            hidden_states[0, query_token_boundary, :].cpu().float().numpy()
                        )

        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()

        return vision_token_reps, query_token_reps

    def _find_token_boundaries(self, inputs: Dict) -> Tuple[int, int]:
        """Find image and query token boundaries

        Returns:
            vision_token_boundary: Index of last image token (before query tokens start)
            query_token_boundary: Index of last query token (end of sequence)
        """
        try:
            input_ids = inputs['input_ids']
            seq_len = input_ids.shape[1] if input_ids.dim() > 1 else input_ids.shape[0]

            # Qwen2.5-VL uses special tokens to mark image regions
            # Try to detect image token positions
            if 'image_grid_thw' in inputs:
                # image_grid_thw contains [temporal, height, width] for each image
                grid = inputs['image_grid_thw']
                if torch.is_tensor(grid) and grid.numel() > 0:
                    # Estimate vision tokens from grid dimensions
                    # Total vision tokens = T * H * W per image
                    vision_token_count = grid[0].prod().item() if grid.dim() > 1 else grid.prod().item()
                    vision_token_boundary = min(vision_token_count, seq_len // 2)
                else:
                    # Fallback: estimate as 1/3 of sequence
                    vision_token_boundary = seq_len // 3
            else:
                # Fallback: estimate as 1/3 of sequence
                vision_token_boundary = seq_len // 3

            # Query token boundary: Last token in sequence
            query_token_boundary = seq_len - 1

            logger.info(f"Token boundaries - Vision (last img token): {vision_token_boundary}, Query (last token): {query_token_boundary}")
            return vision_token_boundary, query_token_boundary

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
        import time

        # Load dataset
        logger.info(f"Loading VQA dataset: {vqa_csv_path}")
        df = pd.read_csv(vqa_csv_path)

        if test_mode:
            logger.info("TEST MODE: Processing 3 random samples")
            df = df.sample(n=3, random_state=42)

        total_samples = len(df)
        logger.info(f"Total samples to process: {total_samples}")

        os.makedirs(output_dir, exist_ok=True)

        # Process samples
        current_batch = {}
        processed_count = 0
        start_time = time.time()
        sample_times = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing VQA"):
            sample_start = time.time()
            try:
                question_id = row['question_id']
                # Support both 'image_id' and 'image_name' columns
                image_id = row.get('image_id') or row.get('image_name')
                question = row['question']
                # Support both 'answer' and 'gt_answer' columns
                gt_answer = row.get('answer') or row.get('gt_answer')

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
                    'vision_token_representation': embeddings['vision_token_representation'],
                    'query_token_representation': embeddings['query_token_representation'],
                    'answer': embeddings['answer'],
                    'ground_truth_answer': gt_answer
                }

                processed_count += 1

                # Track sample processing time
                sample_time = time.time() - sample_start
                sample_times.append(sample_time)

                # Calculate and log progress every 10 samples
                if processed_count % 10 == 0:
                    avg_time = sum(sample_times) / len(sample_times)
                    remaining = total_samples - processed_count
                    eta_seconds = avg_time * remaining
                    eta_hours = eta_seconds / 3600
                    elapsed_hours = (time.time() - start_time) / 3600
                    logger.info(f"Progress: {processed_count}/{total_samples} | "
                              f"Avg time/sample: {avg_time:.2f}s | "
                              f"ETA: {eta_hours:.2f}h | "
                              f"Elapsed: {elapsed_hours:.2f}h")

                # Clear CUDA cache to prevent OOM errors
                torch.cuda.empty_cache()

                # Save checkpoint
                if processed_count % checkpoint_interval == 0:
                    self._save_checkpoint(current_batch, output_dir, processed_count // checkpoint_interval)
                    current_batch = {}
                    torch.cuda.empty_cache()  # Clear cache after checkpoint
                    logger.info(f"Saved checkpoint at {processed_count} samples")

            except Exception as e:
                logger.error(f"Failed to process {question_id}: {e}")
                import traceback
                traceback.print_exc()
                torch.cuda.empty_cache()  # Clear cache on error
                continue

        # Save final batch
        if current_batch:
            final_part = (processed_count // checkpoint_interval) + 1
            self._save_checkpoint(current_batch, output_dir, final_part)
            logger.info(f"Saved final checkpoint with {len(current_batch)} samples")

        # Final statistics
        total_time = time.time() - start_time
        logger.info(f"✅ Processing completed! Total: {processed_count} samples")
        logger.info(f"Total time: {total_time/3600:.2f} hours")
        logger.info(f"Average time per sample: {total_time/processed_count:.2f} seconds")

    def _save_checkpoint(self, batch_data: Dict, output_dir: str, part_num: int):
        """Save batch to HDF5 file

        Output structure per sample:
            question_id/
                image_id: str
                question: str
                ground_truth_answer: str
                vision_only_representation: [D]
                vision_token_representation/
                    layer_0: [D]
                    layer_n/4: [D]
                    layer_n/2: [D]
                    layer_3n/4: [D]
                    layer_n-1: [D]
                query_token_representation/
                    layer_0: [D]
                    layer_n/4: [D]
                    layer_n/2: [D]
                    layer_3n/4: [D]
                    layer_n-1: [D]
                answer: str
        """
        filename = f"qwen25vl_7b_embeddings_part_{part_num:03d}.h5"
        filepath = os.path.join(output_dir, filename)

        with h5py.File(filepath, 'w') as f:
            # Metadata
            f.attrs['model_name'] = self.model_path
            f.attrs['model_type'] = 'qwen2.5-vl-7b-instruct'
            f.attrs['device'] = 'cuda'
            f.attrs['dtype'] = 'bfloat16'
            f.attrs['created_at'] = datetime.now().isoformat()
            f.attrs['num_samples'] = len(batch_data)

            for question_id, data in batch_data.items():
                grp = f.create_group(str(question_id))

                # Store metadata strings
                grp.create_dataset('image_id', data=data['image_id'], dtype=h5py.string_dtype())
                grp.create_dataset('question', data=data['question'], dtype=h5py.string_dtype())
                grp.create_dataset('ground_truth_answer', data=data['ground_truth_answer'], dtype=h5py.string_dtype())

                # Store vision-only representation (from ViT encoder)
                if data['vision_only_representation'] is not None:
                    grp.create_dataset('vision_only_representation',
                                     data=data['vision_only_representation'],
                                     compression='gzip')

                # Store vision token representations (at last image token in decoder)
                if data['vision_token_representation']:
                    rep_grp = grp.create_group('vision_token_representation')
                    for layer_name, embedding in data['vision_token_representation'].items():
                        rep_grp.create_dataset(layer_name, data=embedding, compression='gzip')

                # Store query token representations (at last query token in decoder)
                if data['query_token_representation']:
                    rep_grp = grp.create_group('query_token_representation')
                    for layer_name, embedding in data['query_token_representation'].items():
                        rep_grp.create_dataset(layer_name, data=embedding, compression='gzip')

                # Store generated answer
                grp.create_dataset('answer', data=data['answer'], dtype=h5py.string_dtype())

        logger.info(f"Saved {len(batch_data)} samples to {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL-7B-Instruct VQA Extractor")
    parser.add_argument('--vqa-dataset', required=True, help='Path to VQA CSV dataset')
    parser.add_argument('--images-dir', required=True, help='Directory containing images')
    parser.add_argument('--output-dir', default='./output', help='Output directory for embeddings')
    parser.add_argument('--model', default='Qwen/Qwen2.5-VL-7B-Instruct', help='Model path')
    parser.add_argument('--cache-dir', default='./model_cache', help='Model cache directory')
    parser.add_argument('--checkpoint-interval', type=int, default=1000, help='Save every N samples')
    parser.add_argument('--test', action='store_true', help='Test mode (3 samples)')

    args = parser.parse_args()

    # Initialize extractor
    logger.info("🚀 Initializing Qwen2.5-VL-7B-Instruct Extractor")
    extractor = Qwen25VLExtractorGPU(model_path=args.model, cache_dir=args.cache_dir)

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
