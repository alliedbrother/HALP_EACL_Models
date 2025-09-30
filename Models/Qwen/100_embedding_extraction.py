import sys
import os
import traceback
import time
import numpy as np
from collections import defaultdict
import h5py
import json
import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
import warnings
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Suppress warnings
warnings.filterwarnings("ignore")

def check_gpu_availability():
    """Check GPU availability and set appropriate device."""
    try:
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"CUDA is available with {device_count} GPU(s)")
            
            # Try to create a tensor on GPU to test functionality
            test_tensor = torch.tensor([1.0]).cuda()
            del test_tensor
            torch.cuda.empty_cache()
            
            return "cuda"
        else:
            print("CUDA is not available")
            return "cpu"
    except Exception as e:
        print(f"GPU test failed: {e}")
        print("Falling back to CPU")
        return "cpu"

# Constants
BATCH_SIZE = 1  # Qwen2.5-VL requires batch size 1 for vision processing
PART_SIZE = 250  # Images per part file (as requested)
NUM_WORKERS = 0  # Disabled multiprocessing for Qwen2.5-VL
IMAGE_DIR = "/root/akhil_workspace/coco_val2014"
OUTPUT_DIR = "/root/akhil_workspace/qwen_model/extracted_embeddings_coco2014"
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"  # Can also use "Qwen/Qwen2.5-VL-3B-Instruct"

# Test mode - set to True to process only 2 images for testing
TEST_MODE = True  # Set to False for full processing

# Global model variables (will be initialized in main)
processor = None
model = None
device = None

def get_model_architecture_info(model):
    """Get the total number of layers in the language model."""
    arch_info = {}
    
    # Get language model layers
    if hasattr(model, 'model') and hasattr(model.model, 'config'):
        config = model.model.config
        arch_info['language_layers'] = getattr(config, 'num_hidden_layers', None)
        arch_info['language_hidden_size'] = getattr(config, 'hidden_size', None)
    
    # Get vision model info
    if hasattr(model, 'visual'):
        arch_info['has_vision_tower'] = True
        if hasattr(model.visual, 'config'):
            vision_config = model.visual.config
            arch_info['vision_layers'] = getattr(vision_config, 'num_hidden_layers', None)
            arch_info['vision_hidden_size'] = getattr(vision_config, 'hidden_size', None)
    else:
        arch_info['has_vision_tower'] = False
    
    return arch_info

def get_layer_indices(total_layers):
    """Get layer indices: 0, n//4, n//2, 3n//4, n-1"""
    if total_layers < 5:
        return list(range(total_layers))
    
    indices = [
        0,                          # First layer
        total_layers // 4,          # n//4
        total_layers // 2,          # n//2  
        (3 * total_layers) // 4,    # 3n//4
        total_layers - 1            # Last layer (n-1)
    ]
    
    # Remove duplicates and sort
    return sorted(list(set(indices)))

def detect_and_configure_layers(model):
    """Detect model architecture and configure layer selection."""
    print("🔍 Detecting Qwen2.5-VL model architecture...")
    
    arch_info = get_model_architecture_info(model)
    print(f"Architecture info: {arch_info}")
    
    # Get total language model layers
    total_layers = arch_info.get('language_layers')
    
    if total_layers is None:
        # Default based on model size
        if "3B" in MODEL_NAME:
            total_layers = 36
        else:
            total_layers = 28  # For 7B model
        print(f"Could not determine layer count. Using default: {total_layers}")
    
    # Get target layer indices
    selected_layers = get_layer_indices(total_layers)
    
    config = {
        'model_architecture': arch_info,
        'total_layers': total_layers,
        'selected_layers': selected_layers,
        'has_vision_tower': arch_info.get('has_vision_tower', False),
        'model_name': MODEL_NAME
    }
    
    print(f"📋 Configuration:")
    print(f"   Total language layers: {total_layers}")
    print(f"   Selected layers: {selected_layers}")
    print(f"   Has vision tower: {config['has_vision_tower']}")
    
    return config

def load_image_ids():
    """Load image IDs from the directory."""
    try:
        if not os.path.exists(IMAGE_DIR):
            print(f"Error: Image directory {IMAGE_DIR} not found")
            return []
            
        image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')]
        image_ids = sorted([os.path.splitext(f)[0] for f in image_files])
        
        if TEST_MODE:
            image_ids = image_ids[:2]
            print(f"🧪 TEST MODE: Processing only {len(image_ids)} images")
        
        print(f"Found {len(image_ids)} images in {IMAGE_DIR}")
        return image_ids
    except Exception as e:
        print(f"Error loading image IDs: {e}")
        return []

def safe_tensor_to_numpy(tensor, name="tensor"):
    """Safely convert tensor to numpy array, handling BFloat16 and other dtypes."""
    try:
        tensor_cpu = tensor.cpu()
        
        # Convert BFloat16 to Float32 if needed
        if tensor_cpu.dtype == torch.bfloat16:
            tensor_cpu = tensor_cpu.to(torch.float32)
        elif tensor_cpu.dtype == torch.float16:
            tensor_cpu = tensor_cpu.to(torch.float32)
        
        return tensor_cpu.numpy().tolist()
        
    except Exception as e:
        print(f"Error converting {name} to numpy: {e}")
        return torch.zeros_like(tensor).float().cpu().numpy().tolist()

def find_token_positions(input_ids, processor, vision_token_count):
    """Find the positions of image end and query end tokens."""
    token_ids = input_ids[0].cpu().tolist()
    sequence_length = len(token_ids)
    
    # Image tokens are at the beginning, so image_end_pos is after vision tokens
    image_end_pos = min(vision_token_count, sequence_length - 10)
    
    # Query end is the last token before generation
    query_end_pos = sequence_length - 1
    
    # Ensure positions are valid
    image_end_pos = max(1, min(image_end_pos, sequence_length - 2))
    query_end_pos = max(image_end_pos + 1, min(query_end_pos, sequence_length - 1))
    
    return image_end_pos, query_end_pos

def extract_vision_embeddings_from_sequence(hidden_states, vision_token_count):
    """Extract vision embeddings from the sequence hidden states."""
    try:
        # Vision tokens are typically at the beginning of the sequence
        # We'll extract from the last layer and pool the vision region
        
        final_layer = hidden_states[-1]  # Last layer hidden states
        
        # Extract vision tokens (first vision_token_count tokens)
        vision_tokens = final_layer[0, :vision_token_count, :]  # [vision_tokens, hidden_size]
        
        # Pool across vision tokens to get a single representation
        vision_embedding = vision_tokens.mean(dim=0)  # [hidden_size]
        
        return safe_tensor_to_numpy(vision_embedding, "vision_embeddings_from_sequence")
        
    except Exception as e:
        print(f"Warning: Could not extract vision embeddings from sequence: {e}")
        return None

def estimate_vision_token_count(input_ids, sequence_length):
    """Estimate the number of vision tokens in the sequence."""
    # For Qwen2.5-VL, vision tokens are typically at the beginning
    # Common ranges are 256-1024 tokens depending on image resolution
    
    # Conservative estimate: vision tokens are roughly 10-30% of sequence
    # but usually between 200-800 tokens
    estimated_vision_tokens = min(
        max(200, sequence_length // 4),  # At least 200, at most 1/4 of sequence
        800  # Cap at 800 tokens
    )
    
    return estimated_vision_tokens

def process_image_with_qwen2_5vl(image_path):
    """Process image with Qwen2.5-VL processor."""
    global processor, model
    
    try:
        image = Image.open(image_path).convert("RGB")
        
        # Create messages
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]
        
        # Process following reference code
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        inputs = inputs.to(model.device)
        
        return inputs, text, messages
        
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None, None, None

def extract_targeted_embeddings(image_id, image_path, selected_layers):
    """Extract only the specific embeddings we need."""
    global model, processor, device
    
    try:
        # Process image
        inputs, text, messages = process_image_with_qwen2_5vl(image_path)
        if inputs is None:
            return None
        
        embeddings_data = {}
        
        # Forward pass to get all hidden states
        with torch.no_grad():
            model_outputs = model(**inputs, output_hidden_states=True)
            
            if hasattr(model_outputs, 'hidden_states') and model_outputs.hidden_states:
                hidden_states = model_outputs.hidden_states
                total_available_layers = len(hidden_states)
                
                # Get sequence info
                sequence_length = len(inputs['input_ids'][0])
                
                # Estimate vision token count
                vision_token_count = estimate_vision_token_count(inputs['input_ids'], sequence_length)
                
                # 1. Extract Vision Embeddings from the sequence
                vision_embeddings = extract_vision_embeddings_from_sequence(hidden_states, vision_token_count)
                embeddings_data['vision_embeddings'] = vision_embeddings
                
                # 2. Extract Pre-generation Embeddings at Specific Layers and Positions
                # Validate selected layers
                valid_layers = [layer for layer in selected_layers if layer < total_available_layers]
                if len(valid_layers) != len(selected_layers):
                    invalid_layers = [layer for layer in selected_layers if layer >= total_available_layers]
                    print(f"   ⚠️ Skipping invalid layers {invalid_layers} (max: {total_available_layers-1})")
                
                # Find critical token positions
                image_end_pos, query_end_pos = find_token_positions(inputs['input_ids'], processor, vision_token_count)
                
                # Extract embeddings from selected layers at critical positions
                pre_generation_data = {}
                
                for layer_idx in valid_layers:
                    layer_hidden = hidden_states[layer_idx]
                    
                    # Extract embeddings at the two critical positions
                    after_image_emb = layer_hidden[0, image_end_pos, :]
                    end_query_emb = layer_hidden[0, query_end_pos, :]
                    
                    pre_generation_data[f'layer_{layer_idx}'] = {
                        'after_image_embeddings': safe_tensor_to_numpy(after_image_emb, f"layer_{layer_idx}_after_image"),
                        'end_query_embeddings': safe_tensor_to_numpy(end_query_emb, f"layer_{layer_idx}_end_query")
                    }
                
                embeddings_data['pre_generation'] = pre_generation_data
                
                # Store position info for reference
                embeddings_data['token_positions'] = {
                    'image_end_position': int(image_end_pos),
                    'query_end_position': int(query_end_pos),
                    'sequence_length': int(sequence_length),
                    'estimated_vision_tokens': int(vision_token_count)
                }
        
        # Generate caption for reference
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            generated_caption = output_text[0] if output_text else ""
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            'image_id': image_id,
            'generated_caption': generated_caption,
            'vision_embeddings': embeddings_data.get('vision_embeddings'),
            'pre_generation': embeddings_data.get('pre_generation', {}),
            'token_positions': embeddings_data.get('token_positions', {})
        }
        
    except Exception as e:
        print(f"Error extracting embeddings for {image_id}: {str(e)}")
        return None

def write_embeddings_hdf5(f, image_id, result):
    """Write targeted embeddings to HDF5 file."""
    try:
        group = f.create_group(image_id)
        
        # Basic metadata
        group.create_dataset('image_id', data=result['image_id'].encode('utf-8'))
        group.create_dataset('generated_caption', data=result['generated_caption'].encode('utf-8'))
        
        # Token positions info
        if result['token_positions']:
            pos_group = group.create_group('token_positions')
            for key, value in result['token_positions'].items():
                pos_group.create_dataset(key, data=value)
        
        # Vision embeddings
        if result['vision_embeddings'] is not None:
            vision_emb = np.array(result['vision_embeddings'], dtype=np.float32)
            group.create_dataset('vision_embeddings', 
                               data=vision_emb,
                               chunks=True,
                               compression='gzip')
        
        # Pre-generation embeddings
        pre_gen_group = group.create_group('pre_generation')
        for layer_name, layer_data in result['pre_generation'].items():
            layer_group = pre_gen_group.create_group(layer_name)
            
            # After image embeddings
            if 'after_image_embeddings' in layer_data:
                layer_group.create_dataset('after_image_embeddings',
                                         data=np.array(layer_data['after_image_embeddings'], dtype=np.float32),
                                         chunks=True,
                                         compression='gzip')
            
            # End query embeddings  
            if 'end_query_embeddings' in layer_data:
                layer_group.create_dataset('end_query_embeddings',
                                         data=np.array(layer_data['end_query_embeddings'], dtype=np.float32),
                                         chunks=True,
                                         compression='gzip')
        
        return True
    except Exception as e:
        print(f"Error writing embeddings for image {image_id}: {str(e)}")
        return False

class ImageDataset(Dataset):
    def __init__(self, image_ids, image_dir):
        self.image_ids = image_ids
        self.image_dir = image_dir

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        
        if os.path.exists(image_path):
            return image_id, image_path
        else:
            return image_id, None

def collate_fn(batch):
    """Collate function for DataLoader."""
    valid_batch = [(img_id, img_path) for img_id, img_path in batch if img_path is not None]
    
    if not valid_batch:
        return [], []
    
    image_ids = [item[0] for item in valid_batch]
    image_paths = [item[1] for item in valid_batch]
    
    return image_ids, image_paths

def process_batch(batch_data, selected_layers):
    """Process a batch of images."""
    image_ids, image_paths = batch_data
    results = []
    
    if not image_ids or not image_paths:
        return results
    
    for image_id, image_path in zip(image_ids, image_paths):
        try:
            result = extract_targeted_embeddings(image_id, image_path, selected_layers)
            if result:
                results.append((image_id, result))
        except Exception as e:
            print(f"Error processing image {image_id}: {e}")
            continue
    
    return results

def main():
    """Main function with GPU usage and progress tracking."""
    global processor, model, device
    
    try:
        print("🚀 Starting Qwen2.5-VL Targeted Embeddings Extraction")
        print("=" * 60)
        print(f"📊 Settings: {PART_SIZE} images per H5 file")
        if TEST_MODE:
            print("🧪 RUNNING IN TEST MODE - Processing only 2 images")
        print("=" * 60)
        
        # Check GPU and load model
        device = check_gpu_availability()
        
        print("📥 Loading Qwen2.5-VL model and processor...")
        print(f"Model: {MODEL_NAME}")
        
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_NAME, 
            torch_dtype="auto", 
            device_map="auto"
        )
        
        processor = AutoProcessor.from_pretrained(MODEL_NAME)
        
        device = model.device
        model_dtype = next(model.parameters()).dtype
        print(f"🖥️  Model loaded on device: {device}")
        print(f"🔢 Model dtype: {model_dtype}")
        
        model.eval()
        
        # Configure layers
        config = detect_and_configure_layers(model)
        selected_layers = config['selected_layers']
        
        # Load image IDs
        image_ids = load_image_ids()
        if not image_ids:
            print("No image IDs found. Exiting.")
            return
        
        print(f"Found {len(image_ids)} image IDs")
        
        if TEST_MODE:
            PART_SIZE_ACTUAL = min(PART_SIZE, len(image_ids))
        else:
            PART_SIZE_ACTUAL = PART_SIZE
        
        # Process in parts
        total_parts = (len(image_ids) + PART_SIZE_ACTUAL - 1) // PART_SIZE_ACTUAL
        
        if TEST_MODE:
            print(f"🧪 TEST MODE: Processing {len(image_ids)} images in {total_parts} part(s)")
        
        # Overall progress bar
        overall_pbar = tqdm(total=len(image_ids), desc="Overall Progress", position=0)
        
        for part in range(total_parts):
            start_idx = part * PART_SIZE_ACTUAL
            end_idx = min((part + 1) * PART_SIZE_ACTUAL, len(image_ids))
            current_batch = image_ids[start_idx:end_idx]
            
            # Create output directory
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            
            # Output file
            current_output_path = os.path.join(OUTPUT_DIR, f"qwen2_5_vl_targeted_embeddings_part{part+1}.h5")
            
            print(f"\n📦 Processing part {part+1}/{total_parts}")
            print(f"🖼️  Processing images {start_idx+1} to {end_idx}")
            print(f"💾 Output file: {current_output_path}")
            
            # Create dataset and dataloader
            dataset = ImageDataset(current_batch, IMAGE_DIR)
            dataloader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS,
                shuffle=False,
                collate_fn=collate_fn
            )
            
            # Process images with progress bar
            part_pbar = tqdm(total=len(current_batch), desc=f"Part {part+1}", position=1, leave=False)
            
            with h5py.File(current_output_path, 'w') as f:
                # Store configuration
                f.attrs['model_name'] = MODEL_NAME
                f.attrs['model_config'] = json.dumps(config, default=str)
                f.attrs['extraction_type'] = 'targeted_embeddings_fixed'
                f.attrs['selected_layers'] = json.dumps(selected_layers)
                
                saved_count = 0
                
                for batch_idx, batch_data in enumerate(dataloader):
                    try:
                        batch_results = process_batch(batch_data, selected_layers)
                        
                        for image_id, result in batch_results:
                            try:
                                if write_embeddings_hdf5(f, image_id, result):
                                    saved_count += 1
                            except Exception as e:
                                print(f"Error saving {image_id}: {e}")
                        
                        # Update progress bars
                        batch_size = len(batch_data[0]) if batch_data[0] else 0
                        part_pbar.update(batch_size)
                        overall_pbar.update(batch_size)
                        
                        # GPU memory cleanup
                        if torch.cuda.is_available() and batch_idx % 10 == 0:
                            torch.cuda.empty_cache()
                            
                    except Exception as e:
                        print(f"Error processing batch {batch_idx + 1}: {e}")
                        continue
            
            part_pbar.close()
            
            print(f"✅ Part {part+1} completed: {saved_count} images saved")
            
            if TEST_MODE:
                print("🧪 TEST MODE: Successfully processed test images!")
                break
            
            # Memory cleanup between parts
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        overall_pbar.close()
        
        if TEST_MODE:
            print("\n🎉 TEST MODE COMPLETED SUCCESSFULLY!")
            print("✅ Targeted embeddings extracted without errors")
            print("✅ Ready for full dataset processing (set TEST_MODE = False)")
        else:
            print("\n🎉 All parts processing complete!")
        
    except Exception as e:
        print(f"❌ Error in main function: {e}")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    main()