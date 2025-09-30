import os
import json
import h5py
import pandas as pd
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import glob

def inspect_qwen_h5_structure(embeddings_dir, max_files=2):
    """Inspect Qwen2.5-VL H5 files to understand the exact structure."""
    print("🔍 Inspecting Qwen2.5-VL H5 file structure...")
    
    h5_files = [f for f in os.listdir(embeddings_dir) if f.endswith('.h5')][:max_files]
    structure_info = {}
    
    for h5_file in h5_files:
        file_path = os.path.join(embeddings_dir, h5_file)
        
        try:
            with h5py.File(file_path, 'r') as f:
                # Get first image as sample
                first_image_id = list(f.keys())[0]
                img_group = f[first_image_id]
                
                print(f"📁 Analyzing {h5_file} - Sample image: {first_image_id}")
                
                # Check vision embeddings
                if 'vision_embeddings' in img_group:
                    shape = img_group['vision_embeddings'].shape
                    structure_info['vision_embeddings'] = shape
                    print(f"   ✅ vision_embeddings: {shape}")
                
                # Check pre-generation structure
                if 'pre_generation' in img_group:
                    pre_gen = img_group['pre_generation']
                    layers = list(pre_gen.keys())
                    print(f"   📥 pre_generation layers: {layers}")
                    
                    for layer in layers[:3]:  # Check first 3 layers
                        layer_group = pre_gen[layer]
                        embedding_types = list(layer_group.keys())
                        print(f"     {layer}: {embedding_types}")
                        
                        for emb_type in embedding_types:
                            shape = layer_group[emb_type].shape
                            key = f"pre_generation.{layer}.{emb_type}"
                            structure_info[key] = shape
                
                # Check token positions if available
                if 'token_positions' in img_group:
                    pos_group = img_group['token_positions']
                    pos_keys = list(pos_group.keys())
                    print(f"   📍 token_positions: {pos_keys}")
                
                break  # Only inspect first file for structure
                
        except Exception as e:
            print(f"Error inspecting {h5_file}: {e}")
    
    print(f"\n📋 Discovered Qwen2.5-VL embedding structure:")
    for key, shape in structure_info.items():
        print(f"   {key}: {shape}")
    
    return structure_info

def load_qwen_embeddings_from_h5_files(embeddings_dir):
    """Extract ALL embeddings from Qwen2.5-VL H5 files."""
    print("🔍 Scanning Qwen2.5-VL H5 files for embeddings...")
    
    all_embeddings = {}
    h5_files = [f for f in os.listdir(embeddings_dir) if f.endswith('.h5')]
    
    for h5_file in tqdm(h5_files, desc="Processing H5 files"):
        file_path = os.path.join(embeddings_dir, h5_file)
        
        try:
            with h5py.File(file_path, 'r') as f:
                # Iterate through all image IDs in this H5 file
                for image_id in f.keys():
                    if image_id not in all_embeddings:
                        all_embeddings[image_id] = {}
                    
                    img_group = f[image_id]
                    
                    # Basic metadata
                    if 'generated_caption' in img_group:
                        caption = img_group['generated_caption'][()].decode('utf-8') if isinstance(img_group['generated_caption'][()], bytes) else str(img_group['generated_caption'][()])
                        all_embeddings[image_id]['generated_caption'] = caption
                    
                    # Vision embeddings - direct access
                    if 'vision_embeddings' in img_group:
                        vision_emb = np.array(img_group['vision_embeddings'])
                        all_embeddings[image_id]['vision_embeddings'] = vision_emb
                    
                    # Pre-generation embeddings - ALL layers and types
                    if 'pre_generation' in img_group:
                        pre_gen_data = {}
                        pre_gen_group = img_group['pre_generation']
                        
                        for layer_name in pre_gen_group.keys():
                            layer_group = pre_gen_group[layer_name]
                            layer_data = {}
                            
                            # Extract all embedding types in this layer
                            for emb_type in layer_group.keys():
                                if emb_type.endswith('_embeddings'):
                                    emb_array = np.array(layer_group[emb_type])
                                    layer_data[emb_type] = emb_array
                            
                            pre_gen_data[layer_name] = layer_data
                        
                        all_embeddings[image_id]['pre_generation'] = pre_gen_data
                    
                    # Token positions (for reference)
                    if 'token_positions' in img_group:
                        pos_data = {}
                        pos_group = img_group['token_positions']
                        for key in pos_group.keys():
                            pos_data[key] = int(pos_group[key][()])
                        all_embeddings[image_id]['token_positions'] = pos_data
                        
        except Exception as e:
            print(f"Error processing {h5_file}: {e}")
            continue
    
    print(f"✅ Extracted embeddings for {len(all_embeddings)} images from {len(h5_files)} H5 files")
    return all_embeddings

def generate_qwen_embedding_configs():
    """Generate all possible embedding configurations for Qwen2.5-VL."""
    
    configs = {}
    
    # 1. Vision embeddings
    configs["vision_embeddings"] = {
        "path_components": ["vision_embeddings"],
        "description": "Vision embeddings extracted from sequence hidden states"
    }
    
    # 2. Pre-generation embeddings - based on typical Qwen2.5-VL structure
    # Common layers: 0, 7, 14, 21, 27 (for 28-layer model) or 0, 9, 18, 27, 35 (for 36-layer model)
    # We'll be flexible and extract whatever layers are available
    
    # Typical embedding types for Qwen2.5-VL
    embedding_types = ["after_image_embeddings", "end_query_embeddings"]
    
    # We'll dynamically discover layers, but define common expected layers
    expected_layers = ["layer_0", "layer_7", "layer_14", "layer_21", "layer_27"]  # For 28-layer model
    
    for layer in expected_layers:
        for emb_type in embedding_types:
            config_name = f"pre_generation_{layer}_{emb_type.replace('_embeddings', '')}"
            configs[config_name] = {
                "path_components": ["pre_generation", layer, emb_type],
                "description": f"Pre-generation {layer} {emb_type.replace('_', ' ')}"
            }
    
    return configs

def extract_embedding_by_path(data, path_components):
    """Extract embedding using path components (list of keys)."""
    try:
        current = data
        for component in path_components:
            if isinstance(current, dict) and component in current:
                current = current[component]
            else:
                return None
        
        # Return the numpy array directly
        if isinstance(current, np.ndarray):
            return current
        
        return None
    except Exception as e:
        return None

def normalize_image_id(image_id):
    """Normalize image ID by removing the prefix and keeping only the numeric part."""
    # If the ID contains underscores, extract the numeric part (last part)
    if isinstance(image_id, str) and '_' in image_id:
        return image_id.split('_')[-1]
    
    # If it's a number, convert to zero-padded string
    if isinstance(image_id, (int, float)):
        return str(int(image_id)).zfill(12)
        
    return str(image_id)

def load_qwen_scores_data(scores_csv_path):
    """Load Qwen2.5-VL evaluation scores from CSV."""
    print("📊 Loading Qwen2.5-VL evaluation scores...")
    
    if not os.path.exists(scores_csv_path):
        print(f"❌ Scores file not found: {scores_csv_path}")
        return pd.DataFrame()
    
    try:
        scores_df = pd.read_csv(scores_csv_path)
        print(f"   📋 Loaded {len(scores_df)} rows")
        print(f"   📊 Columns: {list(scores_df.columns)}")
        
        # Normalize image IDs for matching
        scores_df['normalized_image_id'] = scores_df['image_id'].apply(normalize_image_id)
        
        # Show sample data
        print(f"   📋 Sample image IDs: {scores_df['image_id'].head(3).tolist()}")
        print(f"   📋 Sample normalized IDs: {scores_df['normalized_image_id'].head(3).tolist()}")
        
        return scores_df
        
    except Exception as e:
        print(f"   ❌ Error loading scores: {e}")
        return pd.DataFrame()

def create_qwen_comprehensive_dataset(embeddings_dir, scores_csv_path, output_dir):
    """Create comprehensive dataset with Qwen2.5-VL embeddings and scores."""
    
    # First inspect the H5 structure
    structure_info = inspect_qwen_h5_structure(embeddings_dir)
    
    # Load scores data
    scores_df = load_qwen_scores_data(scores_csv_path)
    
    if scores_df.empty:
        print("❌ No scores data available!")
        return
    
    # Load embeddings
    print("📁 Loading ALL Qwen2.5-VL embeddings from H5 files...")
    all_embeddings = load_qwen_embeddings_from_h5_files(embeddings_dir)
    
    if not all_embeddings:
        print("❌ No embeddings found! Check your directory path.")
        return
    
    # Generate embedding configurations dynamically based on available data
    print("🎯 Generating embedding configurations based on available data...")
    
    # First, discover what layers are actually available
    sample_embedding = list(all_embeddings.values())[0]
    available_layers = []
    available_embedding_types = set()
    
    if 'pre_generation' in sample_embedding:
        available_layers = list(sample_embedding['pre_generation'].keys())
        for layer_data in sample_embedding['pre_generation'].values():
            available_embedding_types.update(layer_data.keys())
    
    print(f"   📋 Available layers: {available_layers}")
    print(f"   📋 Available embedding types: {list(available_embedding_types)}")
    
    # Generate configurations based on available data
    embedding_configs = {}
    
    # Vision embeddings
    embedding_configs["vision_embeddings"] = {
        "path_components": ["vision_embeddings"],
        "description": "Vision embeddings extracted from sequence hidden states"
    }
    
    # Pre-generation embeddings
    for layer in available_layers:
        for emb_type in available_embedding_types:
            config_name = f"pre_generation_{layer}_{emb_type.replace('_embeddings', '')}"
            embedding_configs[config_name] = {
                "path_components": ["pre_generation", layer, emb_type],
                "description": f"Pre-generation {layer} {emb_type.replace('_', ' ')}"
            }
    
    print(f"\n🎯 Generated {len(embedding_configs)} embedding configurations:")
    for config_name in list(embedding_configs.keys())[:10]:  # Show first 10
        print(f"   • {config_name}")
    if len(embedding_configs) > 10:
        print(f"   ... and {len(embedding_configs) - 10} more")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create datasets for each embedding configuration
    successful_datasets = 0
    failed_datasets = 0
    
    # Required columns as specified
    required_columns = [
        'image_id', 'corresponding_embedding', 'generated_caption', 
        'ground_truth_captions', 'factual_score', 'has_hallucination',
        'BLEU-1', 'BLEU-4', 'ROUGE-L', 'chair_regression_score'
    ]
    
    for config_name, config in embedding_configs.items():
        print(f"\n🔨 Creating dataset for {config_name}...")
        print(f"   Description: {config['description']}")
        print(f"   Path: {' -> '.join(config['path_components'])}")
        
        # Initialize lists to store data
        dataset_rows = []
        
        # Process each image
        successful_extractions = 0
        for image_id, embedding_data in tqdm(all_embeddings.items(), desc=f"Processing {config_name}", leave=False):
            
            # Extract specific embedding using the path components
            embedding = extract_embedding_by_path(embedding_data, config["path_components"])
            
            if embedding is not None and len(embedding) > 0:
                # Normalize image ID for matching with scores
                normalized_id = normalize_image_id(image_id)
                
                # Find matching scores
                matching_scores = scores_df[scores_df['normalized_image_id'] == normalized_id]
                
                if not matching_scores.empty:
                    score_row = matching_scores.iloc[0]  # Take first match
                    
                    # Create row data with required columns
                    row_data = {
                        'image_id': image_id,
                        'corresponding_embedding': embedding.tolist(),  # Convert to list for CSV storage
                        'generated_caption': score_row.get('generated_caption', ''),
                        'ground_truth_captions': score_row.get('ground_truth_captions', ''),
                        'factual_score': score_row.get('factual_score', 0.0),
                        'has_hallucination': score_row.get('has_hallucination', False),
                        'BLEU-1': score_row.get('BLEU-1', 0.0),
                        'BLEU-4': score_row.get('BLEU-4', 0.0),
                        'ROUGE-L': score_row.get('ROUGE-L', 0.0),
                        'chair_regression_score': score_row.get('chair_regression_score', 0.0)
                    }
                    
                    dataset_rows.append(row_data)
                    successful_extractions += 1
        
        if successful_extractions == 0:
            print(f"   ❌ No embeddings found for path: {' -> '.join(config['path_components'])}")
            failed_datasets += 1
            continue
        
        # Create DataFrame
        df = pd.DataFrame(dataset_rows)
        
        # Verify all required columns are present
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"   ⚠️ Warning: Missing columns: {missing_cols}")
        
        # Save to CSV
        output_file = os.path.join(output_dir, f"qwen_dataset_{config_name}.csv")
        df.to_csv(output_file, index=False)
        
        print(f"   ✅ Saved {len(df)} samples to {output_file}")
        print(f"   📊 Embedding shape: {np.array(dataset_rows[0]['corresponding_embedding']).shape}")
        print(f"   📈 Avg factual score: {df['factual_score'].mean():.4f}")
        print(f"   🔍 Hallucination rate: {df['has_hallucination'].mean():.4f}")
        
        successful_datasets += 1
        
        # Create a summary statistics file
        summary_stats = {
            'dataset_name': f"qwen_dataset_{config_name}",
            'total_samples': len(df),
            'embedding_dimension': len(dataset_rows[0]['corresponding_embedding']),
            'embedding_path': config['path_components'],
            'statistics': {
                'factual_score': {
                    'mean': float(df['factual_score'].mean()),
                    'std': float(df['factual_score'].std()),
                    'min': float(df['factual_score'].min()),
                    'max': float(df['factual_score'].max())
                },
                'evaluation_metrics': {
                    'hallucination_rate': float(df['has_hallucination'].mean()),
                    'avg_bleu1': float(df['BLEU-1'].mean()),
                    'avg_bleu4': float(df['BLEU-4'].mean()),
                    'avg_rouge_l': float(df['ROUGE-L'].mean()),
                    'avg_chair_regression': float(df['chair_regression_score'].mean())
                }
            }
        }
        
        # Save summary
        summary_file = os.path.join(output_dir, f"dataset_summary_{config_name}.json")
        with open(summary_file, 'w') as f:
            json.dump(summary_stats, f, indent=2)
    
    print(f"\n📊 Dataset Creation Summary:")
    print(f"   ✅ Successful datasets: {successful_datasets}")
    print(f"   ❌ Failed datasets: {failed_datasets}")
    print(f"   📁 Total configurations attempted: {len(embedding_configs)}")
    
    return successful_datasets, failed_datasets

def main():
    """Main function to create Qwen2.5-VL comprehensive datasets."""
    
    # Set up paths
    embeddings_dir = "/root/akhil_workspace/qwen_model/extracted_embeddings_coco2014"
    scores_csv_path = "/root/akhil_workspace/qwen_model/evaluation_results/qwen2_comprehensive_scores.csv"
    output_dir = "/root/akhil_workspace/qwen_model/comprehensive_datasets"
    
    print("🚀 Starting Qwen2.5-VL Comprehensive Dataset Creation Pipeline")
    print("=" * 80)
    print("📋 Will extract embeddings:")
    print("   • Vision embeddings")
    print("   • Pre-generation: layers × types (after_image, end_query)")
    print("   • No post-generation embeddings (not available in Qwen2.5-VL)")
    print("=" * 80)
    
    # Check if required directories exist
    if not os.path.exists(embeddings_dir):
        print(f"❌ Embeddings directory not found: {embeddings_dir}")
        return
    
    if not os.path.exists(scores_csv_path):
        print(f"❌ Scores CSV file not found: {scores_csv_path}")
        return
    
    # Create comprehensive datasets
    successful, failed = create_qwen_comprehensive_dataset(embeddings_dir, scores_csv_path, output_dir)
    
    print(f"\n🎉 Qwen2.5-VL comprehensive dataset creation finished!")
    print(f"📁 Datasets saved to: {output_dir}")
    print(f"\n📋 Each dataset contains the required columns:")
    print(f"   • image_id, corresponding_embedding, generated_caption")
    print(f"   • ground_truth_captions, factual_score, has_hallucination")
    print(f"   • BLEU-1, BLEU-4, ROUGE-L, chair_regression_score")
    
    if successful > 0:
        print(f"\n✅ Successfully created {successful} datasets!")
        print(f"🔗 These datasets are ready for probe training and analysis")

if __name__ == "__main__":
    main()