import os
import json
import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm
import gc

def normalize_image_id(image_id):
    """Normalize image ID for matching."""
    if isinstance(image_id, str) and '_' in image_id:
        return image_id.split('_')[-1]  # Extract numeric part from H5
    
    if isinstance(image_id, (int, float)):
        return str(int(image_id)).zfill(12)  # Zero-pad CSV IDs
        
    return str(image_id)

def load_scores_data(csv_path):
    """Load and prepare scores data."""
    print("📊 Loading scores data...")
    
    scores_df = pd.read_csv(csv_path)
    print(f"   Loaded {len(scores_df)} rows")
    
    # Normalize IDs for matching
    scores_df['norm_id'] = scores_df['image_id'].apply(normalize_image_id)
    
    # Create lookup dictionary for fast access
    scores_dict = {}
    for _, row in scores_df.iterrows():
        scores_dict[row['norm_id']] = {
            'generated_caption': row.get('generated_caption', ''),
            'ground_truth_captions': row.get('ground_truth_captions', ''),
            'factual_score': row.get('factual_score', 0.0),
            'has_hallucination': row.get('has_hallucination', False),
            'BLEU-1': row.get('BLEU-1', 0.0),
            'BLEU-4': row.get('BLEU-4', 0.0),
            'ROUGE-L': row.get('ROUGE-L', 0.0),
            'chair_regression_score': row.get('chair_regression_score', 0.0)
        }
    
    print(f"   Created lookup dict for {len(scores_dict)} images")
    return scores_dict

def discover_embedding_structure(embeddings_dir):
    """Discover available embedding types."""
    print("🔍 Discovering embedding structure...")
    
    h5_files = [f for f in os.listdir(embeddings_dir) if f.endswith('.h5')]
    if not h5_files:
        return {}
    
    # Check first file
    sample_file = os.path.join(embeddings_dir, h5_files[0])
    configs = {}
    
    with h5py.File(sample_file, 'r') as f:
        sample_image = list(f.keys())[0]
        img_group = f[sample_image]
        
        print(f"   Sample image: {sample_image}")
        print(f"   Available keys: {list(img_group.keys())}")
        
        # Vision embeddings
        if 'vision_embeddings' in img_group:
            configs['vision_embeddings'] = ['vision_embeddings']
            print(f"   ✅ Found vision_embeddings: {img_group['vision_embeddings'].shape}")
        
        # Pre-generation embeddings
        if 'pre_generation' in img_group:
            pre_gen = img_group['pre_generation']
            layers = list(pre_gen.keys())
            print(f"   📥 Pre-generation layers: {layers}")
            
            for layer in layers:
                layer_group = pre_gen[layer]
                emb_types = list(layer_group.keys())
                print(f"     {layer}: {emb_types}")
                
                for emb_type in emb_types:
                    config_name = f"pre_generation_{layer}_{emb_type.replace('_embeddings', '')}"
                    configs[config_name] = ['pre_generation', layer, emb_type]
    
    print(f"   📋 Discovered {len(configs)} embedding types")
    return configs

def extract_embedding(img_group, path):
    """Extract embedding from H5 group using path."""
    try:
        current = img_group
        for step in path:
            if step in current:
                current = current[step]
            else:
                return None
        
        # Convert to numpy array
        if hasattr(current, 'shape'):
            return np.array(current)
        return None
        
    except Exception as e:
        return None

def process_single_config(config_name, path, embeddings_dir, scores_dict, output_dir, chunk_size=1000):
    """Process a single embedding configuration with chunked processing."""
    print(f"\n🔨 Processing: {config_name}")
    print(f"   Path: {' -> '.join(path)}")
    
    # Get all H5 files
    h5_files = [f for f in os.listdir(embeddings_dir) if f.endswith('.h5')]
    h5_files.sort()  # Process in order
    
    output_file = os.path.join(output_dir, f"qwen_dataset_{config_name}.csv")
    
    # Process files in chunks to avoid memory issues
    all_rows = []
    processed_count = 0
    matched_count = 0
    
    for h5_file in tqdm(h5_files, desc=f"Processing {config_name}"):
        file_path = os.path.join(embeddings_dir, h5_file)
        
        try:
            with h5py.File(file_path, 'r') as f:
                image_ids = list(f.keys())
                
                for image_id in image_ids:
                    try:
                        img_group = f[image_id]
                        
                        # Extract embedding
                        embedding = extract_embedding(img_group, path)
                        
                        if embedding is not None:
                            processed_count += 1
                            
                            # Normalize ID and check for scores
                            norm_id = normalize_image_id(image_id)
                            
                            if norm_id in scores_dict:
                                matched_count += 1
                                scores = scores_dict[norm_id]
                                
                                row = {
                                    'image_id': image_id,
                                    'corresponding_embedding': embedding.tolist(),
                                    'generated_caption': scores['generated_caption'],
                                    'ground_truth_captions': scores['ground_truth_captions'],
                                    'factual_score': scores['factual_score'],
                                    'has_hallucination': scores['has_hallucination'],
                                    'BLEU-1': scores['BLEU-1'],
                                    'BLEU-4': scores['BLEU-4'],
                                    'ROUGE-L': scores['ROUGE-L'],
                                    'chair_regression_score': scores['chair_regression_score']
                                }
                                
                                all_rows.append(row)
                                
                                # Save chunk to avoid memory buildup
                                if len(all_rows) >= chunk_size:
                                    save_chunk(all_rows, output_file, is_first=(matched_count <= chunk_size))
                                    all_rows = []
                                    gc.collect()  # Force garbage collection
                        
                    except Exception as e:
                        continue
                        
        except Exception as e:
            print(f"   Error processing {h5_file}: {e}")
            continue
    
    # Save remaining rows
    if all_rows:
        save_chunk(all_rows, output_file, is_first=(matched_count <= len(all_rows)))
    
    print(f"   ✅ Processed {processed_count} embeddings, matched {matched_count}")
    
    if matched_count > 0:
        print(f"   💾 Saved to: {output_file}")
        
        # Create summary
        create_summary(output_file, config_name, path, matched_count)
        return True
    else:
        print(f"   ❌ No matches found for {config_name}")
        return False

def save_chunk(rows, output_file, is_first=True):
    """Save a chunk of rows to CSV file."""
    df = pd.DataFrame(rows)
    
    # Write header only for first chunk
    mode = 'w' if is_first else 'a'
    header = is_first
    
    df.to_csv(output_file, mode=mode, header=header, index=False)

def create_summary(csv_file, config_name, path, total_samples):
    """Create summary statistics for the dataset."""
    try:
        # Read just a sample to get embedding dimension
        sample_df = pd.read_csv(csv_file, nrows=1)
        embedding_dim = len(eval(sample_df['corresponding_embedding'].iloc[0]))
        
        # Read full file for statistics (only if not too large)
        if total_samples < 10000:
            full_df = pd.read_csv(csv_file)
            
            summary = {
                'dataset_name': f"qwen_dataset_{config_name}",
                'total_samples': total_samples,
                'embedding_dimension': embedding_dim,
                'embedding_path': path,
                'statistics': {
                    'factual_score': {
                        'mean': float(full_df['factual_score'].mean()),
                        'std': float(full_df['factual_score'].std()),
                        'min': float(full_df['factual_score'].min()),
                        'max': float(full_df['factual_score'].max())
                    },
                    'hallucination_rate': float(full_df['has_hallucination'].mean()),
                    'avg_bleu4': float(full_df['BLEU-4'].mean()),
                    'avg_rouge_l': float(full_df['ROUGE-L'].mean())
                }
            }
        else:
            # For large files, create basic summary
            summary = {
                'dataset_name': f"qwen_dataset_{config_name}",
                'total_samples': total_samples,
                'embedding_dimension': embedding_dim,
                'embedding_path': path,
                'note': 'Statistics not computed for large dataset to save memory'
            }
        
        # Save summary
        summary_file = csv_file.replace('.csv', '_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
    except Exception as e:
        print(f"   Warning: Could not create summary: {e}")

def main():
    """Main function - simple and robust."""
    
    # Paths
    embeddings_dir = "/root/akhil_workspace/qwen_model/extracted_embeddings_coco2014"
    scores_csv_path = "/root/akhil_workspace/qwen_model/evaluation_results/qwen2_comprehensive_scores.csv"
    output_dir = "/root/akhil_workspace/qwen_model/comprehensive_datasets"
    
    print("🚀 Simple Qwen Dataset Creation")
    print("=" * 50)
    print("Features:")
    print("  • Memory-efficient chunked processing")
    print("  • No complex GPU operations")
    print("  • Robust error handling")
    print("  • Saves datasets incrementally")
    print("=" * 50)
    
    # Check paths
    if not os.path.exists(embeddings_dir):
        print(f"❌ Embeddings directory not found: {embeddings_dir}")
        return
    
    if not os.path.exists(scores_csv_path):
        print(f"❌ Scores CSV not found: {scores_csv_path}")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load scores data
    scores_dict = load_scores_data(scores_csv_path)
    
    # Discover embedding structure
    configs = discover_embedding_structure(embeddings_dir)
    
    if not configs:
        print("❌ No embedding configurations found!")
        return
    
    print(f"\n🎯 Will create {len(configs)} datasets:")
    for config_name in configs.keys():
        print(f"   • {config_name}")
    
    # Process each configuration
    successful = 0
    failed = 0
    
    for config_name, path in configs.items():
        try:
            success = process_single_config(
                config_name, path, embeddings_dir, scores_dict, output_dir
            )
            
            if success:
                successful += 1
            else:
                failed += 1
                
        except Exception as e:
            print(f"   ❌ Failed {config_name}: {e}")
            failed += 1
            
        # Force cleanup after each config
        gc.collect()
    
    # Final summary
    print(f"\n🎉 Dataset Creation Complete!")
    print(f"   ✅ Successful: {successful}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📁 Output directory: {output_dir}")
    
    if successful > 0:
        print(f"\n📋 Each dataset contains:")
        print(f"   • image_id, corresponding_embedding, generated_caption")
        print(f"   • ground_truth_captions, factual_score, has_hallucination")
        print(f"   • BLEU-1, BLEU-4, ROUGE-L, chair_regression_score")

if __name__ == "__main__":
    main()