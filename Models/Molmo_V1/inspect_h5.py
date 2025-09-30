#!/usr/bin/env python3
"""
HDF5 Inspector for Molmo VQA Embeddings - PYTORCH ONLY
Inspect the structure and content of generated HDF5 files
USES ONLY PYTORCH - NO TENSORFLOW DEPENDENCIES
"""

# Ensure no TensorFlow is imported
try:
    import tensorflow
    raise ImportError("TensorFlow detected! This script uses PYTORCH ONLY. Please uninstall TensorFlow or use a clean environment.")
except ImportError:
    pass  # Good, no TensorFlow

import h5py
import numpy as np
import argparse
import os
from pathlib import Path

def inspect_h5_file(filepath: str, detailed: bool = False):
    """Inspect HDF5 file structure and content"""
    
    print(f"📁 Inspecting: {filepath}")
    print("=" * 60)
    
    try:
        with h5py.File(filepath, 'r') as f:
            # Print file attributes
            print("📋 File Attributes:")
            for key, value in f.attrs.items():
                print(f"  {key}: {value}")
            print()
            
            # Print groups (question IDs)
            question_ids = list(f.keys())
            print(f"🔢 Number of questions: {len(question_ids)}")
            print(f"📝 Sample question IDs: {question_ids[:5]}...")
            print()
            
            # Inspect first question in detail
            if question_ids and detailed:
                first_q = question_ids[0]
                print(f"🔍 Detailed inspection of: {first_q}")
                print("-" * 40)
                
                q_group = f[first_q]
                
                # Basic info
                if 'question' in q_group:
                    question_text = q_group['question'][()].decode('utf-8')
                    print(f"❓ Question: {question_text}")
                
                if 'image_id' in q_group:
                    image_id = q_group['image_id'][()].decode('utf-8')
                    print(f"🖼️  Image ID: {image_id}")
                
                if 'answer' in q_group:
                    answer = q_group['answer'][()].decode('utf-8')
                    print(f"💬 Generated Answer: {answer}")
                
                if 'ground_truth_answer' in q_group:
                    gt_answer = q_group['ground_truth_answer'][()].decode('utf-8')
                    print(f"✅ Ground Truth: {gt_answer}")
                
                print()
                
                # Vision only representation
                if 'vision_only_representation' in q_group:
                    vision_only = q_group['vision_only_representation'][:]
                    print(f"👁️  Vision Only Shape: {vision_only.shape}")
                    print(f"👁️  Vision Only Stats: mean={vision_only.mean():.4f}, std={vision_only.std():.4f}")
                    print()
                
                # Vision token representations
                if 'vision_token_representation' in q_group:
                    print("🎯 Vision Token Representations:")
                    vision_group = q_group['vision_token_representation']
                    for layer_name in vision_group.keys():
                        layer_data = vision_group[layer_name][:]
                        print(f"  {layer_name}: shape={layer_data.shape}, mean={layer_data.mean():.4f}")
                    print()
                
                # Query token representations
                if 'query_token_representation' in q_group:
                    print("❓ Query Token Representations:")
                    query_group = q_group['query_token_representation']
                    for layer_name in query_group.keys():
                        layer_data = query_group[layer_name][:]
                        print(f"  {layer_name}: shape={layer_data.shape}, mean={layer_data.mean():.4f}")
                    print()
            
            # Summary statistics
            print("📊 Summary Statistics:")
            total_size = os.path.getsize(filepath)
            print(f"  File size: {total_size / (1024*1024):.2f} MB")
            print(f"  Questions per MB: {len(question_ids) / (total_size / (1024*1024)):.1f}")
            
    except Exception as e:
        print(f"❌ Error inspecting file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Inspect Molmo VQA HDF5 files")
    parser.add_argument("path", help="Path to HDF5 file or directory containing HDF5 files")
    parser.add_argument("--detailed", "-d", action="store_true", help="Show detailed inspection of first sample")
    parser.add_argument("--all", "-a", action="store_true", help="Inspect all HDF5 files in directory")
    
    args = parser.parse_args()
    
    path = Path(args.path)
    
    if path.is_file() and path.suffix == '.h5':
        # Single file
        inspect_h5_file(str(path), args.detailed)
    elif path.is_dir():
        # Directory
        h5_files = list(path.glob("*.h5"))
        if not h5_files:
            print(f"❌ No HDF5 files found in {path}")
            return
        
        print(f"📁 Found {len(h5_files)} HDF5 files in {path}")
        print()
        
        if args.all:
            for h5_file in sorted(h5_files):
                inspect_h5_file(str(h5_file), args.detailed)
                print("\n" + "="*80 + "\n")
        else:
            # Just inspect the first file
            inspect_h5_file(str(sorted(h5_files)[0]), args.detailed)
            if len(h5_files) > 1:
                print(f"\n💡 Use --all to inspect all {len(h5_files)} files")
    else:
        print(f"❌ Invalid path: {path}")

if __name__ == "__main__":
    main()
