#!/usr/bin/env python3
"""
HDF5 Inspection Utility for Qwen2.5-VL-7B-Instruct Embeddings

Displays structure and contents of embedding HDF5 files.
"""

import h5py
import sys
import numpy as np
from pathlib import Path


def print_attrs(name, obj):
    """Print attributes of an HDF5 object"""
    if obj.attrs:
        print(f"\n  Attributes for {name}:")
        for key, val in obj.attrs.items():
            print(f"    {key}: {val}")


def print_dataset_info(name, obj):
    """Print information about a dataset"""
    if isinstance(obj, h5py.Dataset):
        print(f"  📄 {name}")
        print(f"     Shape: {obj.shape}")
        print(f"     Dtype: {obj.dtype}")

        # If it's a small array, show preview
        if obj.dtype.kind in ['f', 'i', 'u']:  # numeric types
            if obj.size <= 10:
                print(f"     Values: {obj[...]}")
            else:
                # Show statistics for embedding arrays
                data = obj[...]
                if isinstance(data, np.ndarray) and data.size > 0:
                    print(f"     Min: {data.min():.4f}")
                    print(f"     Max: {data.max():.4f}")
                    print(f"     Mean: {data.mean():.4f}")
                    print(f"     Std: {data.std():.4f}")
        elif obj.dtype.kind == 'S' or obj.dtype.kind == 'O':  # string types
            value = obj[()]
            if isinstance(value, bytes):
                value = value.decode('utf-8')
            # Truncate long strings
            if len(str(value)) > 200:
                print(f"     Value: {str(value)[:200]}...")
            else:
                print(f"     Value: {value}")


def inspect_question_group(grp, question_id):
    """Inspect a single question group in detail"""
    print(f"\n{'='*80}")
    print(f"Question ID: {question_id}")
    print(f"{'='*80}")

    # Print basic metadata
    if 'question' in grp:
        question = grp['question'][()]
        if isinstance(question, bytes):
            question = question.decode('utf-8')
        print(f"\n📝 Question: {question}")

    if 'image_id' in grp:
        image_id = grp['image_id'][()]
        if isinstance(image_id, bytes):
            image_id = image_id.decode('utf-8')
        print(f"🖼️  Image: {image_id}")

    if 'answer' in grp:
        answer = grp['answer'][()]
        if isinstance(answer, bytes):
            answer = answer.decode('utf-8')
        print(f"🤖 Generated Answer: {answer}")

    if 'ground_truth_answer' in grp:
        gt_answer = grp['ground_truth_answer'][()]
        if isinstance(gt_answer, bytes):
            gt_answer = gt_answer.decode('utf-8')
        print(f"✅ Ground Truth: {gt_answer}")

    # Print vision-only representation
    if 'vision_only_representation' in grp:
        vision_rep = grp['vision_only_representation']
        print(f"\n🎨 Vision-Only Representation:")
        print(f"   Shape: {vision_rep.shape}")
        print(f"   Dtype: {vision_rep.dtype}")
        data = vision_rep[...]
        print(f"   Stats: min={data.min():.4f}, max={data.max():.4f}, mean={data.mean():.4f}, std={data.std():.4f}")

    # Print vision token representations
    if 'vision_token_representation' in grp:
        print(f"\n🔍 Vision Token Representations:")
        vision_grp = grp['vision_token_representation']
        for layer_name in sorted(vision_grp.keys()):
            layer_data = vision_grp[layer_name]
            data = layer_data[...]
            print(f"   {layer_name}: shape={layer_data.shape}, "
                  f"min={data.min():.4f}, max={data.max():.4f}, "
                  f"mean={data.mean():.4f}")

    # Print query token representations
    if 'query_token_representation' in grp:
        print(f"\n💭 Query Token Representations:")
        query_grp = grp['query_token_representation']
        for layer_name in sorted(query_grp.keys()):
            layer_data = query_grp[layer_name]
            data = layer_data[...]
            print(f"   {layer_name}: shape={layer_data.shape}, "
                  f"min={data.min():.4f}, max={data.max():.4f}, "
                  f"mean={data.mean():.4f}")


def inspect_h5_file(filepath, detailed=False, sample_id=None):
    """Inspect HDF5 file structure and contents"""

    if not Path(filepath).exists():
        print(f"❌ File not found: {filepath}")
        return

    print(f"\n{'='*80}")
    print(f"Inspecting HDF5 File: {filepath}")
    print(f"{'='*80}")

    with h5py.File(filepath, 'r') as f:
        # Print file-level attributes
        print("\n📋 File Metadata:")
        for key, val in f.attrs.items():
            print(f"  {key}: {val}")

        # Get all question IDs
        question_ids = list(f.keys())
        print(f"\n📊 Total Samples: {len(question_ids)}")

        if sample_id:
            # Inspect specific sample
            if sample_id in f:
                inspect_question_group(f[sample_id], sample_id)
            else:
                print(f"❌ Sample ID '{sample_id}' not found in file")
                print(f"Available samples: {question_ids[:10]}...")

        elif detailed:
            # Inspect first 3 samples in detail
            print(f"\n🔍 Inspecting first 3 samples in detail...")
            for qid in question_ids[:3]:
                inspect_question_group(f[qid], qid)

        else:
            # Summary view
            print(f"\n📝 Sample IDs (first 10): {question_ids[:10]}")

            if question_ids:
                # Show structure of first sample
                first_qid = question_ids[0]
                print(f"\n🏗️  Structure (sample: {first_qid}):")

                def print_structure(name, obj, indent=0):
                    prefix = "  " * indent
                    if isinstance(obj, h5py.Group):
                        print(f"{prefix}📁 {name}/")
                    elif isinstance(obj, h5py.Dataset):
                        print(f"{prefix}📄 {name} - shape: {obj.shape}, dtype: {obj.dtype}")

                f[first_qid].visititems(lambda name, obj: print_structure(name, obj, indent=2))

                # Quick stats
                print(f"\n📈 Quick Statistics:")

                # Check vision representation
                if 'vision_only_representation' in f[first_qid]:
                    vision_shape = f[first_qid]['vision_only_representation'].shape
                    print(f"  Vision representation dimension: {vision_shape}")

                # Check layer counts
                if 'vision_token_representation' in f[first_qid]:
                    num_layers = len(f[first_qid]['vision_token_representation'].keys())
                    print(f"  Number of extracted layers: {num_layers}")
                    layer_names = sorted(f[first_qid]['vision_token_representation'].keys())
                    print(f"  Layer names: {layer_names}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_h5.py <h5_file> [--detailed] [--sample <question_id>]")
        print("\nOptions:")
        print("  --detailed       Show detailed information for first 3 samples")
        print("  --sample <id>    Show detailed information for specific sample")
        sys.exit(1)

    filepath = sys.argv[1]
    detailed = '--detailed' in sys.argv

    sample_id = None
    if '--sample' in sys.argv:
        idx = sys.argv.index('--sample')
        if idx + 1 < len(sys.argv):
            sample_id = sys.argv[idx + 1]

    inspect_h5_file(filepath, detailed=detailed, sample_id=sample_id)
    print("\n✅ Inspection complete!\n")


if __name__ == "__main__":
    main()
