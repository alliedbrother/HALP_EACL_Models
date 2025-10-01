#!/usr/bin/env python3
"""
Convert Qwen2.5-VL HDF5 embeddings to JSON format
Preserves the exact structure from the H5 file
"""

import h5py
import json
import numpy as np
import argparse
from pathlib import Path
import sys


def convert_numpy_types(obj):
    """Convert numpy types to native Python types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, bytes):
        return obj.decode('utf-8')
    return obj


def h5_to_dict(h5_file_path):
    """Convert H5 file to nested dictionary maintaining structure"""

    result = {
        'metadata': {},
        'samples': {}
    }

    with h5py.File(h5_file_path, 'r') as f:
        # Extract file-level metadata
        print(f"Reading metadata from {h5_file_path}...")
        for key, val in f.attrs.items():
            result['metadata'][key] = convert_numpy_types(val)

        # Extract all samples
        total_samples = len(f.keys())
        print(f"Found {total_samples} samples")

        for idx, question_id in enumerate(f.keys(), 1):
            print(f"Processing sample {idx}/{total_samples}: {question_id}")

            grp = f[question_id]
            sample_data = {}

            # Extract scalar datasets (strings)
            for key in ['question', 'image_id', 'answer', 'ground_truth_answer']:
                if key in grp:
                    val = grp[key][()]
                    if isinstance(val, bytes):
                        val = val.decode('utf-8')
                    sample_data[key] = val

            # Extract vision_only_representation (2D array)
            if 'vision_only_representation' in grp:
                vision_data = grp['vision_only_representation'][:]
                sample_data['vision_only_representation'] = {
                    'shape': list(vision_data.shape),
                    'data': vision_data.tolist()
                }

            # Extract vision_token_representation (group of 1D arrays)
            if 'vision_token_representation' in grp:
                vision_token_grp = grp['vision_token_representation']
                sample_data['vision_token_representation'] = {}

                for layer_name in sorted(vision_token_grp.keys()):
                    layer_data = vision_token_grp[layer_name][:]
                    sample_data['vision_token_representation'][layer_name] = {
                        'shape': list(layer_data.shape),
                        'data': layer_data.tolist()
                    }

            # Extract query_token_representation (group of 1D arrays)
            if 'query_token_representation' in grp:
                query_token_grp = grp['query_token_representation']
                sample_data['query_token_representation'] = {}

                for layer_name in sorted(query_token_grp.keys()):
                    layer_data = query_token_grp[layer_name][:]
                    sample_data['query_token_representation'][layer_name] = {
                        'shape': list(layer_data.shape),
                        'data': layer_data.tolist()
                    }

            result['samples'][question_id] = sample_data

    return result


def main():
    parser = argparse.ArgumentParser(description='Convert H5 embeddings to JSON')
    parser.add_argument('h5_file', help='Input H5 file path')
    parser.add_argument('--output', '-o', help='Output JSON file path (default: same name as input with .json extension)')
    parser.add_argument('--indent', type=int, default=2, help='JSON indentation (default: 2, use 0 for compact)')
    parser.add_argument('--compact', action='store_true', help='Compact JSON output (no indentation)')

    args = parser.parse_args()

    # Validate input file
    h5_path = Path(args.h5_file)
    if not h5_path.exists():
        print(f"Error: File not found: {h5_path}")
        sys.exit(1)

    # Determine output path
    if args.output:
        json_path = Path(args.output)
    else:
        json_path = h5_path.with_suffix('.json')

    print(f"Converting {h5_path} to {json_path}")
    print("="*80)

    # Convert H5 to dict
    try:
        data = h5_to_dict(h5_path)
    except Exception as e:
        print(f"Error reading H5 file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Write to JSON
    print(f"\nWriting to {json_path}...")
    indent = None if args.compact else args.indent

    try:
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=indent)

        # Print statistics
        json_size = json_path.stat().st_size
        h5_size = h5_path.stat().st_size

        print("="*80)
        print("✅ Conversion complete!")
        print(f"   H5 file size:   {h5_size / 1024:.2f} KB")
        print(f"   JSON file size: {json_size / 1024:.2f} KB")
        print(f"   Size ratio:     {json_size / h5_size:.2f}x")
        print(f"   Total samples:  {len(data['samples'])}")
        print(f"   Output: {json_path}")

    except Exception as e:
        print(f"Error writing JSON file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
