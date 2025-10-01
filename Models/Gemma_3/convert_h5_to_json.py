#!/usr/bin/env python3
"""
Convert HDF5 embeddings to JSON format
"""

import h5py
import json
import numpy as np
import argparse
from pathlib import Path


def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, bytes):
        return obj.decode('utf-8')
    return obj


def convert_h5_to_json(h5_path, json_path):
    """Convert HDF5 file to JSON format"""

    print(f"Converting {h5_path} to JSON...")

    with h5py.File(h5_path, 'r') as h5_file:
        # Read metadata
        metadata = {}
        for key in h5_file.attrs.keys():
            metadata[key] = convert_numpy_types(h5_file.attrs[key])

        # Read all samples
        samples = {}
        for question_id in h5_file.keys():
            sample_group = h5_file[question_id]
            sample_data = {}

            # Extract string fields
            for field in ['question', 'image_id', 'answer', 'ground_truth_answer']:
                if field in sample_group:
                    value = sample_group[field][()]
                    sample_data[field] = convert_numpy_types(value)

            # Extract vision_only_representation
            if 'vision_only_representation' in sample_group:
                sample_data['vision_only_representation'] = sample_group['vision_only_representation'][:].tolist()

            # Extract vision_token_representation layers
            if 'vision_token_representation' in sample_group:
                vision_token = {}
                for layer_name in sample_group['vision_token_representation'].keys():
                    vision_token[layer_name] = sample_group['vision_token_representation'][layer_name][:].tolist()
                sample_data['vision_token_representation'] = vision_token

            # Extract query_token_representation layers
            if 'query_token_representation' in sample_group:
                query_token = {}
                for layer_name in sample_group['query_token_representation'].keys():
                    query_token[layer_name] = sample_group['query_token_representation'][layer_name][:].tolist()
                sample_data['query_token_representation'] = query_token

            samples[question_id] = sample_data

        # Combine metadata and samples
        output = {
            'metadata': metadata,
            'samples': samples
        }

        # Write to JSON
        print(f"Writing to {json_path}...")
        with open(json_path, 'w') as json_file:
            json.dump(output, json_file, indent=2)

        print(f"✅ Conversion complete!")
        print(f"   Metadata keys: {list(metadata.keys())}")
        print(f"   Number of samples: {len(samples)}")

        # Print file sizes
        h5_size = Path(h5_path).stat().st_size / (1024 * 1024)  # MB
        json_size = Path(json_path).stat().st_size / (1024 * 1024)  # MB
        print(f"   H5 file size: {h5_size:.2f} MB")
        print(f"   JSON file size: {json_size:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HDF5 embeddings to JSON")
    parser.add_argument("--h5-file", type=str, required=True, help="Input HDF5 file path")
    parser.add_argument("--json-file", type=str, help="Output JSON file path (default: same name with .json extension)")

    args = parser.parse_args()

    h5_path = args.h5_file
    json_path = args.json_file or h5_path.replace('.h5', '.json')

    convert_h5_to_json(h5_path, json_path)
