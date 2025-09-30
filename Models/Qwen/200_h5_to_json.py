import h5py
import json
import numpy as np
import os
import sys
from datetime import datetime

def convert_to_json_safe(value):
    """Convert any value to JSON-safe format."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, bytes):
        return value.decode('utf-8')
    elif isinstance(value, (np.integer, np.floating)):
        return value.item()
    else:
        return value

def extract_embeddings_from_group(group):
    """Extract all embeddings from H5 group recursively."""
    result = {}
    
    for key in group.keys():
        item = group[key]
        
        if isinstance(item, h5py.Dataset):
            # It's data - extract it
            data = item[()]
            result[key] = convert_to_json_safe(data)
        elif isinstance(item, h5py.Group):
            # It's a group - go deeper
            result[key] = extract_embeddings_from_group(item)
    
    return result

def extract_qwen_embeddings_to_json(h5_file_path):
    """Extract Qwen2.5-VL embeddings from H5 file to JSON."""
    
    # Check if file exists
    if not os.path.exists(h5_file_path):
        print(f"❌ Error: File not found - {h5_file_path}")
        return
    
    # Create output filename
    base_name = os.path.splitext(h5_file_path)[0]
    output_file = f"{base_name}_embeddings.json"
    
    print(f"🔍 Processing: {h5_file_path}")
    print(f"💾 Output: {output_file}")
    
    try:
        # Open H5 file and extract everything
        with h5py.File(h5_file_path, 'r') as f:
            
            print(f"📊 Analyzing H5 file structure...")
            
            # Extract file attributes (model info, config, etc.)
            file_attrs = {}
            for attr_name in f.attrs.keys():
                file_attrs[attr_name] = convert_to_json_safe(f.attrs[attr_name])
            
            # Get all image IDs (top-level keys)
            image_ids = list(f.keys())
            print(f"📸 Found {len(image_ids)} images in the file")
            
            # Extract embeddings for all images
            all_embeddings = {}
            
            for i, image_id in enumerate(image_ids):
                if i % 10 == 0:  # Progress update every 10 images
                    print(f"   Processing image {i+1}/{len(image_ids)}: {image_id}")
                
                image_group = f[image_id]
                image_embeddings = extract_embeddings_from_group(image_group)
                all_embeddings[image_id] = image_embeddings
            
            # Create final structure
            result = {
                'extraction_info': {
                    'source_file': h5_file_path,
                    'extraction_date': datetime.now().isoformat(),
                    'total_images': len(image_ids),
                    'embedding_type': 'qwen2_5_vl_targeted_embeddings'
                },
                'file_attributes': file_attrs,
                'embeddings': all_embeddings
            }
            
            # Save to JSON
            print(f"💾 Saving embeddings to JSON...")
            with open(output_file, 'w') as json_file:
                json.dump(result, json_file, indent=2)
            
            print(f"✅ Success! Extracted embeddings to {output_file}")
            
            # Show file sizes and summary
            h5_size = os.path.getsize(h5_file_path) / 1024 / 1024
            json_size = os.path.getsize(output_file) / 1024 / 1024
            
            print(f"\n📊 Summary:")
            print(f"   📁 H5 file size: {h5_size:.2f} MB")
            print(f"   📁 JSON file size: {json_size:.2f} MB")
            print(f"   📸 Total images: {len(image_ids)}")
            print(f"   🧠 Embeddings per image:")
            
            # Show structure of first image as example
            if image_ids:
                first_image = image_ids[0]
                first_embeddings = all_embeddings[first_image]
                
                if 'vision_embeddings' in first_embeddings:
                    vision_shape = len(first_embeddings['vision_embeddings'])
                    print(f"      • Vision embeddings: {vision_shape} dimensions")
                
                if 'pre_generation' in first_embeddings:
                    layers = list(first_embeddings['pre_generation'].keys())
                    print(f"      • Pre-generation layers: {len(layers)} ({layers})")
                    
                    if layers:
                        first_layer = first_embeddings['pre_generation'][layers[0]]
                        if 'after_image_embeddings' in first_layer:
                            after_img_shape = len(first_layer['after_image_embeddings'])
                            print(f"        - After image embeddings: {after_img_shape} dimensions")
                        if 'end_query_embeddings' in first_layer:
                            end_query_shape = len(first_layer['end_query_embeddings'])
                            print(f"        - End query embeddings: {end_query_shape} dimensions")
                
                if 'token_positions' in first_embeddings:
                    positions = first_embeddings['token_positions']
                    print(f"      • Token positions:")
                    for key, value in positions.items():
                        print(f"        - {key}: {value}")
            
            return output_file
            
    except Exception as e:
        print(f"❌ Error processing H5 file: {e}")
        return None

def inspect_embeddings_structure(json_file_path):
    """Quick inspection of extracted embeddings JSON file."""
    
    if not os.path.exists(json_file_path):
        print(f"❌ JSON file not found: {json_file_path}")
        return
    
    print(f"\n🔍 Inspecting embeddings structure: {json_file_path}")
    print("=" * 60)
    
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        # Show extraction info
        if 'extraction_info' in data:
            info = data['extraction_info']
            print(f"📋 Extraction Info:")
            for key, value in info.items():
                print(f"   {key}: {value}")
        
        # Show file attributes
        if 'file_attributes' in data:
            attrs = data['file_attributes']
            print(f"\n📋 File Attributes:")
            for key, value in attrs.items():
                if isinstance(value, str) and len(value) > 100:
                    print(f"   {key}: {value[:100]}...")
                else:
                    print(f"   {key}: {value}")
        
        # Show embeddings structure
        if 'embeddings' in data:
            embeddings = data['embeddings']
            print(f"\n🧠 Embeddings Structure:")
            print(f"   Total images: {len(embeddings)}")
            
            # Show first image structure
            if embeddings:
                first_image_id = list(embeddings.keys())[0]
                first_image_data = embeddings[first_image_id]
                
                print(f"   Sample image ({first_image_id}):")
                for key, value in first_image_data.items():
                    if key == 'vision_embeddings' and isinstance(value, list):
                        print(f"     • {key}: {len(value)} dimensions")
                    elif key == 'pre_generation' and isinstance(value, dict):
                        print(f"     • {key}: {len(value)} layers")
                        for layer_name, layer_data in value.items():
                            print(f"       - {layer_name}: {list(layer_data.keys())}")
                    elif key == 'token_positions' and isinstance(value, dict):
                        print(f"     • {key}: {list(value.keys())}")
                    else:
                        print(f"     • {key}: {type(value).__name__}")
        
        file_size = os.path.getsize(json_file_path) / 1024 / 1024
        print(f"\n📁 File size: {file_size:.2f} MB")
        
    except Exception as e:
        print(f"❌ Error inspecting JSON file: {e}")

def main():
    """Main function with command line interface."""
    
    if len(sys.argv) < 2:
        print("Qwen2.5-VL Embeddings Extractor")
        print("=" * 40)
        print("\nUsage:")
        print("  python h5_to_json_embeddings.py <h5_file_path>")
        print("\nExample:")
        print("  python h5_to_json_embeddings.py qwen2_5_vl_targeted_embeddings_part1.h5")
        print("\nThis will create:")
        print("  qwen2_5_vl_targeted_embeddings_part1_embeddings.json")
        return
    
    h5_file_path = sys.argv[1]
    
    # Extract embeddings
    output_json = extract_qwen_embeddings_to_json(h5_file_path)
    
    # Inspect the results
    if output_json:
        inspect_embeddings_structure(output_json)
        print(f"\n🎉 Extraction completed successfully!")
        print(f"📁 Output file: {output_json}")

if __name__ == "__main__":
    main()