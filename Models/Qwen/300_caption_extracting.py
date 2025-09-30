import os
import h5py
import json
import glob
from tqdm import tqdm
from collections import defaultdict
import traceback
from datetime import datetime

# Constants
EMBEDDINGS_DIR = "/root/akhil_workspace/qwen_model/extracted_embeddings_coco2014"
OUTPUT_DIR = "/root/akhil_workspace/qwen_model/extracted_captions"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "qwen2_captions.json")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "qwen_extraction_summary.json")

def find_h5_files(directory):
    """Find all Qwen2.5-VL H5 files in the directory."""
    patterns = [
        "embeddings_part*.h5",
        "*.h5"
    ]
    
    found_files = []
    for pattern in patterns:
        files = glob.glob(os.path.join(directory, pattern))
        if files:
            found_files.extend(files)
            break  # Use first pattern that finds files
    
    # Remove duplicates and sort
    found_files = list(set(found_files))
    found_files.sort()
    
    return found_files

def inspect_h5_structure(file_path, max_images=3):
    """Inspect the structure of an H5 file to understand its format."""
    structure_info = {
        'file_path': file_path,
        'file_size_mb': round(os.path.getsize(file_path) / 1024 / 1024, 2),
        'total_images': 0,
        'sample_structures': {},
        'attributes': {},
        'errors': []
    }
    
    try:
        with h5py.File(file_path, 'r') as f:
            # Get file attributes
            for attr_name, attr_value in f.attrs.items():
                try:
                    if isinstance(attr_value, bytes):
                        structure_info['attributes'][attr_name] = attr_value.decode('utf-8')
                    else:
                        structure_info['attributes'][attr_name] = str(attr_value)
                except:
                    structure_info['attributes'][attr_name] = f"<{type(attr_value).__name__}>"
            
            # Get image IDs
            image_ids = list(f.keys())
            structure_info['total_images'] = len(image_ids)
            
            # Inspect first few images
            for i, image_id in enumerate(image_ids[:max_images]):
                try:
                    group = f[image_id]
                    
                    # Get group structure
                    group_info = {
                        'datasets': {},
                        'groups': {},
                        'sample_data': {}
                    }
                    
                    for key in group.keys():
                        if isinstance(group[key], h5py.Dataset):
                            dataset = group[key]
                            group_info['datasets'][key] = {
                                'shape': list(dataset.shape) if hasattr(dataset, 'shape') else 'scalar',
                                'dtype': str(dataset.dtype)
                            }
                            
                            # Try to read sample data for key fields
                            if key in ['image_id', 'generated_caption']:
                                try:
                                    data = dataset[()]
                                    if isinstance(data, bytes):
                                        group_info['sample_data'][key] = data.decode('utf-8')
                                    else:
                                        group_info['sample_data'][key] = str(data)
                                except:
                                    group_info['sample_data'][key] = "<read_error>"
                        
                        elif isinstance(group[key], h5py.Group):
                            subgroup = group[key]
                            group_info['groups'][key] = list(subgroup.keys())
                    
                    structure_info['sample_structures'][f"image_{i+1}_{image_id}"] = group_info
                    
                except Exception as e:
                    structure_info['errors'].append(f"Error inspecting image {image_id}: {str(e)}")
    
    except Exception as e:
        structure_info['errors'].append(f"Error opening file: {str(e)}")
    
    return structure_info

def extract_caption_from_group(group, image_id):
    """Extract caption data from a single image group with error handling."""
    result = {
        'image_id': None,
        'generated_caption': None,
        'error': None
    }
    
    try:
        # Try different possible field names and formats
        caption_fields = ['generated_caption', 'caption', 'text']
        id_fields = ['image_id', 'id']
        
        # Extract image ID
        for field in id_fields:
            if field in group:
                try:
                    data = group[field][()]
                    if isinstance(data, bytes):
                        result['image_id'] = data.decode('utf-8')
                    else:
                        result['image_id'] = str(data)
                    break
                except:
                    continue
        
        # Use group key as fallback
        if result['image_id'] is None:
            result['image_id'] = image_id
        
        # Extract caption
        for field in caption_fields:
            if field in group:
                try:
                    data = group[field][()]
                    if isinstance(data, bytes):
                        result['generated_caption'] = data.decode('utf-8')
                    else:
                        result['generated_caption'] = str(data)
                    break
                except:
                    continue
        
        # Validate results
        if result['generated_caption'] is None:
            result['error'] = "No caption found"
        
    except Exception as e:
        result['error'] = f"Extraction error: {str(e)}"
    
    return result

def extract_qwen_captions():
    """Extract captions from all Qwen2.5-VL H5 files with comprehensive error handling."""
    print("=" * 60)
    print("QWEN2.5-VL CAPTION EXTRACTOR")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Find H5 files
    print(f"🔍 Searching for H5 files in: {EMBEDDINGS_DIR}")
    h5_files = find_h5_files(EMBEDDINGS_DIR)
    
    if not h5_files:
        print(f"❌ No H5 files found in {EMBEDDINGS_DIR}")
        print("Make sure the directory exists and contains H5 files.")
        return
    
    print(f"✅ Found {len(h5_files)} H5 files:")
    for f in h5_files:
        print(f"   📁 {os.path.basename(f)}")
    
    # Inspect file structures
    print(f"\n🔍 Inspecting file structures...")
    structures = {}
    for file_path in h5_files[:2]:  # Inspect first 2 files
        print(f"   Inspecting {os.path.basename(file_path)}...")
        structures[os.path.basename(file_path)] = inspect_h5_structure(file_path)
    
    # Display structure summary
    print(f"\n📊 File Structure Summary:")
    for filename, struct in structures.items():
        print(f"   {filename}:")
        print(f"     Size: {struct['file_size_mb']} MB")
        print(f"     Images: {struct['total_images']}")
        if struct['errors']:
            print(f"     Errors: {len(struct['errors'])}")
        if struct['attributes']:
            print(f"     Attributes: {list(struct['attributes'].keys())}")
    
    # Main extraction
    print(f"\n📥 Starting caption extraction...")
    all_captions = {}
    extraction_stats = {
        'total_files': len(h5_files),
        'processed_files': 0,
        'total_images': 0,
        'successful_extractions': 0,
        'failed_extractions': 0,
        'errors_by_file': defaultdict(list),
        'processing_summary': {}
    }
    
    for file_idx, file_path in enumerate(h5_files):
        filename = os.path.basename(file_path)
        print(f"\n📂 Processing file {file_idx + 1}/{len(h5_files)}: {filename}")
        
        file_stats = {
            'images_found': 0,
            'images_processed': 0,
            'images_failed': 0,
            'errors': []
        }
        
        try:
            with h5py.File(file_path, 'r') as f:
                image_ids = list(f.keys())
                file_stats['images_found'] = len(image_ids)
                extraction_stats['total_images'] += len(image_ids)
                
                print(f"   Found {len(image_ids)} images")
                
                # Process each image with progress bar
                for image_id in tqdm(image_ids, desc=f"  {filename}", leave=False):
                    try:
                        group = f[image_id]
                        
                        # Extract caption data
                        caption_data = extract_caption_from_group(group, image_id)
                        
                        if caption_data['error']:
                            file_stats['errors'].append(f"{image_id}: {caption_data['error']}")
                            file_stats['images_failed'] += 1
                            extraction_stats['failed_extractions'] += 1
                        else:
                            # Store successful extraction
                            all_captions[caption_data['image_id']] = {
                                'generated_caption': caption_data['generated_caption'],
                                'source_file': filename
                            }
                            file_stats['images_processed'] += 1
                            extraction_stats['successful_extractions'] += 1
                    
                    except Exception as e:
                        error_msg = f"{image_id}: {str(e)}"
                        file_stats['errors'].append(error_msg)
                        file_stats['images_failed'] += 1
                        extraction_stats['failed_extractions'] += 1
                
                extraction_stats['processed_files'] += 1
                
        except Exception as e:
            error_msg = f"File error: {str(e)}"
            file_stats['errors'].append(error_msg)
            extraction_stats['errors_by_file'][filename].append(error_msg)
            print(f"   ❌ Error processing file: {error_msg}")
        
        # Store file statistics
        extraction_stats['processing_summary'][filename] = file_stats
        extraction_stats['errors_by_file'][filename].extend(file_stats['errors'])
        
        print(f"   ✅ Processed: {file_stats['images_processed']}")
        print(f"   ❌ Failed: {file_stats['images_failed']}")
    
    # Save results
    print(f"\n💾 Saving results...")
    
    # Create comprehensive output
    output_data = {
        'metadata': {
            'extraction_date': datetime.now().isoformat(),
            'source_directory': EMBEDDINGS_DIR,
            'model_type': 'Qwen2.5-VL',
            'total_files_processed': extraction_stats['processed_files'],
            'total_images_found': extraction_stats['total_images'],
            'successful_extractions': extraction_stats['successful_extractions'],
            'failed_extractions': extraction_stats['failed_extractions'],
            'success_rate': round(extraction_stats['successful_extractions'] / max(extraction_stats['total_images'], 1) * 100, 2)
        },
        'captions': all_captions
    }
    
    # Save main captions file
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Save detailed extraction summary
    summary_data = {
        'extraction_stats': extraction_stats,
        'file_structures': structures
    }
    
    with open(SUMMARY_PATH, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    # Final report
    print(f"\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"📊 Results Summary:")
    print(f"   Total files processed: {extraction_stats['processed_files']}/{extraction_stats['total_files']}")
    print(f"   Total images found: {extraction_stats['total_images']}")
    print(f"   Successful extractions: {extraction_stats['successful_extractions']}")
    print(f"   Failed extractions: {extraction_stats['failed_extractions']}")
    print(f"   Success rate: {output_data['metadata']['success_rate']}%")
    
    print(f"\n📁 Output files:")
    print(f"   Main captions: {OUTPUT_PATH}")
    print(f"   Extraction summary: {SUMMARY_PATH}")
    
    # Show sample captions
    if all_captions:
        print(f"\n📝 Sample captions:")
        sample_keys = list(all_captions.keys())[:3]
        for key in sample_keys:
            caption_info = all_captions[key]
            print(f"   {key}: '{caption_info['generated_caption']}'")
            print(f"      (source: {caption_info['source_file']})")
    
    # Error summary
    if extraction_stats['failed_extractions'] > 0:
        print(f"\n⚠️  Errors occurred in {extraction_stats['failed_extractions']} extractions")
        print(f"   Check {SUMMARY_PATH} for detailed error information")
    
    return output_data

def verify_extraction():
    """Verify the extracted captions and provide statistics."""
    if not os.path.exists(OUTPUT_PATH):
        print(f"❌ Output file not found: {OUTPUT_PATH}")
        return False
    
    print(f"\n🔍 Verifying extraction...")
    
    try:
        with open(OUTPUT_PATH, 'r') as f:
            data = json.load(f)
        
        metadata = data.get('metadata', {})
        captions = data.get('captions', {})
        
        print(f"✅ Verification successful:")
        print(f"   Total captions: {len(captions)}")
        print(f"   Success rate: {metadata.get('success_rate', 'unknown')}%")
        print(f"   Source files: {metadata.get('total_files_processed', 'unknown')}")
        print(f"   Model type: {metadata.get('model_type', 'unknown')}")
        
        if captions:
            # Analyze caption statistics
            caption_lengths = [len(cap['generated_caption']) for cap in captions.values() if cap.get('generated_caption')]
            
            if caption_lengths:
                print(f"   Caption length (chars): {min(caption_lengths)}-{max(caption_lengths)} (avg: {sum(caption_lengths)/len(caption_lengths):.1f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

def extract_specific_images(image_list, output_file=None):
    """Extract captions for specific images only."""
    if output_file is None:
        output_file = OUTPUT_PATH.replace('.json', '_filtered.json')
    
    print(f"🎯 Extracting captions for {len(image_list)} specific images...")
    
    h5_files = find_h5_files(EMBEDDINGS_DIR)
    found_captions = {}
    
    for file_path in h5_files:
        filename = os.path.basename(file_path)
        print(f"   Searching in {filename}...")
        
        try:
            with h5py.File(file_path, 'r') as f:
                for target_image in image_list:
                    if target_image in f:
                        group = f[target_image]
                        caption_data = extract_caption_from_group(group, target_image)
                        
                        if not caption_data['error']:
                            found_captions[caption_data['image_id']] = {
                                'generated_caption': caption_data['generated_caption'],
                                'source_file': filename
                            }
                            print(f"     ✅ Found: {target_image}")
        
        except Exception as e:
            print(f"     ❌ Error reading {filename}: {e}")
    
    # Save filtered results
    with open(output_file, 'w') as f:
        json.dump(found_captions, f, indent=2)
    
    print(f"💾 Saved {len(found_captions)} captions to {output_file}")
    return found_captions

def main():
    """Main function to extract Qwen2.5-VL captions."""
    try:
        print("🚀 Starting Qwen2.5-VL Caption Extraction Pipeline")
        print("=" * 60)
        
        # Main extraction
        result = extract_qwen_captions()
        
        # Verify results
        if result:
            verify_extraction()
        
        print(f"\n✨ Caption extraction pipeline completed successfully!")
        print(f"📁 Captions saved to: {OUTPUT_PATH}")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Extraction interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()