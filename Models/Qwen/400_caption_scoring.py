import sys, os
import traceback
import json
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from collections import defaultdict, Counter
import requests
import zipfile
import re
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch.nn.functional as F

print("Starting Qwen2.5-VL Caption Scoring Pipeline...")

# Download required NLTK data
print("Downloading NLTK data...")
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Paths configuration
BASE_DIR = "/root/akhil_workspace/qwen_model"
CAPTIONS_DIR = os.path.join(BASE_DIR, "extracted_captions")
ANALYSIS_OUTPUT_DIR = os.path.join(BASE_DIR, "factual_analysis")
COCO_DATA_DIR = os.path.join(BASE_DIR, "coco_annotations")
EVALUATION_OUTPUT_DIR = os.path.join(BASE_DIR, "evaluation_results")

# File paths
QWEN_CAPTIONS_PATH = "/root/akhil_workspace/qwen_model/extracted_captions/qwen2_captions.json"
COCO_ANNOTATIONS_PATH = os.path.join(COCO_DATA_DIR, "captions_val2014.json")
OUTPUT_CSV_PATH = os.path.join(EVALUATION_OUTPUT_DIR, "qwen2_comprehensive_scores.csv")
SUMMARY_PATH = os.path.join(EVALUATION_OUTPUT_DIR, "qwen2_scoring_summary.json")

# Create output directories
os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)
os.makedirs(EVALUATION_OUTPUT_DIR, exist_ok=True)
os.makedirs(COCO_DATA_DIR, exist_ok=True)

# Initialize BERT model for factual accuracy
print("Loading BERT model...")
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
bert_model.eval()

# CHAIR Metric Configuration - COCO 80 object categories
MSCOCO_OBJECTS_80 = sorted([
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
])
MSCOCO_OBJECT_SET = set(MSCOCO_OBJECTS_80)

def download_coco_annotations():
    """Download COCO 2014 validation annotations if not present."""
    if os.path.exists(COCO_ANNOTATIONS_PATH):
        print("✅ COCO annotations already exist")
        return True
    
    print("📥 Downloading COCO 2014 validation annotations...")
    
    # COCO 2014 annotations URL
    annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
    zip_path = os.path.join(COCO_DATA_DIR, "annotations_trainval2014.zip")
    
    try:
        # Download annotations
        print("   Downloading annotations zip file...")
        response = requests.get(annotations_url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        with open(zip_path, 'wb') as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit='B',
            unit_scale=True
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        # Extract annotations
        print("   Extracting annotations...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(COCO_DATA_DIR)
        
        # Move the captions file to expected location
        extracted_captions = os.path.join(COCO_DATA_DIR, "annotations", "captions_val2014.json")
        if os.path.exists(extracted_captions):
            os.rename(extracted_captions, COCO_ANNOTATIONS_PATH)
            print("✅ COCO annotations downloaded and extracted successfully")
        
        # Clean up
        os.remove(zip_path)
        return True
        
    except Exception as e:
        print(f"❌ Error downloading COCO annotations: {e}")
        return False

def normalize_image_id(image_id):
    """Normalize image ID by removing the prefix and keeping only the numeric part."""
    if '_' in str(image_id):
        return str(image_id).split('_')[-1]
    return str(image_id)

def load_coco_ground_truth():
    """Load COCO ground truth captions."""
    print("📚 Loading COCO ground truth captions...")
    
    if not os.path.exists(COCO_ANNOTATIONS_PATH):
        print("COCO annotations not found. Attempting to download...")
        if not download_coco_annotations():
            raise FileNotFoundError(f"Could not download COCO annotations")
    
    try:
        with open(COCO_ANNOTATIONS_PATH, 'r') as f:
            coco_data = json.load(f)
        
        # Process COCO annotations
        image_captions = defaultdict(list)
        
        for annotation in coco_data['annotations']:
            image_id = str(annotation['image_id']).zfill(12)
            caption = annotation['caption'].strip()
            
            # Store with normalized ID
            normalized_id = normalize_image_id(image_id)
            image_captions[normalized_id].append(caption)
        
        print(f"   Loaded {len(image_captions)} images with ground truth captions")
        print(f"   Average captions per image: {np.mean([len(caps) for caps in image_captions.values()]):.1f}")
        print(f"   Sample normalized IDs: {list(image_captions.keys())[:5]}")
        
        return dict(image_captions)
        
    except Exception as e:
        print(f"❌ Error loading COCO ground truth: {e}")
        raise

def load_qwen_captions():
    """Load Qwen2.5-VL generated captions."""
    print("🤖 Loading Qwen2.5-VL generated captions...")
    
    if not os.path.exists(QWEN_CAPTIONS_PATH):
        raise FileNotFoundError(f"Qwen captions not found: {QWEN_CAPTIONS_PATH}")
    
    try:
        with open(QWEN_CAPTIONS_PATH, 'r') as f:
            qwen_data = json.load(f)
        
        # Extract captions from our format
        if 'captions' in qwen_data:
            captions_dict = qwen_data['captions']
        else:
            captions_dict = qwen_data
        
        # Convert to simple format with normalized IDs
        qwen_captions = {}
        for image_id, caption_data in captions_dict.items():
            # Normalize the image ID
            normalized_id = normalize_image_id(image_id)
            
            if isinstance(caption_data, dict) and 'generated_caption' in caption_data:
                qwen_captions[normalized_id] = caption_data['generated_caption']
            elif isinstance(caption_data, str):
                qwen_captions[normalized_id] = caption_data
            else:
                print(f"Warning: Unexpected format for image {image_id}")
        
        print(f"   Loaded {len(qwen_captions)} Qwen2.5-VL generated captions")
        print(f"   Sample normalized IDs: {list(qwen_captions.keys())[:5]}")
        
        return qwen_captions
        
    except Exception as e:
        print(f"❌ Error loading Qwen captions: {e}")
        raise

def cosine_similarity_torch(tensor1, tensor2):
    """Compute cosine similarity using PyTorch."""
    tensor1_norm = F.normalize(tensor1.unsqueeze(0), p=2, dim=1)
    tensor2_norm = F.normalize(tensor2.unsqueeze(0), p=2, dim=1)
    similarity = torch.mm(tensor1_norm, tensor2_norm.t())
    return similarity.item()

def get_bert_embedding(text):
    """Get BERT embedding for a text."""
    try:
        with torch.no_grad():
            text = str(text).strip()
            if not text:
                return None
                
            inputs = bert_tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to(device)
            
            outputs = bert_model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]
            return embeddings.squeeze(0)
            
    except Exception as e:
        print(f"Error getting BERT embedding for text '{text[:50]}...': {e}")
        return None

def compute_factual_score(generated_caption, ground_truth_captions):
    """Compute factual score using semantic similarity with ground truth captions."""
    try:
        if not generated_caption or not ground_truth_captions:
            return 0.0
            
        gen_embedding = get_bert_embedding(generated_caption)
        if gen_embedding is None:
            return 0.0
        
        similarities = []
        for gt_caption in ground_truth_captions:
            gt_embedding = get_bert_embedding(gt_caption)
            if gt_embedding is not None:
                similarity = cosine_similarity_torch(gen_embedding, gt_embedding)
                similarities.append(float(similarity))
        
        return float(max(similarities)) if similarities else 0.0
        
    except Exception as e:
        print(f"Error computing factual score: {e}")
        return 0.0

def extract_mscoco_objects_from_caption(caption_text):
    """Extract MSCOCO objects mentioned in a caption using the COCO object list."""
    words = re.findall(r'\b\w+\b', caption_text.lower())
    mentioned_objects = set()
    
    for word in words:
        if word in MSCOCO_OBJECT_SET:
            mentioned_objects.add(word)
    
    return mentioned_objects

def calculate_chair_metrics_individual(generated_caption, ground_truth_captions):
    """Calculate CHAIR metrics for a single image."""
    # Extract ground truth objects
    gt_objects = set()
    for caption in ground_truth_captions:
        gt_objects.update(extract_mscoco_objects_from_caption(caption))
    
    # Extract mentioned objects from generated caption
    mentioned_objects = extract_mscoco_objects_from_caption(generated_caption)
    
    # Find hallucinated objects (mentioned but not in ground truth)
    hallucinated_objects = mentioned_objects - gt_objects
    
    return {
        'mentioned_objects': list(mentioned_objects),
        'hallucinated_objects': list(hallucinated_objects),
        'ground_truth_objects': list(gt_objects),
        'has_hallucination': len(hallucinated_objects) > 0
    }

def calculate_standard_metrics_individual(generated_caption, ground_truth_captions):
    """Calculate standard captioning metrics for a single image."""
    smoothing = SmoothingFunction()
    
    # Tokenize
    gen_tokens = generated_caption.lower().split()
    gt_tokens_list = [gt.lower().split() for gt in ground_truth_captions]
    
    # BLEU scores
    try:
        bleu_1 = sentence_bleu(gt_tokens_list, gen_tokens, 
                             weights=(1.0, 0, 0, 0),
                             smoothing_function=smoothing.method1)
    except:
        bleu_1 = 0.0
    
    try:
        bleu_4 = sentence_bleu(gt_tokens_list, gen_tokens, 
                             weights=(0.25, 0.25, 0.25, 0.25),
                             smoothing_function=smoothing.method1)
    except:
        bleu_4 = 0.0
    
    # Simple ROUGE-L implementation
    def simple_rouge_l(candidate, reference):
        def lcs_length(x, y):
            m, n = len(x), len(y)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if x[i-1] == y[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            
            return dp[m][n]
        
        candidate_words = candidate.split()
        reference_words = reference.split()
        
        if not candidate_words or not reference_words:
            return 0.0
        
        lcs_len = lcs_length(candidate_words, reference_words)
        
        if lcs_len == 0:
            return 0.0
        
        precision = lcs_len / len(candidate_words)
        recall = lcs_len / len(reference_words)
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    
    # ROUGE-L score
    rouge_vals = []
    for gt_caption in ground_truth_captions:
        rouge = simple_rouge_l(generated_caption.lower(), gt_caption.lower())
        rouge_vals.append(rouge)
    rouge_l = max(rouge_vals) if rouge_vals else 0.0
    
    return {
        'BLEU-1': bleu_1,
        'BLEU-4': bleu_4,
        'ROUGE-L': rouge_l
    }

def analyze_captions():
    """Main analysis function."""
    print("🔬 Starting comprehensive caption analysis...")
    
    # Load data
    ground_truth_captions = load_coco_ground_truth()
    qwen_captions = load_qwen_captions()
    
    # Find common images using normalized IDs
    common_images = set(ground_truth_captions.keys()) & set(qwen_captions.keys())
    print(f"📊 Found {len(common_images)} images with both Qwen and ground truth captions")
    
    if len(common_images) == 0:
        print("❌ No common images found even after normalization! Check image ID formats.")
        print(f"Sample Qwen normalized IDs: {list(qwen_captions.keys())[:5]}")
        print(f"Sample GT normalized IDs: {list(ground_truth_captions.keys())[:5]}")
        return
    
    # Analyze each image
    results = []
    
    print("🧮 Computing comprehensive scores...")
    for image_id in tqdm(sorted(common_images), desc="Analyzing images"):
        try:
            generated_caption = qwen_captions[image_id]
            gt_captions = ground_truth_captions[image_id]
            
            # Compute factual score
            factual_score = compute_factual_score(generated_caption, gt_captions)
            
            # Compute CHAIR metrics
            chair_metrics = calculate_chair_metrics_individual(generated_caption, gt_captions)
            
            # Compute standard metrics
            standard_metrics = calculate_standard_metrics_individual(generated_caption, gt_captions)
            
            # Create comprehensive result
            result = {
                'image_id': image_id,
                'generated_caption': generated_caption,
                'ground_truth_captions': '; '.join(gt_captions),
                'factual_score': factual_score,
                'has_hallucination': chair_metrics['has_hallucination'],
                'mentioned_objects': '; '.join(chair_metrics['mentioned_objects']),
                'hallucinated_objects': '; '.join(chair_metrics['hallucinated_objects']),
                'ground_truth_objects': '; '.join(chair_metrics['ground_truth_objects']),
                'BLEU-1': standard_metrics['BLEU-1'],
                'BLEU-4': standard_metrics['BLEU-4'],
                'ROUGE-L': standard_metrics['ROUGE-L']
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"Error analyzing image {image_id}: {e}")
            continue
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Add the requested additional columns
    print("📊 Adding additional computed columns...")
    
    df["hallucinated_objects_count"] = df["hallucinated_objects"].apply(
        lambda x: len(x.split(";")) if pd.notna(x) and x.strip() else 0
    )
    df["mentioned_objects_count"] = df["mentioned_objects"].apply(
        lambda x: len(x.split(";")) if pd.notna(x) and x.strip() else 0
    )
    df["ground_truth_objects_count"] = df["ground_truth_objects"].apply(
        lambda x: len(x.split(";")) if pd.notna(x) and x.strip() else 0
    )
    
    df["chair_regression_score"] = np.where(
        df["mentioned_objects_count"] == 0, 
        0, 
        df["hallucinated_objects_count"] / df["mentioned_objects_count"]
    )
    
    # Compute overall statistics
    stats = {
        'total_images': len(df),
        'average_factual_score': float(df['factual_score'].mean()),
        'hallucination_rate': float(df['has_hallucination'].mean()),
        'average_chair_regression_score': float(df['chair_regression_score'].mean()),
        'average_bleu_1': float(df['BLEU-1'].mean()),
        'average_bleu_4': float(df['BLEU-4'].mean()),
        'average_rouge_l': float(df['ROUGE-L'].mean()),
        'average_mentioned_objects': float(df['mentioned_objects_count'].mean()),
        'average_hallucinated_objects': float(df['hallucinated_objects_count'].mean()),
        'average_ground_truth_objects': float(df['ground_truth_objects_count'].mean())
    }
    
    # Save CSV file
    print(f"💾 Saving comprehensive scores to CSV...")
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    
    # Save summary statistics
    summary_data = {
        'analysis_date': pd.Timestamp.now().isoformat(),
        'model_type': 'Qwen2.5-VL',
        'total_images_analyzed': len(df),
        'statistics': stats,
        'column_descriptions': {
            'factual_score': 'BERT-based semantic similarity with ground truth (0-1)',
            'has_hallucination': 'Boolean indicating presence of hallucinated objects',
            'chair_regression_score': 'Ratio of hallucinated to mentioned objects (0-1)',
            'mentioned_objects_count': 'Number of COCO objects mentioned in caption',
            'hallucinated_objects_count': 'Number of COCO objects mentioned but not in ground truth',
            'ground_truth_objects_count': 'Number of COCO objects in ground truth captions',
            'BLEU-1': 'BLEU-1 score (0-1)',
            'BLEU-4': 'BLEU-4 score (0-1)',
            'ROUGE-L': 'ROUGE-L F1 score (0-1)'
        }
    }
    
    with open(SUMMARY_PATH, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    return df, stats

def print_analysis_summary(df, stats):
    """Print a comprehensive summary of the analysis."""
    
    print("\n" + "=" * 60)
    print("📊 QWEN2.5-VL COMPREHENSIVE ANALYSIS SUMMARY")
    print("=" * 60)
    
    print(f"🔢 Total Images Analyzed: {stats['total_images']}")
    print(f"📈 Average Factual Score: {stats['average_factual_score']:.4f}")
    print(f"🚨 Hallucination Rate: {stats['hallucination_rate']:.4f} ({stats['hallucination_rate']*100:.1f}%)")
    print(f"📊 Average CHAIR Regression Score: {stats['average_chair_regression_score']:.4f}")
    
    print(f"\n📝 Standard Captioning Metrics:")
    print(f"   BLEU-1: {stats['average_bleu_1']:.4f}")
    print(f"   BLEU-4: {stats['average_bleu_4']:.4f}")
    print(f"   ROUGE-L: {stats['average_rouge_l']:.4f}")
    
    print(f"\n🔍 Object Detection Analysis:")
    print(f"   Average Mentioned Objects: {stats['average_mentioned_objects']:.2f}")
    print(f"   Average Hallucinated Objects: {stats['average_hallucinated_objects']:.2f}")
    print(f"   Average Ground Truth Objects: {stats['average_ground_truth_objects']:.2f}")
    
    # Show best and worst examples
    print(f"\n🏆 Best Examples (Highest Factual Score):")
    best_examples = df.nlargest(3, 'factual_score')
    for i, (_, row) in enumerate(best_examples.iterrows()):
        print(f"   {i+1}. Factual Score: {row['factual_score']:.4f}, CHAIR: {row['chair_regression_score']:.4f}")
        print(f"      Generated: '{row['generated_caption'][:100]}...'")
        print(f"      Hallucinated Objects: {row['hallucinated_objects_count']}")
    
    print(f"\n🔍 Worst Examples (Lowest Factual Score):")
    worst_examples = df.nsmallest(3, 'factual_score')
    for i, (_, row) in enumerate(worst_examples.iterrows()):
        print(f"   {i+1}. Factual Score: {row['factual_score']:.4f}, CHAIR: {row['chair_regression_score']:.4f}")
        print(f"      Generated: '{row['generated_caption'][:100]}...'")
        print(f"      Hallucinated Objects: {row['hallucinated_objects_count']}")

def main():
    """Main function."""
    try:
        print("🚀 Starting Qwen2.5-VL Comprehensive Scoring Pipeline")
        print("=" * 60)
        
        # Check if Qwen captions exist
        if not os.path.exists(QWEN_CAPTIONS_PATH):
            print(f"❌ Qwen captions file not found: {QWEN_CAPTIONS_PATH}")
            print("   Please run the caption extraction script first!")
            return
        
        # Run analysis
        df, stats = analyze_captions()
        
        if df is not None and len(df) > 0:
            # Print summary
            print_analysis_summary(df, stats)
            
            print(f"\n📁 Output Files:")
            print(f"   Comprehensive CSV: {OUTPUT_CSV_PATH}")
            print(f"   Summary Statistics: {SUMMARY_PATH}")
            
            print(f"\n📊 CSV File Contains {len(df)} rows with columns:")
            for col in df.columns:
                print(f"   • {col}")
            
            print(f"\n✨ Analysis completed successfully!")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Error in main: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()