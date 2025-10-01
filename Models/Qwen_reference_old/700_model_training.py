import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import json
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import itertools
from datetime import datetime
import logging
import sys
import ast
import glob
import warnings
warnings.filterwarnings('ignore')

# ==========================
# CONFIGURATION
# ==========================

CONFIG = {
    # Paths
    "DATASETS_DIR": "/root/akhil_workspace/blip_model/comprehensive_datasets",
    "MODELS_INFO_DIR": "/root/akhil_workspace/blip_model/models_info",
    "DATASET_SPLITS_DIR": "/root/akhil_workspace/blip_model/models_info/dataset_splits",
    
    # Binary target definitions (1 = hallucination, 0 = no hallucination)
    "BINARY_TARGETS": {
        "has_hallucinations": {
            "source": "chair_hallucinated_objects_count", 
            "rule": "> 0",
            "description": "Direct CHAIR-based hallucination detection"
        },
        "factual_score_hallucination": {
            "source": "factual_score",
            "threshold": 0.7,
            "rule": "<= threshold",
            "description": "Low factual accuracy indicates hallucination"
        },
        "bleu1_hallucination": {
            "source": "BLEU-1", 
            "threshold": 0.2,
            "rule": "<= threshold",
            "description": "Low BLEU-1 indicates poor linguistic quality"
        },
        "bleu4_hallucination": {
            "source": "BLEU-4",
            "threshold": 0.2,
            "rule": "<= threshold",
            "description": "Low BLEU-4 indicates poor linguistic quality"
        },
        "meteor_hallucination": {
            "source": "METEOR",
            "threshold": 0.3,
            "rule": "<= threshold",
            "description": "Low METEOR indicates poor semantic similarity"
        },
        "rouge_l_hallucination": {
            "source": "ROUGE-L",
            "threshold": 0.3,
            "rule": "<= threshold",
            "description": "Low ROUGE-L indicates poor sequence similarity"
        }
    },
    
    # Model hyperparameters
    "HYPERPARAMS": {
        "layer_sizes": [
            [1024, 516, 256],
        ],
        "learning_rates": [.01],
        "batch_sizes": [28],
        "dropout_rates": [0.1]
    },
    
    # Training parameters
    "EPOCHS": 50,
    "EARLY_STOPPING_PATIENCE": 10,
    "MIN_DELTA": 0.001,
    "LR_PATIENCE": 5,
    "LR_FACTOR": 0.5,
    
    # Data splits
    "TRAIN_SIZE": 0.8,
    "TEST_SIZE": 0.1, 
    "VAL_SIZE": 0.1,
    "RANDOM_STATE": 42,
    
    # Device
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu"
}

# ==========================
# EMBEDDING CACHE
# ==========================

class EmbeddingCache:
    """Cache for parsed embeddings to avoid redundant loading."""
    
    def __init__(self):
        self.cache = {}
        self.metadata_cache = {}
    
    def get_embeddings(self, dataset_path):
        """Load embeddings with caching."""
        if dataset_path in self.cache:
            print(f"   📦 Using cached embeddings for {os.path.basename(dataset_path)}")
            return self.cache[dataset_path], self.metadata_cache[dataset_path]
        
        print(f"   📥 Loading and parsing embeddings for {os.path.basename(dataset_path)}")
        df, embeddings = load_dataset(dataset_path)
        
        # Cache the results
        self.cache[dataset_path] = embeddings
        self.metadata_cache[dataset_path] = df
        
        return embeddings, df
    
    def clear(self):
        """Clear cache to free memory."""
        self.cache.clear()
        self.metadata_cache.clear()
        print("   🧹 Cache cleared")

# Global cache instance
embedding_cache = EmbeddingCache()

# ==========================
# UTILITY FUNCTIONS
# ==========================

def setup_directories():
    """Create necessary directories."""
    directories = [
        CONFIG["MODELS_INFO_DIR"],
        CONFIG["DATASET_SPLITS_DIR"]
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def setup_logging(embedding_type, target_name):
    """Setup logging for specific embedding-target combination."""
    log_dir = os.path.join(CONFIG["MODELS_INFO_DIR"], embedding_type, target_name)
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'training_log_{timestamp}.txt')
    
    # Create a new logger for this specific combination
    logger = logging.getLogger(f"{embedding_type}_{target_name}")
    logger.handlers.clear()  # Clear any existing handlers
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger, log_file

def discover_embedding_datasets():
    """Discover vision and pre-generation datasets only (excluding post-generation)."""
    pattern = os.path.join(CONFIG["DATASETS_DIR"], "comprehensive_dataset_*.csv")
    all_files = glob.glob(pattern)
    
    print(f"🔍 Discovering datasets in: {CONFIG['DATASETS_DIR']}")
    print(f"📁 Found {len(all_files)} total comprehensive datasets")
    
    # Filter for vision and pre-generation only
    valid_datasets = {}
    excluded_datasets = []
    
    for file_path in all_files:
        filename = os.path.basename(file_path)
        dataset_name = filename.replace("comprehensive_dataset_", "").replace(".csv", "")
        
        # Include only vision and pre_generation datasets
        if dataset_name == "vision_embeddings" or dataset_name.startswith("pre_generation"):
            valid_datasets[dataset_name] = file_path
            print(f"   ✅ Including: {dataset_name}")
        else:
            excluded_datasets.append(dataset_name)
            print(f"   ❌ Excluding: {dataset_name}")
    
    if excluded_datasets:
        print(f"\n📋 Excluded {len(excluded_datasets)} datasets:")
        for name in excluded_datasets[:5]:  # Show first 5
            print(f"   • {name}")
        if len(excluded_datasets) > 5:
            print(f"   ... and {len(excluded_datasets) - 5} more")
    
    print(f"\n🎯 Selected {len(valid_datasets)} datasets for training:")
    for name in valid_datasets.keys():
        print(f"   • {name}")
    
    return valid_datasets

def load_dataset(dataset_path):
    """Load and preprocess a comprehensive dataset."""
    print(f"     Loading dataset: {os.path.basename(dataset_path)}")
    
    df = pd.read_csv(dataset_path)
    print(f"     Initial dataset shape: {df.shape}")
    
    # Parse embeddings from string representation
    embeddings = []
    valid_indices = []
    
    for idx, emb_str in enumerate(tqdm(df['corresponding_embeddings'], 
                                      desc="     Parsing embeddings", 
                                      leave=False)):
        try:
            embedding = ast.literal_eval(emb_str)
            embeddings.append(np.array(embedding, dtype=np.float32))
            valid_indices.append(idx)
        except Exception as e:
            continue
    
    # Filter dataframe to only valid indices
    df_filtered = df.iloc[valid_indices].reset_index(drop=True)
    embeddings = np.stack(embeddings)
    
    print(f"     Valid samples after filtering: {len(df_filtered)}")
    print(f"     Embedding shape: {embeddings.shape}")
    
    return df_filtered, embeddings

def create_binary_targets(df):
    """Create binary hallucination detection targets."""
    targets = {}
    target_stats = {}
    
    print(f"   🎯 Creating binary targets from {len(df)} samples:")
    
    for target_name, config in CONFIG["BINARY_TARGETS"].items():
        source_col = config["source"]
        
        if source_col not in df.columns:
            print(f"      ⚠️  Warning: Source column '{source_col}' not found in dataset")
            continue
        
        source_data = df[source_col]
        
        # Apply the rule to create binary target
        if config["rule"] == "> 0":
            binary_target = (source_data > 0).astype(int)
        elif config["rule"] == "<= threshold":
            threshold = config["threshold"]
            binary_target = (source_data <= threshold).astype(int)
        else:
            print(f"      ⚠️  Warning: Unknown rule '{config['rule']}' for target {target_name}")
            continue
        
        targets[target_name] = binary_target.values
        
        # Calculate detailed statistics
        pos_count = int(binary_target.sum())
        neg_count = int(len(binary_target) - pos_count)
        pos_ratio = pos_count / len(binary_target)
        
        target_stats[target_name] = {
            "positive_samples": pos_count,
            "negative_samples": neg_count,
            "positive_ratio": pos_ratio,
            "description": config["description"]
        }
        
        # Print target statistics
        print(f"      📊 {target_name}:")
        print(f"         Rule: {source_col} {config['rule'].replace('threshold', str(config.get('threshold', '')))} ")
        print(f"         Class 0 (No Hallucination): {neg_count:,} ({(1-pos_ratio)*100:.1f}%)")
        print(f"         Class 1 (Hallucination):    {pos_count:,} ({pos_ratio*100:.1f}%)")
        
        # Warn about severe class imbalance
        if pos_ratio < 0.05 or pos_ratio > 0.95:
            print(f"         ⚠️  SEVERE CLASS IMBALANCE detected!")
    
    return targets, target_stats

def create_dataset_splits(dataset_name, df):
    """Create and save train/test/val splits for a dataset."""
    splits_file = os.path.join(CONFIG["DATASET_SPLITS_DIR"], f"{dataset_name}_splits.json")
    
    # Check if splits already exist
    if os.path.exists(splits_file):
        print(f"   📂 Loading existing splits for {dataset_name}")
        with open(splits_file, 'r') as f:
            splits_data = json.load(f)
        
        # Print existing split statistics
        print(f"      📊 Split Statistics:")
        print(f"         Total samples: {splits_data['total_samples']}")
        print(f"         Train: {splits_data['split_sizes']['train']} ({splits_data['split_sizes']['train']/splits_data['total_samples']*100:.1f}%)")
        print(f"         Test:  {splits_data['split_sizes']['test']} ({splits_data['split_sizes']['test']/splits_data['total_samples']*100:.1f}%)")
        print(f"         Val:   {splits_data['split_sizes']['val']} ({splits_data['split_sizes']['val']/splits_data['total_samples']*100:.1f}%)")
        
        return splits_data["splits"]
    
    print(f"   📂 Creating new splits for {dataset_name}")
    
    # Create splits
    train_df, temp_df = train_test_split(
        df, test_size=(CONFIG["TEST_SIZE"] + CONFIG["VAL_SIZE"]), 
        random_state=CONFIG["RANDOM_STATE"], stratify=None
    )
    
    test_df, val_df = train_test_split(
        temp_df, test_size=CONFIG["VAL_SIZE"]/(CONFIG["TEST_SIZE"] + CONFIG["VAL_SIZE"]), 
        random_state=CONFIG["RANDOM_STATE"]
    )
    
    splits = {
        "train": train_df['image_id'].tolist(),
        "test": test_df['image_id'].tolist(),
        "val": val_df['image_id'].tolist()
    }
    
    # Save splits with metadata
    splits_data = {
        "dataset_name": dataset_name,
        "split_date": datetime.now().isoformat(),
        "total_samples": len(df),
        "split_ratios": {
            "train": CONFIG["TRAIN_SIZE"],
            "test": CONFIG["TEST_SIZE"], 
            "val": CONFIG["VAL_SIZE"]
        },
        "splits": splits,
        "split_sizes": {
            "train": len(splits["train"]),
            "test": len(splits["test"]),
            "val": len(splits["val"])
        }
    }
    
    with open(splits_file, 'w') as f:
        json.dump(splits_data, f, indent=2)
    
    print(f"      📊 New Split Statistics:")
    print(f"         Total samples: {len(df)}")
    print(f"         Train: {len(splits['train'])} ({len(splits['train'])/len(df)*100:.1f}%)")
    print(f"         Test:  {len(splits['test'])} ({len(splits['test'])/len(df)*100:.1f}%)")
    print(f"         Val:   {len(splits['val'])} ({len(splits['val'])/len(df)*100:.1f}%)")
    
    return splits

def get_split_data(df, embeddings, targets, splits, split_name):
    """Get data for a specific split."""
    split_image_ids = set(splits[split_name])
    
    # Find indices of images in this split
    split_indices = []
    for idx, image_id in enumerate(df['image_id']):
        if image_id in split_image_ids:
            split_indices.append(idx)
    
    split_embeddings = embeddings[split_indices]
    split_targets = {name: target_data[split_indices] for name, target_data in targets.items()}
    
    return split_embeddings, split_targets

def print_split_statistics(X_train, X_test, y_train, y_test, target_name, logger):
    """Print detailed statistics for train/test splits."""
    
    logger.info("📊 SPLIT STATISTICS:")
    logger.info("=" * 50)
    
    # Overall split sizes
    total_samples = len(X_train) + len(X_test)
    train_pct = len(X_train) / total_samples * 100
    test_pct = len(X_test) / total_samples * 100
    
    logger.info(f"📋 Split Sizes:")
    logger.info(f"   Total samples: {total_samples:,}")
    logger.info(f"   Train: {len(X_train):,} ({train_pct:.1f}%)")
    logger.info(f"   Test:  {len(X_test):,} ({test_pct:.1f}%)")
    
    # Class distribution in each split
    train_pos = int(y_train.sum())
    train_neg = len(y_train) - train_pos
    train_pos_pct = train_pos / len(y_train) * 100
    
    test_pos = int(y_test.sum())
    test_neg = len(y_test) - test_pos
    test_pos_pct = test_pos / len(y_test) * 100
    
    logger.info(f"📊 Class Distribution for '{target_name}':")
    logger.info(f"   TRAIN SET:")
    logger.info(f"      Class 0 (No Hallucination): {train_neg:,} ({100-train_pos_pct:.1f}%)")
    logger.info(f"      Class 1 (Hallucination):    {train_pos:,} ({train_pos_pct:.1f}%)")
    logger.info(f"      Positive ratio: {train_pos_pct/100:.3f}")
    
    logger.info(f"   TEST SET:")
    logger.info(f"      Class 0 (No Hallucination): {test_neg:,} ({100-test_pos_pct:.1f}%)")
    logger.info(f"      Class 1 (Hallucination):    {test_pos:,} ({test_pos_pct:.1f}%)")
    logger.info(f"      Positive ratio: {test_pos_pct/100:.3f}")
    
    # Check for class imbalance warnings
    if train_pos_pct < 5 or train_pos_pct > 95:
        logger.warning("⚠️  SEVERE CLASS IMBALANCE in training set!")
    if test_pos_pct < 5 or test_pos_pct > 95:
        logger.warning("⚠️  SEVERE CLASS IMBALANCE in test set!")
    
    # Check for distribution shift between train and test
    ratio_diff = abs(train_pos_pct - test_pos_pct)
    if ratio_diff > 10:
        logger.warning(f"⚠️  DISTRIBUTION SHIFT detected! Train vs Test positive ratio differs by {ratio_diff:.1f}%")
    
    logger.info("=" * 50)

# ==========================
# MODEL DEFINITION
# ==========================

class BinaryClassifier(nn.Module):
    """Binary classifier for hallucination detection."""
    
    def __init__(self, input_dim, layer_sizes, dropout_rate=0.1):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for size in layer_sizes:
            layers.append(nn.Linear(prev_dim, size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = size
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x).squeeze()

# ==========================
# TRAINING FUNCTIONS
# ==========================

def train_model(X_train, y_train, X_test, y_test, hyperparams, input_dim, logger):
    """Train a binary classifier."""
    
    # Convert data to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    # Create data loader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_sizes'], shuffle=True)
    
    # Initialize model
    model = BinaryClassifier(
        input_dim=input_dim,
        layer_sizes=hyperparams['layer_sizes'],
        dropout_rate=hyperparams['dropout_rates']
    ).to(CONFIG["DEVICE"])
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rates'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=CONFIG["LR_FACTOR"], 
                                patience=CONFIG["LR_PATIENCE"], verbose=False)
    criterion = nn.BCELoss()
    
    # Training history
    history = {
        'train_loss': [],
        'test_loss': [],
        'train_acc': [],
        'test_acc': [],
        'learning_rates': []
    }
    
    best_test_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(CONFIG["EPOCHS"]):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(CONFIG["DEVICE"]), y_batch.to(CONFIG["DEVICE"])
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            predicted = (outputs > 0.5).float()
            train_total += y_batch.size(0)
            train_correct += (predicted == y_batch).sum().item()
        
        # Test phase
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            X_test_batch = X_test_tensor.to(CONFIG["DEVICE"])
            y_test_batch = y_test_tensor.to(CONFIG["DEVICE"])
            
            test_outputs = model(X_test_batch)
            test_loss = criterion(test_outputs, y_test_batch).item()
            
            test_predicted = (test_outputs > 0.5).float()
            test_total = y_test_batch.size(0)
            test_correct = (test_predicted == y_test_batch).sum().item()
        
        # Calculate metrics
        avg_train_loss = train_loss / len(train_dataset)
        train_acc = train_correct / train_total
        test_acc = test_correct / test_total
        
        # Update scheduler
        scheduler.step(test_loss)
        
        # Store history
        history['train_loss'].append(avg_train_loss)
        history['test_loss'].append(test_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Early stopping
        if test_loss < best_test_loss - CONFIG["MIN_DELTA"]:
            best_test_loss = test_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["EARLY_STOPPING_PATIENCE"]:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Log progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{CONFIG['EPOCHS']}: "
                       f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                       f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return model, history

def evaluate_model(model, X_test, y_test, save_dir, target_name, logger):
    """Evaluate model and create visualizations."""
    
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(CONFIG["DEVICE"])
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
        
        # Get predictions and probabilities
        outputs = model(X_test_tensor).cpu()
        y_pred_proba = outputs.numpy()
        y_pred = (outputs > 0.5).float().numpy()
        y_true = y_test_tensor.numpy()
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    
    try:
        auc = roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        auc = 0.5  # If only one class present
        logger.warning("Only one class present in test set, setting AUC to 0.5")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auc': float(auc),
        'confusion_matrix': cm.tolist()
    }
    
    logger.info(f"Test Metrics for {target_name}:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1: {f1:.4f}")
    logger.info(f"  AUC: {auc:.4f}")
    
    # Create visualizations
    create_roc_curve(y_true, y_pred_proba, save_dir, auc)
    create_confusion_matrix_plot(cm, save_dir)
    
    return metrics

def create_roc_curve(y_true, y_pred_proba, save_dir, auc):
    """Create and save ROC curve."""
    plt.figure(figsize=(8, 6))
    
    try:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Hallucination Detection')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
    except ValueError:
        plt.text(0.5, 0.5, 'ROC curve not available\n(single class in test set)', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('ROC Curve - Not Available')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_confusion_matrix_plot(cm, save_dir):
    """Create and save confusion matrix plot."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Hallucination', 'Hallucination'],
                yticklabels=['No Hallucination', 'Hallucination'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

def save_model_and_results(model, hyperparams, history, metrics, target_stats, 
                          embedding_type, target_name, input_dim):
    """Save model, hyperparameters, and results."""
    
    save_dir = os.path.join(CONFIG["MODELS_INFO_DIR"], embedding_type, target_name)
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model
    model_path = os.path.join(save_dir, 'model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'hyperparams': hyperparams,
        'input_dim': input_dim,
        'timestamp': timestamp
    }, model_path)
    
    # Save comprehensive results
    results = {
        'embedding_type': embedding_type,
        'target_name': target_name,
        'target_description': CONFIG["BINARY_TARGETS"][target_name]["description"],
        'timestamp': timestamp,
        'hyperparams': hyperparams,
        'input_dim': input_dim,
        'target_statistics': target_stats,
        'test_metrics': metrics,
        'training_history': history,
        'config': CONFIG["BINARY_TARGETS"][target_name]
    }
    
    results_path = os.path.join(save_dir, 'metrics.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return model_path, results_path

def train_with_hyperparameter_search(X_train, y_train, X_test, y_test,
                                   embedding_type, target_name, input_dim,
                                   target_stats, logger):
    """Train model with hyperparameter search."""
    
    # Hyperparameter combinations
    hyperparam_combinations = [
        dict(zip(CONFIG["HYPERPARAMS"].keys(), v)) 
        for v in itertools.product(*CONFIG["HYPERPARAMS"].values())
    ]
    
    logger.info(f"Starting hyperparameter search: {len(hyperparam_combinations)} combinations")
    
    best_auc = 0
    best_model = None
    best_hyperparams = None
    best_history = None
    best_metrics = None
    
    for i, hyperparams in enumerate(hyperparam_combinations, 1):
        logger.info(f"Trying combination {i}/{len(hyperparam_combinations)}: {hyperparams}")
        
        try:
            model, history = train_model(X_train, y_train, X_test, y_test, 
                                       hyperparams, input_dim, logger)
            
            # Evaluate model
            save_dir = os.path.join(CONFIG["MODELS_INFO_DIR"], embedding_type, target_name)
            metrics = evaluate_model(model, X_test, y_test, save_dir, target_name, logger)
           

           # Check if this is the best model
            if metrics['auc'] > best_auc:
               best_auc = metrics['auc']
               best_model = model
               best_hyperparams = hyperparams
               best_history = history
               best_metrics = metrics
               logger.info(f"New best AUC: {best_auc:.4f}")
           
        except Exception as e:
           logger.error(f"Error with hyperparams {hyperparams}: {e}")
           continue
   
    if best_model is None:
        logger.error("No successful training runs!")
        return False
   
   # Save best model and results
    model_path, results_path = save_model_and_results(
        best_model, best_hyperparams, best_history, best_metrics,
        target_stats, embedding_type, target_name, input_dim
    )
   
    logger.info("="*50)
    logger.info("TRAINING COMPLETED SUCCESSFULLY")
    logger.info(f"Best AUC: {best_auc:.4f}")
    logger.info(f"Best hyperparams: {best_hyperparams}")
    logger.info(f"Model saved: {model_path}")
    logger.info(f"Results saved: {results_path}")
    logger.info("="*50)
    
    return True

# ==========================
# OPTIMIZED TRAINING FUNCTIONS
# ==========================

def train_embedding_with_all_targets(embedding_type, dataset_path):
    """Train all target metrics for a single embedding type."""
    
    print(f"\n{'='*80}")
    print(f"🎯 Processing embedding type: {embedding_type}")
    print(f"{'='*80}")
    
    # Load embeddings ONCE for this embedding type
    embeddings, df = embedding_cache.get_embeddings(dataset_path)
    input_dim = embeddings.shape[1]
    
    print(f"   ✅ Loaded {len(df)} samples with {input_dim}D embeddings")
    
    # Create all binary targets ONCE
    targets, target_stats = create_binary_targets(df)
    
    # Print overall target statistics
    print("\n   🎯 OVERALL TARGET STATISTICS:")
    print("   " + "=" * 50)
    for tname, stats in target_stats.items():
        print(f"   {tname}:")
        print(f"      Total samples: {stats['positive_samples'] + stats['negative_samples']:,}")
        print(f"      Class 0: {stats['negative_samples']:,} ({(1-stats['positive_ratio'])*100:.1f}%)")
        print(f"      Class 1: {stats['positive_samples']:,} ({stats['positive_ratio']*100:.1f}%)")
    print("   " + "=" * 50)
    
    # Create/load splits ONCE
    splits = create_dataset_splits(embedding_type, df)
    
    # Get train/test/val data ONCE
    X_train, targets_train = get_split_data(df, embeddings, targets, splits, 'train')
    X_test, targets_test = get_split_data(df, embeddings, targets, splits, 'test')
    
    print(f"\n   📊 Data splits created:")
    print(f"      Train: {len(X_train)} samples")
    print(f"      Test: {len(X_test)} samples")
    
    # Now train models for each target
    successful_targets = 0
    failed_targets = 0
    
    for target_name in CONFIG["BINARY_TARGETS"].keys():
        if target_name not in targets:
            print(f"\n   ⚠️ Target {target_name} not available in dataset")
            failed_targets += 1
            continue
        
        print(f"\n   {'='*60}")
        print(f"   🔨 Training: {embedding_type} -> {target_name}")
        print(f"   {'='*60}")
        
        # Setup logging for this specific combination
        logger, log_file = setup_logging(embedding_type, target_name)
        
        logger.info(f"Target Description: {CONFIG['BINARY_TARGETS'][target_name]['description']}")
        
        try:
            # Extract target data
            y_train = targets_train[target_name]
            y_test = targets_test[target_name]
            
            # Print split statistics
            print_split_statistics(X_train, X_test, y_train, y_test, target_name, logger)
            
            # Train with hyperparameter search
            success = train_with_hyperparameter_search(
                X_train, y_train, X_test, y_test,
                embedding_type, target_name, input_dim,
                target_stats[target_name], logger
            )
            
            if success:
                successful_targets += 1
                print(f"   ✅ Success: {target_name}")
            else:
                failed_targets += 1
                print(f"   ❌ Failed: {target_name}")
                
        except Exception as e:
            logger.error(f"Fatal error training {target_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            failed_targets += 1
            print(f"   ❌ Failed: {target_name} - {str(e)}")
    
    print(f"\n   📊 Summary for {embedding_type}:")
    print(f"      ✅ Successful: {successful_targets}/{len(CONFIG['BINARY_TARGETS'])}")
    print(f"      ❌ Failed: {failed_targets}/{len(CONFIG['BINARY_TARGETS'])}")
    
    return successful_targets, failed_targets

# ==========================
# MAIN FUNCTION
# ==========================

def main():
    """Main training pipeline with optimized embedding loading."""
    print("🚀 Starting Binary Hallucination Detection Probe Training")
    print("="*80)
    print("📋 OPTIMIZATION: Loading embeddings ONCE per embedding type")
    print("="*80)
    
    # Setup
    setup_directories()
    
    # Discover datasets
    datasets = discover_embedding_datasets()
    
    if not datasets:
        print(f"❌ No valid datasets found in {CONFIG['DATASETS_DIR']}")
        return
    
    print(f"\n📁 Found {len(datasets)} embedding datasets")
    print(f"🎯 Will train {len(CONFIG['BINARY_TARGETS'])} binary classifiers per embedding")
    print(f"📊 Total models to train: {len(datasets) * len(CONFIG['BINARY_TARGETS'])}")
    print(f"💻 Using device: {CONFIG['DEVICE']}")
    print(f"⚡ Optimization enabled: Parse embeddings once per type")
    
    # Process each embedding type
    total_successful = 0
    total_failed = 0
    
    start_time = datetime.now()
    
    for idx, (embedding_type, dataset_path) in enumerate(datasets.items(), 1):
        print(f"\n{'='*80}")
        print(f"📦 Processing embedding {idx}/{len(datasets)}: {embedding_type}")
        print(f"{'='*80}")
        
        embedding_start_time = datetime.now()
        
        successful, failed = train_embedding_with_all_targets(embedding_type, dataset_path)
        total_successful += successful
        total_failed += failed
        
        embedding_time = (datetime.now() - embedding_start_time).total_seconds()
        print(f"\n⏱️  Time for {embedding_type}: {embedding_time:.1f} seconds")
        
        # Optional: Clear cache between embedding types if memory is limited
        # embedding_cache.clear()
    
    total_time = (datetime.now() - start_time).total_seconds()
    
    # Final summary
    print(f"\n{'='*80}")
    print("🎉 TRAINING PIPELINE COMPLETED")
    print(f"{'='*80}")
    print(f"✅ Total successful trainings: {total_successful}")
    print(f"❌ Total failed trainings: {total_failed}")
    print(f"⏱️  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"📁 Models saved in: {CONFIG['MODELS_INFO_DIR']}")
    print(f"📋 Dataset splits saved in: {CONFIG['DATASET_SPLITS_DIR']}")
    
    if total_successful > 0:
        avg_time_per_model = total_time / (total_successful + total_failed)
        print(f"\n📊 Performance Statistics:")
        print(f"   Average time per model: {avg_time_per_model:.1f} seconds")
        print(f"   Models trained per minute: {60/avg_time_per_model:.1f}")
        
        print(f"\n📋 Each successful model includes:")
        print(f"   • model.pt (trained model)")
        print(f"   • metrics.json (comprehensive results)")
        print(f"   • roc_curve.png (ROC-AUC visualization)")
        print(f"   • confusion_matrix.png (confusion matrix)")
        print(f"   • training_log_*.txt (detailed training log)")
    
    # Clear cache at the end
    embedding_cache.clear()

if __name__ == "__main__":
    main()