"""
Data preparation pipeline module.
Handles model loading, dataset loading, and embedding extraction.
"""

import torch
import numpy as np

from src.models.clip_model import CLIPModel
from src.data.dataset import DatasetLoader


def setup_model_and_data(config, device):
    """
    Load CLIP model and dataset, extract embeddings.
    
    Args:
        config: Configuration object/module
        device: torch.device
        
    Returns:
        dict containing model, loaders, embeddings, labels, etc.
    """
    print("\n" + "=" * 70)
    print("STEP 1-3: MODEL & DATA PREPARATION")
    print("=" * 70)
    
    # 1. Load CLIP Model
    print(f"\nDevice: {device}")
    print(f"Loading CLIP model: {config.MODEL_NAME} ({config.MODEL_PRETRAINED})")
    
    clip_model = CLIPModel(
        model_name=config.MODEL_NAME,
        pretrained=config.MODEL_PRETRAINED,
        device=device,
        cache_dir=config.CACHE_DIR
    )
    print("✓ Model loaded successfully")
    
    # 2. Load Dataset
    print(f"\nLoading dataset: {config.DATASET_NAME}")
    dataset_loader = DatasetLoader(
        dataset_name=config.DATASET_NAME,
        root=str(config.DATA_DIR),
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        val_split=config.VAL_SPLIT,
        seed=config.RANDOM_SEED
    )
    
    labels = dataset_loader.labels
    print(f"✓ Dataset loaded: {len(labels)} classes")
    print(f"  Classes: {labels}")
    
    # Precompute class text embeddings
    class_prompts = [f"A photo of a {lab}" for lab in labels]
    class_text_embs = clip_model.encode_text(class_prompts)
    print(f"✓ Class text embeddings computed: {class_text_embs.shape}")
    
    # 3. Extract Embeddings
    print("\n--- Extracting embeddings ---")
    print("Computing validation embeddings...")
    val_imgs, val_labels_arr, val_embs = dataset_loader.extract_embeddings(
        dataset_loader.val_loader, clip_model
    )
    print(f"✓ Val embeddings: {val_embs.shape}")
    
    print("Computing test embeddings...")
    test_imgs, test_labels_arr, test_embs = dataset_loader.extract_embeddings(
        dataset_loader.test_loader, clip_model
    )
    print(f"✓ Test embeddings: {test_embs.shape}")
    
    return {
        'clip_model': clip_model,
        'dataset_loader': dataset_loader,
        'labels': labels,
        'class_text_embs': class_text_embs,
        'val_imgs': val_imgs,
        'val_labels': val_labels_arr,
        'val_embs': val_embs,
        'test_imgs': test_imgs,
        'test_labels': test_labels_arr,
        'test_embs': test_embs
    }
