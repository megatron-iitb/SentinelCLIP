"""
Ensemble uncertainty estimation using augmentations.
"""
import numpy as np
from tqdm.auto import tqdm
from src.utils.math_utils import compute_entropy
from src.data.dataset import create_augmentations


def compute_augmentation_ensemble(images, clip_model, class_text_embs, temperature, 
                                  augmentation_scales=[0.9, 0.8]):
    """
    Compute ensemble predictions using augmentations.
    
    Args:
        images: List of PIL images
        clip_model: CLIPModel instance
        class_text_embs: Text embeddings for classes
        temperature: Temperature for scaling
        augmentation_scales: Scales for center crops
    
    Returns:
        aug_probs_list: List of probability arrays (one per image)
        aug_mean_probs: Mean probabilities (N x K)
        ensemble_conf_std: Std-based confidence (N,)
    """
    from src.utils.math_utils import softmax
    
    aug_probs_list = []
    
    print("Computing augmentation ensemble stats...")
    for idx in tqdm(range(len(images)), desc="Processing augmentations"):
        aug_imgs = create_augmentations(images[idx], scales=augmentation_scales)
        emb = clip_model.encode_images(aug_imgs)
        sims = emb @ class_text_embs.T
        probs = softmax(sims / temperature, axis=1)
        aug_probs_list.append(probs)
    
    # Compute statistics
    aug_mean_probs = np.stack([p.mean(axis=0) for p in aug_probs_list], axis=0)
    
    # Std-based confidence: 1 - std(chosen class)
    per_sample_std = []
    for p in aug_probs_list:
        mean_probs = p.mean(axis=0)
        chosen = mean_probs.argmax()
        per_sample_std.append(p[:, chosen].std())
    per_sample_std = np.array(per_sample_std)
    ensemble_conf_std = 1.0 - per_sample_std
    
    return aug_probs_list, aug_mean_probs, ensemble_conf_std


def compute_entropy_ensemble_conf(aug_probs_list):
    """
    Compute ensemble confidence using predictive vs expected entropy (MI proxy).
    
    Args:
        aug_probs_list: List of probability arrays from augmentations
    
    Returns:
        conf: Ensemble confidence scores (N,)
        mi_proxy: Mutual information proxy values (N,)
    """
    entropies_pred = []
    entropies_exp = []
    
    for p in aug_probs_list:
        # Mean probability over augmentations
        mean_p = p.mean(axis=0)
        
        # Predictive entropy: H(mean distribution)
        H_pred = compute_entropy(mean_p[np.newaxis, :])[0]
        
        # Expected entropy: mean of individual entropies
        H_exp = compute_entropy(p).mean()
        
        entropies_pred.append(H_pred)
        entropies_exp.append(H_exp)
    
    entropies_pred = np.array(entropies_pred)
    entropies_exp = np.array(entropies_exp)
    
    # Mutual information proxy (measures disagreement)
    mi_proxy = entropies_pred - entropies_exp
    
    # Normalize to [0, 1] and invert (high MI -> low confidence)
    max_ent = np.log(aug_probs_list[0].shape[1])  # log(num_classes)
    mi_normalized = np.clip(mi_proxy / max_ent, 0.0, 1.0)
    
    # Confidence = 1 - normalized_disagreement
    conf = 1.0 - mi_normalized
    
    return conf, mi_proxy


def combine_ensemble_confidences(conf_std, conf_entropy, strategy="combined"):
    """
    Combine different ensemble confidence measures.
    
    Args:
        conf_std: Std-based confidence
        conf_entropy: Entropy-based confidence
        strategy: Combination strategy ("std", "entropy", "combined", "geometric")
    
    Returns:
        Combined ensemble confidence
    """
    if strategy == "std":
        return conf_std
    elif strategy == "entropy":
        return conf_entropy
    elif strategy == "combined":
        return 0.5 * conf_std + 0.5 * conf_entropy
    elif strategy == "geometric":
        return np.sqrt(conf_std * conf_entropy)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
