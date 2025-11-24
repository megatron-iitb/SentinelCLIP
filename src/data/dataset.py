"""
Data loading and preprocessing utilities.
"""
import numpy as np
from PIL import Image, ImageOps
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch
from tqdm.auto import tqdm


class DatasetLoader:
    """Handle dataset loading and preprocessing."""
    
    def __init__(self, dataset_name="CIFAR10", root="./data", 
                 batch_size=256, num_workers=2, val_split=0.1, seed=42):
        """
        Initialize dataset loader.
        
        Args:
            dataset_name: Name of dataset (currently supports CIFAR10)
            root: Root directory for data storage
            batch_size: Batch size for dataloaders
            num_workers: Number of worker processes
            val_split: Fraction of training data for validation
            seed: Random seed for reproducibility
        """
        self.dataset_name = dataset_name
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.seed = seed
        
        self._load_dataset()
    
    def _load_dataset(self):
        """Load and split dataset."""
        if self.dataset_name == "CIFAR10":
            # Load CIFAR-10
            cifar_train = datasets.CIFAR10(root=self.root, train=True, download=True)
            cifar_test = datasets.CIFAR10(root=self.root, train=False, download=True)
            self.labels = cifar_train.classes
            
            # Create train/val split
            full_train = datasets.CIFAR10(
                root=self.root, train=True, download=False, 
                transform=transforms.ToTensor()
            )
            train_size = int((1 - self.val_split) * len(full_train))
            val_size = len(full_train) - train_size
            
            train_ds, val_ds = random_split(
                full_train, [train_size, val_size],
                generator=torch.Generator().manual_seed(self.seed)
            )
            
            test_ds = datasets.CIFAR10(
                root=self.root, train=False, download=False,
                transform=transforms.ToTensor()
            )
            
            # Create dataloaders
            self.train_loader = DataLoader(
                train_ds, batch_size=self.batch_size, 
                shuffle=True, num_workers=self.num_workers
            )
            self.val_loader = DataLoader(
                val_ds, batch_size=self.batch_size,
                shuffle=False, num_workers=self.num_workers
            )
            self.test_loader = DataLoader(
                test_ds, batch_size=self.batch_size,
                shuffle=False, num_workers=self.num_workers
            )
            
            print(f"Labels: {self.labels}")
            print(f"Train size: {train_size}, Val size: {val_size}, Test size: {len(test_ds)}")
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported")
    
    @staticmethod
    def tensor_to_pil(t):
        """Convert tensor to PIL image."""
        arr = t.numpy().transpose(1, 2, 0)
        arr = (arr * 255).astype('uint8')
        return Image.fromarray(arr)
    
    def extract_embeddings(self, dataloader, clip_model, max_items=None):
        """
        Extract CLIP embeddings for entire dataset.
        
        Args:
            dataloader: PyTorch dataloader
            clip_model: CLIPModel instance
            max_items: Maximum number of items to process (None for all)
        
        Returns:
            images (list), labels (numpy array), embeddings (numpy array)
        """
        imgs = []
        labels_list = []
        
        for x, y in dataloader:
            for i in range(x.shape[0]):
                imgs.append(self.tensor_to_pil(x[i]))
                labels_list.append(int(y[i]))
                if max_items is not None and len(imgs) >= max_items:
                    break
            if max_items is not None and len(imgs) >= max_items:
                break
        
        # Batch process embeddings
        batch_size = 128
        emb_list = []
        for i in tqdm(range(0, len(imgs), batch_size), desc="Computing embeddings"):
            batch = imgs[i:i+batch_size]
            emb = clip_model.encode_images(batch)
            emb_list.append(emb)
        
        embeddings = np.vstack(emb_list)
        return imgs, np.array(labels_list), embeddings


def create_augmentations(img_pil, scales=[0.9, 0.8]):
    """
    Create augmented versions of a PIL image.
    
    Args:
        img_pil: PIL image
        scales: List of crop scales
    
    Returns:
        List of augmented PIL images
    """
    crops = []
    crops.append(img_pil)  # Original
    crops.append(ImageOps.mirror(img_pil))  # Horizontal flip
    
    w, h = img_pil.size
    for f in scales:
        new_w, new_h = int(w * f), int(h * f)
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        cropped = img_pil.crop((left, top, left + new_w, top + new_h))
        crops.append(cropped.resize((w, h)))
    
    return crops
