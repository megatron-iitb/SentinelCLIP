"""
CLIP model wrapper for loading and inference.
"""
import torch
import torch.nn.functional as F
import open_clip
from pathlib import Path
import os


class CLIPModel:
    """Wrapper for CLIP model with caching support."""
    
    def __init__(self, model_name="ViT-B-32", pretrained="openai", device="cuda", cache_dir=None):
        """
        Initialize CLIP model.
        
        Args:
            model_name: CLIP model architecture
            pretrained: Pretrained weights tag
            device: Device to load model on
            cache_dir: Cache directory for offline mode
        """
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        
        print(f"Loading CLIP model: {model_name} ({pretrained})")
        print(f"Using cache directory: {cache_dir}")
        print(f"Offline mode: HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE', 'not set')}")
        
        try:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained,
                cache_dir=str(cache_dir)
            )
            print("✓ CLIP model loaded successfully from cache")
        except Exception as e:
            print(f"Error loading CLIP model: {e}")
            print("Attempting to load without pretrained weights...")
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=None
            )
            print("⚠ Warning: Using model without pretrained weights")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
    
    @torch.no_grad()
    def encode_images(self, images):
        """
        Encode PIL images to normalized embeddings.
        
        Args:
            images: List of PIL images
        
        Returns:
            Normalized image embeddings (numpy array)
        """
        if not isinstance(images, list):
            images = [images]
        
        img_tensors = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        img_emb = self.model.encode_image(img_tensors)
        img_emb = F.normalize(img_emb, dim=-1)
        return img_emb.cpu().numpy()
    
    @torch.no_grad()
    def encode_text(self, texts):
        """
        Encode text prompts to normalized embeddings.
        
        Args:
            texts: List of text strings
        
        Returns:
            Normalized text embeddings (numpy array)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        text_tokens = self.tokenizer(texts).to(self.device)
        txt_emb = self.model.encode_text(text_tokens)
        txt_emb = F.normalize(txt_emb, dim=-1)
        return txt_emb.cpu().numpy()
    
    def compute_similarities(self, image_embeddings, text_embeddings):
        """
        Compute cosine similarity between image and text embeddings.
        
        Args:
            image_embeddings: Image embeddings (N x D)
            text_embeddings: Text embeddings (M x D)
        
        Returns:
            Similarity matrix (N x M)
        """
        return image_embeddings @ text_embeddings.T
