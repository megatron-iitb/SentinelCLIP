"""
Question-based semantic reasoning module.
"""
import numpy as np
from src.utils.math_utils import softmax, compute_entropy


class QuestionEngine:
    """Handle question-based semantic reasoning."""
    
    def __init__(self, question_prompts, clip_model):
        """
        Initialize question engine.
        
        Args:
            question_prompts: Dictionary of question prompts
            clip_model: CLIPModel instance
        """
        self.question_prompts = question_prompts
        self.clip_model = clip_model
        
        # Precompute text embeddings for questions
        self.question_keys = []
        all_question_texts = []
        
        for qk, (t_yes, t_no) in question_prompts.items():
            all_question_texts.extend([t_yes, t_no])
            self.question_keys.append(qk)
        
        self.question_text_embs = clip_model.encode_text(all_question_texts)
        self.Q_count = len(self.question_keys)
        
        print(f"Initialized {self.Q_count} semantic questions")
    
    def get_question_confidences(self, img_embs):
        """
        Compute question confidences for images.
        
        Args:
            img_embs: Image embeddings (N x D)
        
        Returns:
            prob_yes: Probability of "yes" answer (N x Q)
            entropy_q: Entropy per question (N x Q)
            probs_q: Full probability distribution (N x Q x 2)
        """
        sims = img_embs @ self.question_text_embs.T
        sims_q = sims.reshape(sims.shape[0], self.Q_count, 2)
        probs_q = softmax(sims_q, axis=2)
        
        prob_yes = probs_q[:, :, 0]
        entropy_q = compute_entropy(probs_q)
        
        return prob_yes, entropy_q, probs_q
    
    def get_question_index(self):
        """Return dictionary mapping question keys to indices."""
        return {self.question_keys[i]: i for i in range(self.Q_count)}
