# text_embedder.py

import numpy as np
import torch


class TextEmbedder:
    modality = "text"

    def __init__(self, clip_model, clip_tokenizer, device="cpu"):
        self.device = device
        self.clip_model = clip_model
        self.clip_tokenizer = clip_tokenizer
        self.embedding_dim = 512

    def embed(self, text: str):

        if not text.strip():
            vec = np.zeros(self.embedding_dim)
        else:
            with torch.no_grad():
                tokens = self.clip_tokenizer([text]).to(self.device)
                features = self.clip_model.encode_text(tokens)
                features = features / features.norm(dim=-1, keepdim=True)
                vec = features.cpu().numpy().flatten()

        # normalize (important for clustering)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        return {
            "modality": self.modality,
            "vector": vec,
            "dim": self.embedding_dim
        }