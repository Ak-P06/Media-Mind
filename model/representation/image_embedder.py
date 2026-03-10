# image_embedder.py

import numpy as np
import torch


class ImageEmbedder:
    modality = "image"

    def __init__(self, clip_model, preprocess, device="cpu"):
        self.clip_model = clip_model
        self.preprocess = preprocess
        self.device = device
        self.embedding_dim = 512

    def embed(self, pil_image):

        with torch.no_grad():
            image_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            features = self.clip_model.encode_image(image_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
            vec = features.cpu().numpy().flatten()

        # normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        return {
            "modality": self.modality,
            "vector": vec,
            "dim": self.embedding_dim
        }