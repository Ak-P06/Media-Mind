# audio_embedder.py

import torch
import laion_clap
import librosa
import numpy as np


class AudioEmbedder:
    modality = "audio"

    def __init__(self, device="cpu"):
        self.device = device

        self.clap_model = laion_clap.CLAP_Module(enable_fusion=False)
        self.clap_model.load_ckpt()
        self.clap_model.to(device)
        self.clap_model.eval()

        self.embedding_dim = 512

    def embed(self, audio_path: str):

        audio, sr = librosa.load(audio_path, sr=48000, mono=True)

        # force 10 seconds
        target_len = sr * 10
        if len(audio) > target_len:
            audio = audio[:target_len]
        else:
            audio = np.pad(audio, (0, target_len - len(audio)))
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            embed = self.clap_model.get_audio_embedding_from_data(
                x=audio_tensor,
                use_tensor=True
            )

        vec = embed.squeeze().cpu().numpy()

        # normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        return {
            "modality": self.modality,
            "vector": vec,
            "dim": self.embedding_dim
        }