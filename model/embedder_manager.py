import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

from model.representation.image_embedder import ImageEmbedder
from model.representation.audio_embedder import AudioEmbedder
from model.representation.text_embedder import TextEmbedder
from model.representation.video_embedder import VideoRepresentation

from model.Input import HandleInput
from model.preprocess import Preprocessor

import clip

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP
clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)

text_embedder = TextEmbedder(
    clip_model=clip_model,
    clip_tokenizer=clip.tokenize,
    device=DEVICE
)

image_embedder = ImageEmbedder(
    clip_model=clip_model,
    preprocess=preprocess,
    device=DEVICE
)

audio_embedder = AudioEmbedder(device=DEVICE)

video_embedder = VideoRepresentation(
    image_embedder=image_embedder,
    audio_embedder=audio_embedder
)

embedding_progress = {"current": 0, "total": 1, "status": "idle"}


class EmbedManager:

    def __init__(self):
        self.image_embedder = image_embedder
        self.text_embedder = text_embedder
        self.audio_embedder = audio_embedder
        self.video_embedder = video_embedder
        self.handler = HandleInput()
        self.preprocessor = Preprocessor()

    def embed_image_batch(self, image_paths):
        """
        Batch embed images using CLIP
        """
        images = []

        for p in image_paths:
            try:
                img = Image.open(p).convert("RGB")
                img = self.image_embedder.preprocess(img)
                images.append(img)
            except Exception as e:
                print("Skipping image:", p, e)
                images.append(None)

        valid_indices = [i for i, v in enumerate(images) if v is not None]

        if not valid_indices:
            return {p: None for p in image_paths}

        batch = torch.stack([images[i] for i in valid_indices]).to(DEVICE)

        with torch.no_grad():
            features = self.image_embedder.clip_model.encode_image(batch)

        features = features / features.norm(dim=-1, keepdim=True)
        features = features.cpu().numpy()

        results = {}

        idx = 0
        for i, path in enumerate(image_paths):
            if i in valid_indices:
                results[path] = features[idx]
                idx += 1
            else:
                results[path] = None

        return results

    def get_embedding(self, media_type, data):

        if media_type == "text":
            return self.text_embedder.embed(data)

        elif media_type == "image":
            from PIL import Image

            try:
                img = Image.open(data).convert("RGB")
                emb = self.image_embedder.embed(img)

                if emb is None:
                    return None

                return emb["vector"]

            except Exception as e:
                print("Image embedding failed:", e)
                return None

        elif media_type == "audio":
            return self.audio_embedder.embed(data)

        elif media_type == "video":

            video_data = self.handler.video_input(data)

            if video_data is None:
                return None

            processed = self.preprocessor.preprocess_video(video_data)

            if processed is None:
                return None

            emb = self.video_embedder.embed(processed)

            if emb is None:
                return None

            return emb["vector"]

        else:
            raise ValueError("Unsupported media type")

    def embed_files(self, files, show_progress=True, batch_size=16):

        embeddings = {}
        total = len(files)

        embedding_progress["current"] = 0
        embedding_progress["total"] = total
        embedding_progress["status"] = "embedding"

        iterator = tqdm(files, desc="Embedding files") if show_progress else files

        image_batch = []
        image_paths = []

        for file_data in iterator:

            path = file_data["path"]
            media_type = file_data["type"]

            try:

                if media_type == "image":

                    image_batch.append(path)
                    image_paths.append(path)

                    if len(image_batch) >= batch_size:

                        batch_results = self.embed_image_batch(image_batch)

                        embeddings.update(batch_results)

                        image_batch = []
                        image_paths = []

                else:

                    emb = self.get_embedding(media_type, path)

                    embeddings[path] = emb

            except Exception as e:

                print(f"Skipping {path}: {e}")
                embeddings[path] = None

            embedding_progress["current"] += 1

        # process remaining image batch
        if image_batch:

            batch_results = self.embed_image_batch(image_batch)

            embeddings.update(batch_results)

        embedding_progress["status"] = "done"

        return embeddings

    def compute_similarity_matrix(self, embeddings):

        file_list = [f for f, v in embeddings.items() if v is not None]

        if not file_list:
            return None, []

        matrix = np.stack([embeddings[f] for f in file_list])

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)

        matrix_normed = matrix / (norms + 1e-8)

        sim_matrix = matrix_normed @ matrix_normed.T

        return sim_matrix, file_list