# Video embedder

import numpy as np,os

class VideoRepresentation:

    def __init__(self, image_embedder, audio_embedder, embedding_dim=512):
        self.image_embedder = image_embedder
        self.audio_embedder = audio_embedder
        self.embedding_dim = embedding_dim

    def embed(self, video_preprocessed):
        embedding_dim = self.embedding_dim

        # ---- FRAME EMBEDDINGS ----
        frame_vectors = []
        for frame in video_preprocessed.get("frames", []):
            pil_image = frame.get("frame")
            if pil_image is not None:
                emb = self.image_embedder.embed(pil_image)
                vec = emb["vector"]
                if len(vec) < embedding_dim:
                    vec = np.pad(vec, (0, embedding_dim - len(vec)))
                frame_vectors.append(vec)

        visual_vec = np.mean(frame_vectors, axis=0) if frame_vectors else np.zeros(embedding_dim)

        # ---- AUDIO EMBEDDING ----
        audio_vec = np.zeros(embedding_dim)
        audio_data = video_preprocessed.get("audio")
        if audio_data and "waveform" in audio_data:
            # use your audio embedder on temporary waveform
            try:
                # Save waveform temporarily
                temp_path = "temp_video_audio.wav"
                import soundfile as sf
                sf.write(temp_path, audio_data["waveform"], audio_data.get("sample_rate", 16000))
                emb = self.audio_embedder.embed(temp_path)
                vec = emb["vector"]
                if len(vec) < embedding_dim:
                    vec = np.pad(vec, (0, embedding_dim - len(vec)))
                audio_vec = vec
                os.remove(temp_path)
            except:
                audio_vec = np.zeros(embedding_dim)

        # ---- COMBINED ----
        combined = (visual_vec + audio_vec) / 2

        # ensure final vector is always 512-dim
        combined = np.array(combined, dtype=np.float32).flatten()
        if len(combined) < embedding_dim:
            combined = np.pad(combined, (0, embedding_dim - len(combined)))
        elif len(combined) > embedding_dim:
            combined = combined[:embedding_dim]

        return {"modality": "video", "vector": combined, "dim": embedding_dim}